"""LLM-mediated conversation with EBM/Laya-only diagnosis and stopping.

The LLM extracts facts, phrases follow-up questions, and explains a fixed final
decision. It never selects a diagnosis, escalation category, or stopping point.
"""

import argparse
import asyncio
import json
from pathlib import Path

import joblib

from diagnostic_model.live_conversation import (
    EBMLivePredictor,
    EvidenceState,
    GREETING,
    LayaLivePredictor,
    MAX_PATIENT_ANSWERS,
    emergency_findings,
)
from diagnostic_model.stopping import stopping_probabilities
from llm_service.app.explainer import explain
from llm_service.app.extractor import extract_symptoms
from llm_service.app.llm_client import LLMClient
from llm_service.app.merge import merge_profile
from llm_service.app.questioner import phrase_context_question, phrase_question
from llm_service.app.schema import ConditionEstimate, DecisionRecord, Profile
from llm_service.app.vocab import get_vocab


def apply_profile_to_evidence(profile: Profile, evidence: EvidenceState) -> EvidenceState:
    """Translate LLM-extracted facts into the leakage-safe model evidence state."""
    for symptom_id, state in profile.symptoms.items():
        if state.presence in {"present", "absent"}:
            evidence.symptoms[symptom_id] = state.presence == "present"
        if state.presence != "present":
            continue
        if state.duration_days is not None:
            evidence.duration_days = state.duration_days
        if state.severity is not None:
            evidence.severity = "normal" if state.severity == "moderate" else state.severity
        if state.progression is not None:
            evidence.progression = state.progression
    return evidence


def select_question_target(profile: Profile, evidence: EvidenceState, answer_number: int):
    """Deterministically select what to ask; the LLM controls wording only."""
    present = [item for item in profile.symptoms.values() if item.presence == "present"]
    if not present:
        return "context", "presenting_symptoms", None

    focus_order = {"duration": 0, "severity": 1, "progression": 2, "presence": 3}
    missing = []
    for item in profile.missing_info:
        if ":" not in item:
            continue
        symptom_id, focus = item.rsplit(":", 1)
        if get_vocab().is_valid(symptom_id):
            missing.append((focus_order.get(focus, 9), symptom_id, focus))
    if missing:
        _, symptom_id, focus = min(missing)
        return "symptom", symptom_id, focus

    if answer_number >= 4 and (
        evidence.age is None or evidence.temperature_c is None or evidence.sex is None
    ):
        return "context", "demographics", None

    entries = get_vocab()._entries
    unasked = [sid for sid, item in entries.items()
               if sid not in profile.symptoms and item.get("is_red_flag")]
    if not unasked:
        unasked = [sid for sid in entries if sid not in profile.symptoms]
    if unasked:
        return "symptom", unasked[0], "presence"
    return "context", "medical_history", None


class LLMDrivenConsultation:
    """One live consultation with strict language/decision separation."""

    def __init__(
        self,
        predictor,
        stopping,
        llm,
        minimum_answers=2,
        safety_profile="none",
    ):
        if safety_profile not in {"none", "sg_moh"}:
            raise ValueError(f"Unknown safety profile: {safety_profile}")
        bundle = joblib.load(stopping) if isinstance(stopping, (str, Path)) else stopping
        self.predictor = predictor
        self.stop_model = bundle["model"]
        self.stop_threshold = float(bundle["threshold"])
        self.llm = llm
        self.minimum_answers = minimum_answers
        self.safety_profile = safety_profile
        self.profile = Profile()
        self.evidence = EvidenceState()
        self.turns = [{"role": "doctor", "content": GREETING}]
        self.trace = []
        self.finished = False

    async def receive(self, patient_message: str):
        if self.finished:
            raise RuntimeError("The consultation has already stopped")
        extraction = await extract_symptoms(self.llm, patient_message, self.profile)
        self.profile = merge_profile(self.profile, extraction)
        self.evidence.update(patient_message)
        apply_profile_to_evidence(self.profile, self.evidence)
        self.turns.append({"role": "patient", "content": patient_message})
        answer_number = sum(turn["role"] == "patient" for turn in self.turns)

        answers = self.predictor.predict(self.turns, self.evidence, answer_number)
        checkpoint = {
            "patient_answer": answer_number,
            "latest_response": patient_message,
            "extraction": extraction.model_dump(),
            "profile": self.profile.model_dump(),
            "answers": answers,
            "diagnostic_backend": self.predictor.name,
        }
        self.trace.append(checkpoint)
        stop_probability = stopping_probabilities(self.stop_model, [self.trace])[0][-1]
        model_stop = answer_number >= MAX_PATIENT_ANSWERS or (
            answer_number >= self.minimum_answers and stop_probability >= self.stop_threshold
        )
        findings = (emergency_findings(patient_message)
                    if self.safety_profile == "sg_moh" else [])
        safety_override = bool(findings)
        stop = safety_override or model_stop
        checkpoint.update({
            "stop_probability": round(float(stop_probability), 4),
            "stop_threshold": self.stop_threshold,
            "model_stop": bool(model_stop),
            "safety_profile": self.safety_profile,
            "safety_override_applied": safety_override,
            "safety_findings": findings,
            "stop": stop,
        })

        if stop:
            self.finished = True
            response = await self._final_response(checkpoint)
            checkpoint["question_target"] = None
            checkpoint["question_fallback_used"] = response["fallback_used"]
            checkpoint["doctor_response"] = " ".join(
                (response["headline"], response["explanation"], response["disclaimer"])
            )
        else:
            kind, target, focus = select_question_target(
                self.profile, self.evidence, answer_number
            )
            if kind == "symptom":
                question, fallback = await phrase_question(self.llm, target, focus)
            else:
                question, fallback = await phrase_context_question(self.llm, target)
            checkpoint["question_target"] = {
                "kind": kind, "target": target, "focus": focus
            }
            checkpoint["question_fallback_used"] = fallback
            checkpoint["doctor_response"] = question
            self.turns.append({"role": "doctor", "content": question})
        return checkpoint

    async def _final_response(self, checkpoint):
        condition = checkpoint["answers"]["condition"]
        escalation = checkpoint["answers"]["escalation"]
        ranked = sorted(
            condition["probabilities"].items(), key=lambda item: item[1], reverse=True
        )[:3]
        record = DecisionRecord(
            recommended_action=(
                "urgent_medical_review"
                if checkpoint["safety_override_applied"] or escalation["choice"] == "urgent"
                else "routine_teleconsultation"
            ),
            responses=[turn["content"] for turn in self.turns if turn["role"] == "patient"],
            preliminary_estimates=[
                ConditionEstimate(condition=name, score=float(score))
                for name, score in ranked
            ],
            risk_factors=checkpoint["safety_findings"],
            triggered_rules=(
                ["jurisdictional safety override"]
                if checkpoint["safety_override_applied"] else
                [f"{self.predictor.name} escalation={escalation['choice']}"]
            ),
            missing_information=self.profile.missing_info,
            limitations=[
                "Research prototype evaluated on synthetic source labels",
                "Preliminary model estimate, not a clinical diagnosis",
            ],
            sources=[self.predictor.name, "LLM symptom extraction and question phrasing"],
        )
        return (await explain(self.llm, record, "patient")).model_dump()


async def _run(args):
    llm = LLMClient()
    predictor = (EBMLivePredictor(args.model) if args.backend == "ebm"
                 else LayaLivePredictor(args.model, args.device))
    session = LLMDrivenConsultation(
        predictor, args.stopping, llm, safety_profile=args.safety_profile
    )
    print(f"Doctor: {GREETING}")
    events = []
    while not session.finished:
        message = input("Patient: ").strip()
        checkpoint = await session.receive(message)
        events.append(checkpoint)
        print(json.dumps({
            "backend": checkpoint["diagnostic_backend"],
            "diagnosis": checkpoint["answers"]["condition"],
            "escalation": checkpoint["answers"]["escalation"],
            "stop_probability": checkpoint["stop_probability"],
            "stop": checkpoint["stop"],
        }, indent=2))
        print(f"Doctor: {checkpoint['doctor_response']}")
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps({"turns": session.turns, "events": events}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("ebm", "laya"), default="ebm")
    parser.add_argument("--model", required=True)
    parser.add_argument("--stopping", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--safety-profile", choices=("none", "sg_moh"), default="none")
    parser.add_argument("--log", type=Path)
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
