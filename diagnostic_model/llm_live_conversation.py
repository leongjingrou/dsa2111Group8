"""LLM-mediated conversation with EBM/Laya-only diagnosis and stopping.

The LLM extracts facts, phrases follow-up questions, and explains a fixed final
decision. It never selects a diagnosis, escalation category, or stopping point.
"""

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import joblib

from diagnostic_model.live_conversation import (
    GREETING,
    EBMLivePredictor,
    EvidenceState,
    LayaLivePredictor,
)
from diagnostic_model.live_conversation import (
    MAX_PATIENT_ANSWERS as MAX_MODEL_ANSWERS,
)
from diagnostic_model.stopping import stopping_probabilities
from knowledge_base.kb_service import retrieve_knowledge
from llm_service.app.explainer import explain
from llm_service.app.extractor import extract_symptoms
from llm_service.app.llm_client import LLMClient
from llm_service.app.merge import merge_profile
from llm_service.app.questioner import phrase_context_question, phrase_question
from llm_service.app.schema import (
    ConditionEstimate,
    Contradiction,
    DecisionRecord,
    Profile,
)
from llm_service.app.vocab import get_vocab
from llm_service.intake import bind_answer, empty_extraction, safety_question
from safety_layer import SafetyEngine, guard_decision
from safety_layer.integration import result_from_safety

MAX_INTAKE_ANSWERS = 64


def apply_profile_to_evidence(
    profile: Profile, evidence: EvidenceState
) -> EvidenceState:
    """Translate LLM-extracted facts into the leakage-safe model evidence state."""
    for symptom_id, state in profile.symptoms.items():
        if state.presence in {"present", "absent"}:
            evidence.symptoms[symptom_id] = state.presence == "present"
        if state.presence != "present":
            continue
        if state.duration_days is not None:
            evidence.duration_days = state.duration_days
        if state.severity is not None:
            evidence.severity = (
                "normal" if state.severity == "moderate" else state.severity
            )
        if state.progression is not None:
            evidence.progression = state.progression
    return evidence


def select_question_target(
    profile: Profile, evidence: EvidenceState, answer_number: int
):
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

    if answer_number >= 4:
        if not evidence.temperature_asked:
            return "context", "temperature", None
        if not evidence.sex_asked:
            return "context", "sex", None

    entries = get_vocab()._entries
    unasked = [
        symptom_id
        for symptom_id, item in entries.items()
        if symptom_id not in profile.symptoms and item.get("is_red_flag")
    ]
    if not unasked:
        unasked = [
            symptom_id for symptom_id in entries if symptom_id not in profile.symptoms
        ]
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
        safety_profile="rules_v1",
        safety_engine=None,
    ):
        if safety_profile not in {"rules_v1", "sg_moh"}:
            raise ValueError(f"Unknown safety profile: {safety_profile}")
        bundle = (
            joblib.load(stopping) if isinstance(stopping, (str, Path)) else stopping
        )
        self.predictor = predictor
        self.stop_model = bundle["model"]
        self.stop_threshold = float(bundle["threshold"])
        self.llm = llm
        self.minimum_answers = minimum_answers
        self.safety_profile = "rules_v1"
        self.safety_engine = (
            safety_engine if safety_engine is not None else SafetyEngine()
        )
        self.profile = Profile()
        self.evidence = EvidenceState()
        self.turns = [{"role": "doctor", "content": GREETING}]
        self.trace = []
        self.finished = False
        self.pending_question = None

    def snapshot(self):
        return {
            "profile": self.profile.model_dump(),
            "evidence": asdict(self.evidence),
            "turns": self.turns,
            "trace": self.trace,
            "finished": self.finished,
            "pending_question": self.pending_question,
        }

    def restore(self, state):
        self.profile = Profile.model_validate(state["profile"])
        self.evidence = EvidenceState(**state["evidence"])
        self.turns = state["turns"]
        self.trace = state["trace"]
        self.finished = state["finished"]
        self.pending_question = state["pending_question"]

    def _checkpoint(self, patient_message, extraction):
        return {
            "patient_answer": sum(turn["role"] == "patient" for turn in self.turns),
            "latest_response": patient_message,
            "extraction": extraction.model_dump(),
            "profile": self.profile.model_dump(),
            "answers": None,
            "diagnostic_backend": self.predictor.name,
            "decision_record": None,
            "knowledge_base": None,
            "stop_probability": None,
            "stop_threshold": self.stop_threshold,
            "model_stop": False,
            "safety_profile": self.safety_profile,
        }

    def _respond_to_safety(self, checkpoint, decision):
        result = result_from_safety(
            decision, source_urls=self.safety_engine.source_urls
        )
        stop = decision.disposition != "needs_clarification"
        checkpoint.update(
            {
                "safety": decision.to_dict(),
                "safety_findings": [h.reason for h in decision.triggered_rules],
                "safety_override_applied": stop,
                "stop": stop,
                "decision_record": result.record,
                "question_fallback_used": True,
            }
        )
        if stop:
            self.finished = True
            self.pending_question = None
            checkpoint.update(
                question_target=None, doctor_response=result.patient_message
            )
        else:
            self.pending_question, question = safety_question(decision, self.profile)
            checkpoint.update(
                question_target=self.pending_question, doctor_response=question
            )
            self.turns.append({"role": "doctor", "content": question})
        self.trace.append(checkpoint)
        return checkpoint

    async def receive(self, patient_message: str):
        """Screen facts before prediction, then guard the proposed model decision."""
        if self.finished:
            raise RuntimeError("The consultation has already stopped")

        previous_question = self.pending_question
        bound_extraction = bind_answer(
            patient_message, previous_question, self.profile, self.evidence
        )
        self._record_patient_answer(patient_message)

        # Explicit emergency evidence must not wait for a remote language provider.
        emergency_decision = self._check_emergency_before_extraction()
        if emergency_decision is not None:
            checkpoint = self._checkpoint(patient_message, empty_extraction())
            return self._respond_to_safety(checkpoint, emergency_decision)

        extraction = bound_extraction
        if extraction is None:
            extraction = await extract_symptoms(
                self.llm, patient_message, self.profile, previous_question
            )
        self._merge_patient_facts(extraction, bound_extraction, previous_question)
        checkpoint = self._checkpoint(patient_message, extraction)

        screening = self._screen_before_model(checkpoint["patient_answer"])
        if not screening.can_continue:
            return self._respond_to_safety(checkpoint, screening)

        # Training used five partial evidence states. Checklist turns must not
        # create an out-of-range feature or force an incomplete safety screen.
        answer_number = min(
            1 + sum(item["answers"] is not None for item in self.trace),
            MAX_MODEL_ANSWERS,
        )
        diagnostic_evidence = EvidenceState(**asdict(self.evidence))
        # The compact safety screen can mark many emergency findings absent in
        # one answer. Those facts belong to the rule engine, but presenting all
        # of them as first-step diagnostic features is outside the sequential
        # model's training distribution and can create false referrals.
        diagnostic_evidence.symptoms = {
            symptom_id: present
            for symptom_id, present in self.evidence.symptoms.items()
            if present or symptom_id not in self.safety_engine.required_screening
        }
        model_answers = self.predictor.predict(
            self.turns, diagnostic_evidence, answer_number
        )
        checkpoint["model_answer"] = answer_number
        checkpoint["answers"] = model_answers
        proposed_record = self._build_model_record(checkpoint)
        guarded_decision = guard_decision(
            self.profile,
            proposed_record,
            candidate_condition=model_answers["condition"]["choice"],
            evidence=self.evidence,
            engine=self.safety_engine,
            # A preliminary model label must not end the consultation before
            # the learned stopping policy has enough evidence. Explicit safety
            # rules and out-of-scope predictions still take effect immediately.
            preserve_model_referral=False,
        )
        if not guarded_decision.safety.can_continue:
            checkpoint["answers"] = None
            return self._respond_to_safety(checkpoint, guarded_decision.safety)

        checkpoint.update(
            safety=guarded_decision.safety.to_dict(),
            decision_record=guarded_decision.record,
        )
        self._apply_stopping_policy(checkpoint)

        if (
            checkpoint["stop"]
            and proposed_record.recommended_action == "urgent_medical_review"
        ):
            final_decision = guard_decision(
                self.profile,
                proposed_record,
                candidate_condition=model_answers["condition"]["choice"],
                evidence=self.evidence,
                engine=self.safety_engine,
            )
            checkpoint["answers"] = None
            return self._respond_to_safety(checkpoint, final_decision.safety)

        self.trace.append(checkpoint)
        await self._explain_or_ask_followup(checkpoint)
        return checkpoint

    def _record_patient_answer(self, patient_message):
        self.evidence.update(patient_message)
        if self.profile.age is None and self.evidence.age is not None:
            self.profile.age = self.evidence.age
        if self.profile.pregnancy_status is None and self.evidence.pregnant is not None:
            self.profile.pregnancy_status = (
                "pregnant" if self.evidence.pregnant else "not_pregnant"
            )
        self.turns.append({"role": "patient", "content": patient_message})

    def _check_emergency_before_extraction(self):
        has_warning_sign = any(
            self.evidence.symptoms.get(symptom_id) is True
            for symptom_id in self.safety_engine.required_screening
        )
        if has_warning_sign:
            decision = self.safety_engine.evaluate(self.profile, evidence=self.evidence)
            if decision.disposition == "emergency":
                return decision
        return None

    def _merge_patient_facts(self, extraction, bound_extraction, previous_question):
        previous_profile = self.profile
        self.profile = merge_profile(self.profile, extraction)
        # Generic merging retains definite answers. Safety needs the current
        # uncertainty and any earlier positive finding to remain visible.
        for symptom in extraction.symptoms:
            if symptom.presence == "uncertain":
                previous_symptom = previous_profile.symptoms.get(symptom.symptom_id)
                if (
                    previous_symptom is not None
                    and previous_symptom.presence != "uncertain"
                ):
                    self.profile.contradictions.append(
                        Contradiction(
                            symptom_id=symptom.symptom_id,
                            previous=previous_symptom.presence,
                            new="uncertain",
                        )
                    )
                self.profile.symptoms[symptom.symptom_id].presence = "uncertain"
        self._resolve_confirmed_non_safety_conflict(bound_extraction, previous_question)
        apply_profile_to_evidence(self.profile, self.evidence)

    def _resolve_confirmed_non_safety_conflict(self, bound_extraction, question):
        # Only an explicit answer to the server's clarification question can
        # resolve an ordinary symptom conflict. Prior positive red flags persist.
        if bound_extraction is None or question is None:
            return
        symptom_id = question["target"]
        if symptom_id in self.safety_engine.required_screening:
            return
        is_confirmed = any(
            symptom.symptom_id == symptom_id and symptom.presence != "uncertain"
            for symptom in bound_extraction.symptoms
        )
        if is_confirmed:
            self.profile.contradictions = [
                conflict
                for conflict in self.profile.contradictions
                if conflict.symptom_id != symptom_id
            ]

    def _screen_before_model(self, answer_number):
        decision = self.safety_engine.evaluate(self.profile, evidence=self.evidence)
        if (
            decision.disposition == "needs_clarification"
            and answer_number >= MAX_INTAKE_ANSWERS
        ):
            # An unresolved intake ends with a clinician referral, never a
            # diagnostic guess based on incomplete screening.
            return self.safety_engine.evaluate(
                self.profile,
                evidence=self.evidence,
                proposed_action="in_person_assessment",
            )
        return decision

    def _apply_stopping_policy(self, checkpoint):
        model_trace = [item for item in self.trace if item["answers"] is not None]
        model_trace.append(checkpoint)
        stop_probability = stopping_probabilities(self.stop_model, [model_trace])[0][-1]
        answer_number = checkpoint["model_answer"]
        model_stop = answer_number >= MAX_MODEL_ANSWERS or (
            answer_number >= self.minimum_answers
            and stop_probability >= self.stop_threshold
        )
        checkpoint.update(
            {
                "stop_probability": round(float(stop_probability), 4),
                "stop_threshold": self.stop_threshold,
                "model_stop": bool(model_stop),
                "safety_profile": self.safety_profile,
                "safety_override_applied": False,
                "safety_findings": [],
                "stop": bool(model_stop),
            }
        )

    async def _explain_or_ask_followup(self, checkpoint):
        if checkpoint["stop"]:
            self.finished = True
            self.pending_question = None
            record = DecisionRecord.model_validate(checkpoint["decision_record"])
            explanation = (await explain(self.llm, record, "patient")).model_dump()
            checkpoint["question_target"] = None
            checkpoint["question_fallback_used"] = explanation["fallback_used"]
            checkpoint["doctor_response"] = " ".join(
                (
                    explanation["headline"],
                    explanation["explanation"],
                    explanation["disclaimer"],
                )
            )
            return

        question_kind, target, focus = select_question_target(
            self.profile, self.evidence, checkpoint["patient_answer"]
        )
        if question_kind == "symptom":
            question, fallback_used = await phrase_question(self.llm, target, focus)
        else:
            question, fallback_used = await phrase_context_question(self.llm, target)
        self.pending_question = {
            "kind": question_kind,
            "target": target,
            "focus": focus,
        }
        checkpoint.update(
            question_target=self.pending_question,
            question_fallback_used=fallback_used,
            doctor_response=question,
        )
        self.turns.append({"role": "doctor", "content": question})

    def _build_model_record(self, checkpoint):
        condition = checkpoint["answers"]["condition"]
        escalation = checkpoint["answers"]["escalation"]
        ranked = sorted(
            condition["probabilities"].items(), key=lambda item: item[1], reverse=True
        )
        # ``out_of_scope`` is the scope classifier's sentinel, not a diagnosis.
        # The top candidate is still passed separately to the safety guard, so a
        # genuinely out-of-scope prediction remains blocking.
        ranked = [item for item in ranked if item[0] != "out_of_scope"][:3]
        sources = [self.predictor.name, "LLM symptom extraction and question phrasing"]
        if self.safety_engine.condition_is_in_scope(condition["choice"]):
            query = " ".join(
                get_vocab().name(symptom_id)
                for symptom_id, state in self.profile.symptoms.items()
                if state.presence == "present"
            )
            knowledge = retrieve_knowledge(query)
            checkpoint["knowledge_base"] = {
                "version": knowledge["kb_version"],
                "results": knowledge["results"],
            }
            sources.extend(
                source["url"]
                for item in knowledge["results"]
                for source in item["sources"]
            )
        record = DecisionRecord(
            recommended_action=(
                "urgent_medical_review"
                if escalation["choice"] == "urgent"
                else "routine_teleconsultation"
            ),
            responses=[
                turn["content"] for turn in self.turns if turn["role"] == "patient"
            ],
            preliminary_estimates=[
                ConditionEstimate(condition=name, score=float(score))
                for name, score in ranked
            ],
            risk_factors=[],
            triggered_rules=[
                f"{self.predictor.name} escalation={escalation['choice']}"
            ],
            missing_information=self.profile.missing_info,
            limitations=[
                "Research prototype evaluated on synthetic source labels",
                "Preliminary model estimate, not a clinical diagnosis",
            ],
            sources=list(dict.fromkeys(sources)),
        )
        return record


async def _run(args):
    llm = LLMClient()
    predictor = (
        EBMLivePredictor(args.model)
        if args.backend == "ebm"
        else LayaLivePredictor(args.model, args.device)
    )
    session = LLMDrivenConsultation(
        predictor, args.stopping, llm, safety_profile=args.safety_profile
    )
    print(f"Doctor: {GREETING}")
    events = []
    while not session.finished:
        message = input("Patient: ").strip()
        checkpoint = await session.receive(message)
        events.append(checkpoint)
        print(
            json.dumps(
                {
                    "backend": checkpoint["diagnostic_backend"],
                    "diagnosis": (checkpoint["answers"] or {}).get("condition"),
                    "escalation": (checkpoint["answers"] or {}).get("escalation"),
                    "stop_probability": checkpoint["stop_probability"],
                    "stop": checkpoint["stop"],
                },
                indent=2,
            )
        )
        print(f"Doctor: {checkpoint['doctor_response']}")
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(
            json.dumps({"turns": session.turns, "events": events}, indent=2)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("ebm", "laya"), default="ebm")
    parser.add_argument("--model", required=True)
    parser.add_argument("--stopping", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--safety-profile", choices=("rules_v1", "sg_moh"), default="rules_v1"
    )
    parser.add_argument("--log", type=Path)
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
