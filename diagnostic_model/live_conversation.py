"""Run a clinician-led consultation with live confidence and stopping decisions.

This remains a research prototype.  The language parser is deliberately
conservative: an unmentioned finding stays unknown and is never treated as a
negative answer.
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data.generate import SYMPTOMS
from diagnostic_model.laya_model import consultation_state, questions
from diagnostic_model.stopping import stopping_probabilities
from llm_service.app.negation import is_negated

GREETING = "Hello, what seems to be the problem today?"
MAX_PATIENT_ANSWERS = 5
QUESTION_SEQUENCE = (
    "How long has this been going on, how severe is it, and is it getting better or worse?",
    "What other symptoms have you noticed?",
    (
        "I need to check for warning signs. Do you have severe or chest pain, trouble "
        "breathing, uncontrolled bleeding, confusion, fainting, weakness or numbness, "
        "vision changes, vomiting blood, or continuous vomiting or diarrhoea?"
    ),
    "Lastly, what is your age and temperature? If relevant, are you pregnant?",
)

# Singapore MOH, Getting medical help, updated 22 April 2025.  These phrases
# are a transparent safety override, separate from learned model confidence.
SG_MOH_EMERGENCY_PATTERNS = {
    "chest pain": r"\b(chest pain|pressure (?:on|in) (?:my|the) chest)\b",
    "breathing difficulty": r"\b(short(?:ness)? of breath|trouble breathing|difficulty breathing|cannot breathe)\b",
    "uncontrolled bleeding": r"\b(uncontrolled|won'?t stop) bleeding\b",
    "confusion": r"\b(confus(?:ed|ion)|disorient(?:ed|ation))\b",
    "fainting": r"\b(faint(?:ed|ing)?|loss of consciousness|passed out)\b",
    "neurological change": r"\b(sudden (?:dizziness|numbness|weakness|vision change)|one-sided weakness)\b",
    "blood in cough or vomit": r"\b(coughing|vomiting|vomited) blood\b",
    "continuous vomiting or diarrhoea": r"\b(continuous|nonstop|won'?t stop) (?:vomiting|diarrh(?:ea|oea))\b",
}

ALIASES = {
    "shortness_of_breath": (
        "shortness of breath",
        "trouble breathing",
        "difficulty breathing",
    ),
    "painful_urination": ("painful urination", "burning urination", "burns when i pee"),
    "frequent_urination": ("frequent urination", "peeing often", "urinating often"),
    "throbbing_headache": ("throbbing headache", "pulsing headache"),
    "band_like_headache": ("band-like headache", "tight headache"),
    "confusion_drowsiness": ("confusion", "confused", "very drowsy"),
    "persistent_vomiting": (
        "persistent vomiting",
        "continuous vomiting",
        "keep vomiting",
    ),
    "severe_abdominal_pain": (
        "severe stomach pain",
        "severe belly pain",
        "severe tummy pain",
        "severe stomach ache",
    ),
}


def _normalise(text):
    return re.sub(r"\s+", " ", text.lower().replace("’", "'")).strip()


def _is_negated(text, start):
    return is_negated(text, start)


def emergency_findings(text):
    text = _normalise(text)
    return [
        name
        for name, pattern in SG_MOH_EMERGENCY_PATTERNS.items()
        if any(
            not _is_negated(text, match.start()) for match in re.finditer(pattern, text)
        )
    ]


@dataclass
class EvidenceState:
    symptoms: dict = field(default_factory=dict)
    duration_days: float | None = None
    severity: str | None = None
    progression: str | None = None
    age: float | None = None
    temperature_c: float | None = None
    temperature_asked: bool = False
    sex: str | None = None
    sex_asked: bool = False
    pregnant: bool | None = None

    def update(self, message):
        text = _normalise(message)
        symptom_names = {
            symptom_id: name.lower() for symptom_id, name, _, _ in SYMPTOMS
        }
        for symptom_id, name in symptom_names.items():
            phrases = ALIASES.get(symptom_id, ()) + (name,)
            matches = [
                match
                for phrase in phrases
                for match in re.finditer(rf"\b{re.escape(phrase)}\b", text)
            ]
            if matches:
                self.symptoms[symptom_id] = any(
                    not _is_negated(text, match.start()) for match in matches
                )

        duration = re.search(r"\b(\d+(?:\.\d+)?)\s*(day|week|month)s?\b", text)
        if duration:
            factor = {"day": 1, "week": 7, "month": 30}[duration.group(2)]
            self.duration_days = float(duration.group(1)) * factor
        for value in ("mild", "moderate", "normal", "severe"):
            if re.search(rf"\b{value}\b", text):
                self.severity = "normal" if value == "moderate" else value
        if re.search(r"\b(worse|worsening|getting worse)\b", text):
            self.progression = "worsening"
        elif re.search(r"\b(better|improving|getting better)\b", text):
            self.progression = "improving"
        elif re.search(r"\b(stable|same|unchanged)\b", text):
            self.progression = "stable"

        temperature = re.search(
            r"\b(?:temperature(?: is| of)?\s*)?(3[5-9]|4[0-2])(?:\.(\d))?\s*(?:°?c|degrees)?\b",
            text,
        )
        if temperature and (
            "temperature" in temperature.group(0)
            or "c" in temperature.group(0)
            or "degree" in temperature.group(0)
        ):
            self.temperature_c = float(
                f"{temperature.group(1)}.{temperature.group(2) or 0}"
            )
            self.temperature_asked = True
        age = re.search(
            r"\b(?:i am|i'm|age(?: is)?|aged)\s+(\d{1,3})(?:\s+years? old)?\b", text
        )
        if age is None:
            # Patients often start with "31, not pregnant" or "31 years old".
            # Require the number at the start so a symptom duration is not age.
            age = re.search(
                r"^(\d{1,3})(?:\s+years? old\b|\s*,\s*(?:not\s+)?pregnant\b)",
                text,
            )
        if age:
            candidate = int(age.group(1))
            if 0 < candidate < 120:
                self.age = float(candidate)
        if re.search(r"\b(female|woman)\b", text):
            self.sex = "F"
            self.sex_asked = True
        elif re.search(r"\b(male|man)\b", text):
            self.sex = "M"
            self.sex_asked = True
        if re.search(r"\b(not pregnant|i am not pregnant|i'm not pregnant)\b", text):
            self.pregnant = False
        elif re.search(r"\b(pregnant|expecting a baby)\b", text):
            self.pregnant = True


def _answer(choice, labels, probabilities):
    mapped = {str(label): float(value) for label, value in zip(labels, probabilities)}
    return {
        "choice": str(choice),
        "probabilities": mapped,
        "answer_confidence": max(mapped.values(), default=0.0),
    }


class EBMLivePredictor:
    name = "ebm"

    def __init__(self, model_path):
        self.bundle = joblib.load(model_path)

    def predict(self, turns, evidence, answer_number):
        values = {feature: 0.0 for feature in self.bundle["features"]}
        for symptom_id, present in evidence.symptoms.items():
            asked = f"symptom_{symptom_id}_asked"
            value = f"symptom_{symptom_id}_present"
            if asked in values:
                values[asked] = 1.0
                values[value] = float(present)
        if evidence.duration_days is not None:
            values["duration_days"] = evidence.duration_days
            values["history_observed"] = 1.0
        if evidence.severity:
            values[f"severity_{evidence.severity}"] = 1.0
            values["history_observed"] = 1.0
        if evidence.progression:
            values[f"progression_{evidence.progression}"] = 1.0
            values["history_observed"] = 1.0
        if evidence.age is not None:
            values["age"] = evidence.age
            values["demographics_observed"] = 1.0
        if evidence.temperature_c is not None:
            values["temperature_c"] = evidence.temperature_c
            values["demographics_observed"] = 1.0
        if evidence.sex:
            values["sex_female"] = float(evidence.sex == "F")
            values["demographics_observed"] = 1.0
        if evidence.pregnant is not None:
            values["pregnant"] = float(evidence.pregnant)
            values["demographics_observed"] = 1.0
        values["patient_answer"] = answer_number / MAX_PATIENT_ANSWERS
        frame = pd.DataFrame([values], columns=self.bundle["features"])

        scope_model = self.bundle["scope"]
        diagnosis_model = self.bundle["diagnosis"]
        escalation_model = self.bundle["escalation"]
        scope = scope_model.predict_proba(frame)[0, list(scope_model.classes_).index(1)]
        diagnosis = diagnosis_model.predict_proba(frame)[0]
        labels = self.bundle["condition_labels"]
        combined = np.zeros(len(labels))
        combined[labels.index("out_of_scope")] = 1 - scope
        for index, label in enumerate(diagnosis_model.classes_):
            combined[labels.index(label)] = scope * diagnosis[index]
        escalation = escalation_model.predict_proba(frame)[0]
        escalation_labels = [
            "urgent" if int(label) else "routine" for label in escalation_model.classes_
        ]
        return {
            "condition": _answer(labels[int(combined.argmax())], labels, combined),
            "escalation": _answer(
                escalation_labels[int(escalation.argmax())],
                escalation_labels,
                escalation,
            ),
        }


class LayaLivePredictor:
    name = "laya"

    def __init__(self, model_dir, device="cuda"):
        import laya

        self.agent = laya.load(model_dir, device=device)

    def predict(self, turns, evidence, answer_number):
        state = consultation_state({}, turns=turns, evidence_mode="conversation")
        return self.agent.predict_batch([state], questions(), batch_size=1)[0][
            "answers"
        ]


class LiveConsultation:
    def __init__(
        self, predictor, stopping_path, minimum_answers=2, safety_profile="none"
    ):
        if safety_profile not in {"none", "sg_moh"}:
            raise ValueError(f"Unknown safety profile: {safety_profile}")
        stopping = joblib.load(stopping_path)
        self.predictor = predictor
        self.stop_model = stopping["model"]
        self.stop_threshold = float(stopping["threshold"])
        self.minimum_answers = minimum_answers
        self.safety_profile = safety_profile
        self.evidence = EvidenceState()
        self.turns = [{"role": "doctor", "content": GREETING}]
        self.trace = []
        self.finished = False

    def receive(self, patient_message):
        if self.finished:
            raise RuntimeError("The consultation has already stopped")
        self.turns.append({"role": "patient", "content": patient_message})
        self.evidence.update(patient_message)
        answer_number = sum(turn["role"] == "patient" for turn in self.turns)
        answers = self.predictor.predict(self.turns, self.evidence, answer_number)
        checkpoint = {
            "patient_answer": answer_number,
            "latest_response": patient_message,
            "answers": answers,
        }
        self.trace.append(checkpoint)
        stop_probability = stopping_probabilities(self.stop_model, [self.trace])[0][-1]
        model_stop = answer_number >= MAX_PATIENT_ANSWERS or (
            answer_number >= self.minimum_answers
            and stop_probability >= self.stop_threshold
        )
        safety_findings = (
            emergency_findings(patient_message)
            if self.safety_profile == "sg_moh"
            else []
        )
        safety_override = bool(safety_findings)
        stop = safety_override or model_stop
        reason = (
            "safety_profile_sg_moh"
            if safety_override
            else (
                "maximum_questions"
                if answer_number >= MAX_PATIENT_ANSWERS
                else "learned_stop_threshold"
                if stop
                else "more_information_needed"
            )
        )
        checkpoint.update(
            {
                "stop_probability": round(float(stop_probability), 4),
                "stop_threshold": self.stop_threshold,
                "model_stop": bool(model_stop),
                "safety_profile": self.safety_profile,
                "safety_override_applied": safety_override,
                "safety_findings": safety_findings,
                "stop": stop,
                "stop_reason": reason,
            }
        )
        if stop:
            self.finished = True
            next_message = self._summary(checkpoint)
        else:
            next_message = QUESTION_SEQUENCE[answer_number - 1]
            self.turns.append({"role": "doctor", "content": next_message})
        checkpoint["doctor_response"] = next_message
        return checkpoint

    @staticmethod
    def _summary(checkpoint):
        condition = checkpoint["answers"]["condition"]
        escalation = checkpoint["answers"]["escalation"]
        ranked = sorted(
            condition["probabilities"].items(), key=lambda item: item[1], reverse=True
        )[:3]
        differential = ", ".join(f"{name} {value:.0%}" for name, value in ranked)
        if checkpoint["safety_override_applied"]:
            action = "The safety screen indicates urgent emergency assessment in Singapore; call 995 if this is happening now."
        else:
            action = f"The model's current escalation category is {escalation['choice']} ({escalation['answer_confidence']:.0%})."
        return (
            f"I have enough information for the prototype to stop asking questions. "
            f"Current differential: {differential}. {action} This is research output, not a diagnosis."
        )


def _print_checkpoint(checkpoint):
    condition = checkpoint["answers"]["condition"]
    escalation = checkpoint["answers"]["escalation"]
    print(
        json.dumps(
            {
                "patient_answer": checkpoint["patient_answer"],
                "condition": condition["choice"],
                "condition_confidence": round(condition["answer_confidence"], 4),
                "escalation": escalation["choice"],
                "escalation_confidence": round(escalation["answer_confidence"], 4),
                "stop_probability": checkpoint["stop_probability"],
                "stop": checkpoint["stop"],
                "stop_reason": checkpoint["stop_reason"],
            },
            indent=2,
        )
    )
    print(f"Doctor: {checkpoint['doctor_response']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("ebm", "laya"), default="ebm")
    parser.add_argument(
        "--model", required=True, help="EBM joblib file or Laya model directory"
    )
    parser.add_argument("--stopping", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--safety-profile",
        choices=("none", "sg_moh"),
        default="none",
        help="Optional jurisdictional safety guard; always logged separately from model stopping",
    )
    parser.add_argument(
        "--script",
        type=Path,
        help="JSON list of patient replies; omit for interactive input",
    )
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    predictor = (
        EBMLivePredictor(args.model)
        if args.backend == "ebm"
        else LayaLivePredictor(args.model, args.device)
    )
    session = LiveConsultation(
        predictor, args.stopping, safety_profile=args.safety_profile
    )
    scripted = (
        json.loads(args.script.read_text(encoding="utf-8")) if args.script else None
    )
    print(f"Doctor: {GREETING}")
    events = []
    while not session.finished:
        if scripted is None:
            message = input("Patient: ").strip()
        else:
            if not scripted:
                raise ValueError("Script ended before the model stopped")
            message = scripted.pop(0)
            print(f"Patient: {message}")
        checkpoint = session.receive(message)
        events.append(checkpoint)
        _print_checkpoint(checkpoint)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(
            json.dumps(
                {
                    "safety_profile": session.safety_profile,
                    "turns": session.turns,
                    "checkpoints": events,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
