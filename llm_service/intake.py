"""Bind short replies to the question chosen by the server.

A bare "no" answers one pending symptom question. It never supplies negative
answers for the rest of the safety checklist.
"""

import re

from llm_service.app.schema import ExtractedSymptom, ExtractionResult
from llm_service.app.vocab import get_vocab

UNCERTAIN_ANSWERS = {"not sure", "unsure", "i don't know"}
NEGATIVE_GROUP_ANSWERS = {"no", "nope", "none", "none of those", "not any of those"}
PRESENCE_ANSWERS = {
    "yes": "present",
    "no": "absent",
    **dict.fromkeys(UNCERTAIN_ANSWERS, "uncertain"),
}
PREGNANCY_ANSWERS = {
    "yes": "pregnant",
    "pregnant": "pregnant",
    "no": "not_pregnant",
    "not pregnant": "not_pregnant",
    "not applicable": "not_applicable",
    **dict.fromkeys(UNCERTAIN_ANSWERS, "uncertain"),
}


def empty_extraction():
    return ExtractionResult(
        symptoms=[],
        unmapped_mentions=[],
        medical_history=[],
        allergies=[],
        medications=[],
    )


def safety_question(decision, profile=None):
    """Ask about a conflict first, otherwise the next missing safety field."""
    conflicts = [
        hit for hit in decision.triggered_rules if hit.rule_id == "DATA_CONTRADICTION"
    ]
    if conflicts and conflicts[0].fields:
        question_field = conflicts[0].fields[0]
    else:
        question_field = next(iter(decision.missing_information), "presenting_symptoms")

    if profile is not None and not any(
        symptom.presence == "present" for symptom in profile.symptoms.values()
    ):
        return (
            {"kind": "context", "target": "presenting_symptoms", "focus": None},
            "Could you describe what feels wrong in your own words, including where it is, when it started, how severe it feels, and whether it is changing?",
        )

    if question_field == "age":
        target = {"kind": "safety", "target": "age", "focus": None}
        return target, "What is your age in years?"
    if question_field == "pregnancy_status":
        target = {"kind": "safety", "target": "pregnancy_status", "focus": None}
        return target, (
            "Are you pregnant? Please answer pregnant, not pregnant, "
            "not applicable, or not sure."
        )

    missing_symptoms = [
        field.removeprefix("symptoms.")
        for field in decision.missing_information
        if field.startswith("symptoms.")
        and get_vocab().is_valid(field.removeprefix("symptoms."))
    ]
    if not conflicts and len(missing_symptoms) > 1:
        target = {
            "kind": "safety",
            "target": "emergency_warning_signs",
            "targets": missing_symptoms,
            "focus": "presence",
        }
        return target, (
            "Before we continue, are you having any emergency warning signs such as "
            "trouble breathing or chest pain; fainting, a seizure, or severe confusion; "
            "heavy bleeding; face droop, new speech trouble, or one-sided weakness; "
            "tongue or throat swelling; a sudden worst-ever headache; or severe abdominal pain? "
            "Tell me which one, or say none."
        )

    symptom_id = question_field.removeprefix("symptoms.")
    if get_vocab().is_valid(symptom_id):
        prefix = (
            "Your answers differ. Please confirm: "
            if conflicts
            else "I need to check a warning sign: "
        )
        question = (
            prefix + f"do you have {get_vocab().name(symptom_id).lower()}? "
            "Answer yes, no, or not sure."
        )
        return {"kind": "safety", "target": symptom_id, "focus": "presence"}, question
    return (
        {"kind": "context", "target": "presenting_symptoms", "focus": None},
        "Please describe the main symptoms you are experiencing.",
    )


def bind_answer(message, target, profile, evidence):
    """Return structured facts for a recognized reply, or None for LLM extraction.

    Demographic replies update the profile and evidence directly. Symptom
    replies are returned as an extraction so normal conflict handling still runs.
    """
    if target is None:
        return None
    normalized_answer = message.lower().strip().rstrip(".!?")
    question_field = target["target"]
    grouped_targets = target.get("targets", [])
    if question_field == "emergency_warning_signs" and grouped_targets:
        if normalized_answer in NEGATIVE_GROUP_ANSWERS:
            for symptom_id in grouped_targets:
                evidence.symptoms[symptom_id] = False
            return ExtractionResult(
                symptoms=[
                    ExtractedSymptom(
                        symptom_id=symptom_id,
                        presence="absent",
                        severity=None,
                        duration_days=None,
                        progression=None,
                        evidence=message,
                    )
                    for symptom_id in grouped_targets
                ],
                unmapped_mentions=[],
                medical_history=[],
                allergies=[],
                medications=[],
            )
        return None
    if target["kind"] == "safety":
        if question_field == "age" and _bind_age(normalized_answer, profile, evidence):
            return empty_extraction()
        if question_field == "pregnancy_status" and _bind_pregnancy(
            normalized_answer, profile, evidence
        ):
            return empty_extraction()

    if (
        get_vocab().is_valid(question_field)
        and target.get("focus") == "presence"
        and normalized_answer in PRESENCE_ANSWERS
    ):
        presence = PRESENCE_ANSWERS[normalized_answer]
        if presence == "uncertain":
            evidence.symptoms.pop(question_field, None)
        else:
            evidence.symptoms[question_field] = presence == "present"
        return _symptom_extraction(question_field, message, presence)

    symptom = profile.symptoms.get(question_field)
    if symptom is not None and symptom.presence == "present":
        details = _parse_symptom_detail(normalized_answer, target.get("focus"))
        if details:
            return _symptom_extraction(question_field, message, "present", **details)
    return None


def _bind_age(answer, profile, evidence):
    if re.fullmatch(r"\d{1,3}(?:\.\d+)?(?: years? old)?", answer):
        age = float(answer.split()[0])
        if 0 <= age <= 120:
            profile.age = evidence.age = age
        return True
    return answer in UNCERTAIN_ANSWERS


def _bind_pregnancy(answer, profile, evidence):
    if answer not in PREGNANCY_ANSWERS:
        return False
    status = PREGNANCY_ANSWERS[answer]
    profile.pregnancy_status = status
    if status == "pregnant":
        evidence.pregnant = True
    elif status in {"not_pregnant", "not_applicable"}:
        evidence.pregnant = False
    else:
        evidence.pregnant = None
    return True


def _parse_symptom_detail(answer, focus):
    if focus == "duration":
        match = re.fullmatch(
            r"(?:for )?(\d+(?:\.\d+)?)\s*(days?|weeks?|months?)", answer
        )
        if match:
            days_per_unit = {"d": 1, "w": 7, "m": 30}
            return {"duration_days": float(match[1]) * days_per_unit[match[2][0]]}
    elif focus == "severity" and answer in {"mild", "moderate", "severe"}:
        return {"severity": answer}
    elif focus == "progression" and answer in {"improving", "stable", "worsening"}:
        return {"progression": answer}
    return {}


def _symptom_extraction(symptom_id, message, presence, **details):
    return ExtractionResult(
        symptoms=[
            ExtractedSymptom(
                symptom_id=symptom_id,
                presence=presence,
                severity=details.get("severity"),
                duration_days=details.get("duration_days"),
                progression=details.get("progression"),
                evidence=message,
            )
        ],
        unmapped_mentions=[],
        medical_history=[],
        allergies=[],
        medications=[],
    )
