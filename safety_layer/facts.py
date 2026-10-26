"""Normalize structured patient facts without inventing negative answers."""

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

PRESENCES = {"present", "absent", "uncertain"}
PRESENCE_PRIORITY = {"absent": 0, "uncertain": 1, "present": 2}


def as_mapping(value: Any) -> Mapping | None:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if callable(getattr(value, "model_dump", None)):
        result = value.model_dump()
        return result if isinstance(result, Mapping) else None
    return None


def symptom_presence(value: Any) -> str | None:
    state = as_mapping(value)
    if state is not None:
        value = state.get("presence")
    if type(value) is bool:
        return "present" if value else "absent"
    if isinstance(value, str) and value in PRESENCES:
        return value
    return None


def pregnancy_state(value: Any) -> str | None:
    if type(value) is bool:
        return "pregnant" if value else "not_pregnant"
    if isinstance(value, str) and value in {
        "pregnant",
        "not_pregnant",
        "not_applicable",
        "unknown",
        "uncertain",
    }:
        return value
    return None


def is_finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


@dataclass
class PatientFacts:
    """Validated observations and data-quality issues from all supplied sources.

    Positive history is kept separately from current presence so a later denial
    cannot make an earlier emergency disappear.
    """

    symptom_states: dict[str, str] = field(default_factory=dict)
    positive_symptoms: set[str] = field(default_factory=set)
    conflicting_fields: set[str] = field(default_factory=set)
    invalid_fields: list[str] = field(default_factory=list)
    ages: list[float] = field(default_factory=list)
    pregnancy_states: list[str] = field(default_factory=list)
    pregnancy_uncertain: bool = False
    has_unmapped_findings: bool = False

    @classmethod
    def from_inputs(cls, profile, evidence, known_symptoms):
        facts = cls()
        inputs = [as_mapping(profile)]
        if evidence is not None:
            inputs.append(as_mapping(evidence))
        for patient_data in inputs:
            if patient_data is None:
                facts.invalid_fields.append("profile_or_evidence")
                continue
            facts._collect_symptoms(patient_data.get("symptoms", {}), known_symptoms)
            facts._collect_contradictions(
                patient_data.get("contradictions", []), known_symptoms
            )
            facts._collect_unmapped_mentions(patient_data.get("unmapped_mentions", []))
            facts._collect_demographics(patient_data)
        if len(set(facts.ages)) > 1:
            facts.conflicting_fields.add("age")
        if len(set(facts.pregnancy_states)) > 1:
            facts.conflicting_fields.add("pregnancy_status")
        return facts

    def _collect_symptoms(self, symptoms, known_symptoms):
        if not isinstance(symptoms, Mapping):
            self.invalid_fields.append("symptoms")
            return
        for symptom_id, value in symptoms.items():
            if symptom_id not in known_symptoms:
                self.has_unmapped_findings = True
                continue
            presence = symptom_presence(value)
            details = as_mapping(value)
            if details and details.get("symptom_id", symptom_id) != symptom_id:
                self.invalid_fields.append(f"symptoms.{symptom_id}.symptom_id")
            if presence is None:
                self.invalid_fields.append(f"symptoms.{symptom_id}.presence")
                continue
            previous_presence = self.symptom_states.get(symptom_id)
            if (
                previous_presence in {"present", "absent"}
                and presence in {"present", "absent"}
                and previous_presence != presence
            ):
                self.conflicting_fields.add(symptom_id)
            if presence == "present":
                self.positive_symptoms.add(symptom_id)
            # Combining parsers must never turn uncertainty into a negative.
            if (
                previous_presence is None
                or PRESENCE_PRIORITY[presence] > PRESENCE_PRIORITY[previous_presence]
            ):
                self.symptom_states[symptom_id] = presence

    def _collect_contradictions(self, contradictions, known_symptoms):
        if not isinstance(contradictions, list):
            self.invalid_fields.append("contradictions")
            return
        for contradiction in contradictions:
            details = as_mapping(contradiction)
            if (
                details is None
                or not isinstance(details.get("symptom_id"), str)
                or details["symptom_id"] not in known_symptoms
                or not isinstance(details.get("previous"), str)
                or not isinstance(details.get("new"), str)
                or details["previous"] not in PRESENCES
                or details["new"] not in PRESENCES
            ):
                self.invalid_fields.append("contradictions")
                continue
            symptom_id = details["symptom_id"]
            self.conflicting_fields.add(symptom_id)
            if "present" in (details["previous"], details["new"]):
                self.positive_symptoms.add(symptom_id)

    def _collect_unmapped_mentions(self, mentions):
        if not isinstance(mentions, list) or not all(
            isinstance(mention, str) for mention in mentions
        ):
            self.invalid_fields.append("unmapped_mentions")
        elif any(mention.strip() for mention in mentions):
            self.has_unmapped_findings = True

    def _collect_demographics(self, patient_data):
        age = patient_data.get("age")
        if age is not None:
            if is_finite_number(age) and 0 <= age <= 120:
                self.ages.append(age)
            else:
                self.invalid_fields.append("age")
        for field_name in ("pregnancy_status", "pregnant"):
            value = patient_data.get(field_name)
            if value is None:
                continue
            pregnancy = pregnancy_state(value)
            if pregnancy is None:
                self.invalid_fields.append("pregnancy_status")
            elif pregnancy in {"unknown", "uncertain"}:
                self.pregnancy_uncertain = True
            else:
                # An explicit not-applicable answer and EvidenceState's False
                # both confirm a negative pregnancy screen; they do not conflict.
                normalized_state = (
                    "not_pregnant" if pregnancy == "not_applicable" else pregnancy
                )
                self.pregnancy_states.append(normalized_state)
