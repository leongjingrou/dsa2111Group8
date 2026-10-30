import pytest

from llm_service.app.schema import Profile, SymptomState
from llm_service.condition_followups import (
    followup_config, next_followup, supported_condition_matches,
)


@pytest.mark.parametrize(
    ("symptoms", "condition_id"),
    [
        (["sore_throat"], "common_cold"),
        (["fever", "chills"], "influenza"),
        (["upper_abdominal_pain"], "gastroenteritis"),
        (["headache"], "migraine"),
        (["itchy_nose"], "allergic_rhinitis"),
        (["painful_urination"], "urinary_tract_infection"),
    ],
)
def test_every_supported_condition_has_a_relevant_followup(symptoms, condition_id):
    profile = Profile(symptoms={
        symptom_id: SymptomState(symptom_id=symptom_id, presence="present")
        for symptom_id in symptoms
    })
    plan = next_followup(profile, [])
    assert plan is not None
    assert plan[0] == condition_id
    assert all(symptom_id in followup_config()[condition_id]["followups"] for symptom_id in plan[1])
    assert next_followup(profile, [{"question_target": {"target": "condition_symptoms"}}]) is None


@pytest.mark.parametrize(
    ("symptoms", "condition_id"),
    [
        (["sore_throat", "cough"], "common_cold"),
        (["fever", "chills"], "influenza"),
        (["upper_abdominal_pain", "nausea"], "gastroenteritis"),
        (["headache", "nausea"], "migraine"),
        (["itchy_nose", "watery_eyes"], "allergic_rhinitis"),
        (["painful_urination", "frequent_urination"], "urinary_tract_infection"),
    ],
)
def test_patient_matches_require_corroborated_symptoms_for_each_condition(symptoms, condition_id):
    profile = Profile(symptoms={
        symptom_id: SymptomState(symptom_id=symptom_id, presence="present")
        for symptom_id in symptoms
    })
    matches = supported_condition_matches(profile)
    assert condition_id in {item["condition_id"] for item in matches["results"]}


def test_nausea_without_migraine_anchor_does_not_create_migraine_match():
    profile = Profile(symptoms={
        symptom_id: SymptomState(symptom_id=symptom_id, presence="present")
        for symptom_id in ("upper_abdominal_pain", "nausea")
    })
    matches = supported_condition_matches(profile)
    assert "migraine" not in {item["condition_id"] for item in matches["results"]}
