import csv
import json

import joblib
import numpy as np

from diagnostic_model.grounding import convert_ddxplus, load_registry
from diagnostic_model.export_raw_samples import export_samples
from diagnostic_model.live_conversation import (
    EvidenceState,
    LiveConsultation,
    emergency_findings,
)


class FakePredictor:
    def predict(self, turns, evidence, answer_number):
        confidence = min(0.55 + answer_number * 0.1, 0.95)
        return {
            "condition": {
                "choice": "influenza",
                "probabilities": {"influenza": confidence, "common_cold": 1 - confidence},
                "answer_confidence": confidence,
            },
            "escalation": {
                "choice": "routine",
                "probabilities": {"routine": confidence, "urgent": 1 - confidence},
                "answer_confidence": confidence,
            },
        }


class FakeStopModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, frame):
        probability = np.where(frame["answer_fraction"].to_numpy() >= 0.6, 0.9, 0.1)
        return np.column_stack([1 - probability, probability])


def _stopping_file(tmp_path):
    path = tmp_path / "stopping.joblib"
    joblib.dump({"model": FakeStopModel(), "threshold": 0.75}, path)
    return path


def test_evidence_parser_preserves_unknowns_and_explicit_negation():
    state = EvidenceState()
    state.update("I have a cough but no chest pain. It has been 2 weeks and is getting worse.")
    assert state.symptoms["cough"] is True
    assert state.symptoms["chest_pain"] is False
    assert "fever" not in state.symptoms
    assert state.duration_days == 14
    assert state.progression == "worsening"


def test_singapore_emergency_override_respects_negation():
    assert emergency_findings("I have chest pain and feel unwell") == ["chest pain"]
    assert emergency_findings("I have no chest pain and no difficulty breathing") == []
    assert emergency_findings("No chest pain, confusion, fainting, or vomiting blood.") == []
    assert emergency_findings(
        "No chest pain, confusion, fainting, weakness, vomiting blood, or continuous vomiting or diarrhoea."
    ) == []
    assert emergency_findings("No chest pain, but I am confused.") == ["confusion"]


def test_live_chain_stops_when_controller_crosses_threshold(tmp_path):
    session = LiveConsultation(FakePredictor(), _stopping_file(tmp_path))
    first = session.receive("I have a cough and body aches")
    second = session.receive("It has been 3 days, is severe, and is getting worse")
    third = session.receive("I also have chills and fatigue")
    assert first["stop"] is False
    assert second["stop"] is False
    assert third["stop"] is True
    assert third["stop_reason"] == "learned_stop_threshold"
    assert len(session.trace) == 3


def test_emergency_override_stops_before_model_minimum(tmp_path):
    session = LiveConsultation(
        FakePredictor(), _stopping_file(tmp_path), safety_profile="sg_moh"
    )
    event = session.receive("I have chest pain and trouble breathing")
    assert event["stop"] is True
    assert event["model_stop"] is False
    assert event["stop_reason"] == "safety_profile_sg_moh"
    assert set(event["safety_findings"]) == {"chest pain", "breathing difficulty"}


def test_safety_profile_is_explicit_and_disabled_by_default(tmp_path):
    session = LiveConsultation(FakePredictor(), _stopping_file(tmp_path))
    event = session.receive("I have chest pain and trouble breathing")
    assert event["safety_profile"] == "none"
    assert event["safety_override_applied"] is False
    assert event["stop"] is False


def test_dataset_registry_has_separate_grounding_roles():
    registry = load_registry()
    roles = {source["role"] for source in registry["sources"]}
    assert "diagnosis_and_differential_training" in roles
    assert "jurisdictional_safety_override" in roles
    assert "cultural_and_language_robustness_only" in roles


def test_ddxplus_conversion_preserves_split_and_differential(tmp_path):
    metadata = {
        "E1": {"question_en": "Do you have a cough?", "value_meaning": {}},
        "E2": {"question_en": "How severe is it?", "value_meaning": {"V2": "severe"}},
    }
    evidence_path = tmp_path / "evidences.json"
    evidence_path.write_text(json.dumps(metadata), encoding="utf-8")
    csv_path = tmp_path / "patients.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "PATIENT_ID", "AGE", "SEX", "PATHOLOGY", "EVIDENCES",
            "INITIAL_EVIDENCE", "DIFFERENTIAL_DIAGNOSIS",
        ])
        writer.writeheader()
        writer.writerow({
            "PATIENT_ID": "p1", "AGE": "33", "SEX": "F", "PATHOLOGY": "Influenza",
            "EVIDENCES": "['E1', 'E2_@_V2']", "INITIAL_EVIDENCE": "E1",
            "DIFFERENTIAL_DIAGNOSIS": "[['Influenza', 0.7], ['URTI', 0.3]]",
        })
    output = tmp_path / "converted.jsonl"
    assert convert_ddxplus(csv_path, evidence_path, output, "test") == 1
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["source_split"] == "test"
    assert record["differential_diagnosis"][0] == ["Influenza", 0.7]
    assert sum(turn["role"] == "patient" for turn in record["conversation"]) == 5


def test_ddxplus_multichoice_values_share_one_question(tmp_path):
    metadata = {"E1": {"question_en": "Where is the pain?", "value_meaning": {
        "V1": {"en": "chest"}, "V2": {"en": "back"},
    }}}
    evidence_path = tmp_path / "evidences.json"
    evidence_path.write_text(json.dumps(metadata), encoding="utf-8")
    csv_path = tmp_path / "patients.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "AGE", "SEX", "PATHOLOGY", "EVIDENCES", "INITIAL_EVIDENCE", "DIFFERENTIAL_DIAGNOSIS",
        ])
        writer.writeheader()
        writer.writerow({"AGE": "40", "SEX": "M", "PATHOLOGY": "Example",
                         "EVIDENCES": "['E1_@_V1', 'E1_@_V2']", "INITIAL_EVIDENCE": "E0",
                         "DIFFERENTIAL_DIAGNOSIS": "[]"})
    output = tmp_path / "converted.jsonl"
    convert_ddxplus(csv_path, evidence_path, output, "test")
    record = json.loads(output.read_text(encoding="utf-8"))
    matching = [turn for turn in record["conversation"] if turn["role"] == "patient" and "chest, back" in turn["content"]]
    assert len(matching) == 1


def test_raw_sample_export_keeps_prediction_and_source_truth_separate(tmp_path):
    answer = {
        "condition": {"choice": "influenza", "probabilities": {"influenza": 0.8, "common_cold": 0.2}, "answer_confidence": 0.8},
        "escalation": {"choice": "routine", "probabilities": {"routine": 0.9, "urgent": 0.1}, "answer_confidence": 0.9},
    }
    record = {
        "scenario_id": "SYN-1", "split": "test",
        "conversation": [{"role": "doctor", "content": "Hello"}, {"role": "patient", "content": "Cough"}],
        "truth": {"condition": "influenza", "escalation": "routine"},
        "confidence_trace": [{"patient_answer": 1, "answers": answer, "stop_probability": 0.8, "policy_stop": True}],
        "selected_answer": 1, "selected_answers": answer,
    }
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(json.dumps(record) + "\n", encoding="utf-8")
    output = export_samples(decisions, tmp_path / "samples.json", "EBM", count=1)
    sample = output["samples"][0]
    assert sample["final_model_response"]["diagnosis"] == "influenza"
    assert sample["source_ground_truth"]["condition"] == "influenza"
    assert sample["comparison"]["joint_correct"] is True
