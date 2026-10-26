import json

import pandas as pd

from diagnostic_model.laya_finetune import write_rows
from diagnostic_model.laya_model import (
    consultation_state,
    evaluate_laya,
    held_out_rows,
    prioritized_evidence,
    questions,
)


class FakeAgent:
    def predict_batch(self, states, model_questions, batch_size):
        answers = {}
        for name, question in model_questions.items():
            labels = list(question["criteria"])
            answers[name] = {
                "choice": labels[0],
                "probabilities": {label: 1 / len(labels) for label in labels},
                "answer_confidence": 1 / len(labels),
            }
        return [{"answers": answers} for _ in states]


def sample_scenarios(path="data/generated/synthetic_scenarios.csv"):
    data = pd.read_csv(path)
    return pd.concat(
        [group.head(4) for _, group in data.groupby(["condition_id", "escalation"], sort=False)]
    ).reset_index(drop=True)


def test_laya_questions_include_abstention_and_escalation():
    model_questions = questions()
    assert "out_of_scope" in model_questions["condition"]["criteria"]
    assert set(model_questions["escalation"]["criteria"]) == {"routine", "urgent"}


def test_prioritized_evidence_is_label_free_and_respects_turn_boundary():
    row = sample_scenarios().iloc[0].to_dict()
    early = prioritized_evidence(row, patient_answers=1)
    full = prioritized_evidence(row, patient_answers=5)

    assert early
    assert all(item.startswith("P") for item in full)
    assert not any("temperature:" in item or "duration:" in item for item in early)
    assert any("temperature:" in item for item in full)
    assert row["diagnosis"].lower() not in " ".join(full).lower()
    assert "priority_evidence" in consultation_state(row, evidence_mode="prioritized")


def test_laya_rows_and_evaluation_use_shared_holdout(tmp_path):
    data = sample_scenarios()
    train_path, test_path = write_rows(data, tmp_path, test_size=0.5)
    train = [json.loads(line) for line in train_path.read_text().splitlines()]
    test = [json.loads(line) for line in test_path.read_text().splitlines()]

    assert len(train) + len(test) == len(data)
    assert {item["expected"]["escalation"] for item in train} == {"routine", "urgent"}
    assert "out_of_scope" in {item["expected"]["condition"] for item in train}

    report = evaluate_laya(
        data,
        tmp_path,
        FakeAgent(),
        batch_size=8,
        runs=1,
        trace_conversation=True,
        test_size=0.5,
    )
    assert report["rows"] == len(held_out_rows(data, test_size=0.5))
    assert report["test_size"] == 0.5
    assert {"condition", "scope", "escalation"}.issubset(report)
    assert report["conversation_trace"]["states"] == report["rows"] * 5
    assert len(report["conversation_trace"]["by_patient_answer"]) == 5
    assert len(report["conversation_trace"]["pathway_map"]["nodes"]) == 5
    assert len(report["conversation_trace"]["pathway_map"]["edges"]) == 4
    assert (tmp_path / "laya_decisions.jsonl").is_file()
    decision = json.loads((tmp_path / "laya_decisions.jsonl").read_text().splitlines()[0])
    assert len(decision["confidence_trace"]) == 5
    assert set(decision["confidence_trace"][-1]["answers"]) == {"condition", "escalation"}
