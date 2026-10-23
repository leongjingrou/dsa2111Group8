import json

import pandas as pd

from diagnostic_model.laya_finetune import write_rows
from diagnostic_model.laya_model import evaluate_laya, held_out_rows, questions


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


def test_laya_rows_and_evaluation_use_shared_holdout(tmp_path):
    data = sample_scenarios()
    train_path, test_path = write_rows(data, tmp_path)
    train = [json.loads(line) for line in train_path.read_text().splitlines()]
    test = [json.loads(line) for line in test_path.read_text().splitlines()]

    assert len(train) + len(test) == len(data)
    assert {item["expected"]["escalation"] for item in train} == {"routine", "urgent"}
    assert "out_of_scope" in {item["expected"]["condition"] for item in train}

    report = evaluate_laya(data, tmp_path, FakeAgent(), batch_size=8, runs=1)
    assert report["rows"] == len(held_out_rows(data))
    assert {"condition", "scope", "escalation"}.issubset(report)
    assert (tmp_path / "laya_decisions.jsonl").is_file()
