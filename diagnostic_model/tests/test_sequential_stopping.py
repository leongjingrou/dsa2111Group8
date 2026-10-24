import json

import pandas as pd

from diagnostic_model.laya_finetune import write_sequential_rows
from diagnostic_model.scenario_data import scenario_train_validation_test_split
from diagnostic_model.sequential_ebm import progressive_scenario_features
from diagnostic_model.stopping import (
    MAX_PATIENT_ANSWERS,
    apply_stop_policy,
    policy_metrics,
    trace_feature_rows,
)


def sample_scenarios(path="data/generated/synthetic_scenarios.csv"):
    data = pd.read_csv(path)
    return pd.concat(
        [group.head(5) for _, group in data.groupby(["condition_id", "escalation"], sort=False)]
    ).reset_index(drop=True)


def _answer(condition, escalation, confidence):
    return {
        "condition": {
            "choice": condition,
            "probabilities": {condition: confidence, "other": 1 - confidence},
            "answer_confidence": confidence,
        },
        "escalation": {
            "choice": escalation,
            "probabilities": {escalation: confidence, "other": 1 - confidence},
            "answer_confidence": confidence,
        },
    }


def test_three_way_split_is_disjoint_and_complete():
    data = sample_scenarios()
    split = scenario_train_validation_test_split(data)
    sets = {name: set(index) for name, index in split.items()}
    assert not sets["train"] & sets["validation"]
    assert not sets["train"] & sets["test"]
    assert not sets["validation"] & sets["test"]
    assert sum(map(len, sets.values())) == len(data)
    for index in split.values():
        assert set(data.loc[index, "condition_id"]) == set(data["condition_id"])


def test_progressive_ebm_features_respect_question_boundaries():
    data = sample_scenarios().head(3)
    first, symptom_ids = progressive_scenario_features(data, 1)
    final, _ = progressive_scenario_features(data, 5, symptom_ids=symptom_ids)
    assert first["history_observed"].eq(0).all()
    assert first["demographics_observed"].eq(0).all()
    assert first["age"].eq(0).all()
    assert final["history_observed"].eq(1).all()
    assert final["demographics_observed"].eq(1).all()
    assert final["age"].gt(0).all()
    assert first.filter(like="_asked").sum(axis=1).le(1).all()


def test_laya_sequential_export_has_five_partial_states_per_case(tmp_path):
    data = sample_scenarios()
    paths = write_sequential_rows(data, tmp_path)
    assert set(paths) == {"train", "validation", "test"}
    total = 0
    for split, path in paths.items():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) % MAX_PATIENT_ANSWERS == 0
        assert {row["patient_answer"] for row in rows} == set(range(1, 6))
        assert {row["split"] for row in rows} == {split}
        total += len(rows)
    assert total == len(data) * MAX_PATIENT_ANSWERS


def test_stop_policy_uses_earliest_threshold_crossing_and_forces_final():
    traces = [[
        {"answers": _answer("cold", "routine", 0.55)},
        {"answers": _answer("cold", "routine", 0.70)},
        {"answers": _answer("cold", "routine", 0.80)},
        {"answers": _answer("cold", "routine", 0.90)},
        {"answers": _answer("cold", "routine", 0.95)},
    ]]
    features = trace_feature_rows(traces)
    assert len(features) == 5
    assert features.loc[1, "joint_stable"] == 1
    assert apply_stop_policy(traces, [[0.1, 0.4, 0.8, 0.9, 0.95]], 0.75) == [2]
    assert apply_stop_policy(traces, [[0.1] * 5], 0.75) == [4]
    metrics = policy_metrics(
        traces, [{"condition": "cold", "escalation": "routine"}], [2]
    )
    assert metrics["joint_accuracy"] == 1.0
    assert metrics["mean_patient_answers"] == 3.0
