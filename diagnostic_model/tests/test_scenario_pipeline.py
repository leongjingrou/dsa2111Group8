import json

import pandas as pd

from data.generate import DATASET_VERSION, DISEASES, N_PER_CLASS, SYMPTOMS, generate
from diagnostic_model.scenario_data import (
    regenerate_scenarios,
    scenario_features,
    scenario_split,
    scenario_targets,
)
from diagnostic_model.training import train_scenario_models


def sample_scenarios(path="data/generated/synthetic_scenarios.csv"):
    data = pd.read_csv(path)
    return pd.concat(
        [group.head(4) for _, group in data.groupby(["condition_id", "escalation"], sort=False)]
    ).reset_index(drop=True)


def test_scenario_adapter_prevents_label_leakage_and_preserves_unknown():
    data = sample_scenarios()
    features, symptom_ids = scenario_features(data)

    assert symptom_ids
    assert not {"diagnosis", "condition_id", "scope_status", "scenario_id"} & set(features)
    assert all(features.dtypes.eq(float))
    assert any(name.endswith("_asked") for name in features)
    assert any(name.endswith("_present") for name in features)

    raw = json.loads(data.loc[0, "symptoms"])
    unasked = next(symptom for symptom in symptom_ids if symptom not in raw)
    assert features.loc[0, f"symptom_{unasked}_asked"] == 0
    assert features.loc[0, f"symptom_{unasked}_present"] == 0


def test_scenario_split_is_matched_and_disjoint():
    data = sample_scenarios()
    train, test = scenario_split(data)
    assert set(train).isdisjoint(test)
    assert len(train) + len(test) == len(data)
    assert set(data.loc[train, "condition_id"]) == set(data.loc[test, "condition_id"])

    half_train, half_test = scenario_split(data, test_size=0.5)
    assert len(half_train) == len(half_test)
    assert set(half_train).isdisjoint(half_test)


def test_scenario_models_write_three_artifacts(tmp_path):
    data = sample_scenarios()
    report = train_scenario_models(data, tmp_path, tune=False)

    assert set(report["tasks"]) == {"diagnosis", "scope", "escalation"}
    for task in report["tasks"]:
        assert (tmp_path / f"{task}_model.joblib").is_file()
    assert (tmp_path / "scenario_metrics.json").is_file()


def test_scenario_targets_have_expected_balance():
    targets = scenario_targets(pd.read_csv("data/generated/synthetic_scenarios.csv"))
    assert int(targets["scope"].sum()) == 600
    assert int(targets["escalation"].sum()) == 850


def test_data_generator_is_deterministic_and_pipeline_compatible(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_symptoms, first_scenarios = generate(first)
    second_symptoms, second_scenarios = generate(second)

    assert first_symptoms.read_bytes() == second_symptoms.read_bytes()
    assert first_scenarios.read_bytes() == second_scenarios.read_bytes()
    assert b"\r\n" not in first_symptoms.read_bytes()
    assert b"\r\n" not in first_scenarios.read_bytes()

    data = regenerate_scenarios(first / "synthetic_scenarios.csv")
    assert len(data) == len(DISEASES) * 2 * N_PER_CLASS
    assert data["dataset_version"].astype(str).unique().tolist() == [DATASET_VERSION]
    assert data["scenario_id"].is_unique
    assert len(pd.read_csv(first_symptoms)) == len(SYMPTOMS)
    assert set(data["scope_status"]) == {"in_scope", "out_of_scope"}
    assert {"conversation", "conversation_text"}.issubset(data.columns)
    for _, row in data.head(20).iterrows():
        turns = json.loads(row["conversation"])
        assert len(turns) == 10
        assert [turn["role"] for turn in turns] == ["doctor", "patient"] * 5
        assert row["diagnosis"].lower() not in row["conversation_text"].lower()
