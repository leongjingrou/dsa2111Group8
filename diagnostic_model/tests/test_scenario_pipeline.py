import json

import pandas as pd

from diagnostic_model.scenario_data import scenario_features, scenario_split, scenario_targets
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
