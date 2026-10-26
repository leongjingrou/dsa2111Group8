from pathlib import Path

from diagnostic_model.demo_data import make_demo_dataset
from diagnostic_model.predict import predict_with_explanation
from diagnostic_model.training import split_features, train_and_evaluate


def test_demo_data_uses_all_knowledge_base_conditions():
    data = make_demo_dataset(rows_per_condition=5)

    assert len(data) == 30
    assert data["condition_id"].nunique() == 6
    assert any(column.startswith("symptom_") for column in data.columns)


def test_split_excludes_identifiers():
    features, labels = split_features(make_demo_dataset(rows_per_condition=5))

    assert "condition_id" not in features
    assert "vignette_id" not in features
    assert len(labels) == len(features)


def test_training_writes_required_outputs(tmp_path: Path):
    data = make_demo_dataset(rows_per_condition=8)
    report = train_and_evaluate(data, tmp_path, tune=False)

    assert {"ebm", "logistic_regression"}.issubset(report)
    assert (tmp_path / "ebm_model.joblib").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "feature_importance.png").is_file()
    assert (tmp_path / "calibration.png").is_file()

    record = data.drop(columns=["condition_id", "vignette_id"]).iloc[0].to_dict()
    prediction = predict_with_explanation(tmp_path / "ebm_model.joblib", record)
    assert prediction["condition"] in set(data["condition_id"])
    assert prediction["contributions"]
