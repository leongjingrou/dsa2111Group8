from diagnostic_model.confidence_narrowing import analyse_records


def _answer(choice, probabilities):
    return {
        "choice": choice,
        "probabilities": probabilities,
        "answer_confidence": max(probabilities.values()),
    }


def _record(split, truth_condition, predicted):
    trace = []
    for turn in range(1, 6):
        confidence = 0.5 + turn * 0.08
        other = 1 - confidence
        choice = truth_condition if predicted else "other"
        probabilities = (
            {truth_condition: confidence, "other": other}
            if predicted else {truth_condition: other, "other": confidence}
        )
        trace.append({
            "answers": {
                "condition": _answer(choice, probabilities),
                "escalation": _answer("routine", {"routine": 0.8, "urgent": 0.2}),
            }
        })
    return {
        "split": split,
        "truth": {"condition": truth_condition, "escalation": "routine"},
        "confidence_trace": trace,
    }


def test_confidence_analysis_labels_bayesian_and_gaussian_intervals():
    records = [
        _record("train", "cold", True),
        _record("train", "cold", False),
        _record("validation", "cold", True),
        _record("test", "cold", True),
        _record("test", "flu", False),
    ]
    report = analyse_records(records)
    assert set(report["splits"]) == {"train", "validation", "test"}
    assert "Beta(1,1)" in report["intervals"]["accuracy"]
    assert "Gaussian" in report["intervals"]["confidence"]
    test_turns = report["splits"]["test"]["by_patient_answer"]
    assert len(test_turns) == 5
    assert test_turns[0]["diagnosis_accuracy"]["rows"] == 2
    assert len(test_turns[0]["diagnosis_accuracy"]["credible_interval_95"]) == 2
    assert len(test_turns[0]["top_confidence"]["gaussian_interval_95"]) == 2
    assert set(report["splits"]["test"]["by_true_diagnosis"]) == {"cold", "flu"}
