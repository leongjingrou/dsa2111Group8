import json

from diagnostic_model.decision_analysis import analyse_decisions, wilson_interval


def _answer(condition, escalation):
    return {
        "condition": {"choice": condition},
        "escalation": {"choice": escalation},
    }


def test_decision_analysis_builds_intervals_and_paths(tmp_path):
    path = tmp_path / "decisions.jsonl"
    rows = [
        {
            "truth": {"condition": "common_cold", "escalation": "routine"},
            "confidence_trace": [
                {"answers": _answer("out_of_scope", "routine")},
                {"answers": _answer("common_cold", "routine")},
            ],
        },
        {
            "truth": {"condition": "out_of_scope", "escalation": "urgent"},
            "confidence_trace": [
                {"answers": _answer("out_of_scope", "routine")},
                {"answers": _answer("out_of_scope", "urgent")},
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))

    report = analyse_decisions(path)

    assert report["rows"] == 2
    assert report["turn_metrics"][0]["condition"]["accuracy"] == 0.5
    assert report["turn_metrics"][1]["condition"]["accuracy"] == 1.0
    assert len(report["pathways"]["condition"]["nodes"]) == 3
    assert len(report["pathways"]["condition"]["links"]) == 2
    assert wilson_interval(50, 100) == [0.4038, 0.5962]
