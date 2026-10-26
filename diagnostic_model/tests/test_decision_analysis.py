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
    assert len(report["pathways"]["condition"]["nodes"]) == 5
    assert len(report["pathways"]["condition"]["links"]) == 4
    outcomes = {node["outcome"] for node in report["pathways"]["condition"]["nodes"]}
    assert outcomes == {"correct", "incorrect"}
    assert {link["target_outcome"] for link in report["pathways"]["condition"]["links"]} == {"correct"}
    terminals = [node for node in report["pathways"]["condition"]["nodes"] if node["terminal"]]
    assert terminals == [{
        "id": "6|finalised_answer_2|correct", "patient_answer": 6,
        "label": "finalised_answer_2", "display_label": "Finalised after answer 2",
        "outcome": "correct", "correct": True, "terminal": True,
        "finalized_at_answer": 2, "count": 2,
    }]
    assert wilson_interval(50, 100) == [0.4038, 0.5962]
