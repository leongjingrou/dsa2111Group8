"""Evaluate Laya on all five Q&A states and train its stopping controller."""

import argparse
import json
from pathlib import Path

import joblib

from diagnostic_model.laya_model import (
    OUT_OF_SCOPE,
    confidence_traces,
    conversation_turns,
)
from diagnostic_model.scenario_data import (
    load_scenarios,
    scenario_targets,
    scenario_train_validation_test_split,
)
from diagnostic_model.stopping import (
    annotate_records,
    apply_stop_policy,
    fit_stop_controller,
    policy_metrics,
    select_threshold,
    stopping_probabilities,
)


def _truths(rows):
    escalation = scenario_targets(rows)["escalation"]
    return [
        {
            "condition": (
                row["condition_id"] if row["scope_status"] == "in_scope" else OUT_OF_SCOPE
            ),
            "escalation": "urgent" if bool(urgent) else "routine",
        }
        for (_, row), urgent in zip(rows.iterrows(), escalation)
    ]


def evaluate_laya_sequential(
    data, agent, output_dir, batch_size=16, validation_size=0.2,
    test_size=0.2, evidence_mode="conversation", model_name=None,
):
    """Fit a stop head on Laya traces and report every data split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = scenario_train_validation_test_split(
        data, validation_size=validation_size, test_size=test_size
    )
    traces = {}
    truths = {}
    records = {}
    for split, split_index in indices.items():
        rows = data.loc[split_index].reset_index(drop=True)
        traces[split] = confidence_traces(
            rows, agent, batch_size=batch_size, evidence_mode=evidence_mode
        )
        truths[split] = _truths(rows)
        records[split] = [
            {
                "scenario_id": row["scenario_id"],
                "split": split,
                "conversation": conversation_turns(row),
                "truth": truth,
                "confidence_trace": trace,
                "dataset_version": row["dataset_version"],
            }
            for (_, row), truth, trace in zip(rows.iterrows(), truths[split], traces[split])
        ]

    stop_model = fit_stop_controller(traces["train"], truths["train"], family="laya")
    threshold, threshold_search = select_threshold(
        stop_model, traces["validation"], truths["validation"]
    )
    report = {
        "model": model_name or "Laya with learned confidence-trace stopping controller",
        "fixed_patient_answers": 5,
        "split": {name: len(value) for name, value in indices.items()},
        "selected_stop_threshold": threshold,
        "threshold_objective": (
            "joint_accuracy - 0.03*question_fraction - 0.5*urgent_miss_rate"
        ),
        "splits": {},
        "warning": "Synthetic generator data; not clinical validation.",
    }
    decisions_path = output_dir / "laya_sequential_decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8", newline="\n") as log:
        for split in ("train", "validation", "test"):
            probabilities = stopping_probabilities(stop_model, traces[split])
            selected = apply_stop_policy(traces[split], probabilities, threshold)
            split_metrics = policy_metrics(traces[split], truths[split], selected)
            split_metrics["by_patient_answer"] = [
                {
                    "patient_answer": answer,
                    **policy_metrics(
                        traces[split], truths[split], [answer - 1] * len(traces[split])
                    ),
                }
                for answer in range(1, 6)
            ]
            split_metrics["answer_5_baseline"] = split_metrics["by_patient_answer"][-1]
            report["splits"][split] = split_metrics
            for record in annotate_records(records[split], stop_model, threshold):
                log.write(json.dumps(record) + "\n")
    joblib.dump(
        {"model": stop_model, "threshold": threshold},
        output_dir / "laya_stopping_controller.joblib",
    )
    (output_dir / "laya_sequential_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "laya_threshold_search.json").write_text(
        json.dumps(threshold_search, indent=2), encoding="utf-8"
    )
    return report


def main():
    import laya

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evidence-mode", choices=("conversation", "prioritized"), default="conversation")
    args = parser.parse_args()
    agent = laya.load(args.model_dir, device=args.device)
    report = evaluate_laya_sequential(
        load_scenarios(args.data), agent, args.output_dir,
        batch_size=args.batch_size, evidence_mode=args.evidence_mode,
        model_name=args.model_dir,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
