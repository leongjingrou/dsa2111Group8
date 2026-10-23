import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from diagnostic_model.scenario_data import (
    knowledge_base_condition_ids,
    load_scenarios,
    regenerate_scenarios,
    scenario_features,
    scenario_split,
    scenario_targets,
)
from knowledge_base.retrieval import load_json


OUT_OF_SCOPE = "out_of_scope"


def questions():
    conditions = load_json("conditions.json")["conditions"]
    condition_criteria = {item["id"]: item["description"] for item in conditions}
    condition_criteria[OUT_OF_SCOPE] = (
        "The vignette is not adequately represented by the supported condition list."
    )
    return {
        "condition": {
            "type": "choice",
            "instructions": (
                "Choose the best supported condition, or out_of_scope when none is adequate."
            ),
            "criteria": condition_criteria,
        },
        "escalation": {
            "type": "choice",
            "instructions": "Does this presentation require urgent escalation?",
            "criteria": {
                "routine": "No urgent escalation signal is present.",
                "urgent": "Escalation is required because of risk factors or red flags.",
            },
        },
    }


def held_out_rows(data):
    _, test_index = scenario_split(data)
    return data.loc[test_index].copy().reset_index(drop=True)


def expected_condition(rows):
    return np.where(rows["scope_status"].eq("in_scope"), rows["condition_id"], OUT_OF_SCOPE)


def _normalise(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    totals = probabilities.sum(axis=1, keepdims=True)
    empty = totals[:, 0] <= 0
    probabilities[empty] = 1.0
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def multiclass_score(truth, predictions, probabilities, labels):
    probabilities = _normalise(probabilities)
    expected = pd.get_dummies(pd.Series(truth)).reindex(columns=labels, fill_value=0).to_numpy()
    return {
        "accuracy": round(accuracy_score(truth, predictions), 4),
        "f1_macro": round(f1_score(truth, predictions, labels=labels, average="macro"), 4),
        "log_loss": round(log_loss(truth, probabilities, labels=labels), 4),
        "brier": round(np.mean(np.sum((probabilities - expected) ** 2, axis=1)), 4),
    }


def binary_score(truth, predictions, positive_probabilities):
    truth = np.asarray(truth, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    positive_probabilities = np.asarray(positive_probabilities, dtype=float)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": round(accuracy_score(truth, predictions), 4),
        "f1": round(f1_score(truth, predictions, zero_division=0), 4),
        "precision": round(precision_score(truth, predictions, zero_division=0), 4),
        "recall": round(recall_score(truth, predictions, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(truth, positive_probabilities), 4),
        "average_precision": round(average_precision_score(truth, positive_probabilities), 4),
        "brier": round(float(np.mean((positive_probabilities - truth) ** 2)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _answer_matrix(answers, name, labels):
    return _normalise(
        [[answer[name]["probabilities"].get(label, 0.0) for label in labels] for answer in answers]
    )


def evaluate_laya(data, output_dir, agent, batch_size=16, runs=5, name="laya", model_name=None):
    rows = held_out_rows(data)
    condition_labels = sorted(knowledge_base_condition_ids() + [OUT_OF_SCOPE])
    escalation_labels = ["routine", "urgent"]
    states = [{"vignette": text} for text in rows["vignette_text"]]
    agent.predict_batch(states[:1], questions(), batch_size=1)
    timings = []
    for _ in range(runs):
        started = perf_counter()
        results = agent.predict_batch(states, questions(), batch_size=batch_size)
        timings.append((perf_counter() - started) * 1000)
    elapsed_ms = median(timings)

    answers = [result["answers"] for result in results]
    condition_probabilities = _answer_matrix(answers, "condition", condition_labels)
    escalation_probabilities = _answer_matrix(answers, "escalation", escalation_labels)
    condition_predictions = [answer["condition"]["choice"] for answer in answers]
    escalation_predictions = [answer["escalation"]["choice"] == "urgent" for answer in answers]
    condition_truth = expected_condition(rows)
    escalation_truth = scenario_targets(rows)["escalation"].to_numpy()

    report = {
        "rows": len(rows),
        "model": model_name or "convaiinnovations/laya:typed-decisions",
        "batch_size": batch_size,
        "latency_ms_total_median": round(elapsed_ms, 2),
        "latency_ms_per_row": round(elapsed_ms / len(rows), 3),
        "condition": multiclass_score(
            condition_truth, condition_predictions, condition_probabilities, condition_labels
        ),
        "scope": binary_score(
            np.asarray(condition_truth) != OUT_OF_SCOPE,
            np.asarray(condition_predictions) != OUT_OF_SCOPE,
            1 - condition_probabilities[:, condition_labels.index(OUT_OF_SCOPE)],
        ),
        "escalation": binary_score(
            escalation_truth,
            escalation_predictions,
            escalation_probabilities[:, escalation_labels.index("urgent")],
        ),
        "warning": "Synthetic generator holdout; not clinical validation.",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}_metrics.json").write_text(json.dumps(report, indent=2))
    with (output_dir / f"{name}_decisions.jsonl").open("w") as log:
        for row, answer in zip(rows.to_dict("records"), answers):
            record = {
                "scenario_id": row["scenario_id"],
                "truth": {
                    "condition": row["condition_id"] if row["scope_status"] == "in_scope" else OUT_OF_SCOPE,
                    "escalation": "urgent" if bool(row["escalation"]) else "routine",
                },
                "answers": answer,
                "dataset_version": row["dataset_version"],
            }
            log.write(json.dumps(record) + "\n")
    return report


def evaluate_ebm_on_laya_holdout(data, artifacts_dir):
    rows = held_out_rows(data)
    targets = scenario_targets(rows)
    bundles = {
        task: joblib.load(artifacts_dir / f"{task}_model.joblib")
        for task in ("diagnosis", "scope", "escalation")
    }
    x, _ = scenario_features(rows, symptom_ids=bundles["scope"]["symptom_ids"])
    scope_model = bundles["scope"]["model"]
    escalation_model = bundles["escalation"]["model"]
    diagnosis_model = bundles["diagnosis"]["model"]

    scope_probability = scope_model.predict_proba(x)[:, list(scope_model.classes_).index(1)]
    escalation_probability = escalation_model.predict_proba(x)[:, list(escalation_model.classes_).index(1)]
    diagnosis_probability = diagnosis_model.predict_proba(x)
    condition_labels = sorted(knowledge_base_condition_ids() + [OUT_OF_SCOPE])
    combined = np.zeros((len(rows), len(condition_labels)))
    combined[:, condition_labels.index(OUT_OF_SCOPE)] = 1 - scope_probability
    for column, label in enumerate(diagnosis_model.classes_):
        combined[:, condition_labels.index(label)] = scope_probability * diagnosis_probability[:, column]
    condition_predictions = np.asarray(condition_labels)[combined.argmax(axis=1)]
    condition_truth = expected_condition(rows)

    return {
        "rows": len(rows),
        "condition": multiclass_score(
            condition_truth, condition_predictions, combined, condition_labels
        ),
        "scope": binary_score(
            targets["scope"], scope_probability >= 0.5, scope_probability
        ),
        "escalation": binary_score(
            targets["escalation"], escalation_probability >= 0.5, escalation_probability
        ),
    }


def compare_ebm(data, artifacts_dir, laya_report, output_dir, name="model_comparison"):
    comparison = {
        "matched_holdout_rows": len(held_out_rows(data)),
        "laya": laya_report,
        "ebm": evaluate_ebm_on_laya_holdout(data, artifacts_dir),
        "warning": "Matched synthetic holdout; not clinical validation.",
    }
    (output_dir / f"{name}.json").write_text(json.dumps(comparison, indent=2))
    return comparison


if __name__ == "__main__":
    import laya

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostic_model/artifacts"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--subfolder", default="typed-decisions")
    parser.add_argument(
        "--generate-scenarios",
        action="store_true",
        help="Regenerate the scenario CSV with data/generate.py before evaluation.",
    )
    args = parser.parse_args()
    data = (
        regenerate_scenarios(args.data)
        if args.generate_scenarios
        else load_scenarios(args.data)
    )
    model = laya.load(args.model, subfolder=args.subfolder)
    report = evaluate_laya(
        data,
        args.output_dir,
        model,
        args.batch_size,
        model_name=args.model,
    )
    comparison = compare_ebm(data, args.output_dir, report, args.output_dir)
    print(json.dumps(comparison, indent=2))
