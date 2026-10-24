"""Analyse how diagnostic confidence narrows over five patient answers.

Accuracy uses a Beta(1, 1) posterior and equal-tailed 95% credible intervals.
Continuous confidence measures use a Gaussian sampling approximation for the
mean (mean +/- 1.96 standard errors).  The two interval types are deliberately
labelled separately in the report and figures.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import beta

matplotlib.use("Agg")
from matplotlib import pyplot as plt


TURNS = range(1, 6)


def _bayesian_accuracy(hits, total):
    lower, upper = beta.ppf([0.025, 0.975], hits + 1, total - hits + 1)
    return {
        "observed": hits / total,
        "posterior_mean": (hits + 1) / (total + 2),
        "credible_interval_95": [float(lower), float(upper)],
        "hits": int(hits),
        "rows": int(total),
    }


def _gaussian_mean(values, bounded=True):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    lower, upper = mean - 1.96 * standard_error, mean + 1.96 * standard_error
    if bounded:
        lower, upper = max(0.0, lower), min(1.0, upper)
    return {
        "mean": mean,
        "gaussian_interval_95": [lower, upper],
        "standard_error": standard_error,
        "rows": int(len(values)),
    }


def _diagnostic_measures(answer, truth):
    probabilities = np.asarray(list(answer["probabilities"].values()), dtype=float)
    probabilities = probabilities / probabilities.sum()
    ordered = np.sort(probabilities)[::-1]
    entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1))))
    maximum_entropy = math.log(len(probabilities)) if len(probabilities) > 1 else 1.0
    return {
        "top_confidence": float(ordered[0]),
        "top_two_margin": float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0]),
        "concentration": 1 - entropy / maximum_entropy,
        "effective_diagnoses": math.exp(entropy),
        "true_diagnosis_probability": float(answer["probabilities"].get(truth, 0.0)),
    }


def analyse_records(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["split"]].append(record)
    report = {
        "intervals": {
            "accuracy": "Beta(1,1) posterior; equal-tailed 95% credible interval",
            "confidence": "Gaussian approximation for the mean; mean +/- 1.96 standard errors",
        },
        "splits": {},
        "warning": "Synthetic generator data; intervals are not clinical uncertainty bounds.",
    }
    for split, split_records in grouped.items():
        turn_rows = []
        by_diagnosis = defaultdict(list)
        for turn in TURNS:
            diagnosis_hits = escalation_hits = 0
            measures = defaultdict(list)
            diagnosis_measures = defaultdict(lambda: defaultdict(list))
            diagnosis_hits_by_label = defaultdict(int)
            diagnosis_totals = defaultdict(int)
            for record in split_records:
                truth = record["truth"]
                answers = record["confidence_trace"][turn - 1]["answers"]
                condition = answers["condition"]
                escalation = answers["escalation"]
                diagnosis_hits += condition["choice"] == truth["condition"]
                escalation_hits += escalation["choice"] == truth["escalation"]
                row_measures = _diagnostic_measures(condition, truth["condition"])
                for name, value in row_measures.items():
                    measures[name].append(value)
                    diagnosis_measures[truth["condition"]][name].append(value)
                diagnosis_totals[truth["condition"]] += 1
                diagnosis_hits_by_label[truth["condition"]] += condition["choice"] == truth["condition"]
            turn_rows.append({
                "patient_answer": turn,
                "diagnosis_accuracy": _bayesian_accuracy(diagnosis_hits, len(split_records)),
                "escalation_accuracy": _bayesian_accuracy(escalation_hits, len(split_records)),
                **{
                    name: _gaussian_mean(values, bounded=name != "effective_diagnoses")
                    for name, values in measures.items()
                },
            })
            for label in sorted(diagnosis_totals):
                by_diagnosis[label].append({
                    "patient_answer": turn,
                    "diagnosis_accuracy": _bayesian_accuracy(
                        diagnosis_hits_by_label[label], diagnosis_totals[label]
                    ),
                    **{
                        name: _gaussian_mean(
                            values, bounded=name != "effective_diagnoses"
                        )
                        for name, values in diagnosis_measures[label].items()
                    },
                })
        report["splits"][split] = {
            "rows": len(split_records),
            "by_patient_answer": turn_rows,
            "by_true_diagnosis": dict(by_diagnosis),
        }
    return report


def _plot_interval(axis, turns, rows, key, estimate, interval, label, colour, linestyle="-"):
    values = np.asarray([row[key][estimate] for row in rows])
    bounds = np.asarray([row[key][interval] for row in rows])
    axis.plot(turns, values, marker="o", color=colour, label=label, linestyle=linestyle)
    axis.fill_between(turns, bounds[:, 0], bounds[:, 1], color=colour, alpha=0.16)


def draw_split_confidence(report, output, approach):
    splits = [name for name in ("train", "validation", "test") if name in report["splits"]]
    figure, axes = plt.subplots(len(splits), 2, figsize=(14, 4.1 * len(splits)), squeeze=False)
    for row_index, split in enumerate(splits):
        rows = report["splits"][split]["by_patient_answer"]
        turns = [row["patient_answer"] for row in rows]
        left, right = axes[row_index]
        _plot_interval(left, turns, rows, "diagnosis_accuracy", "posterior_mean", "credible_interval_95", "Diagnosis correct", "#2F6FED")
        _plot_interval(left, turns, rows, "escalation_accuracy", "posterior_mean", "credible_interval_95", "Escalation correct", "#E17A2D")
        left.set_title(f"{split.title()}: Bayesian correctness")
        left.set_ylabel("Posterior probability")
        left.set_ylim(0, 1.02)
        left.legend(loc="lower right")
        _plot_interval(right, turns, rows, "top_confidence", "mean", "gaussian_interval_95", "Top diagnosis confidence", "#7656C9")
        _plot_interval(right, turns, rows, "true_diagnosis_probability", "mean", "gaussian_interval_95", "True-diagnosis probability", "#238B67")
        _plot_interval(right, turns, rows, "top_two_margin", "mean", "gaussian_interval_95", "Top-two margin", "#238F9D")
        _plot_interval(right, turns, rows, "concentration", "mean", "gaussian_interval_95", "Distribution concentration", "#C74343")
        right.set_title(f"{split.title()}: differential narrows")
        right.set_ylabel("Mean probability / concentration")
        right.set_ylim(0, 1.02)
        right.legend(loc="upper left", fontsize=8)
        for axis in (left, right):
            axis.set_xticks(list(TURNS))
            axis.set_xlabel("Patient answer")
            axis.grid(alpha=0.22)
    figure.suptitle(
        f"{approach}: confidence and diagnostic narrowing over five answers\n"
        "Shading: Bayesian 95% credible intervals (left); Gaussian 95% intervals for means (right)",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def draw_diagnosis_confidence(report, output, approach, split="test"):
    diagnoses = report["splits"][split]["by_true_diagnosis"]
    columns = 2
    rows_count = math.ceil(len(diagnoses) / columns)
    figure, axes = plt.subplots(rows_count, columns, figsize=(14, 3.6 * rows_count), squeeze=False)
    for axis, (diagnosis, rows) in zip(axes.flat, sorted(diagnoses.items())):
        turns = [row["patient_answer"] for row in rows]
        _plot_interval(axis, turns, rows, "diagnosis_accuracy", "posterior_mean", "credible_interval_95", "Decision correct", "#2F6FED")
        _plot_interval(axis, turns, rows, "true_diagnosis_probability", "mean", "gaussian_interval_95", "Probability on truth", "#238B67", "--")
        axis.set_title(diagnosis.replace("_", " ").title())
        axis.set_ylim(0, 1.02)
        axis.set_xticks(list(TURNS))
        axis.set_xlabel("Patient answer")
        axis.grid(alpha=0.22)
        axis.legend(loc="lower right", fontsize=8)
    for axis in axes.flat[len(diagnoses):]:
        axis.axis("off")
    figure.suptitle(
        f"{approach}: diagnosis-specific confidence narrowing on {split}\n"
        "Solid: Bayesian correctness posterior; dashed: Gaussian interval for mean probability assigned to truth",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--approach", required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.decisions.read_text(encoding="utf-8").splitlines()]
    report = analyse_records(records)
    safe_name = args.approach.lower().replace(" ", "-")
    report_path = args.output_dir / f"{safe_name}-confidence-narrowing.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    draw_split_confidence(report, args.output_dir / f"{safe_name}-confidence-over-time.png", args.approach)
    draw_diagnosis_confidence(report, args.output_dir / f"{safe_name}-diagnosis-narrowing.png", args.approach)
    print(json.dumps({"report": str(report_path), "images": 2}, indent=2))


if __name__ == "__main__":
    main()
