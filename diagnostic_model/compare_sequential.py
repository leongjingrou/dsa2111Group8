"""Write a compact EBM/Laya stopping-policy comparison bundle."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def build_comparison(ebm, laya):
    models = {"EBM": ebm, "Laya": laya}
    comparison = {"models": {}, "warning": "Synthetic generator data; not clinical validation."}
    for name, report in models.items():
        comparison["models"][name] = {
            "stop_threshold": report["selected_stop_threshold"],
            "splits": {
                split: {
                    key: values[key]
                    for key in (
                        "condition_accuracy", "escalation_accuracy", "joint_accuracy",
                        "urgent_recall", "mean_patient_answers", "stopped_before_final",
                    )
                }
                for split, values in report["splits"].items()
            },
        }
    return comparison


def write_markdown(comparison, output):
    lines = [
        "# Sequential EBM/Laya comparison",
        "",
        "| Model | Split | Diagnosis | Escalation | Joint | Urgent recall | Mean answers | Early stop |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, model_report in comparison["models"].items():
        for split in ("train", "validation", "test"):
            row = model_report["splits"][split]
            lines.append(
                f"| {model} | {split} | {row['condition_accuracy']:.2%} | "
                f"{row['escalation_accuracy']:.2%} | {row['joint_accuracy']:.2%} | "
                f"{row['urgent_recall']:.2%} | {row['mean_patient_answers']:.3f} | "
                f"{row['stopped_before_final']:.2%} |"
            )
    lines.extend(["", comparison["warning"], ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def draw_comparison(comparison, output):
    metrics = ["condition_accuracy", "escalation_accuracy", "joint_accuracy", "urgent_recall"]
    labels = ["Diagnosis", "Escalation", "Joint", "Urgent recall"]
    models = ["EBM", "Laya"]
    x = np.arange(len(metrics))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for index, model in enumerate(models):
        row = comparison["models"][model]["splits"]["test"]
        axes[0].bar(x + (index - 0.5) * 0.34, [row[key] for key in metrics], 0.34, label=model)
    axes[0].set_xticks(x, labels, rotation=12)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Test proportion")
    axes[0].set_title("Stopped-decision quality")
    axes[0].legend()
    answers = [comparison["models"][model]["splits"]["test"]["mean_patient_answers"] for model in models]
    early = [comparison["models"][model]["splits"]["test"]["stopped_before_final"] for model in models]
    axes[1].bar(np.arange(2) - 0.18, answers, 0.36, label="Mean answers")
    axes[1].bar(np.arange(2) + 0.18, np.asarray(early) * 5, 0.36, label="Early-stop rate × 5")
    axes[1].set_xticks(np.arange(2), models)
    axes[1].set_ylim(0, 5)
    axes[1].set_title("Question efficiency")
    axes[1].legend()
    figure.suptitle("Sequential EBM vs Laya on the identical synthetic test split")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ebm", type=Path, required=True)
    parser.add_argument("--laya", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ebm = json.loads(args.ebm.read_text(encoding="utf-8"))
    laya = json.loads(args.laya.read_text(encoding="utf-8"))
    comparison = build_comparison(ebm, laya)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    write_markdown(comparison, args.output_dir / "MODEL_COMPARISON.md")
    draw_comparison(comparison, args.output_dir / "model-comparison.png")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
