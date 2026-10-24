"""Build model-neutral decision paths and confidence intervals from traces."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

OUT_OF_SCOPE = "out_of_scope"


def wilson_interval(successes, total, z=1.96):
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total <= 0:
        return [None, None]
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]


def _pathway(records, dimension):
    node_counts = Counter()
    edge_counts = Counter()
    for record in records:
        previous = None
        selected_answer = record.get("selected_answer", len(record["confidence_trace"]))
        selected_answer = min(selected_answer, len(record["confidence_trace"]))
        for step, checkpoint in enumerate(record["confidence_trace"][:selected_answer], 1):
            label = checkpoint["answers"][dimension]["choice"]
            correct = label == record["truth"][dimension]
            outcome = "correct" if correct else "incorrect"
            node = f"{step}|{label}|{outcome}"
            node_counts[node] += 1
            if previous is not None:
                edge_counts[(previous, node)] += 1
            previous = node
        final_outcome = "correct" if previous.endswith("|correct") else "incorrect"
        terminal = f"6|finalised_answer_{selected_answer}|{final_outcome}"
        node_counts[terminal] += 1
        edge_counts[(previous, terminal)] += 1
    nodes = []
    for node, count in sorted(node_counts.items()):
        step, label, outcome = node.split("|", 2)
        terminal = label.startswith("finalised_answer_")
        final_answer = int(label.rsplit("_", 1)[1]) if terminal else None
        nodes.append({
            "id": node,
            "patient_answer": int(step),
            "label": label,
            "display_label": (
                f"Finalised after answer {final_answer}"
                if terminal else label.replace("_", " ").title()
            ),
            "outcome": outcome,
            "correct": outcome == "correct",
            "terminal": terminal,
            "finalized_at_answer": final_answer,
            "count": count,
        })
    links = [
        {
            "source": source,
            "target": target,
            "target_outcome": target.rsplit("|", 1)[1],
            "count": count,
        }
        for (source, target), count in sorted(edge_counts.items())
    ]
    return {"nodes": nodes, "links": links}


def analyse_decisions(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    if not records:
        raise ValueError("Decision log is empty")
    steps = max(len(record["confidence_trace"]) for record in records)
    turn_metrics = []
    for step in range(steps):
        available = [record for record in records if len(record["confidence_trace"]) > step]
        condition_hits = scope_hits = escalation_hits = 0
        for record in available:
            truth = record["truth"]
            answers = record["confidence_trace"][step]["answers"]
            predicted_condition = answers["condition"]["choice"]
            condition_hits += predicted_condition == truth["condition"]
            scope_hits += ((predicted_condition != OUT_OF_SCOPE)
                           == (truth["condition"] != OUT_OF_SCOPE))
            escalation_hits += answers["escalation"]["choice"] == truth["escalation"]
        total = len(available)
        turn_metrics.append({
            "patient_answer": step + 1,
            "rows": total,
            "condition": {
                "accuracy": round(condition_hits / total, 4),
                "accuracy_95_ci": wilson_interval(condition_hits, total),
            },
            "scope": {
                "accuracy": round(scope_hits / total, 4),
                "accuracy_95_ci": wilson_interval(scope_hits, total),
            },
            "escalation": {
                "accuracy": round(escalation_hits / total, 4),
                "accuracy_95_ci": wilson_interval(escalation_hits, total),
            },
        })
    return {
        "rows": len(records),
        "turn_metrics": turn_metrics,
        "pathways": {
            "condition": _pathway(records, "condition"),
            "escalation": _pathway(records, "escalation"),
        },
        "interval": "Wilson 95% confidence interval for accuracy",
        "warning": "Synthetic generator holdout; not clinical validation.",
        "pathway_semantics": (
            "Paths end at the learned/forced selected answer and flow to an explicit "
            "correct or incorrect finalisation node. Correctness is against source labels."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyse_decisions(args.decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"rows": report["rows"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
