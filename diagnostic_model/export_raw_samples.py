"""Export a few auditable raw decisions with source truth and final response."""

import argparse
import json
from pathlib import Path


def _category(record):
    predicted = record["selected_answers"]
    truth = record["truth"]
    condition_correct = predicted["condition"]["choice"] == truth["condition"]
    escalation_correct = predicted["escalation"]["choice"] == truth["escalation"]
    if not condition_correct:
        return "diagnosis_miss"
    if not escalation_correct:
        return "escalation_miss"
    if record["selected_answer"] == 5:
        return "correct_forced_final"
    return "correct_learned_stop"


def _final_response(record):
    answers = record["selected_answers"]
    ranked = sorted(answers["condition"]["probabilities"].items(),
                    key=lambda item: item[1], reverse=True)[:3]
    return {
        "diagnosis": answers["condition"]["choice"],
        "diagnosis_confidence": answers["condition"]["answer_confidence"],
        "differential_top_3": [
            {"diagnosis": label, "probability": probability}
            for label, probability in ranked
        ],
        "escalation": answers["escalation"]["choice"],
        "escalation_confidence": answers["escalation"]["answer_confidence"],
        "patient_answers_used": record["selected_answer"],
        "disclaimer": "Research prototype output; not a clinical diagnosis.",
    }


def select_records(records, count=4, split="test"):
    eligible = [record for record in records if record.get("split") == split]
    wanted = ("correct_learned_stop", "correct_forced_final", "diagnosis_miss", "escalation_miss")
    selected = []
    seen_ids = set()
    for category in wanted:
        match = next((record for record in eligible
                      if record["scenario_id"] not in seen_ids and _category(record) == category), None)
        if match:
            selected.append(match)
            seen_ids.add(match["scenario_id"])
        if len(selected) >= count:
            break
    for record in eligible:
        if len(selected) >= count:
            break
        if record["scenario_id"] not in seen_ids:
            selected.append(record)
            seen_ids.add(record["scenario_id"])
    return selected


def export_samples(decisions_path, output_path, model, count=4, split="test"):
    records = [json.loads(line) for line in Path(decisions_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = select_records(records, count=count, split=split)
    output = {
        "schema_version": "1.0",
        "model": model,
        "split": split,
        "ground_truth_semantics": "Synthetic/source label, not a clinician-confirmed diagnosis.",
        "samples": [],
    }
    for record in selected:
        final = _final_response(record)
        truth = record["truth"]
        output["samples"].append({
            "sample_type": _category(record),
            "scenario_id": record["scenario_id"],
            "conversation": record["conversation"],
            "raw_checkpoints": record["confidence_trace"],
            "final_model_response": final,
            "source_ground_truth": truth,
            "comparison": {
                "diagnosis_correct": final["diagnosis"] == truth["condition"],
                "escalation_correct": final["escalation"] == truth["escalation"],
                "joint_correct": (
                    final["diagnosis"] == truth["condition"]
                    and final["escalation"] == truth["escalation"]
                ),
            },
        })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decisions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    output = export_samples(args.decisions, args.output, args.model, args.count, args.split)
    print(json.dumps({"samples": len(output["samples"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
