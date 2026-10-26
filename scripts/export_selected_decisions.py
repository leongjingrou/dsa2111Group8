"""Export compact, reproducible correct/incorrect final-decision examples."""

import argparse
import json
from pathlib import Path


def selected_record(record, approach):
    selected = record["selected_answers"]
    truth = record["truth"]
    diagnosis = selected["condition"]["choice"]
    escalation = selected["escalation"]["choice"]
    return {
        "approach": approach,
        "scenario_id": record["scenario_id"],
        "split": record["split"],
        "actual": truth,
        "final": {
            "patient_answers": record["selected_answer"],
            "diagnosis": diagnosis,
            "diagnosis_confidence": selected["condition"]["answer_confidence"],
            "diagnosis_probabilities": selected["condition"]["probabilities"],
            "escalation": escalation,
            "escalation_confidence": selected["escalation"]["answer_confidence"],
            "escalation_probabilities": selected["escalation"]["probabilities"],
            "diagnosis_correct": diagnosis == truth["condition"],
            "escalation_correct": escalation == truth["escalation"],
        },
        "confidence_over_time": [
            {
                "patient_answer": item.get("patient_answer", item.get("patient_turn", 0) // 2),
                "diagnosis": item["answers"]["condition"]["choice"],
                "diagnosis_confidence": item["answers"]["condition"]["answer_confidence"],
                "escalation": item["answers"]["escalation"]["choice"],
                "escalation_confidence": item["answers"]["escalation"]["answer_confidence"],
                "stop_probability": item["stop_probability"],
                "policy_stop": item["policy_stop"],
            }
            for item in record["confidence_trace"][: record["selected_answer"]]
        ],
    }


def choose(path, approach):
    correct = incorrect = None
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            output = selected_record(record, approach)
            is_correct = output["final"]["diagnosis_correct"]
            if is_correct and correct is None:
                correct = output
            if not is_correct and incorrect is None:
                incorrect = output
            if correct and incorrect:
                break
    return [item for item in (correct, incorrect) if item]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ebm", required=True)
    parser.add_argument("--laya", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema": "selected-final-decisions/v1",
        "note": "Source labels are evaluation targets, not clinician-confirmed diagnoses.",
        "samples": choose(args.ebm, "EBM") + choose(args.laya, "Laya"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
