"""Replay labelled synthetic consultations through the orchestrated HTTP API."""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import requests


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://diagnosis-api")
    parser.add_argument(
        "--data", type=Path, default=Path("data/generated/synthetic_scenarios.csv")
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    latencies = []
    with args.data.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        all_rows = list(reader)
        has_split = "split" in (reader.fieldnames or [])
    rows = (
        [row for row in all_rows if row["split"] == args.split]
        if has_split
        else all_rows[-max(args.limit, len(all_rows) // 5) :]
    )

    for row in rows[: args.limit]:
        started = time.perf_counter()
        start = requests.post(args.base_url + "/api/conversations", json={}, timeout=30)
        start.raise_for_status()
        session_id = start.json()["session_id"]
        trace = []
        for turn in json.loads(row["conversation"]):
            if turn["role"] != "patient":
                continue
            turn_started = time.perf_counter()
            response = requests.post(
                f"{args.base_url}/api/conversations/{session_id}/turn",
                json={"patient_message": turn["content"]},
                timeout=30,
            )
            response.raise_for_status()
            body = response.json()
            body["processing_ms_client"] = round(
                (time.perf_counter() - turn_started) * 1000, 3
            )
            latencies.append(body["processing_ms_client"])
            trace.append(body)
            if body["stop"]:
                break
        final = trace[-1]
        actual_condition = (
            "out_of_scope"
            if row["scope_status"] == "out_of_scope"
            else row["condition_id"]
        )
        actual_escalation = (
            "urgent" if row["escalation"].lower() == "true" else "routine"
        )
        predicted_condition = (final.get("diagnosis") or {}).get("choice")
        predicted_escalation = (final.get("escalation") or {}).get("choice")
        records.append(
            {
                "scenario_id": row["scenario_id"],
                "actual": {
                    "condition": actual_condition,
                    "escalation": actual_escalation,
                },
                "predicted": {
                    "condition": predicted_condition,
                    "escalation": predicted_escalation,
                },
                "complete": final["stop"],
                "safety_disposition": final["safety"]["disposition"],
                "diagnosis_correct": predicted_condition == actual_condition
                if predicted_condition
                else None,
                "escalation_correct": predicted_escalation == actual_escalation
                if predicted_escalation
                else None,
                "consultation_ms_client": round(
                    (time.perf_counter() - started) * 1000, 3
                ),
                "trace": trace,
            }
        )
        requests.delete(
            f"{args.base_url}/api/conversations/{session_id}", timeout=30
        ).raise_for_status()

    correct = [record for record in records if record["diagnosis_correct"]]
    wrong = [record for record in records if record["diagnosis_correct"] is False]
    assessed = [record for record in records if record["diagnosis_correct"] is not None]
    output = {
        "schema": "orchestrated-server-evaluation/v2",
        "method": "labelled synthetic patient replies; deterministic local LLM adapter; configured diagnosis backend",
        "summary": {
            "consultations": len(records),
            "turn_requests": len(latencies),
            "model_decisions": len(assessed),
            "incomplete_screens": sum(not item["complete"] for item in records),
            "diagnosis_accuracy_on_model_decisions": sum(
                item["diagnosis_correct"] for item in assessed
            )
            / len(assessed)
            if assessed
            else None,
            "escalation_accuracy_on_model_decisions": sum(
                item["escalation_correct"] for item in assessed
            )
            / len(assessed)
            if assessed
            else None,
            "turn_latency_ms": {
                "mean": statistics.fmean(latencies),
                "p50": percentile(latencies, 0.5),
                "p95": percentile(latencies, 0.95),
                "p99": percentile(latencies, 0.99),
            },
        },
        "correct_examples": correct[:2],
        "wrong_examples": wrong[:2],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
