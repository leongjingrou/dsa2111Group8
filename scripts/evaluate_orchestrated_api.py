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
    parser.add_argument("--data", type=Path, default=Path("data/generated/synthetic_scenarios.csv"))
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
    rows = ([row for row in all_rows if row["split"] == args.split] if has_split
            else all_rows[-max(args.limit, len(all_rows) // 5):])

    for row in rows[:args.limit]:
        started = time.perf_counter()
        start = requests.post(args.base_url + "/api/conversations",
                              json={"safety_profile": "none"}, timeout=30)
        start.raise_for_status()
        session_id = start.json()["session_id"]
        trace = []
        for turn in json.loads(row["conversation"]):
            if turn["role"] != "patient":
                continue
            turn_started = time.perf_counter()
            response = requests.post(
                f"{args.base_url}/api/conversations/{session_id}/turn",
                json={"patient_message": turn["content"]}, timeout=30,
            )
            response.raise_for_status()
            body = response.json()
            body["processing_ms_client"] = round((time.perf_counter() - turn_started) * 1000, 3)
            latencies.append(body["processing_ms_client"])
            trace.append(body)
            if body["stop"]:
                break
        final = trace[-1]
        actual_condition = "out_of_scope" if row["scope_status"] == "out_of_scope" else row["condition_id"]
        actual_escalation = "urgent" if row["escalation"].lower() == "true" else "routine"
        records.append({
            "scenario_id": row["scenario_id"],
            "actual": {"condition": actual_condition, "escalation": actual_escalation},
            "predicted": {"condition": final["diagnosis"]["choice"],
                          "escalation": final["escalation"]["choice"]},
            "diagnosis_correct": final["diagnosis"]["choice"] == actual_condition,
            "escalation_correct": final["escalation"]["choice"] == actual_escalation,
            "consultation_ms_client": round((time.perf_counter() - started) * 1000, 3),
            "trace": trace,
        })

    correct = [record for record in records if record["diagnosis_correct"]]
    wrong = [record for record in records if not record["diagnosis_correct"]]
    output = {
        "schema": "orchestrated-server-evaluation/v1",
        "method": "labelled synthetic patient replies; deterministic local LLM adapter; EBM diagnosis",
        "summary": {
            "consultations": len(records),
            "turn_requests": len(latencies),
            "diagnosis_accuracy": sum(item["diagnosis_correct"] for item in records) / len(records),
            "escalation_accuracy": sum(item["escalation_correct"] for item in records) / len(records),
            "turn_latency_ms": {"mean": statistics.fmean(latencies),
                                "p50": percentile(latencies, .5),
                                "p95": percentile(latencies, .95),
                                "p99": percentile(latencies, .99)},
        },
        "correct_examples": correct[:2],
        "wrong_examples": wrong[:2],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
