"""Run a bounded concurrent load check against the conversation API.

The workload follows the question target returned by the server, so it tests the
same history-aware path as the browser instead of replaying a fixed, stale turn
sequence.  Defaults are intentionally modest for a developer workstation.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


MAX_CONCURRENCY = 32
MAX_TURNS = 12


def request_json(method: str, url: str, payload: dict | None = None) -> tuple[dict, float]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read()) if response.status != 204 else {}
    return body, (time.perf_counter() - started) * 1000


def answer_for(target: dict | None) -> str:
    target = target or {}
    name = target.get("target")
    focus = target.get("focus")
    if name == "age":
        return "29"
    if name == "pregnancy_status":
        return "not pregnant"
    if name == "emergency_warning_signs":
        return "none of those"
    if name == "condition_symptoms":
        return "sneezing, no sore throat"
    if name == "medical_history":
        return "none"
    if name == "temperature":
        return "36.8 C"
    if focus == "duration":
        return "same two days"
    if focus == "severity":
        return "mild"
    if focus == "progression":
        return "stable"
    return "runny nose and mild cough for 2 days, no fever, breathing okay"


def run_consultation(base_url: str, ordinal: int) -> dict:
    latencies = []
    session_id = None
    try:
        started, latency = request_json("POST", f"{base_url}/api/conversations", {})
        latencies.append({"operation": "start", "latency_ms": latency})
        session_id = started["session_id"]
        target = {"target": "presenting_symptoms", "focus": None}
        final = None
        for _ in range(MAX_TURNS):
            final, latency = request_json(
                "POST",
                f"{base_url}/api/conversations/{session_id}/turn",
                {"patient_message": answer_for(target)},
            )
            latencies.append({"operation": "turn", "latency_ms": latency})
            target = final.get("question_target")
            if final.get("stop"):
                break
        return {
            "ordinal": ordinal,
            "completed": bool(final and final.get("stop")),
            "turns": sum(item["operation"] == "turn" for item in latencies),
            "latencies": latencies,
            "error": None,
        }
    except (KeyError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {
            "ordinal": ordinal,
            "completed": False,
            "turns": sum(item["operation"] == "turn" for item in latencies),
            "latencies": latencies,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if session_id:
            try:
                request_json("DELETE", f"{base_url}/api/conversations/{session_id}")
            except (TimeoutError, urllib.error.URLError):
                pass


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)
    return round(ordered[index], 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--consultations", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        parser.error(f"--concurrency must be between 1 and {MAX_CONCURRENCY}")
    if not 1 <= args.consultations <= 500:
        parser.error("--consultations must be between 1 and 500")

    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(run_consultation, args.base_url.rstrip("/"), ordinal)
            for ordinal in range(args.consultations)
        ]
        consultations = [future.result() for future in as_completed(futures)]
    wall_seconds = time.perf_counter() - wall_started
    latencies = [
        item["latency_ms"]
        for consultation in consultations
        for item in consultation["latencies"]
    ]
    report = {
        "schema": "bounded-chat-load/v1",
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "consultations": args.consultations,
        "completed_consultations": sum(item["completed"] for item in consultations),
        "errors": sum(item["error"] is not None for item in consultations),
        "requests": len(latencies),
        "wall_seconds": round(wall_seconds, 3),
        "throughput_requests_per_second": round(len(latencies) / wall_seconds, 3),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": round(max(latencies), 3) if latencies else None,
        },
        "failed_examples": [item for item in consultations if item["error"]][:3],
        "warning": "Synthetic bounded developer load; not a production SLA or clinical validation.",
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
