"""Concurrent multi-device, multi-turn load simulation with JSON/PNG output."""

import argparse
import csv
import json
import math
import random
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEVICES = ("mobile", "tablet", "desktop")


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[index], 3)


def post_json(url, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
        return result, (time.perf_counter() - started) * 1000, None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return None, (time.perf_counter() - started) * 1000, type(exc).__name__


def run_session(base_url, scenario_id, ordinal, timeout):
    client_id = f"device-{ordinal:05d}"
    device_type = DEVICES[ordinal % len(DEVICES)]
    requests = []
    for answer in range(1, 6):
        payload = {
            "client_id": client_id,
            "request_id": str(uuid.uuid4()),
            "scenario_id": scenario_id,
            "patient_answers": answer,
            "device_type": device_type,
        }
        response, latency_ms, error = post_json(
            f"{base_url.rstrip('/')}/v1/simulations/consultations/turn",
            payload,
            timeout,
        )
        requests.append({
            "patient_answers": answer,
            "latency_ms": latency_ms,
            "error": error,
            "replica": None if response is None else response.get("replica"),
        })
        if error or response.get("should_stop"):
            break
    return {
        "client_id": client_id,
        "device_type": device_type,
        "scenario_id": scenario_id,
        "requests": requests,
        "stopped_at": requests[-1]["patient_answers"],
    }


def summarize(concurrency, sessions, elapsed):
    requests = [request for session in sessions for request in session["requests"]]
    latencies = [request["latency_ms"] for request in requests if not request["error"]]
    errors = [request for request in requests if request["error"]]
    return {
        "concurrency": concurrency,
        "sessions": len(sessions),
        "requests": len(requests),
        "elapsed_seconds": round(elapsed, 3),
        "throughput_requests_per_second": round(len(requests) / elapsed, 3),
        "error_rate": round(len(errors) / len(requests), 6) if requests else 0.0,
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": round(max(latencies), 3) if latencies else None,
        },
        "replica_requests": dict(sorted(Counter(
            request["replica"] for request in requests if request["replica"]
        ).items())),
        "device_sessions": dict(sorted(Counter(
            session["device_type"] for session in sessions
        ).items())),
        "stop_turns": dict(sorted(Counter(
            str(session["stopped_at"]) for session in sessions
        ).items())),
    }


def draw_report(report, path):
    width, height = 1500, 850
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    title = ImageFont.load_default(size=34)
    draw.text((50, 35), "Concurrent conversational inference", fill="#14213d", font=title)
    experiments = report["experiments"]
    max_latency = max(item["latency_ms"]["p99"] or 1 for item in experiments)
    max_rate = max(item["throughput_requests_per_second"] for item in experiments) or 1
    colors = {"p50": "#2a9d8f", "p95": "#f4a261", "p99": "#e76f51"}
    for panel, (label, maximum) in enumerate((("Latency (ms)", max_latency), ("Throughput (req/s)", max_rate))):
        left = 70 + panel * 740
        top = 125
        draw.text((left, top), label, fill="#14213d", font=font)
        baseline = 700
        draw.line((left, baseline, left + 620, baseline), fill="#6c757d", width=2)
        for index, item in enumerate(experiments):
            x = left + 35 + index * max(100, 560 // max(1, len(experiments)))
            if panel == 0:
                for offset, key in enumerate(("p50", "p95", "p99")):
                    value = item["latency_ms"][key] or 0
                    bar_height = 500 * value / maximum
                    draw.rectangle((x + offset * 22, baseline - bar_height, x + offset * 22 + 18, baseline), fill=colors[key])
            else:
                value = item["throughput_requests_per_second"]
                bar_height = 500 * value / maximum
                draw.rectangle((x, baseline - bar_height, x + 55, baseline), fill="#3a86ff")
            draw.text((x, baseline + 15), str(item["concurrency"]), fill="#14213d", font=font)
        draw.text((left + 210, 770), "Concurrent devices", fill="#495057", font=font)
    draw.text((70, 810), "Latency bars: green p50, amber p95, red p99", fill="#495057", font=font)
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--data", type=Path, default=Path("data/generated/synthetic_scenarios.csv"))
    parser.add_argument("--concurrency", default="1,4,16,32")
    parser.add_argument("--sessions", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=2111)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.data.open(newline="", encoding="utf-8") as source:
        scenario_ids = [row["scenario_id"] for row in csv.DictReader(source)]
    rng = random.Random(args.seed)
    experiments = []
    for concurrency in [int(value) for value in args.concurrency.split(",")]:
        chosen = [rng.choice(scenario_ids) for _ in range(args.sessions)]
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(run_session, args.base_url, scenario_id, index, args.timeout)
                for index, scenario_id in enumerate(chosen)
            ]
            sessions = [future.result() for future in as_completed(futures)]
        experiments.append(summarize(concurrency, sessions, time.perf_counter() - started))
    report = {
        "base_url": args.base_url,
        "seed": args.seed,
        "sessions_per_experiment": args.sessions,
        "experiments": experiments,
        "warning": "Synthetic load and synthetic patient data; not a clinical or capacity SLA.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "load-test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    draw_report(report, args.output_dir / "load-test.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
