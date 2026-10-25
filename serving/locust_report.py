"""Summarize Locust full-history CSV and draw a compact saturation chart."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _number(row, *names, default=0.0):
    for name in names:
        if row.get(name) not in (None, ""):
            try:
                return float(row[name])
            except ValueError:
                # Locust reports percentile fields as N/A before the first
                # request. Those warm-up rows are valid zero-valued samples.
                continue
    return default


def summarize_history(path, workers=None, api_replica_ceiling=None):
    with Path(path).open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if row.get("Name") == "Aggregated"]
    if not rows:
        raise ValueError("Locust history contains no Aggregated rows")
    points = []
    grouped = defaultdict(list)
    for row in rows:
        point = {
            "timestamp": int(float(row["Timestamp"])),
            "users": int(_number(row, "User Count")),
            "requests_per_second": _number(row, "Requests/s", "Current RPS"),
            "failures_per_second": _number(row, "Failures/s", "Current Failures/s"),
            "p50_ms": _number(row, "50%"),
            "p95_ms": _number(row, "95%"),
            "p99_ms": _number(row, "99%"),
        }
        points.append(point)
        grouped[point["users"]].append(point)
    stages = []
    for users, values in sorted(grouped.items()):
        stages.append({
            "users": users,
            "samples": len(values),
            "mean_requests_per_second": round(sum(v["requests_per_second"] for v in values) / len(values), 3),
            "peak_requests_per_second": round(max(v["requests_per_second"] for v in values), 3),
            "mean_failures_per_second": round(sum(v["failures_per_second"] for v in values) / len(values), 3),
            "median_p95_ms": round(sorted(v["p95_ms"] for v in values)[len(values) // 2], 3),
            "max_p99_ms": round(max(v["p99_ms"] for v in values), 3),
        })
    report = {
        "maximum_users_observed": max(point["users"] for point in points),
        "points": points,
        "stages": stages,
        "warning": "Synthetic workload on one server; not a production capacity SLA or clinical validation.",
    }
    if workers is not None:
        report["locust_worker_pods"] = workers
    if api_replica_ceiling is not None:
        report["api_replica_ceiling"] = api_replica_ceiling
    return report


def draw_report(report, output):
    width, height = 1700, 1000
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=19)
    title = ImageFont.load_default(size=34)
    draw.text((55, 35), "Kubernetes conversational inference ramp", fill="#172033", font=title)
    points = report["points"]
    start = points[0]["timestamp"]
    duration = max(1, points[-1]["timestamp"] - start)
    panels = [
        ("Concurrent users", "users", "#2F6FED"),
        ("Requests / second", "requests_per_second", "#238B67"),
        ("Tail latency p95 (ms)", "p95_ms", "#C74343"),
    ]
    for index, (label, key, colour) in enumerate(panels):
        top = 125 + index * 270
        left, right, bottom = 125, 1625, top + 205
        maximum = max(1, max(point[key] for point in points))
        draw.text((55, top), label, fill="#172033", font=font)
        draw.rectangle((left, top + 35, right, bottom), outline="#DCE2EA", width=2)
        coordinates = []
        for point in points:
            x = left + (right - left) * (point["timestamp"] - start) / duration
            y = bottom - (bottom - top - 35) * point[key] / maximum
            coordinates.append((x, y))
        if len(coordinates) > 1:
            draw.line(coordinates, fill=colour, width=4, joint="curve")
        draw.text((right - 160, top + 43), f"max {maximum:.1f}", fill=colour, font=font)
    draw.text(
        (55, 945),
        "Green/red in diagnostic pathway figures denote source-label match/mismatch; red here denotes latency only.",
        fill="#627087",
        font=font,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG", optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--api-replica-ceiling", type=int)
    args = parser.parse_args()
    report = summarize_history(args.history, args.workers, args.api_replica_ceiling)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "locust-summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    draw_report(report, args.output_dir / "locust-timeseries.png")
    print(json.dumps({"maximum_users_observed": report["maximum_users_observed"]}, indent=2))


if __name__ == "__main__":
    main()
