"""Export human-readable PNGs from a decision_analysis JSON report.

The renderer intentionally depends only on Pillow so it can run on the server
without a browser or a plotting stack.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


INK = "#172033"
MUTED = "#596579"
GRID = "#D9DFE8"
BLUE = "#2F6FED"
ORANGE = "#E17A2D"
GREEN = "#238B67"
RED = "#C74343"
PURPLE = "#7656C9"
TEAL = "#238F9D"
PALETTE = [BLUE, ORANGE, GREEN, PURPLE, TEAL, RED, "#8A6D3B"]


def _font(size, bold=False):
    names = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def _rgba(hex_colour, alpha):
    value = hex_colour.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def _title(draw, title, subtitle, width):
    draw.text((70, 42), title, fill=INK, font=_font(42, True))
    draw.text((70, 98), subtitle, fill=MUTED, font=_font(22))
    draw.line((70, 143, width - 70, 143), fill=GRID, width=2)


def _bezier(draw, points, fill, width):
    p0, p1, p2, p3 = points
    sampled = []
    for index in range(41):
        t = index / 40
        u = 1 - t
        sampled.append((
            u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0],
            u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1],
        ))
    draw.line(sampled, fill=fill, width=max(2, int(width)), joint="curve")


def draw_pathway(report, dimension, output, mode, approach, minimum=20):
    width, height = 2500, 1300
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image, "RGBA")
    label = "diagnosis" if dimension == "condition" else "escalation"
    _title(
        draw,
        f"{approach} {label} decision pathways — {mode}",
        f"Green = prediction matches source answer · red = mismatch · transitions with at least {minimum} cases",
        width,
    )
    pathway = report["pathways"][dimension]
    links = [link for link in pathway["links"] if link["count"] >= minimum]
    used = {value for link in links for value in (link["source"], link["target"])}
    nodes = [node for node in pathway["nodes"] if node["id"] in used]
    by_step = defaultdict(list)
    for node in nodes:
        by_step[node["patient_answer"]].append(node)
    top, bottom, gap = 210, 1180, 18
    # Leave enough room after the fifth column for full diagnosis labels.
    xs = {step: 120 + (step - 1) * 395 for step in range(1, 7)}
    layout = {}
    for step in range(1, 7):
        peers = sorted(by_step[step], key=lambda item: item["label"])
        total = sum(node["count"] for node in peers) or 1
        available = bottom - top - gap * max(0, len(peers) - 1)
        y = top
        for node in peers:
            node_height = max(8, available * node["count"] / total)
            layout[node["id"]] = (xs[step], y, y + node_height, node)
            y += node_height + gap
        heading = f"Answer {step}" if step <= 5 else "Finalised"
        draw.text((xs[step] - 36, 164), heading, fill=INK, font=_font(22, True))

    source_offsets = defaultdict(float)
    target_offsets = defaultdict(float)
    outgoing = defaultdict(int)
    incoming = defaultdict(int)
    for link in links:
        outgoing[link["source"]] += link["count"]
        incoming[link["target"]] += link["count"]
    for link in sorted(links, key=lambda item: (item["source"], item["target"])):
        sx, sy0, sy1, _ = layout[link["source"]]
        tx, ty0, ty1, _ = layout[link["target"]]
        source_scale = (sy1 - sy0) / max(1, outgoing[link["source"]])
        target_scale = (ty1 - ty0) / max(1, incoming[link["target"]])
        source_y = sy0 + source_offsets[link["source"]] + link["count"] * source_scale / 2
        target_y = ty0 + target_offsets[link["target"]] + link["count"] * target_scale / 2
        source_offsets[link["source"]] += link["count"] * source_scale
        target_offsets[link["target"]] += link["count"] * target_scale
        thickness = max(2, link["count"] * min(source_scale, target_scale) * 0.72)
        colour = GREEN if link["target_outcome"] == "correct" else RED
        _bezier(
            draw,
            ((sx + 20, source_y), (sx + 180, source_y), (tx - 160, target_y), (tx, target_y)),
            _rgba(colour, 85),
            thickness,
        )

    for _, (x, y0, y1, node) in layout.items():
        colour = GREEN if node["correct"] else RED
        draw.rounded_rectangle((x, y0, x + 20, y1), radius=5, fill=colour)
        text_y = (y0 + y1) / 2 - 11
        symbol = "CORRECT" if node["correct"] else "WRONG"
        draw.text(
            (x + 29, text_y),
            f"{node['display_label']} · {symbol}  {node['count']}",
            fill=INK,
            font=_font(16, True),
        )

    draw.text(
        (70, 1243),
        "Every path ends where that model finalised. Correctness is against the source label, not a clinician-confirmed diagnosis.",
        fill=MUTED,
        font=_font(18),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG", optimize=True)


def draw_intervals(report, output, mode, approach):
    width, height = 1800, 1080
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image, "RGBA")
    _title(
        draw,
        f"{approach} accuracy by conversation turn — {mode}",
        "Point estimate with Wilson 95% confidence interval · n = 850 conversations",
        width,
    )
    left, right, top, bottom = 155, 1710, 220, 905
    draw.rectangle((left, top, right, bottom), outline=GRID, width=2)
    for pct in range(0, 101, 10):
        y = bottom - (bottom - top) * pct / 100
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((85, y - 12), f"{pct}%", fill=MUTED, font=_font(18))
    steps = [metric["patient_answer"] for metric in report["turn_metrics"]]
    x_positions = {step: left + (right - left) * (step - 1) / 4 for step in steps}
    series = [("condition", "Diagnosis accuracy", BLUE), ("scope", "In/out-of-scope accuracy", GREEN), ("escalation", "Escalation accuracy", ORANGE)]
    for step in steps:
        x = x_positions[step]
        draw.line((x, top, x, bottom), fill=_rgba(GRID, 120), width=1)
        draw.text((x - 40, bottom + 24), f"Answer {step}", fill=INK, font=_font(19, True))
    for key, _, colour in series:
        points = []
        for metric in report["turn_metrics"]:
            x = x_positions[metric["patient_answer"]]
            value = metric[key]["accuracy"] * 100
            low, high = [bound * 100 for bound in metric[key]["accuracy_95_ci"]]
            y = bottom - (bottom - top) * value / 100
            y_low = bottom - (bottom - top) * low / 100
            y_high = bottom - (bottom - top) * high / 100
            points.append((x, y))
            draw.line((x, y_high, x, y_low), fill=colour, width=5)
            draw.line((x - 9, y_high, x + 9, y_high), fill=colour, width=4)
            draw.line((x - 9, y_low, x + 9, y_low), fill=colour, width=4)
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=colour)
            draw.text((x + 14, y - 25), f"{value:.1f}%", fill=INK, font=_font(17, True))
        draw.line(points, fill=colour, width=5, joint="curve")
    legend_x = 210
    for _, display, colour in series:
        draw.line((legend_x, 975, legend_x + 46, 975), fill=colour, width=6)
        draw.text((legend_x + 58, 962), display, fill=INK, font=_font(20))
        legend_x += 505
    draw.text((70, 1035), "Synthetic holdout; intervals quantify sampling uncertainty only and are not clinical validation.", fill=MUTED, font=_font(18))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG", optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True, help="Short input-mode label for titles and filenames")
    parser.add_argument("--approach", default="Laya", help="Model name shown in titles")
    parser.add_argument("--minimum-transition", type=int, default=20)
    args = parser.parse_args()
    report = json.loads(args.analysis.read_text())
    safe_name = args.name.lower().replace(" ", "-")
    for dimension in ("condition", "escalation"):
        draw_pathway(
            report,
            dimension,
            args.output_dir / f"{safe_name}-{dimension}-pathways.png",
            args.name,
            args.approach,
            args.minimum_transition,
        )
    draw_intervals(
        report,
        args.output_dir / f"{safe_name}-confidence-intervals.png",
        args.name,
        args.approach,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "files": 3}, indent=2))


if __name__ == "__main__":
    main()
