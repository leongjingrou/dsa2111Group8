"""Render split-level and individual stopping-policy maps as PNG images."""

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


INK = "#172033"
MUTED = "#627087"
GRID = "#DCE2EA"
BLUE = "#2F6FED"
GREEN = "#238B67"
ORANGE = "#E17A2D"
RED = "#C74343"


def _font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _title(draw, title, subtitle, width):
    draw.text((65, 40), title, fill=INK, font=_font(40, True))
    draw.text((65, 94), subtitle, fill=MUTED, font=_font(21))
    draw.line((65, 136, width - 65, 136), fill=GRID, width=2)


def draw_split_overview(report, output, approach):
    width, height = 1800, 1150
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    _title(
        draw,
        f"{approach}: learned stopping across train / validation / test",
        "Five-answer maximum · threshold chosen on validation · urgent misses explicitly penalized",
        width,
    )
    splits = ["train", "validation", "test"]
    colours = [BLUE, ORANGE, GREEN]
    left, top, chart_height = 165, 220, 560
    group_width = 500
    for split_index, (split, colour) in enumerate(zip(splits, colours)):
        metrics = report["splits"][split]
        x0 = left + split_index * group_width
        draw.text((x0, 165), split.title(), fill=INK, font=_font(27, True))
        values = [
            ("Diagnosis", metrics["condition_accuracy"]),
            ("Escalation", metrics["escalation_accuracy"]),
            ("Joint", metrics["joint_accuracy"]),
            ("Urgent recall", metrics["urgent_recall"]),
        ]
        for row, (label, value) in enumerate(values):
            y = top + row * 120
            draw.text((x0, y), label, fill=INK, font=_font(19, True))
            draw.rounded_rectangle((x0, y + 34, x0 + 380, y + 65), 12, fill="#EEF2F7")
            draw.rounded_rectangle((x0, y + 34, x0 + 380 * value, y + 65), 12, fill=colour)
            draw.text((x0 + 392, y + 37), f"{value:.1%}", fill=INK, font=_font(18, True))
        draw.text(
            (x0, top + chart_height),
            f"Mean answers: {metrics['mean_patient_answers']:.2f} / 5",
            fill=INK,
            font=_font(21, True),
        )
        draw.text(
            (x0, top + chart_height + 38),
            f"Stopped early: {metrics['stopped_before_final']:.1%}",
            fill=MUTED,
            font=_font(19),
        )
        distribution = metrics["stop_distribution"]
        total = sum(distribution.values())
        y = top + chart_height + 93
        draw.text((x0, y), "Stop answer", fill=INK, font=_font(17, True))
        for answer in range(1, 6):
            count = distribution.get(str(answer), 0)
            bar = 300 * count / max(1, total)
            yy = y + 31 + answer * 26
            draw.text((x0, yy), str(answer), fill=MUTED, font=_font(15))
            draw.rectangle((x0 + 28, yy + 2, x0 + 28 + bar, yy + 17), fill=colour)
            draw.text((x0 + 340, yy), str(count), fill=MUTED, font=_font(15))
    draw.text(
        (65, 1110),
        f"Selected stop threshold: {report['selected_stop_threshold']:.3f}. Synthetic data; not clinical validation.",
        fill=MUTED,
        font=_font(17),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG", optimize=True)


def _selected_samples(records, count=4):
    test = [record for record in records if record.get("split") == "test"]
    early = [record for record in test if record["selected_answer"] < 5]
    urgent = [record for record in early if record["truth"]["escalation"] == "urgent"]
    routine = [record for record in early if record["truth"]["escalation"] == "routine"]
    final = [record for record in test if record["selected_answer"] == 5]
    wrong = []
    for record in test:
        answer = record["selected_answers"]
        if (
            answer["condition"]["choice"] != record["truth"]["condition"]
            or answer["escalation"]["choice"] != record["truth"]["escalation"]
        ):
            wrong.append(record)
    chosen = []
    for pool in (urgent, routine, final, wrong, test):
        for record in pool:
            if record not in chosen:
                chosen.append(record)
                break
    return chosen[:count]


def draw_sample_maps(records, output, approach):
    samples = _selected_samples(records)
    width, panel_height = 2200, 510
    height = 155 + panel_height * len(samples) + 55
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    _title(
        draw,
        f"{approach}: individual five-answer decision maps",
        "Each tile shows the live diagnosis / escalation confidence and the learned probability of stopping",
        width,
    )
    for sample_index, record in enumerate(samples):
        y0 = 165 + sample_index * panel_height
        selected = record["selected_answer"]
        truth = record["truth"]
        draw.text(
            (55, y0),
            f"{record['scenario_id']}  ·  truth: {truth['condition']} / {truth['escalation']}  ·  stop at answer {selected}",
            fill=INK,
            font=_font(23, True),
        )
        tile_width = 405
        for turn_index, checkpoint in enumerate(record["confidence_trace"]):
            x0 = 55 + turn_index * 425
            tile_y = y0 + 47
            stopped = turn_index + 1 == selected
            border = GREEN if stopped else GRID
            draw.rounded_rectangle(
                (x0, tile_y, x0 + tile_width, tile_y + 405),
                radius=16,
                fill="#F8FAFC" if not stopped else "#EFFAF6",
                outline=border,
                width=5 if stopped else 2,
            )
            draw.text((x0 + 18, tile_y + 16), f"Answer {turn_index + 1}", fill=INK, font=_font(20, True))
            if stopped:
                draw.text((x0 + 300, tile_y + 18), "STOP", fill=GREEN, font=_font(18, True))
            response = " ".join(textwrap.wrap(checkpoint["latest_response"], width=43)[:3])
            draw.multiline_text(
                (x0 + 18, tile_y + 55), textwrap.fill(response, 43),
                fill=MUTED, font=_font(16), spacing=5,
            )
            answers = checkpoint["answers"]
            condition = answers["condition"]
            escalation = answers["escalation"]
            draw.text((x0 + 18, tile_y + 158), "Diagnosis", fill=MUTED, font=_font(15, True))
            draw.text(
                (x0 + 18, tile_y + 183),
                f"{condition['choice']}  {condition['answer_confidence']:.0%}",
                fill=INK,
                font=_font(18, True),
            )
            draw.text((x0 + 18, tile_y + 229), "Escalation", fill=MUTED, font=_font(15, True))
            escalation_colour = RED if escalation["choice"] == "urgent" else INK
            draw.text(
                (x0 + 18, tile_y + 254),
                f"{escalation['choice']}  {escalation['answer_confidence']:.0%}",
                fill=escalation_colour,
                font=_font(18, True),
            )
            stop_probability = checkpoint.get("stop_probability", 0.0)
            draw.text((x0 + 18, tile_y + 306), "Stop probability", fill=MUTED, font=_font(15, True))
            draw.rounded_rectangle((x0 + 18, tile_y + 337, x0 + 370, tile_y + 361), 9, fill="#E4EAF2")
            draw.rounded_rectangle(
                (x0 + 18, tile_y + 337, x0 + 18 + 352 * stop_probability, tile_y + 361),
                9,
                fill=GREEN if stopped else BLUE,
            )
            draw.text((x0 + 315, tile_y + 370), f"{stop_probability:.0%}", fill=INK, font=_font(16, True))
        draw.line((55, y0 + 475, width - 55, y0 + 475), fill=GRID, width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG", optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--approach", required=True)
    args = parser.parse_args()
    report = json.loads(args.metrics.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in args.decisions.read_text(encoding="utf-8").splitlines()]
    safe_name = args.approach.lower().replace(" ", "-")
    draw_split_overview(report, args.output_dir / f"{safe_name}-split-overview.png", args.approach)
    draw_sample_maps(records, args.output_dir / f"{safe_name}-sample-maps.png", args.approach)
    print(json.dumps({"files": 2, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
