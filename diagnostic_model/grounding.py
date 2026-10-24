"""Audit external sources and convert DDXPlus rows into bounded conversations."""

import argparse
import ast
import csv
import json
from pathlib import Path


REGISTRY_PATH = Path(__file__).with_name("dataset_sources.json")


def load_registry(path=REGISTRY_PATH):
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"id", "name", "url", "region", "license", "role", "status", "notes"}
    ids = []
    for source in registry["sources"]:
        missing = required - set(source)
        if missing:
            raise ValueError(f"{source.get('id', '<unknown>')} missing {sorted(missing)}")
        ids.append(source["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset source IDs must be unique")
    return registry


def _structured(value):
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def _evidence_parts(token):
    pieces = str(token).split("_@_", 1)
    return pieces[0], pieces[1] if len(pieces) == 2 else None


def _group_evidences(tokens):
    """Combine DDXPlus multi-choice values that share one question code."""
    order = []
    values = {}
    for token in tokens:
        code, value = _evidence_parts(token)
        if code not in values:
            order.append(code)
            values[code] = []
        if value is not None:
            values[code].append(value)
    return [
        code if not values[code] else f"{code}_@_{'|||'.join(values[code])}"
        for code in order
    ]


def _answer_for(token, metadata):
    code, value = _evidence_parts(token)
    detail = metadata.get(code, {})
    if value is None:
        return "Yes."
    meanings = detail.get("value_meaning", {})
    rendered = []
    for item in value.split("|||"):
        meaning = meanings.get(item, item)
        if isinstance(meaning, dict):
            meaning = meaning.get("en", item)
        rendered.append(str(meaning))
    return f"{', '.join(rendered)}."


def _question_for(token, metadata):
    code, _ = _evidence_parts(token)
    detail = metadata.get(code, {})
    return detail.get("question_en") or f"Can you tell me whether {code} applies?"


def ddxplus_record(row, evidence_metadata, split, maximum_answers=5):
    """Create an auditable fixed-cap transcript; it is reconstructed, not recorded."""
    evidences = list(_structured(row["EVIDENCES"]))
    initial = str(row.get("INITIAL_EVIDENCE", evidences[0] if evidences else ""))
    initial_code = _evidence_parts(initial)[0]
    grouped = _group_evidences(evidences)
    ordered = [initial] + [item for item in grouped if _evidence_parts(item)[0] != initial_code]
    selected = ordered[:maximum_answers]
    turns = [{"role": "doctor", "content": "Hello, what seems to be the problem today?"}]
    for index, token in enumerate(selected):
        question = _question_for(token, evidence_metadata)
        if index:
            turns.append({"role": "doctor", "content": question})
            response = _answer_for(token, evidence_metadata)
        else:
            response = f"Regarding '{question}', my answer is {_answer_for(token, evidence_metadata).lower()}"
        turns.append({"role": "patient", "content": response})
    while sum(turn["role"] == "patient" for turn in turns) < maximum_answers:
        turns.extend([
            {"role": "doctor", "content": "Is there anything else important that has not been covered?"},
            {"role": "patient", "content": "No additional information is recorded in this source case."},
        ])
    return {
        "source": "ddxplus",
        "source_split": split,
        "source_patient_id": row.get("PATIENT_ID"),
        "age": float(row["AGE"]),
        "sex": row["SEX"],
        "target_condition": row["PATHOLOGY"],
        "differential_diagnosis": _structured(row.get("DIFFERENTIAL_DIAGNOSIS", "[]")),
        "source_evidence_tokens": evidences,
        "initial_evidence": initial,
        "conversation": turns,
        "reconstruction": True,
        "warning": "Synthetic DDXPlus evidence rendered as dialogue; not a recorded consultation or clinical validation.",
    }


def convert_ddxplus(csv_path, evidences_path, output_path, split, maximum_answers=5, limit=None):
    metadata = json.loads(Path(evidences_path).read_text(encoding="utf-8"))
    count = 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(csv_path).open(encoding="utf-8", newline="") as source, output_path.open("w", encoding="utf-8", newline="\n") as target:
        for row in csv.DictReader(source):
            target.write(json.dumps(ddxplus_record(row, metadata, split, maximum_answers), ensure_ascii=False) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    convert = subparsers.add_parser("convert-ddxplus")
    convert.add_argument("csv", type=Path)
    convert.add_argument("evidences", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--split", required=True, choices=("train", "validate", "test"))
    convert.add_argument("--maximum-answers", type=int, default=5)
    convert.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "audit":
        print(json.dumps(load_registry(), indent=2))
    else:
        count = convert_ddxplus(args.csv, args.evidences, args.output, args.split,
                                args.maximum_answers, args.limit)
        print(json.dumps({"rows": count, "output": str(args.output)}))


if __name__ == "__main__":
    main()
