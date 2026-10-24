"""Build deterministic, provenance-bearing samples from medical and SEA seeds.

DDXPlus supplies the diagnosis, differential and evidence.  Audited Southeast
Asian records supply only region/language/style metadata: they never change a
medical label or add a symptom.  This separation prevents a non-medical style
corpus from silently becoming a clinical source.
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from diagnostic_model.grounding import load_registry


SEED = 2111
GREETINGS = (
    "Hello, what seems to be the problem today?",
    "Hello. What brings you in today?",
    "Hi, could you tell me what has been troubling you?",
)


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stable_split(value):
    bucket = int(hashlib.sha256(str(value).encode()).hexdigest()[:8], 16) % 10
    if bucket < 2:
        return "test"
    if bucket == 2:
        return "validation"
    return "train"


def _normal_split(value):
    return "validation" if value == "validate" else value


def validate_regional_seeds(records, registry=None):
    registry = registry or load_registry()
    sources = {source["id"]: source for source in registry["sources"]}
    accepted = []
    for index, record in enumerate(records):
        required = {"source_id", "record_id", "country", "language", "text"}
        missing = required - set(record)
        if missing:
            raise ValueError(f"regional seed {index} missing {sorted(missing)}")
        source = sources.get(record["source_id"])
        if source is None:
            raise ValueError(f"unknown source_id: {record['source_id']}")
        if source["status"].startswith("hold_"):
            raise ValueError(f"source is on hold and cannot seed generation: {record['source_id']}")
        role = source["role"]
        if "language" not in role and "cultural" not in role:
            raise ValueError(f"source is not approved for regional style grounding: {record['source_id']}")
        item = dict(record)
        item["assigned_split"] = record.get("assigned_split") or _stable_split(record["record_id"])
        item["source_url"] = source["url"]
        item["source_license"] = source["license"]
        item["source_role"] = role
        accepted.append(item)
    return accepted


def _sample_id(medical, regional, variant):
    identity = "|".join((
        str(medical.get("source_patient_id")),
        regional["source_id"],
        str(regional["record_id"]),
        str(variant),
    ))
    return "GRD-" + hashlib.sha256(identity.encode()).hexdigest()[:16]


def build_grounded_samples(ddxplus_records, regional_records, variants_per_case=1, seed=SEED):
    regional_records = validate_regional_seeds(regional_records)
    by_split = defaultdict(list)
    for item in regional_records:
        by_split[_normal_split(item["assigned_split"])].append(item)
    output = []
    for case_index, medical in enumerate(ddxplus_records):
        split = _normal_split(medical["source_split"])
        candidates = by_split[split]
        if not candidates:
            raise ValueError(f"no Southeast Asian seed assigned to {split}; cross-split reuse is disabled")
        for variant in range(variants_per_case):
            chooser = random.Random(f"{seed}:{case_index}:{variant}:{medical.get('source_patient_id')}")
            regional = candidates[chooser.randrange(len(candidates))]
            conversation = [dict(turn) for turn in medical["conversation"]]
            conversation[0]["content"] = GREETINGS[chooser.randrange(len(GREETINGS))]
            excerpt = " ".join(str(regional["text"]).split())[:280]
            output.append({
                "sample_id": _sample_id(medical, regional, variant),
                "split": split,
                "conversation": conversation,
                "target_condition": medical["target_condition"],
                "differential_diagnosis": medical["differential_diagnosis"],
                "source_evidence_tokens": medical["source_evidence_tokens"],
                "grounding": {
                    "medical_seed": {
                        "source": medical["source"],
                        "source_url": medical["source_url"],
                        "source_citation": medical.get("source_citation"),
                        "source_patient_id": medical.get("source_patient_id"),
                        "source_split": medical["source_split"],
                    },
                    "regional_style_seed": {
                        "source_id": regional["source_id"],
                        "source_url": regional["source_url"],
                        "source_license": regional["source_license"],
                        "source_role": regional["source_role"],
                        "record_id": regional["record_id"],
                        "country": regional["country"],
                        "language": regional["language"],
                        "assigned_split": regional["assigned_split"],
                        "text_sha256": hashlib.sha256(str(regional["text"]).encode()).hexdigest(),
                        "text_excerpt": excerpt,
                    },
                    "generation_seed": seed,
                    "variant": variant,
                    "constraints": {
                        "diagnosis_and_evidence_from_ddxplus_only": True,
                        "regional_seed_used_for_style_metadata_only": True,
                        "regional_text_injected_into_clinical_facts": False,
                        "cross_split_seed_reuse": False,
                    },
                },
                "warning": (
                    "Synthetic DDXPlus case paired with an audited Southeast Asian style seed; "
                    "not a recorded consultation or clinical validation."
                ),
            })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ddxplus_jsonl", type=Path)
    parser.add_argument("regional_seeds_jsonl", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variants-per-case", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    records = build_grounded_samples(
        _read_jsonl(args.ddxplus_jsonl),
        _read_jsonl(args.regional_seeds_jsonl),
        args.variants_per_case,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(records), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
