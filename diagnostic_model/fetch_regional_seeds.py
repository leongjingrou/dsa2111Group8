"""Fetch small, auditable SEA style-seed previews from Hugging Face."""

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path


SOURCES = (
    {
        "source_id": "vimed_sft",
        "dataset": "quannguyen204/medical_sft_crawl_vi_10k_v1",
        "country": "Vietnam",
        "language": "vi",
    },
    {
        "source_id": "indohealth_ncd_sample",
        "dataset": "IndoHealth-NLP/NCD_Instruct-Tuning_Medical_QA_Indonesian",
        "country": "Indonesia",
        "language": "id",
    },
)


def _preview(dataset):
    query = urllib.parse.urlencode({
        "dataset": dataset, "config": "default", "split": "train",
    })
    request = urllib.request.Request(
        f"https://datasets-server.huggingface.co/first-rows?{query}",
        headers={"User-Agent": "dsa2111-grounding-audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())["rows"]


def _text(source_id, row):
    if source_id == "vimed_sft":
        users = [message["content"] for message in row["messages"] if message["role"] == "user"]
        return "\n".join(users)
    return row["instruction"]


def fetch_regional_seeds():
    records = []
    for source in SOURCES:
        for item in _preview(source["dataset"]):
            records.append({
                "source_id": source["source_id"],
                "record_id": f"{source['source_id']}:{item['row_idx']}",
                "country": source["country"],
                "language": source["language"],
                "text": _text(source["source_id"], item["row"]),
                "retrieval": "Hugging Face datasets-server first-rows API",
            })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = fetch_regional_seeds()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(records), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
