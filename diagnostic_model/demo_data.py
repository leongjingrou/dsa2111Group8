import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from knowledge_base.retrieval import load_json


def feature_name(text):
    return "symptom_" + re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def make_demo_dataset(rows_per_condition=30, seed=2111):
    rng = np.random.default_rng(seed)
    conditions = load_json("conditions.json")["conditions"]
    keywords = sorted({word for condition in conditions for word in condition["keywords"]})
    rows = []

    for condition in conditions:
        for i in range(rows_per_condition):
            row = {
                "vignette_id": f"synthetic-{condition['id']}-{i + 1:03d}",
                "age": rng.integers(18, 81),
                "duration_days": rng.integers(1, 11),
                "temperature_c": round(rng.normal(37, 0.55), 1),
                "condition_id": condition["id"],
            }
            for word in keywords:
                chance = 0.72 if word in condition["keywords"] else 0.06
                row[feature_name(word)] = int(rng.random() < chance)
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("diagnostic_model/demo_data.csv"))
    parser.add_argument("--rows-per-condition", type=int, default=30)
    args = parser.parse_args()
    data = make_demo_dataset(args.rows_per_condition)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Wrote {len(data)} rows to {args.output}")
