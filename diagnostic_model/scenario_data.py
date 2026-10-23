import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from knowledge_base.retrieval import load_json


SEED = 2111
DEFAULT_SCENARIO_PATH = Path("data/generated/synthetic_scenarios.csv")
IDENTIFIER_COLUMNS = {"scenario_id", "vignette_id"}
LABEL_COLUMNS = {"diagnosis", "condition_id", "escalation", "scope_status"}
METADATA_COLUMNS = {"synthetic", "dataset_version", "vignette_text", "symptoms"}


def knowledge_base_condition_ids():
    return [item["id"] for item in load_json("conditions.json")["conditions"]]


def _as_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _symptom_ids(data):
    ids = set()
    for raw in data["symptoms"].fillna("{}").astype(str):
        ids.update(json.loads(raw))
    return sorted(ids)


def scenario_features(data, symptom_ids=None, include_anthropometrics=False):
    """Convert the scenario schema to leakage-safe numeric model features.

    Symptom answers are represented by two columns. ``asked`` distinguishes an
    explicit negative answer from a symptom that was never assessed, while
    ``present`` stores the observed yes/no value.
    """
    required = {
        "age", "sex", "pregnancy_status", "temperature", "duration_days",
        "severity", "progression", "symptoms", "condition_id", "escalation",
        "scope_status",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Scenario data is missing required columns: {missing}")

    symptom_ids = list(symptom_ids or _symptom_ids(data))
    columns = {
        "age": pd.to_numeric(data["age"], errors="raise"),
        "temperature_c": pd.to_numeric(data["temperature"], errors="raise"),
        "duration_days": pd.to_numeric(data["duration_days"], errors="raise"),
        "sex_female": data["sex"].astype(str).str.upper().eq("F").astype(int),
        "pregnant": data["pregnancy_status"].map(_as_bool).astype(int),
    }

    if include_anthropometrics:
        columns["weight_kg"] = pd.to_numeric(data["weight_kg"], errors="raise")
        columns["height_cm"] = pd.to_numeric(data["height_cm"], errors="raise")

    for value in ("mild", "normal", "severe"):
        columns[f"severity_{value}"] = data["severity"].astype(str).eq(value).astype(int)
    for value in ("improving", "stable", "worsening"):
        columns[f"progression_{value}"] = data["progression"].astype(str).eq(value).astype(int)

    parsed = [json.loads(raw) for raw in data["symptoms"].fillna("{}").astype(str)]
    for symptom_id in symptom_ids:
        columns[f"symptom_{symptom_id}_asked"] = [int(symptom_id in row) for row in parsed]
        columns[f"symptom_{symptom_id}_present"] = [
            int(_as_bool(row.get(symptom_id, False))) for row in parsed
        ]

    return pd.DataFrame(columns, index=data.index).astype(float), symptom_ids


def scenario_targets(data):
    return {
        "diagnosis": data["condition_id"].astype(str),
        "scope": data["scope_status"].astype(str).eq("in_scope").astype(int),
        "escalation": data["escalation"].map(_as_bool).astype(int),
    }


def scenario_split(data, test_size=0.25, seed=SEED):
    """Return one matched split shared by the EBM and Laya evaluations."""
    targets = scenario_targets(data)
    strata = data["condition_id"].astype(str) + "|" + targets["escalation"].astype(str)
    train_index, test_index = train_test_split(
        data.index.to_numpy(),
        test_size=test_size,
        random_state=seed,
        stratify=strata,
    )
    return np.sort(train_index), np.sort(test_index)


def load_scenarios(path=DEFAULT_SCENARIO_PATH):
    data = pd.read_csv(path)
    # Validate eagerly so failures happen before model fitting.
    scenario_features(data)
    return data


def regenerate_scenarios(path=DEFAULT_SCENARIO_PATH):
    """Run the data-branch-derived generator, then validate and load its output."""
    from data.generate import generate

    path = Path(path)
    if path.name != DEFAULT_SCENARIO_PATH.name:
        raise ValueError(
            f"Generated scenario path must end with {DEFAULT_SCENARIO_PATH.name!r}: {path}"
        )
    _, generated_path = generate(path.parent)
    return load_scenarios(generated_path)
