"""Load fictional safety cases with an explicitly documented negative screen."""

import copy
import json
from pathlib import Path

VIGNETTES = Path(__file__).parent / "tests" / "vignettes.json"


def load_vignettes(path=VIGNETTES):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def materialize_case(dataset, case):
    profile = copy.deepcopy(
        dataset["baseline_profile"] if case.get("use_baseline", True) else {}
    )
    for key, value in case.get("profile", {}).items():
        if key == "symptoms" and isinstance(value, dict):
            profile.setdefault(key, {}).update(value)
        else:
            profile[key] = copy.deepcopy(value)
    for path in case.get("remove", []):
        parts = path.split(".")
        parent = profile
        for part in parts[:-1]:
            parent = parent[part]
        parent.pop(parts[-1], None)
    return profile
