"""app/vocab.py

Loads the symptom vocabulary from data/symptoms.csv (or data/symptoms.json as a
fallback). This is the single source of truth for which symptom IDs exist.
The LLM may only use IDs from here.

CSV columns (header row required, names are case-insensitive):
  symptom_id            required (an "id" column also works); must match Role 3's data
  name                  display name
  category              optional; used to group the list shown to the LLM
  is_red_flag           optional (True/False); kept for other code, NEVER shown to the LLM
  associated_diagnoses  optional; ignored here, NEVER shown to the LLM
  synonyms              optional; separated by | or ;

The LLM only ever sees id, name, category and synonyms. It must not see which
symptoms are red flags or which diagnoses they point to, because it must not
diagnose or judge urgency.
"""

import csv
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CSV_PATH = DATA_DIR / "generated" / "symptoms.csv"
JSON_PATH = DATA_DIR / "symptoms.json"
SAFETY_PATH = Path(__file__).resolve().parents[1] / "data" / "safety_symptoms.json"
JSON_PATH = Path(__file__).resolve().parents[1] / "data" / "symptoms.json"
SYNONYM_SPLIT = re.compile(r"[|;]")
TRUE_VALUES = {"true", "1", "yes", "y"}


class UnknownSymptomError(ValueError):
    """Raised when a symptom ID is not in the vocabulary."""


class Vocab:
    def __init__(self, entries: list[dict]) -> None:
        if not entries:
            raise ValueError("Symptom vocabulary is empty")
        dupes = [i for i, n in Counter(e["id"] for e in entries).items() if n > 1]
        if dupes:
            raise ValueError(f"Duplicate symptom ids: {dupes}")
        self._entries = {e["id"]: e for e in entries}

    def ids(self) -> set[str]:
        return set(self._entries)

    def is_valid(self, symptom_id: str) -> bool:
        return symptom_id in self._entries

    def name(self, symptom_id: str) -> str:
        if symptom_id not in self._entries:
            raise UnknownSymptomError(symptom_id)
        entry = self._entries[symptom_id]
        return entry.get("name") or symptom_id.replace("_", " ")

    def phrases(self, symptom_id: str) -> list[str]:
        """Return explicit names a patient might use for a known symptom."""
        if symptom_id not in self._entries:
            raise UnknownSymptomError(symptom_id)
        entry = self._entries[symptom_id]
        return list(dict.fromkeys(
            [symptom_id.replace("_", " "), entry.get("name", "")]
            + list(entry.get("synonyms") or [])
        ))

    def is_red_flag(self, symptom_id: str) -> bool:
        """For the safety layer / orchestrator. Not used by the LLM prompts."""
        if symptom_id not in self._entries:
            raise UnknownSymptomError(symptom_id)
        return bool(self._entries[symptom_id].get("is_red_flag"))

    def prompt_listing(self) -> str:
        """Symptom list for the extract prompt, grouped by category when available."""
        groups: dict[str, list[str]] = {}
        for sid, entry in self._entries.items():
            line = f"- {sid}: {entry.get('name') or sid.replace('_', ' ')}"
            synonyms = entry.get("synonyms") or []
            if synonyms:
                line += f" (also: {', '.join(synonyms)})"
            groups.setdefault(entry.get("category") or "", []).append(line)
        blocks = []
        for category, lines in groups.items():
            header = f"[{category}]\n" if category else ""
            blocks.append(header + "\n".join(lines))
        return "\n\n".join(blocks)


def _load_csv(path: Path) -> list[dict]:
    entries = []
    with path.open(
        newline="", encoding="utf-8-sig"
    ) as f:  # utf-8-sig strips Excel's BOM
        for row in csv.DictReader(f):
            row = {
                (k or "").strip().lower(): (v.strip() if isinstance(v, str) else "")
                for k, v in row.items()
            }
            name = row.get("name") or row.get("symptom") or ""
            sid = row.get("symptom_id") or row.get("id") or ""
            if not sid and name:
                sid = re.sub(r"\W+", "_", name.lower()).strip("_")
            if not sid:
                continue  # skip blank lines
            synonyms = [
                s.strip()
                for s in SYNONYM_SPLIT.split(row.get("synonyms", ""))
                if s.strip()
            ]
            entries.append(
                {
                    "id": sid,
                    "name": name or sid.replace("_", " "),
                    "synonyms": synonyms,
                    "category": row.get("category", ""),
                    "is_red_flag": row.get("is_red_flag", "").lower() in TRUE_VALUES,
                }
            )
    return entries


def _load_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["symptoms"] if isinstance(data, dict) else data


@lru_cache
def get_vocab() -> Vocab:
    if CSV_PATH.exists():
        entries = _load_csv(CSV_PATH)
    elif JSON_PATH.exists():
        entries = _load_json(JSON_PATH)
    else:
        raise FileNotFoundError(
            f"No symptom file found. Expected {CSV_PATH} or {JSON_PATH}"
        )
    known = {entry["id"] for entry in entries}
    entries.extend(
        entry for entry in _load_json(SAFETY_PATH) if entry["id"] not in known
    )
    return Vocab(entries)
