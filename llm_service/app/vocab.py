"""app/vocab.py

Loads data/symptoms.json. This is the single source of truth for which
symptom IDs exist. The LLM may only use IDs from here.

Accepted file shapes:
  [ {"id": "fever", "name": "Fever", "synonyms": ["high temperature"]}, ... ]
  { "symptoms": [ ...same as above... ] }
"""
import json
from functools import lru_cache
from pathlib import Path

SYMPTOMS_PATH = Path(__file__).resolve().parents[1] / "data" / "symptoms.json"


class UnknownSymptomError(ValueError):
    """Raised when a symptom ID is not in the vocabulary."""


class Vocab:
    def __init__(self, entries: list[dict]) -> None:
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

    def prompt_listing(self) -> str:
        """One line per symptom, for inclusion in the extract prompt."""
        lines = []
        for sid, entry in self._entries.items():
            line = f"- {sid}: {entry.get('name', sid.replace('_', ' '))}"
            synonyms = entry.get("synonyms") or []
            if synonyms:
                line += f" (also: {', '.join(synonyms)})"
            lines.append(line)
        return "\n".join(lines)


@lru_cache
def get_vocab() -> Vocab:
    data = json.loads(SYMPTOMS_PATH.read_text(encoding="utf-8"))
    entries = data["symptoms"] if isinstance(data, dict) else data
    return Vocab(entries)
