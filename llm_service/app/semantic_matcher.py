"""Small local semantic retriever for patient phrasing.

Model2Vec supplies CPU-only dense embeddings.  The retrieved label is evidence
for a clarification unless the server has just offered a small explicit set of
symptoms and the reply clearly selects one of them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .vocab import get_vocab

DESCRIPTIONS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "semantic_symptom_descriptions.json"
)


@dataclass(frozen=True)
class SemanticMatch:
    symptom_id: str
    score: float
    margin: float


def _model_source() -> str:
    return os.getenv("SEMANTIC_MODEL_PATH", "").strip()


@lru_cache(maxsize=1)
def _descriptions() -> dict[str, str]:
    return json.loads(DESCRIPTIONS_PATH.read_text(encoding="utf-8"))


def _document(symptom_id: str) -> str:
    vocab = get_vocab()
    phrases = ", ".join(vocab.phrases(symptom_id))
    description = _descriptions().get(symptom_id, "")
    return f"Symptom: {vocab.name(symptom_id)}. Known wording: {phrases}. {description}".strip()


@lru_cache(maxsize=2)
def _index(model_source: str):
    if not model_source:
        return None
    try:
        from model2vec import StaticModel

        model = StaticModel.from_pretrained(model_source, normalize=True)
    except (ImportError, OSError, ValueError):
        return None
    symptom_ids = sorted(get_vocab().ids())
    matrix = np.asarray(
        model.encode(
            [_document(symptom_id) for symptom_id in symptom_ids],
            use_multiprocessing=False,
        ),
        dtype=np.float32,
    )
    return model, symptom_ids, matrix


def semantic_match(
    text: str,
    candidate_ids: list[str] | None = None,
) -> SemanticMatch | None:
    """Return a well-separated local semantic match, never a diagnosis."""
    indexed = _index(_model_source())
    if indexed is None or not text.strip():
        return None
    model, symptom_ids, matrix = indexed
    allowed = set(candidate_ids or symptom_ids)
    indices = [index for index, symptom_id in enumerate(symptom_ids) if symptom_id in allowed]
    if not indices:
        return None
    query = np.asarray(
        model.encode([text], use_multiprocessing=False), dtype=np.float32
    )[0]
    scores = matrix[indices] @ query
    order = np.argsort(scores)[::-1]
    best_position = int(order[0])
    best_score = float(scores[best_position])
    second_score = float(scores[int(order[1])]) if len(order) > 1 else 0.0
    margin = best_score - second_score
    # A restricted question provides useful context, so it permits a slightly
    # lower absolute threshold while retaining a strong separation requirement.
    threshold = 0.25 if candidate_ids else 0.43
    required_margin = 0.10 if candidate_ids else 0.13
    if best_score < threshold or margin < required_margin:
        return None
    return SemanticMatch(symptom_ids[indices[best_position]], best_score, margin)
