import os

import pytest

from llm_service.app.semantic_matcher import semantic_match


pytestmark = pytest.mark.skipif(
    not os.getenv("SEMANTIC_MODEL_PATH"),
    reason="local semantic model is not configured",
)


@pytest.mark.parametrize(
    ("wording", "candidates", "expected"),
    [
        ("yeah a bit tired", ["cough", "chills", "fatigue"], "fatigue"),
        ("damn shag, no energy", ["cough", "chills", "fatigue"], "fatigue"),
        ("shivering like mad", ["cough", "chills", "fatigue"], "chills"),
        (
            "nose damn blocked",
            ["runny_nose", "nasal_congestion", "sneezing"],
            "nasal_congestion",
        ),
    ],
)
def test_local_semantic_model_handles_paraphrases_and_singlish(
    wording, candidates, expected
):
    assert semantic_match(wording, candidates).symptom_id == expected


@pytest.mark.parametrize(
    ("wording", "candidates"),
    [
        ("feel like puking", ["nausea", "vomiting", "diarrhoea"]),
        (
            "keep needing to pee",
            ["painful_urination", "frequent_urination", "urinary_urgency"],
        ),
        ("little sick", None),
    ],
)
def test_local_semantic_model_declines_ambiguous_or_vague_wording(
    wording, candidates
):
    assert semantic_match(wording, candidates) is None
