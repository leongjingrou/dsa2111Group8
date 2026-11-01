import asyncio
import json

from llm_service.app import keyword_llm
from llm_service.app.keyword_llm import KeywordLLM
from llm_service.app.schema import ExtractionResult
from llm_service.app.semantic_matcher import SemanticMatch


def test_keyword_fallback_understands_reordered_multiword_symptom():
    result = asyncio.run(
        KeywordLLM().complete_structured(
            "extract",
            {"patient_message": "nose all runny since ytd"},
            ExtractionResult,
        )
    )

    runny_nose = next(item for item in result.symptoms if item.symptom_id == "runny_nose")
    assert runny_nose.presence == "present"
    assert "nose all runny" in runny_nose.evidence


def test_keyword_fallback_distinguishes_presence_from_progression_wording():
    llm = KeywordLLM()
    progression = asyncio.run(
        llm.complete_text(
            "ask",
            {
                "symptom_name": "runny nose",
                "focus_instruction": "whether it is getting better, staying the same or getting worse",
            },
        )
    )
    presence = asyncio.run(
        llm.complete_text(
            "ask",
            {
                "symptom_name": "runny nose",
                "focus_instruction": "whether the patient currently has this symptom",
            },
        )
    )

    assert "getting better" in progression
    assert "noticed" in presence


def test_keyword_extractor_understands_natural_stable_progression():
    result = asyncio.run(
        KeywordLLM().complete_structured(
            "extract",
            {
                "patient_message": (
                    "moderate fever since yesterday and it is staying the same"
                )
            },
            ExtractionResult,
        )
    )

    fever = next(item for item in result.symptoms if item.symptom_id == "fever")
    assert fever.severity == "moderate"
    assert fever.duration_days == 1.0
    assert fever.progression == "stable"


def test_keyword_vector_match_is_only_an_uncertain_clarification_candidate():
    result = asyncio.run(
        KeywordLLM().complete_structured(
            "extract",
            {"patient_message": "back pain and cramps a bit"},
            ExtractionResult,
        )
    )

    candidate = next(
        symptom for symptom in result.symptoms if symptom.symptom_id == "flank_pain"
    )
    assert candidate.presence == "uncertain"


def test_keyword_vector_fallback_handles_free_wording_without_forcing_weak_match():
    llm = KeywordLLM()
    matched = asyncio.run(
        llm.complete_structured(
            "extract", {"patient_message": "cannot breathe properly"}, ExtractionResult
        )
    )
    vague = asyncio.run(
        llm.complete_structured(
            "extract", {"patient_message": "little sick"}, ExtractionResult
        )
    )

    candidate = next(
        symptom
        for symptom in matched.symptoms
        if symptom.symptom_id == "shortness_of_breath"
    )
    assert candidate.presence == "uncertain"
    assert vague.symptoms == []


def test_grouped_question_uses_local_semantics_for_a_clear_paraphrase(monkeypatch):
    monkeypatch.setattr(
        keyword_llm,
        "semantic_match",
        lambda text, candidates=None: SemanticMatch("fatigue", 0.47, 0.29),
    )
    result = asyncio.run(
        KeywordLLM().complete_structured(
            "extract",
            {
                "patient_message": "yeah a bit tired",
                "question_target": json.dumps(
                    {
                        "target": "condition_symptoms",
                        "targets": ["cough", "chills", "fatigue"],
                    }
                ),
            },
            ExtractionResult,
        )
    )

    assert [(item.symptom_id, item.presence) for item in result.symptoms] == [
        ("fatigue", "present")
    ]


def test_unconstrained_semantic_match_still_requires_patient_clarification(monkeypatch):
    monkeypatch.setattr(
        keyword_llm,
        "semantic_match",
        lambda text, candidates=None: SemanticMatch("fatigue", 0.5, 0.2),
    )
    result = asyncio.run(
        KeywordLLM().complete_structured(
            "extract", {"patient_message": "damn shag today"}, ExtractionResult
        )
    )

    assert result.symptoms[0].symptom_id == "fatigue"
    assert result.symptoms[0].presence == "uncertain"


def test_grouped_semantic_negation_is_preserved(monkeypatch):
    monkeypatch.setattr(
        keyword_llm,
        "semantic_match",
        lambda text, candidates=None: SemanticMatch("fatigue", 0.48, 0.25),
    )
    result = asyncio.run(
        KeywordLLM().complete_structured(
            "extract",
            {
                "patient_message": "not tired at all",
                "question_target": json.dumps(
                    {
                        "target": "condition_symptoms",
                        "targets": ["cough", "chills", "fatigue"],
                    }
                ),
            },
            ExtractionResult,
        )
    )

    assert result.symptoms[0].symptom_id == "fatigue"
    assert result.symptoms[0].presence == "absent"
