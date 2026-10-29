import asyncio

from llm_service.app.keyword_llm import KeywordLLM
from llm_service.app.schema import ExtractionResult


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


def test_keyword_fallback_understands_alternative_display_label_wording():
    result = asyncio.run(
        KeywordLLM().complete_structured(
            "extract",
            {"patient_message": "back pain and cramps a bit"},
            ExtractionResult,
        )
    )

    assert "flank_pain" in {symptom.symptom_id for symptom in result.symptoms}


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

    assert "shortness_of_breath" in {
        symptom.symptom_id for symptom in matched.symptoms
    }
    assert vague.symptoms == []
