import asyncio

from llm_service.app.explainer import explain
from llm_service.app.keyword_llm import KeywordLLM
from llm_service.app.schema import ConditionEstimate, DecisionRecord


def test_completed_assessment_names_preliminary_match_and_doctor_review():
    record = DecisionRecord(
        recommended_action="routine_teleconsultation",
        preliminary_estimates=[ConditionEstimate(condition="common_cold", score=0.72)],
    )

    result = asyncio.run(explain(KeywordLLM(), record))

    assert "common cold" in result.explanation.lower()
    assert "doctor" in result.explanation.lower()
    assert "confirm" in result.explanation.lower()
    assert "not a diagnosis" in result.disclaimer.lower()
