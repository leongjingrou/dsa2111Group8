"""app/explainer.py

Turns a DecisionRecord into plain language. The recommended action is decided
elsewhere; the headline and disclaimer are FIXED text chosen by code, so the
LLM can never soften or change the urgency. The LLM only writes the explanatory
body, and a deterministic fallback is used if it fails.
"""
from .llm_client import LLM, LLMError
from .schema import Audience, DecisionRecord, ExplainResponse

PATIENT_HEADLINES: dict[str, str] = {
    "self_care": "Based on what you have shared, self-care at home may be suitable for now.",
    "routine_teleconsultation": "We recommend a routine teleconsultation with a doctor.",
    "specialist_review": "We recommend a review by a specialist.",
    "in_person_assessment": "We recommend that you be seen in person by a healthcare professional.",
    "urgent_medical_review": (
        "Your symptoms need urgent medical review. Please seek care now. "
        "If you feel severely unwell, call 995."
    ),
}

CLINICIAN_HEADLINES: dict[str, str] = {
    "self_care": "Recommended pathway: self-care guidance.",
    "routine_teleconsultation": "Recommended pathway: routine teleconsultation.",
    "specialist_review": "Recommended pathway: specialist review.",
    "in_person_assessment": "Recommended pathway: in-person assessment.",
    "urgent_medical_review": "Recommended pathway: URGENT medical review.",
}

DISCLAIMERS: dict[str, str] = {
    "patient": (
        "This is an automated pre-screening, not a diagnosis. "
        "A healthcare professional makes the final decision."
    ),
    "clinician": (
        "Automated pre-screening output for decision support only. "
        "Clinical judgement remains with the treating clinician."
    ),
}


def _fallback_body(record: DecisionRecord) -> str:
    parts = []
    if record.triggered_rules:
        parts.append("Reasons: " + "; ".join(record.triggered_rules) + ".")
    if record.missing_information:
        parts.append("Information not yet confirmed: " + "; ".join(record.missing_information) + ".")
    if record.limitations:
        parts.append("Limitations: " + "; ".join(record.limitations) + ".")
    return " ".join(parts) or "Please speak to a healthcare professional about your concerns."


async def explain(llm: LLM, record: DecisionRecord, audience: Audience = "patient") -> ExplainResponse:
    headlines = PATIENT_HEADLINES if audience == "patient" else CLINICIAN_HEADLINES
    try:
        body = await llm.complete_text(
            "explain",
            {"audience": audience, "decision_record": record.model_dump_json()},
            temperature=0.3,
        )
        fallback_used = False
    except LLMError:
        body = _fallback_body(record)
        fallback_used = True
    return ExplainResponse(
        headline=headlines[record.recommended_action],
        explanation=body.strip(),
        disclaimer=DISCLAIMERS[audience],
        fallback_used=fallback_used,
    )
