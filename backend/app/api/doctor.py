"""Read-only, structured consultation trace for the local clinician view."""

from fastapi import APIRouter, HTTPException, Request

from llm_service.app.schema import Profile
from llm_service.condition_followups import supported_condition_matches
from llm_service.consultation_service import consultation_response
from llm_service.session_store import SessionMissing

router = APIRouter()


@router.get("/conversations/{session_id}/review")
async def review_consultation(session_id: str, request: Request):
    try:
        _, state = request.app.state.store.load(session_id)
    except SessionMissing:
        raise HTTPException(404, "Unknown or expired consultation") from None

    trace = state["trace"]
    doctor_turns = [turn["content"] for turn in state["turns"] if turn["role"] == "doctor"]
    pathway = []
    for index, checkpoint in enumerate(trace):
        # The stored trace is the actual server decision path. The preceding
        # doctor turn is the question that elicited this answer.
        pathway.append(
            {
                "turn": checkpoint["patient_answer"],
                "question_asked": doctor_turns[index],
                "patient_answer": checkpoint["latest_response"],
                "response": checkpoint["doctor_response"],
                "question_target": checkpoint.get("question_target"),
                "answer_target": checkpoint.get("answer_target"),
                "extraction": checkpoint["extraction"],
                "safety": checkpoint["safety"],
                "diagnosis": (checkpoint.get("answers") or {}).get("condition"),
                "escalation": (checkpoint.get("answers") or {}).get("escalation"),
                "decision_record": checkpoint.get("decision_record"),
                "stop_probability": checkpoint.get("stop_probability"),
                "stop_threshold": checkpoint["stop_threshold"],
                "stop": checkpoint["stop"],
            }
        )

    # A scope referral withholds model estimates. Show only text matches with
    # corroborating positive symptoms; retrieval scores are not probabilities.
    knowledge_context = None
    if trace and "SCOPE_CONDITION" in {
        hit["rule_id"] for hit in trace[-1]["safety"]["triggered_rules"]
    }:
        profile = Profile.model_validate(state["profile"])
        knowledge_context = supported_condition_matches(profile)

    return {
        "session_id": session_id,
        "finished": state["finished"],
        "pathway": pathway,
        "knowledge_context": knowledge_context,
        "latest": consultation_response(session_id, trace[-1]) if trace else None,
    }
