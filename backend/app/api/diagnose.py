from fastapi import APIRouter, HTTPException, Request

from llm_service.orchestrated_main import receive_turn
from llm_service.session_store import SessionCapacity

from ..schemas import DiagnoseRequest, Diagnosis

router = APIRouter()


@router.post("/diagnose", response_model=Diagnosis)
async def diagnose(payload: DiagnoseRequest, request: Request):
    """Process one patient answer through the shared persisted consultation."""
    service = request.app.state.consultations
    try:
        session_id = payload.conversation_id or service.start()
    except SessionCapacity as exc:
        raise HTTPException(503, str(exc)) from exc
    message = (
        payload.patient_message
        if payload.patient_message is not None
        else payload.messages[-1].text
    )
    return await receive_turn(service, session_id, message, payload.request_id)
