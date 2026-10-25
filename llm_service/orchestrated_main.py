"""FastAPI service joining LLM conversation with EBM/Laya decision layers."""

import asyncio
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from diagnostic_model.live_conversation import EBMLivePredictor, GREETING, LayaLivePredictor
from diagnostic_model.llm_live_conversation import LLMDrivenConsultation
from llm_service.app.llm_client import LLMClient
from llm_service.app.router import router as llm_router


class StartRequest(BaseModel):
    safety_profile: str = Field(default="none", pattern="^(none|sg_moh)$")


class TurnRequest(BaseModel):
    patient_message: str = Field(min_length=1, max_length=2000)


def _predictor_from_environment():
    backend = os.getenv("MODEL_BACKEND", "ebm").lower()
    if backend == "ebm":
        return EBMLivePredictor(os.environ["EBM_LIVE_MODEL"])
    if backend == "laya":
        return LayaLivePredictor(
            os.environ["LAYA_MODEL_DIR"], os.getenv("LAYA_DEVICE", "cuda")
        )
    raise ValueError(f"Unsupported MODEL_BACKEND: {backend}")


def create_orchestrated_app(predictor=None, stopping=None, llm=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = predictor or _predictor_from_environment()
        app.state.stopping = stopping or os.environ["STOPPING_MODEL"]
        app.state.llm = llm or LLMClient()
        yield

    app = FastAPI(
        title="LLM-mediated EBM/Laya consultation API",
        lifespan=lifespan,
    )
    app.state.sessions = {}
    app.state.session_locks = {}
    app.include_router(llm_router, prefix="/api")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/api/conversations")
    async def start(request: StartRequest):
        maximum = int(os.getenv("MAX_ACTIVE_SESSIONS", "4096"))
        if len(app.state.sessions) >= maximum:
            raise HTTPException(status_code=503, detail="active session capacity reached")
        session_id = str(uuid.uuid4())
        app.state.sessions[session_id] = LLMDrivenConsultation(
            app.state.predictor,
            app.state.stopping,
            app.state.llm,
            safety_profile=request.safety_profile,
        )
        app.state.session_locks[session_id] = asyncio.Lock()
        return {
            "session_id": session_id,
            "backend": app.state.predictor.name,
            "doctor_response": GREETING,
            "safety_profile": request.safety_profile,
        }

    @app.post("/api/conversations/{session_id}/turn")
    async def turn(session_id: str, request: TurnRequest):
        session = app.state.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown conversation")
        async with app.state.session_locks[session_id]:
            if session.finished:
                raise HTTPException(status_code=409, detail="conversation already finalised")
            checkpoint = await session.receive(request.patient_message)
        return {
            "session_id": session_id,
            "backend": checkpoint["diagnostic_backend"],
            "patient_answer": checkpoint["patient_answer"],
            "extraction": checkpoint["extraction"],
            "diagnosis": checkpoint["answers"]["condition"],
            "escalation": checkpoint["answers"]["escalation"],
            "stop_probability": checkpoint["stop_probability"],
            "stop_threshold": checkpoint["stop_threshold"],
            "stop": checkpoint["stop"],
            "question_target": checkpoint["question_target"],
            "question_fallback_used": checkpoint["question_fallback_used"],
            "doctor_response": checkpoint["doctor_response"],
            "safety_profile": checkpoint["safety_profile"],
            "safety_override_applied": checkpoint["safety_override_applied"],
        }

    return app


app = create_orchestrated_app()
