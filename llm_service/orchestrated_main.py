"""Consultation HTTP API with mandatory versioned safety and replica-safe state."""

import os
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from diagnostic_model.live_conversation import (
    GREETING,
    EBMLivePredictor,
    LayaLivePredictor,
)
from llm_service.api_schema import ConsultationResponse
from llm_service.app.keyword_llm import KeywordLLM
from llm_service.app.llm_client import LLMClient, LLMError
from llm_service.consultation_service import (
    ConsultationFinished,
    ConsultationService,
    RequestReplayConflict,
)
from llm_service.session_store import (
    SessionCapacity,
    SessionConflict,
    SessionMissing,
    SessionStore,
)
from safety_layer import AuditLogError, SafetyEngine


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    safety_profile: Literal["rules_v1", "sg_moh"] = "rules_v1"


class TurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_message: str = Field(min_length=1, max_length=2000, pattern=r"\S")
    request_id: str | None = Field(default=None, min_length=1, max_length=64)


def _predictor_from_environment():
    backend = os.getenv("MODEL_BACKEND", "ebm").lower()
    if backend == "ebm":
        return EBMLivePredictor(os.environ["EBM_LIVE_MODEL"])
    if backend == "laya":
        return LayaLivePredictor(
            os.environ["LAYA_MODEL_DIR"], os.getenv("LAYA_DEVICE", "cuda")
        )
    raise ValueError(f"Unsupported MODEL_BACKEND: {backend}")


def _llm_from_environment():
    backend = os.getenv("LLM_BACKEND", "remote").lower()
    if backend == "remote":
        return LLMClient()
    if backend == "keyword":
        return KeywordLLM()
    raise ValueError(f"Unsupported LLM_BACKEND: {backend}")


async def receive_turn(service, session_id, message, request_id=None):
    try:
        return await service.turn(session_id, message, request_id)
    except SessionMissing:
        raise HTTPException(404, "Unknown or expired consultation") from None
    except (ConsultationFinished, SessionConflict, RequestReplayConflict) as exc:
        raise HTTPException(409, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(
            503,
            {
                "message": "The language provider is unavailable; assessment has not completed.",
                "session_id": session_id,
            },
        ) from exc
    except AuditLogError as exc:
        raise HTTPException(
            503,
            {
                "message": "Safety audit storage is unavailable.",
                "safety": exc.decision.to_dict(),
                "reply": exc.decision.patient_message,
            },
        ) from exc


def create_orchestrated_app(
    predictor=None, stopping=None, llm=None, *, safety_engine=None, session_store=None
):
    @asynccontextmanager
    async def lifespan(app):
        audit_path = Path(
            os.getenv(
                "SAFETY_AUDIT_LOG",
                f"runtime/audit-{socket.gethostname()}-{os.getpid()}.jsonl",
            )
        )
        engine = (
            safety_engine
            if safety_engine is not None
            else SafetyEngine(audit_log=audit_path)
        )
        if engine.audit_log is not None:
            engine.audit_log.parent.mkdir(parents=True, exist_ok=True)
            with engine.audit_log.open("a", encoding="utf-8"):
                pass
        app.state.predictor = (
            predictor if predictor is not None else _predictor_from_environment()
        )
        stop_bundle = stopping if stopping is not None else os.environ["STOPPING_MODEL"]
        app.state.stopping = (
            joblib.load(stop_bundle)
            if isinstance(stop_bundle, (str, Path))
            else stop_bundle
        )
        app.state.llm = llm if llm is not None else _llm_from_environment()
        app.state.safety_engine = engine
        app.state.store = (
            session_store
            if session_store is not None
            else SessionStore(
                os.getenv("CONSULTATION_DB", "runtime/consultations.sqlite3"),
                ttl_seconds=int(os.getenv("CONSULTATION_TTL_SECONDS", "3600")),
                maximum=int(os.getenv("MAX_ACTIVE_SESSIONS", "4096")),
            )
        )
        app.state.consultations = ConsultationService(
            app.state.predictor,
            app.state.stopping,
            app.state.llm,
            engine,
            app.state.store,
        )
        yield

    app = FastAPI(
        title="Safety-screened telemedicine consultation API", lifespan=lifespan
    )

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        return {
            "status": "ready",
            "backend": app.state.predictor.name,
            "safety_profile": "rules_v1",
            "policy_version": app.state.safety_engine.version,
            "policy_sha256": app.state.safety_engine.policy_sha256,
        }

    @app.post("/api/conversations")
    async def start(request: StartRequest = StartRequest()):
        try:
            session_id = app.state.consultations.start()
        except SessionCapacity as exc:
            raise HTTPException(503, str(exc)) from exc
        return {
            "session_id": session_id,
            "conversation_id": session_id,
            "backend": app.state.predictor.name,
            "doctor_response": GREETING,
            "safety_profile": "rules_v1",
            "policy_version": app.state.safety_engine.version,
        }

    @app.post(
        "/api/conversations/{session_id}/turn", response_model=ConsultationResponse
    )
    async def turn(session_id: str, request: TurnRequest):
        return await receive_turn(
            app.state.consultations,
            session_id,
            request.patient_message,
            request.request_id,
        )

    @app.delete("/api/conversations/{session_id}", status_code=204)
    async def discard(session_id: str):
        if not app.state.store.delete(session_id):
            raise HTTPException(404, "Unknown consultation")

    return app


app = create_orchestrated_app()
