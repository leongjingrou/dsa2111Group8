"""app/router.py

HTTP layer only
Mounted by main.py:  app.include_router(llm_router, prefix="/api")
-> /api/llm/extract, /api/llm/next-question, /api/llm/explain, /api/llm/health
The service is stateless: the caller sends the current profile and gets the new one back.
"""
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException

from .explainer import explain
from .extractor import extract_symptoms
from .llm_client import LLM, LLMClient, LLMError
from .merge import merge_profile
from .questioner import phrase_question
from .schema import (
    ExplainRequest,
    ExplainResponse,
    ExtractRequest,
    ExtractResponse,
    NextQuestionRequest,
    NextQuestionResponse,
)
from .vocab import UnknownSymptomError

router = APIRouter(prefix="/llm", tags=["llm"])


@lru_cache
def get_llm_client() -> LLM:
    return LLMClient()  # override in tests with FakeLLMClient


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest, llm: LLM = Depends(get_llm_client)):
    try:
        extraction = await extract_symptoms(llm, req.patient_message, req.current_profile)
    except LLMError:
        raise HTTPException(status_code=502, detail="LLM provider error during extraction")
    profile = merge_profile(req.current_profile, extraction)
    return ExtractResponse(profile=profile, extraction=extraction)


@router.post("/next-question", response_model=NextQuestionResponse)
async def next_question(req: NextQuestionRequest, llm: LLM = Depends(get_llm_client)):
    try:
        question, fallback_used = await phrase_question(llm, req.symptom_id, req.focus)
    except UnknownSymptomError:
        raise HTTPException(status_code=422, detail=f"Unknown symptom_id: {req.symptom_id}")
    return NextQuestionResponse(question=question, fallback_used=fallback_used)


@router.post("/explain", response_model=ExplainResponse)
async def explain_endpoint(req: ExplainRequest, llm: LLM = Depends(get_llm_client)):
    return await explain(llm, req.record, req.audience)
