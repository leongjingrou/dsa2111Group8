"""Run llm_service with NO API key, using a crude keyword-matching fake LLM.

Run from the REPO ROOT:
    uvicorn llm_service.dev_fake_main:app --reload --port 8001
Then open http://localhost:8001/docs and try POST /api/llm/extract with e.g.
    {"patient_message": "I have a fever for 2 days but no cough"}

This only exercises your pipeline (vocab check, merge, endpoints). It says
nothing about how a real model will perform: use a real key for that.
"""
from llm_service.app.router import get_llm_client, router
from llm_service.app.keyword_llm import KeywordLLM

from fastapi import FastAPI

app = FastAPI(title="llm_service (fake dev)")

app.include_router(router, prefix="/api")
app.dependency_overrides[get_llm_client] = lambda: KeywordLLM()
