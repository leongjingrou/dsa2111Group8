"""Run llm_service with NO API key, using a crude keyword-matching fake LLM.

Run from the REPO ROOT:
    uvicorn llm_service.dev_fake_main:app --reload --port 8001
Then open http://localhost:8001/docs and try POST /api/llm/extract with e.g.
    {"patient_message": "I have a fever for 2 days but no cough"}

This only exercises your pipeline (vocab check, merge, endpoints). It says
nothing about how a real model will perform: use a real key for that.
"""
import re

from llm_service.app.router import get_llm_client, router
from llm_service.app.schema import ExtractedSymptom, ExtractionResult
from llm_service.app.vocab import get_vocab

from fastapi import FastAPI

app = FastAPI(title="llm_service (fake dev)")

NEGATIONS = ("no ", "not ", "don't", "dont", "haven't", "havent", "without")
SEVERITY = {"severe": "severe", "terrible": "severe", "very bad": "severe",
            "moderate": "moderate", "mild": "mild", "slight": "mild"}


class KeywordLLM:
    async def complete_structured(self, prompt_name, variables, output_model, temperature=0.0):
        msg = variables["patient_message"].lower()
        severity = next((v for k, v in SEVERITY.items() if k in msg), None)  # crude: applies to all
        m = re.search(r"(\d+)\s*day", msg)
        days = float(m.group(1)) if m else None

        symptoms = []
        for sid, entry in get_vocab()._entries.items():
            terms = [sid.replace("_", " "), entry.get("name", "").lower(),
                     *[s.lower() for s in entry.get("synonyms", [])]]
            hit = next((t for t in terms if t and t in msg), None)
            if not hit:
                continue
            before = msg[max(0, msg.index(hit) - 15): msg.index(hit)]
            present = not any(n in before for n in NEGATIONS)
            symptoms.append(ExtractedSymptom(
                symptom_id=sid,
                presence="present" if present else "absent",
                severity=severity if present else None,
                duration_days=days if present else None,
                progression=None,
                evidence=hit,
            ))
        return ExtractionResult(symptoms=symptoms, unmapped_mentions=[],
                                medical_history=[], allergies=[], medications=[])

    async def complete_text(self, prompt_name, variables, temperature=0.3):
        if prompt_name == "ask":
            return f"Could you tell me more about your {variables['symptom_name']}?"
        return "This is a placeholder explanation from the fake LLM."

app.include_router(router, prefix="/api")
app.dependency_overrides[get_llm_client] = lambda: KeywordLLM()
