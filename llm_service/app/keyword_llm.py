"""Deterministic language adapter for integration and load testing.

This adapter deliberately performs no diagnosis or urgency assessment.  It is
useful when measuring the local orchestration stack without including latency,
cost, rate limits, or availability from an external LLM provider.
"""

import re

from .negation import is_negated
from .schema import ExtractedSymptom, ExtractionResult
from .vocab import get_vocab

SEVERITY = {
    "severe": "severe",
    "terrible": "severe",
    "very bad": "severe",
    "moderate": "moderate",
    "mild": "mild",
    "slight": "mild",
}


class KeywordLLM:
    """OpenAI-client-compatible deterministic benchmark double."""

    async def complete_structured(
        self, prompt_name, variables, output_model, temperature=0.0
    ):
        if prompt_name != "extract":
            raise ValueError(f"Unsupported structured prompt: {prompt_name}")
        message = variables["patient_message"]
        text = message.lower()
        severity = next((value for key, value in SEVERITY.items() if key in text), None)
        duration = re.search(r"(\d+(?:\.\d+)?)\s*(day|week|month)s?", text)
        days = None
        if duration:
            days = (
                float(duration.group(1))
                * {"day": 1, "week": 7, "month": 30}[duration.group(2)]
            )
        progression = next(
            (
                value
                for key, value in (
                    ("wors", "worsening"),
                    ("improv", "improving"),
                    ("stable", "stable"),
                    ("unchanged", "stable"),
                )
                if key in text
            ),
            None,
        )

        symptoms = []
        for symptom_id, entry in get_vocab()._entries.items():
            terms = [
                symptom_id.replace("_", " "),
                entry.get("name", "").lower(),
                *[item.lower() for item in entry.get("synonyms", [])],
            ]
            hits = [
                (match.start(), term)
                for term in terms
                if term
                for match in re.finditer(rf"\b{re.escape(term)}\b", text)
            ]
            if not hits:
                continue
            positive_hits = [
                (start, term) for start, term in hits if not is_negated(text, start)
            ]
            start, hit = min(positive_hits or hits)
            present = bool(positive_hits)
            symptoms.append(
                ExtractedSymptom(
                    symptom_id=symptom_id,
                    presence="present" if present else "absent",
                    severity=severity if present else None,
                    duration_days=days if present else None,
                    progression=progression if present else None,
                    evidence=message[start : start + len(hit)],
                )
            )
        return ExtractionResult(
            symptoms=symptoms,
            unmapped_mentions=[],
            medical_history=[],
            allergies=[],
            medications=[],
        )

    async def complete_text(self, prompt_name, variables, temperature=0.3):
        if prompt_name == "ask":
            return f"Could you tell me {variables['focus_instruction']} for your {variables['symptom_name']}?"
        if prompt_name == "ask_context":
            return variables["question_instruction"].rstrip(".") + "?"
        if prompt_name == "explain":
            return "The diagnostic model used the symptoms reported so far and its calibrated decision threshold."
        raise ValueError(f"Unsupported text prompt: {prompt_name}")
