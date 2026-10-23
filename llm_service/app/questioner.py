"""app/questioner.py

The question controller decides WHICH symptom and WHAT aspect to ask about.
This module only words the question naturally. If the LLM fails or returns
something odd, a fixed template is used, so the conversation never stalls.
"""
from .llm_client import LLM, LLMError
from .schema import Focus
from .vocab import get_vocab

FOCUS_INSTRUCTIONS: dict[str, str] = {
    "presence": "whether the patient currently has this symptom",
    "severity": "how severe it is (mild, moderate or severe)",
    "duration": "how long they have had it",
    "progression": "whether it is getting better, staying the same or getting worse",
}

FALLBACK_TEMPLATES: dict[str, str] = {
    "presence": "Do you have {name}?",
    "severity": "How severe is your {name}: mild, moderate or severe?",
    "duration": "How long have you had {name}?",
    "progression": "Is your {name} getting better, staying the same or getting worse?",
}


def _acceptable(text: str) -> bool:
    text = text.strip()
    return "?" in text and "\n" not in text and len(text) <= 250


async def phrase_question(llm: LLM, symptom_id: str, focus: Focus = "presence") -> tuple[str, bool]:
    """Return (question, fallback_used). Raises UnknownSymptomError for a bad ID."""
    name = get_vocab().name(symptom_id).lower()  # raises UnknownSymptomError if invalid
    try:
        text = await llm.complete_text(
            "ask",
            {"symptom_name": name, "focus_instruction": FOCUS_INSTRUCTIONS[focus]},
            temperature=0.3,
        )
    except LLMError:
        return FALLBACK_TEMPLATES[focus].format(name=name), True
    text = text.strip().strip('"')
    if not _acceptable(text):
        return FALLBACK_TEMPLATES[focus].format(name=name), True
    return text, False
