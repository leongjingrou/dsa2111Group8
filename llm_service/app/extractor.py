"""app/extractor.py

Patient message -> ExtractionResult, with every symptom ID checked against
the vocabulary. IDs the LLM invents are dropped and their text is kept in
unmapped_mentions instead.
"""
import json

from .llm_client import LLM
from .schema import ExtractionResult, Profile
from .vocab import Vocab, get_vocab


def _profile_for_prompt(profile: Profile) -> str:
    """Compact view of what we already know (no evidence text, no bookkeeping)."""
    return json.dumps({
        "symptoms": {
            sid: {
                "presence": s.presence,
                "severity": s.severity,
                "duration_days": s.duration_days,
                "progression": s.progression,
            }
            for sid, s in profile.symptoms.items()
        },
        "medical_history": profile.medical_history,
        "allergies": profile.allergies,
        "medications": profile.medications,
    })


def sanitize(result: ExtractionResult, vocab: Vocab) -> ExtractionResult:
    """Drop symptoms whose ID is not in the vocabulary; keep their text as unmapped."""
    kept = {}
    unmapped = list(result.unmapped_mentions)
    for s in result.symptoms:
        if not vocab.is_valid(s.symptom_id):
            unmapped.append(s.evidence or s.symptom_id)
            continue
        kept[s.symptom_id] = s  # if repeated within one message, the last mention wins
    return result.model_copy(update={"symptoms": list(kept.values()), "unmapped_mentions": unmapped})


async def extract_symptoms(llm: LLM, message: str, profile: Profile) -> ExtractionResult:
    vocab = get_vocab()
    result = await llm.complete_structured(
        "extract",
        {
            "patient_message": message.strip(),
            "known_symptoms": vocab.prompt_listing(),
            "current_profile": _profile_for_prompt(profile),
        },
        ExtractionResult,
    )
    return sanitize(result, vocab)
