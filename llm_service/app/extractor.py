"""app/extractor.py

Patient message -> ExtractionResult, with every symptom ID checked against
the vocabulary. IDs the LLM invents are dropped and their text is kept in
unmapped_mentions instead.
"""
import json
import re

from .llm_client import LLM
from .schema import ExtractionResult, Profile
from .vocab import Vocab, get_vocab

VAGUE_PRESENTING_REPLIES = re.compile(
    r"^(?:(?:i\s+)?(?:do\s*not|don't|dunno)\s+know|not sure|unsure|"
    r"same(?: thing)?|nothing|no idea|hard to say|just sick|feel(?:ing)? sick)"
    r"[.! ]*$",
    re.IGNORECASE,
)


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


async def extract_symptoms(
    llm: LLM, message: str, profile: Profile, question_target: dict | None = None
) -> ExtractionResult:
    vocab = get_vocab()
    result = await llm.complete_structured(
        "extract",
        {
            "patient_message": message.strip(),
            "known_symptoms": vocab.prompt_listing(),
            "current_profile": _profile_for_prompt(profile),
            "question_target": json.dumps(question_target),
        },
        ExtractionResult,
    )
    result = sanitize(result, vocab)
    # Once the server explicitly asks for the presenting complaint, never
    # silently discard a substantive answer merely because an extractor cannot
    # map it to the model's closed vocabulary. Preserving the patient's words as
    # an unmapped finding lets safety route it to clinician review instead of
    # repeating the opening prompt.
    is_presenting_answer = (
        question_target is not None
        and question_target.get("target") == "presenting_symptoms"
    )
    if (
        is_presenting_answer
        and not result.symptoms
        and not result.unmapped_mentions
        and not VAGUE_PRESENTING_REPLIES.fullmatch(message.strip())
    ):
        result = result.model_copy(update={"unmapped_mentions": [message.strip()]})
    return result
