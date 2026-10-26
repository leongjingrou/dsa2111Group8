"""app/merge.py

Deterministic (no LLM) merge of a new ExtractionResult into the running Profile.
Pure function: the input profile is never mutated.

Rules:
- A definite answer (present/absent) overrides an earlier one; a flip is
  recorded in profile.contradictions so the safety layer can see it.
- "uncertain" never overwrites a definite answer.
- New non-null details (severity, duration, progression) override old ones;
  null never erases an old detail.
- Marking a symptom "absent" clears its severity/duration/progression.
- missing_info is recomputed every time.
"""
from .schema import Contradiction, ExtractionResult, Profile, SymptomState

DEFINITE = {"present", "absent"}

# Profile field -> focus name used by /next-question
DETAIL_FOCUS = {"severity": "severity", "duration_days": "duration", "progression": "progression"}


def _union(existing: list[str], new: list[str]) -> list[str]:
    """Append new items, ignoring blanks and case-insensitive duplicates."""
    out = list(existing)
    seen = {x.lower() for x in out}
    for item in new:
        item = item.strip()
        if item and item.lower() not in seen:
            out.append(item)
            seen.add(item.lower())
    return out


def compute_missing_info(profile: Profile) -> list[str]:
    missing: list[str] = []
    states = profile.symptoms.values()
    if not any(s.presence == "present" for s in states):
        missing.append("presenting_symptoms")
    for s in states:
        if s.presence == "uncertain":
            missing.append(f"{s.symptom_id}:presence")
        elif s.presence == "present":
            for field, focus in DETAIL_FOCUS.items():
                if getattr(s, field) is None:
                    missing.append(f"{s.symptom_id}:{focus}")
    return missing


def merge_profile(profile: Profile, extraction: ExtractionResult) -> Profile:
    new = profile.model_copy(deep=True)

    for s in extraction.symptoms:
        old = new.symptoms.get(s.symptom_id)
        if old is None:
            state = SymptomState(**s.model_dump())
        else:
            presence = old.presence
            if s.presence in DEFINITE:
                if old.presence in DEFINITE and s.presence != old.presence:
                    c = Contradiction(symptom_id=s.symptom_id, previous=old.presence, new=s.presence)
                    if c not in new.contradictions:
                        new.contradictions.append(c)
                presence = s.presence
            state = SymptomState(
                symptom_id=s.symptom_id,
                presence=presence,
                severity=s.severity or old.severity,
                duration_days=s.duration_days if s.duration_days is not None else old.duration_days,
                progression=s.progression or old.progression,
                evidence=s.evidence or old.evidence,
            )
        if state.presence == "absent":
            state.severity = state.duration_days = state.progression = None
        new.symptoms[s.symptom_id] = state

    new.medical_history = _union(new.medical_history, extraction.medical_history)
    new.allergies = _union(new.allergies, extraction.allergies)
    new.medications = _union(new.medications, extraction.medications)
    new.unmapped_mentions = _union(new.unmapped_mentions, extraction.unmapped_mentions)
    new.missing_info = compute_missing_info(new)
    new.turn_count += 1
    return new
