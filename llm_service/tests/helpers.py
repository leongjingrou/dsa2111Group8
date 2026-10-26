from llm_service.app.schema import ExtractedSymptom, ExtractionResult
from llm_service.app.vocab import get_vocab


def real_ids(count):
    return sorted(get_vocab().ids())[:count]


def sym(
    symptom_id,
    presence="present",
    severity=None,
    duration_days=None,
    progression=None,
):
    return ExtractedSymptom(
        symptom_id=symptom_id,
        presence=presence,
        severity=severity,
        duration_days=duration_days,
        progression=progression,
        evidence=symptom_id,
    )


def ext(*symptoms, allergies=None, meds=None):
    return ExtractionResult(
        symptoms=list(symptoms),
        unmapped_mentions=[],
        medical_history=[],
        allergies=allergies or [],
        medications=meds or [],
    )
