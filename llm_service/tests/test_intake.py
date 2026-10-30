import asyncio
import json

import pytest

from diagnostic_model.live_conversation import EvidenceState
from llm_service.app.extractor import extract_symptoms
from llm_service.app.llm_client import FakeLLMClient
from llm_service.app.schema import Profile, SymptomState
from llm_service.intake import (
    answer_target_for_reply,
    bind_answer,
    duration_without_unit,
    empty_extraction,
)


def test_bare_temperature_is_bound_to_temperature_question():
    evidence = EvidenceState()

    extraction = bind_answer(
        "37",
        {"kind": "context", "target": "temperature", "focus": None},
        Profile(),
        evidence,
    )

    assert extraction is not None
    assert evidence.temperature_c == 37.0
    assert evidence.temperature_asked is True


def test_temperature_and_sex_declines_are_recorded_as_asked():
    evidence = EvidenceState()
    profile = Profile()

    assert bind_answer(
        "not measured",
        {"kind": "context", "target": "temperature", "focus": None},
        profile,
        evidence,
    ) is not None
    assert bind_answer(
        "prefer not to say",
        {"kind": "context", "target": "sex", "focus": None},
        profile,
        evidence,
    ) is not None

    assert evidence.temperature_c is None
    assert evidence.temperature_asked is True
    assert evidence.sex is None
    assert evidence.sex_asked is True


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("I'm twenty-five years old", 25.0), ("I recently turned 42", 42.0)],
)
def test_age_question_accepts_natural_answers(answer, expected):
    evidence = EvidenceState()
    profile = Profile()
    assert bind_answer(
        answer, {"kind": "safety", "target": "age", "focus": None}, profile, evidence
    ) is not None
    assert profile.age == evidence.age == expected


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("No, I'm definitely not pregnant", False),
        ("nah cannot be pregnant leh", False),
        ("there's no chance of being pregnant", False),
        ("Yes, I am expecting", True),
        ("That does not apply to me", False),
    ],
)
def test_pregnancy_question_accepts_natural_answers(answer, expected):
    evidence = EvidenceState()
    profile = Profile()
    assert bind_answer(
        answer,
        {"kind": "safety", "target": "pregnancy_status", "focus": None},
        profile,
        evidence,
    ) is not None
    assert evidence.pregnant is expected


def test_temperature_accepts_a_sentence_and_fahrenheit():
    evidence = EvidenceState()
    assert bind_answer(
        "It was about 98.6 F when I checked",
        {"kind": "context", "target": "temperature", "focus": None},
        Profile(),
        evidence,
    ) is not None
    assert evidence.temperature_c == 37.0


@pytest.mark.parametrize(
    ("focus", "answer", "field", "expected"),
    [
        ("presence", "not that I've noticed", "presence", "absent"),
        ("duration", "for about two weeks now", "duration_days", 14.0),
        ("duration", "since yday morning, give or take", "duration_days", 1.0),
        ("duration", "since ytd lor", "duration_days", 1.0),
        ("severity", "it's not too bad, just slight", "severity", "mild"),
        ("severity", "quite jialat leh", "severity", "severe"),
        ("progression", "it has stayed about the same", "progression", "stable"),
        ("progression", "same lor", "progression", "stable"),
    ],
)
def test_symptom_questions_accept_free_form_replies(focus, answer, field, expected):
    symptom_id = "wheezing"
    profile = Profile(
        symptoms={symptom_id: SymptomState(symptom_id=symptom_id, presence="present")}
    )
    extraction = bind_answer(
        answer,
        {"kind": "symptom", "target": symptom_id, "focus": focus},
        profile,
        EvidenceState(),
    )
    assert extraction is not None
    assert getattr(extraction.symptoms[0], field) == expected


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("don't have leh", "absent"), ("bo lah", "absent"), ("got lor", "present")],
)
def test_symptom_presence_accepts_common_singlish(answer, expected):
    symptom_id = "wheezing"
    extraction = bind_answer(
        answer,
        {"kind": "symptom", "target": symptom_id, "focus": "presence"},
        Profile(),
        EvidenceState(),
    )
    assert extraction is not None
    assert extraction.symptoms[0].presence == expected


def test_remote_extractor_receives_pending_question_context():
    llm = FakeLLMClient(structured={"extract": empty_extraction()})
    target = {"kind": "symptom", "target": "wheezing", "focus": "duration"}
    history = [
        {"role": "doctor", "content": "How long?"},
        {"role": "patient", "content": "about a week"},
    ]

    asyncio.run(
        extract_symptoms(
            llm, "about a week", Profile(), target, conversation_history=history
        )
    )

    assert json.loads(llm.calls[0][1]["question_target"]) == target
    assert json.loads(llm.calls[0][1]["conversation_history"]) == history


def test_substantive_unmapped_presenting_answer_is_preserved_not_discarded():
    llm = FakeLLMClient(structured={"extract": empty_extraction()})
    target = {"kind": "context", "target": "presenting_symptoms", "focus": None}

    result = asyncio.run(
        extract_symptoms(llm, "strange cramps behind my shoulder", Profile(), target)
    )

    assert result.unmapped_mentions == ["strange cramps behind my shoulder"]


def test_vague_presenting_answer_can_still_be_clarified():
    llm = FakeLLMClient(structured={"extract": empty_extraction()})
    target = {"kind": "context", "target": "presenting_symptoms", "focus": None}

    result = asyncio.run(extract_symptoms(llm, "not sure", Profile(), target))

    assert result.unmapped_mentions == []


def test_colloquial_grouped_safety_denial_is_understood():
    evidence = EvidenceState()
    targets = ["chest_pain", "shortness_of_breath"]
    extraction = bind_answer(
        "nope none of that thank u",
        {
            "kind": "safety",
            "target": "emergency_warning_signs",
            "targets": targets,
            "focus": "presence",
        },
        Profile(),
        evidence,
    )
    assert extraction is not None
    assert all(evidence.symptoms[target] is False for target in targets)


def test_short_reply_can_explicitly_refer_to_a_recent_symptom_question():
    earlier = {"kind": "symptom", "target": "cough", "focus": "duration"}
    pending = {"kind": "safety", "target": "age", "focus": None}
    assert answer_target_for_reply("2d for the cough", pending, [earlier, pending]) == earlier
    assert answer_target_for_reply("30", pending, [earlier, pending]) == pending


def test_brief_duration_with_explicit_unit_is_bound_without_guessing():
    profile = Profile(symptoms={"cough": SymptomState(symptom_id="cough", presence="present")})
    target = {"kind": "symptom", "target": "cough", "focus": "duration"}
    extraction = bind_answer("2d for the cough", target, profile, EvidenceState())
    assert extraction.symptoms[0].duration_days == 2.0
    assert bind_answer("2", target, profile, EvidenceState()) is None


def test_named_yes_no_answer_can_bind_to_a_recent_presence_question():
    target = {"kind": "symptom", "target": "cough", "focus": "presence"}
    evidence = EvidenceState()
    extraction = bind_answer("no cough", target, Profile(), evidence)
    assert extraction.symptoms[0].presence == "absent"
    assert evidence.symptoms["cough"] is False


def test_bare_duration_is_clarified_and_unit_only_reply_uses_saved_number():
    profile = Profile(symptoms={"cough": SymptomState(symptom_id="cough", presence="present")})
    target = {"kind": "symptom", "target": "cough", "focus": "duration"}
    assert duration_without_unit("2", target) == "2"
    assert duration_without_unit("2d", target) is None
    clarified = {**target, "clarifying_number": "2"}
    extraction = bind_answer("days", clarified, profile, EvidenceState())
    assert extraction.symptoms[0].duration_days == 2.0
    assert extraction.symptoms[0].evidence == "days"
