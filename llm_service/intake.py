"""Bind short replies to the question chosen by the server.

A bare "no" answers one pending symptom question. It never supplies negative
answers for the rest of the safety checklist.
"""

import re

from llm_service.app.keyword_llm import ranked_clarification_candidates
from llm_service.app.schema import ExtractedSymptom, ExtractionResult
from llm_service.app.vocab import get_vocab

UNCERTAIN_ANSWERS = {"not sure", "unsure", "i don't know"}
NEGATIVE_GROUP_ANSWERS = {"no", "nope", "none", "none of those", "not any of those"}
PRESENCE_ANSWERS = {
    "yes": "present",
    "no": "absent",
    **dict.fromkeys(UNCERTAIN_ANSWERS, "uncertain"),
}
PREGNANCY_ANSWERS = {
    "yes": "pregnant",
    "pregnant": "pregnant",
    "no": "not_pregnant",
    "not pregnant": "not_pregnant",
    "not applicable": "not_applicable",
    **dict.fromkeys(UNCERTAIN_ANSWERS, "uncertain"),
}
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def empty_extraction():
    return ExtractionResult(
        symptoms=[],
        unmapped_mentions=[],
        medical_history=[],
        allergies=[],
        medications=[],
    )


def answer_target_for_reply(message, pending_target, recent_targets):
    """Use an earlier question only when a short reply names its symptom.

    A bare yes/no or number still belongs to the pending question. This avoids
    silently turning an ambiguous answer into a different clinical fact.
    """
    if pending_target is None or len(message.split()) > 12:
        return pending_target
    normalized = _normalize_conversational_answer(message)
    matches = []
    for target in reversed(recent_targets[-3:]):
        if not target or target == pending_target:
            continue
        symptom_id = target.get("target", "")
        if not get_vocab().is_valid(symptom_id):
            continue
        if any(
            phrase and re.search(rf"\b{re.escape(phrase.lower())}\b", normalized)
            for phrase in get_vocab().phrases(symptom_id)
        ):
            matches.append(target)
    # Several named symptoms need full extraction, not a guessed binding.
    return matches[0] if len(matches) == 1 else pending_target


def duration_without_unit(message, target):
    """Identify a bare amount that cannot safely be converted to days."""
    if target is None or target.get("focus") != "duration":
        return None
    answer = _normalize_conversational_answer(message)
    if re.fullmatch(r"\d+(?:\.\d+)?", answer) or answer in NUMBER_WORDS:
        return answer
    return None


def vector_clarification(message):
    """Offer closely related vocabulary terms without asserting any of them."""
    candidates, source = ranked_clarification_candidates(message)
    if not candidates:
        return None
    labels = [get_vocab().name(symptom_id).lower() for symptom_id in candidates]
    choices = "; ".join(f"{index}. {label}" for index, label in enumerate(labels, 1))
    return (
        {
            "kind": "semantic_clarification",
            "target": "symptom_choice",
            "focus": "presence",
            "candidates": candidates,
            "candidate_source": source,
            "original_text": message,
        },
        f"To make sure I understood, which best matches what you described: {choices}? "
        "You can reply with a number, describe it in your own words, or say none of these.",
    )


def resolve_vector_choice(message, target):
    """Bind a brief choice only when it selects one candidate unambiguously."""
    original = target.get("original_text", "")
    reply = _normalize_conversational_answer(message)
    candidates = target.get("candidates", [])
    selected = None
    number = re.fullmatch(r"(?:option |number |#)?([1-9])", reply)
    if number and int(number[1]) <= len(candidates):
        selected = candidates[int(number[1]) - 1]
    else:
        matches = []
        for symptom_id in candidates:
            distinctive = set(symptom_id.split("_")) - {"pain", "abdominal", "headache"}
            if any(re.search(rf"\b{re.escape(word)}\b", reply) for word in distinctive):
                matches.append(symptom_id)
        if len(matches) == 1:
            selected = matches[0]
        elif len(matches) == 2 and any(sid.startswith("severe_") for sid in matches):
            # "upper and severe" identifies a location plus severity, rather
            # than two competing locations. Preserve both facts for screening.
            selected = next((sid for sid in matches if not sid.startswith("severe_")), None)
        elif reply in {get_vocab().name(symptom_id).lower() for symptom_id in candidates}:
            named = [sid for sid in candidates if get_vocab().name(sid).lower() == reply]
            if len(named) == 1:
                selected = named[0]
    if selected is None:
        return ExtractionResult(
            symptoms=[],
            unmapped_mentions=[original or message],
            medical_history=[],
            allergies=[],
            medications=[],
        )
    combined = f"{original} {message}".lower()
    duration = _parse_symptom_detail(combined, "duration")
    progression = _parse_symptom_detail(combined, "progression")
    severity = _parse_symptom_detail(combined, "severity")
    symptoms = [
        ExtractedSymptom(
            symptom_id=selected,
            presence="present",
            severity=severity.get("severity"),
            duration_days=duration.get("duration_days"),
            progression=progression.get("progression"),
            evidence=f"{original} | {message}",
        )
    ]
    if severity.get("severity") == "severe" and "abdominal" in selected:
        symptoms.append(
            ExtractedSymptom(
                symptom_id="severe_abdominal_pain",
                presence="present",
                severity="severe",
                duration_days=duration.get("duration_days"),
                progression=progression.get("progression"),
                evidence=f"{original} | {message}",
            )
        )
    return ExtractionResult(
        symptoms=symptoms,
        unmapped_mentions=[],
        medical_history=[],
        allergies=[],
        medications=[],
    )


def safety_question(decision, profile=None):
    """Ask about a conflict first, otherwise the next missing safety field."""
    conflicts = [
        hit for hit in decision.triggered_rules if hit.rule_id == "DATA_CONTRADICTION"
    ]
    if conflicts and conflicts[0].fields:
        question_field = conflicts[0].fields[0]
    else:
        question_field = next(iter(decision.missing_information), "presenting_symptoms")

    uncertain = [] if profile is None else [
        symptom
        for symptom in profile.symptoms.values()
        if symptom.presence == "uncertain"
    ]
    if uncertain:
        symptom = uncertain[0]
        evidence = (symptom.evidence or "that description").strip()
        if symptom.symptom_id == "flank_pain":
            question = (
                f"I want to make sure I understood '{evidence}'. Is the pain specifically "
                "in your flank—the side of your back just below the ribs—or is it elsewhere?"
            )
        else:
            question = (
                f"I want to make sure I understood '{evidence}'. Did you mean "
                f"{get_vocab().name(symptom.symptom_id).lower()}, or something different? "
                "Please describe where it is and what it feels like."
            )
        return (
            {
                "kind": "semantic_clarification",
                "target": symptom.symptom_id,
                "focus": "presence",
            },
            question,
        )

    if profile is not None and not any(
        symptom.presence == "present" for symptom in profile.symptoms.values()
    ):
        return (
            {"kind": "context", "target": "presenting_symptoms", "focus": None},
            "Could you describe what feels wrong in your own words, including where it is, when it started, how severe it feels, and whether it is changing?",
        )

    if question_field == "age":
        target = {"kind": "safety", "target": "age", "focus": None}
        return target, "How old are you?"
    if question_field == "pregnancy_status":
        target = {"kind": "safety", "target": "pregnancy_status", "focus": None}
        return target, "Is there any chance you could be pregnant?"

    missing_symptoms = [
        field.removeprefix("symptoms.")
        for field in decision.missing_information
        if field.startswith("symptoms.")
        and get_vocab().is_valid(field.removeprefix("symptoms."))
    ]
    if not conflicts and len(missing_symptoms) > 1:
        target = {
            "kind": "safety",
            "target": "emergency_warning_signs",
            "targets": missing_symptoms,
            "focus": "presence",
        }
        return target, (
            "Before we continue, are you having any emergency warning signs such as "
            "trouble breathing or chest pain; fainting, a seizure, or severe confusion; "
            "heavy bleeding; face droop, new speech trouble, or one-sided weakness; "
            "tongue or throat swelling; a sudden worst-ever headache; or severe abdominal pain? "
            "Tell me which one, or say none."
        )

    symptom_id = question_field.removeprefix("symptoms.")
    if get_vocab().is_valid(symptom_id):
        prefix = (
            "Your answers differ. Please confirm: "
            if conflicts
            else "I need to check a warning sign: "
        )
        question = (
            prefix + f"do you have {get_vocab().name(symptom_id).lower()}? "
            "Answer yes, no, or not sure."
        )
        return {"kind": "safety", "target": symptom_id, "focus": "presence"}, question
    return (
        {"kind": "context", "target": "presenting_symptoms", "focus": None},
        "Please describe the main symptoms you are experiencing.",
    )


def bind_answer(message, target, profile, evidence):
    """Return structured facts for a recognized reply, or None for LLM extraction.

    Demographic replies update the profile and evidence directly. Symptom
    replies are returned as an extraction so normal conflict handling still runs.
    """
    if target is None:
        return None
    normalized_answer = _normalize_conversational_answer(message)
    if target.get("clarifying_number") and normalized_answer in {
        "day", "days", "week", "weeks", "month", "months"
    }:
        normalized_answer = f"{target['clarifying_number']} {normalized_answer}"
    question_field = target["target"]
    grouped_targets = target.get("targets", [])
    if question_field == "emergency_warning_signs" and grouped_targets:
        if normalized_answer in NEGATIVE_GROUP_ANSWERS or re.fullmatch(
            r"(?:nah[, ]*)?(?:no(?:pe)?[, ]*)?(?:none|none of (?:that|them|those)|not any of (?:them|those))(?:[, ]+(?:thanks?|thank you|thankfully|thank u|tbh))?",
            normalized_answer,
        ):
            for symptom_id in grouped_targets:
                evidence.symptoms[symptom_id] = False
            return ExtractionResult(
                symptoms=[
                    ExtractedSymptom(
                        symptom_id=symptom_id,
                        presence="absent",
                        severity=None,
                        duration_days=None,
                        progression=None,
                        evidence=message,
                    )
                    for symptom_id in grouped_targets
                ],
                unmapped_mentions=[],
                medical_history=[],
                allergies=[],
                medications=[],
            )
        return None
    if target["kind"] == "context":
        if question_field == "temperature" and _bind_temperature(
            normalized_answer, evidence
        ):
            return empty_extraction()
        if question_field == "sex" and _bind_sex(normalized_answer, evidence):
            return empty_extraction()
    if target["kind"] == "safety":
        if question_field == "age" and _bind_age(normalized_answer, profile, evidence):
            return empty_extraction()
        if question_field == "pregnancy_status" and _bind_pregnancy(
            normalized_answer, profile, evidence
        ):
            return empty_extraction()

    if (
        get_vocab().is_valid(question_field)
        and target.get("focus") == "presence"
    ):
        presence = _presence_from_answer(normalized_answer)
        if presence is None:
            presence = _presence_from_answer(
                _without_target_phrase(normalized_answer, question_field)
            )
        if presence is None:
            return None
        if presence == "uncertain":
            evidence.symptoms.pop(question_field, None)
        else:
            evidence.symptoms[question_field] = presence == "present"
        return _symptom_extraction(question_field, message, presence)

    symptom = profile.symptoms.get(question_field)
    if symptom is not None and symptom.presence == "present":
        details = _parse_symptom_detail(normalized_answer, target.get("focus"))
        if details:
            return _symptom_extraction(question_field, message, "present", **details)
    return None


def _bind_age(answer, profile, evidence):
    age = _number_from_text(answer)
    if age is not None and 0 < age < 120:
        profile.age = evidence.age = age
        return True
    return answer in UNCERTAIN_ANSWERS


def _bind_pregnancy(answer, profile, evidence):
    status = PREGNANCY_ANSWERS.get(answer)
    if status is None:
        if re.search(
            r"\b(?:(?:not|isn'?t|am not|can(?:not|'t)|could(?: not|n't))\s+(?:be\s+)?pregnant|no chance(?:\s+of\s+being)?\s+pregnant)\b",
            answer,
        ):
            status = "not_pregnant"
        elif re.search(r"\b(pregnant|expecting)\b", answer):
            status = "pregnant"
        elif re.search(
            r"\b(not applicable|doesn'?t apply|does not apply|no uterus)\b", answer
        ):
            status = "not_applicable"
        else:
            return False
    profile.pregnancy_status = status
    if status == "pregnant":
        evidence.pregnant = True
    elif status in {"not_pregnant", "not_applicable"}:
        evidence.pregnant = False
    else:
        evidence.pregnant = None
    return True


def _bind_temperature(answer, evidence):
    match = re.fullmatch(
        r".*?((?:3[5-9]|4[0-2])(?:\.\d+)?)\s*(?:°?c|degrees?\s*c?)?.*",
        answer,
    )
    fahrenheit = re.search(
        r"\b((?:9[5-9]|10[0-8])(?:\.\d+)?)\s*(?:°?f|fahrenheit)\b",
        answer,
    )
    if fahrenheit:
        evidence.temperature_c = round((float(fahrenheit.group(1)) - 32) * 5 / 9, 1)
        evidence.temperature_asked = True
        return True
    if match:
        evidence.temperature_c = float(match.group(1))
        evidence.temperature_asked = True
        return True
    if re.search(
        r"\b(no|not measured|never measure[ds]?|haven'?t measured|don'?t know|not sure|unknown)\b",
        answer,
    ):
        evidence.temperature_asked = True
        return True
    return False


def _bind_sex(answer, evidence):
    normalized = {
        "f": "F",
        "female": "F",
        "woman": "F",
        "m": "M",
        "male": "M",
        "man": "M",
    }
    if re.search(r"\b(prefer not|rather not|decline|not sure)\b", answer):
        evidence.sex_asked = True
        return True
    matched = next(
        (value for word, value in normalized.items() if re.search(rf"\b{word}\b", answer)),
        None,
    )
    if matched is None:
        return False
    evidence.sex = matched
    evidence.sex_asked = True
    return True


def _parse_symptom_detail(answer, focus):
    if focus == "duration":
        match = re.search(
            r"\b(\d+(?:\.\d+)?)\s*(days?|d|weeks?|wks?|w|months?|mos?|m)\b",
            answer,
        )
        if match:
            days_per_unit = {"d": 1, "w": 7, "m": 30}
            return {"duration_days": float(match[1]) * days_per_unit[match[2][0]]}
        word_match = re.search(
            r"\b([a-z-]+(?:\s+[a-z-]+)?)\s*(days?|weeks?|months?)\b", answer
        )
        if word_match:
            amount = _number_from_text(word_match[1])
            if amount is not None:
                days_per_unit = {"d": 1, "w": 7, "m": 30}
                return {"duration_days": amount * days_per_unit[word_match[2][0]]}
        if re.search(r"\b(?:yesterday|yday|yest(?:er)?day)\b", answer):
            return {"duration_days": 1.0}
        if re.search(r"\b(?:today|tdy|this morning|last night)\b", answer):
            return {"duration_days": 0.5}
    elif focus == "severity":
        for phrase, severity in (
            (r"\b(severe|terrible|unbearable|very bad|jialat)\b", "severe"),
            (r"\b(moderate|medium)\b", "moderate"),
            (r"\b(mild|slight|a little|not too bad)\b", "mild"),
        ):
            if re.search(phrase, answer):
                return {"severity": severity}
    elif focus == "progression":
        if re.search(r"\b(better|improving|easing)\b", answer):
            return {"progression": "improving"}
        if re.search(r"\b(stable|same|unchanged|about the same)\b", answer):
            return {"progression": "stable"}
        if re.search(r"\b(worse|worsening|getting bad)\b", answer):
            return {"progression": "worsening"}
    return {}


def _without_target_phrase(answer, symptom_id):
    """Remove one explicit symptom reference from a brief presence answer."""
    for phrase in sorted(get_vocab().phrases(symptom_id), key=len, reverse=True):
        if phrase:
            answer = re.sub(
                rf"\b(?:for|about|to)?\s*(?:the|my)?\s*{re.escape(phrase.lower())}\b",
                " ",
                answer,
                count=1,
            )
    return re.sub(r"\s+", " ", answer).strip(" ,")


def _presence_from_answer(answer):
    if answer in PRESENCE_ANSWERS:
        return PRESENCE_ANSWERS[answer]
    if re.fullmatch(
        r"(?:no(?:pe)?|not really|i (?:do not|don'?t|haven'?t)|don'?t have|not that i(?:'?ve| have) noticed)(?:[, ]+(?:thanks?|tbh))?",
        answer,
    ):
        return "absent"
    if re.fullmatch(r"(?:yes|yeah|yep|i do|i have|got|a little|sometimes)", answer):
        return "present"
    if re.fullmatch(r"(?:maybe|possibly|i(?:'m| am) not sure|hard to say)", answer):
        return "uncertain"
    return None


def _normalize_conversational_answer(message):
    """Normalize common shorthand and Singlish particles without changing evidence."""
    answer = message.lower().strip().rstrip(".!?")
    replacements = (
        (r"\bytd\b", "yesterday"),
        (r"\btdy\b", "today"),
        (r"\babit\b", "a bit"),
        (r"\bdunno\b", "not sure"),
        (r"\bboh?\b", "no"),
        (r"\bnvr\b", "never"),
    )
    for pattern, replacement in replacements:
        answer = re.sub(pattern, replacement, answer)
    # Sentence-final particles carry tone rather than clinical meaning.
    answer = re.sub(r"(?:[, ]+)(?:lah|leh|lor|sia|mah|hor)+$", "", answer).strip()
    return re.sub(r"\s+", " ", answer)


def _number_from_text(answer):
    numeric = re.search(r"\b\d{1,3}(?:\.\d+)?\b", answer)
    if numeric:
        return float(numeric.group())
    words = answer.replace("-", " ").split()
    values = [NUMBER_WORDS[word] for word in words if word in NUMBER_WORDS]
    if not values:
        return None
    total = 0
    for value in values:
        if value >= 20 and value % 10 == 0:
            total += value
        elif total and total % 10 == 0:
            total += value
        else:
            total = value
    return float(total)


def _symptom_extraction(symptom_id, message, presence, **details):
    return ExtractionResult(
        symptoms=[
            ExtractedSymptom(
                symptom_id=symptom_id,
                presence=presence,
                severity=details.get("severity"),
                duration_days=details.get("duration_days"),
                progression=details.get("progression"),
                evidence=message,
            )
        ],
        unmapped_mentions=[],
        medical_history=[],
        allergies=[],
        medications=[],
    )
