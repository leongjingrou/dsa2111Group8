"""Deterministic language adapter for integration and load testing.

This adapter deliberately performs no diagnosis or urgency assessment.  It is
useful when measuring the local orchestration stack without including latency,
cost, rate limits, or availability from an external LLM provider.
"""

import json
import re
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    "jialat": "severe",
}

TERM_STOPWORDS = {"a", "an", "of", "the"}


def _term_hits(text, term):
    """Find exact terms plus close, reordered multi-word phrasing.

    The latter keeps the deterministic fallback useful for conversational input
    such as ``nose all runny`` without maintaining a phrase-by-phrase rule list.
    """
    exact = [(match.start(), match.end()) for match in re.finditer(rf"\b{re.escape(term)}\b", text)]
    if exact:
        return exact
    wanted = [token for token in re.findall(r"[a-z0-9]+", term) if token not in TERM_STOPWORDS]
    if len(wanted) < 2:
        return []
    tokens = list(re.finditer(r"\b[a-z0-9]+\b", text))
    width = len(wanted) + 2
    for index in range(len(tokens)):
        window = tokens[index : index + width]
        words = {match.group() for match in window}
        if all(token in words for token in wanted):
            return [(window[0].start(), window[-1].end())]
    return []


def _entry_terms(symptom_id, entry):
    """Return deterministic phrases, including alternatives in display labels."""
    name = entry.get("name", "").lower()
    terms = [symptom_id.replace("_", " "), name]
    terms.extend(item.lower() for item in entry.get("synonyms", []))
    # Labels such as "Flank / back pain" express alternative patient wording.
    # Matching only the entire display label discarded ordinary "back pain".
    if "/" in name:
        terms.extend(part.strip(" ()") for part in name.split("/"))
    return list(dict.fromkeys(term for term in terms if term))


@lru_cache
def _vector_index():
    """Build a small offline text-embedding index over labels and synonyms."""
    phrases = []
    symptom_ids = []
    for symptom_id, entry in get_vocab()._entries.items():
        terms = _entry_terms(symptom_id, entry)
        phrases.extend(terms)
        symptom_ids.extend([symptom_id] * len(terms))
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        strip_accents="unicode",
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(phrases)
    return vectorizer, matrix, symptom_ids


def _vector_match(text):
    """Return a high-separation vector match, otherwise decline to guess."""
    if re.search(r"\b(?:no|not|without|don'?t|never|bo)\b", text):
        return None
    vectorizer, matrix, symptom_ids = _vector_index()
    similarities = cosine_similarity(vectorizer.transform([text]), matrix)[0]
    by_symptom = {}
    for symptom_id, score in zip(symptom_ids, similarities):
        by_symptom[symptom_id] = max(by_symptom.get(symptom_id, 0.0), float(score))
    ranked = sorted(by_symptom.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) < 2:
        return None
    (best_id, best_score), (_, second_score) = ranked[:2]
    if best_score < 0.20 or best_score - second_score < 0.08:
        return None
    return best_id, best_score


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
            terms = _entry_terms(symptom_id, entry)
            hits = [
                (start, end, term)
                for term in terms
                if term
                for start, end in _term_hits(text, term)
            ]
            if not hits:
                continue
            positive_hits = [
                (start, end, term)
                for start, end, term in hits
                if not is_negated(text, start)
            ]
            start, end, _ = min(positive_hits or hits)
            present = bool(positive_hits)
            symptoms.append(
                ExtractedSymptom(
                    symptom_id=symptom_id,
                    presence="present" if present else "absent",
                    severity=severity if present else None,
                    duration_days=days if present else None,
                    progression=progression if present else None,
                    evidence=message[start:end],
                )
            )
        if not symptoms:
            vector_match = _vector_match(text)
            if vector_match is not None:
                symptom_id, _ = vector_match
                symptoms.append(
                    ExtractedSymptom(
                        symptom_id=symptom_id,
                        presence="present",
                        severity=severity,
                        duration_days=days,
                        progression=progression,
                        evidence=message,
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
            name = variables["symptom_name"]
            focus = variables["focus_instruction"]
            if "getting better" in focus:
                return f"Is the {name} getting better, staying the same, or worsening?"
            if focus.startswith("how severe"):
                return f"How severe is the {name}: mild, moderate, or severe?"
            if focus.startswith("how long"):
                return f"How long has the {name} been going on?"
            return f"Have you also noticed {name}?"
        if prompt_name == "ask_context":
            instruction = variables["question_instruction"]
            if "main problem" in instruction:
                return "Could you describe what feels wrong in your own words?"
            if "current temperature" in instruction:
                return "Have you measured your temperature? If so, what is it?"
            if "patient's sex" in instruction:
                return "For this health assessment, what sex were you assigned at birth? You can also choose not to say."
            if "medical history" in instruction:
                return "Do you have any relevant medical conditions, medicines, or allergies?"
            return "Could you tell me a little more about that?"
        if prompt_name == "explain":
            record = json.loads(variables["decision_record"])
            estimates = record.get("preliminary_estimates", [])
            if estimates:
                leading = estimates[0]
                label = leading["condition"].replace("_", " ")
                score = leading.get("score")
                confidence = f" ({score:.0%} model confidence)" if score is not None else ""
                return (
                    f"The leading preliminary match is {label}{confidence}. "
                    "A doctor should review this assessment and confirm the diagnosis."
                )
            return "A healthcare professional should review the assessment and decide the next step."
        raise ValueError(f"Unsupported text prompt: {prompt_name}")
