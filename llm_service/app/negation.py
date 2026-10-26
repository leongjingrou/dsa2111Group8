"""Limited clause-aware negation shared by the deterministic text adapters.

This is a prototype parser, not a clinically validated language recognizer.
"""

import re


def is_negated(text, start):
    prefix = text[:start]
    segment = re.split(r"[.;!?]|\b(?:but|however)\b|\band\s+(?:i|my)\b", prefix)[-1]
    # Pregnancy status does not negate a different symptom in the same clause.
    segment = re.sub(r"\bnot pregnant\b", "", segment)
    return bool(
        re.search(
            r"\b(no|not|never|without|deny|denied|don'?t|do not|haven'?t|have not)\b",
            segment,
        )
    )
