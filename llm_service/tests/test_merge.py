from helpers import ext, real_ids, sym

from llm_service.app.merge import merge_profile
from llm_service.app.schema import Profile

A, B = real_ids(2)


def test_adds_new_symptom_and_counts_turn():
    p = merge_profile(Profile(), ext(sym(A, severity="mild")))
    assert p.symptoms[A].presence == "present"
    assert p.symptoms[A].severity == "mild"
    assert p.turn_count == 1


def test_flip_records_contradiction_and_latest_wins():
    p = merge_profile(Profile(), ext(sym(A, "absent")))
    p = merge_profile(p, ext(sym(A, "present")))
    assert p.symptoms[A].presence == "present"
    assert len(p.contradictions) == 1
    assert p.contradictions[0].previous == "absent"
    assert p.contradictions[0].new == "present"


def test_uncertain_does_not_overwrite_definite_answer():
    p = merge_profile(Profile(), ext(sym(A, "present")))
    p = merge_profile(p, ext(sym(A, "uncertain")))
    assert p.symptoms[A].presence == "present"
    assert p.contradictions == []


def test_null_details_do_not_erase_old_details():
    p = merge_profile(Profile(), ext(sym(A, severity="severe", duration_days=2)))
    p = merge_profile(p, ext(sym(A, "present")))
    assert p.symptoms[A].severity == "severe"
    assert p.symptoms[A].duration_days == 2


def test_absent_clears_details():
    p = merge_profile(Profile(), ext(sym(A, severity="severe", duration_days=2)))
    p = merge_profile(p, ext(sym(A, "absent")))
    assert p.symptoms[A].severity is None
    assert p.symptoms[A].duration_days is None


def test_missing_info_uses_focus_names():
    p = merge_profile(Profile(), ext(sym(A, severity="mild")))
    assert f"{A}:duration" in p.missing_info
    assert f"{A}:progression" in p.missing_info
    assert f"{A}:severity" not in p.missing_info


def test_no_symptoms_means_presenting_symptoms_missing():
    p = merge_profile(Profile(), ext())
    assert p.missing_info == ["presenting_symptoms"]


def test_lists_are_deduplicated_case_insensitively():
    p = merge_profile(Profile(), ext(allergies=["Penicillin"], meds=["Panadol"]))
    p = merge_profile(p, ext(allergies=["penicillin", "Peanuts"]))
    assert p.allergies == ["Penicillin", "Peanuts"]
    assert p.medications == ["Panadol"]


def test_merge_does_not_mutate_input():
    original = Profile()
    merge_profile(original, ext(sym(A)))
    assert original.symptoms == {}
    assert original.turn_count == 0
