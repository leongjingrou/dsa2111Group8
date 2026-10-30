"""The clinician view must reflect the persisted, guarded decision path."""

from fastapi.testclient import TestClient

from backend.tests.conftest import RecordingPredictor, start, turn


def test_review_tracks_questions_and_safety_without_inventing_model_output(client):
    session_id = start(client)
    response = turn(client, session_id, "I have a runny nose")
    review = client.get(f"/api/conversations/{session_id}/review")

    assert review.status_code == 200
    data = review.json()
    assert data["finished"] is False
    assert data["latest"]["reply"] == response["reply"]
    assert data["pathway"][0]["question_asked"].startswith("Hello")
    assert data["pathway"][0]["patient_answer"] == "I have a runny nose"
    assert data["pathway"][0]["diagnosis"] is None

    next_answer = turn(client, session_id, "30")
    refreshed = client.get(f"/api/conversations/{session_id}/review").json()
    assert len(refreshed["pathway"]) == 2
    assert refreshed["pathway"][1]["question_asked"] == response["reply"]
    assert refreshed["latest"]["reply"] == next_answer["reply"]


def test_review_of_emergency_keeps_model_results_empty(client):
    session_id = start(client)
    turn(client, session_id, "I have chest pain")
    review = client.get(f"/api/conversations/{session_id}/review").json()

    assert review["finished"] is True
    assert review["latest"]["status"] == "urgent"
    assert review["pathway"][0]["safety"]["disposition"] == "emergency"
    assert review["pathway"][0]["diagnosis"] is None
    assert review["pathway"][0]["escalation"] is None


def test_review_requires_existing_session(client):
    assert client.get("/api/conversations/missing/review").status_code == 404


def test_review_retains_qualitative_context_before_and_after_symptom_choice(client):
    session_id = start(client)
    turn(client, session_id, "stomach pain")
    repeated = turn(client, session_id, "stomach hurting for past few days, getting a bit worse")
    assert "past few days" in repeated["reply"].lower()
    assert "getting worse" in repeated["reply"].lower()
    pending = client.get(f"/api/conversations/{session_id}/review").json()["pathway"][1]
    details = pending["question_target"]["reported_details"]
    assert details["duration_text"] == "for past few days"
    assert details["progression"] == "worsening"
    turn(client, session_id, "upper")
    resolved = client.get(f"/api/conversations/{session_id}/review").json()["pathway"][2]
    assert resolved["answer_target"]["reported_details"] == details
    assert resolved["extraction"]["symptoms"][0]["progression"] == "worsening"


def test_all_over_abdominal_pain_does_not_repeat_location_question(client):
    session_id = start(client)
    first = turn(client, session_id, "stomach pain")
    assert "upper abdomen" in first["reply"].lower()
    response = turn(client, session_id, "It hurts all over actually")
    assert response["question_target"] is None or response["question_target"]["target"] != "symptom_choice"
    assert response["extraction"]["unmapped_mentions"] == [
        "stomach pain | It hurts all over actually"
    ]


def test_scope_referral_shows_separate_gastroenteritis_text_match(app_factory):
    with TestClient(app_factory(predictor=RecordingPredictor(condition="out_of_scope"))) as client:
        session_id = start(client)
        for answer in ("stomach ache", "upper", "24", "no", "none", "diarrhea and nausea", "2 days, mild, stable"):
            result = turn(client, session_id, answer)
        review = client.get(f"/api/conversations/{session_id}/review").json()

    assert result["diagnosis"] is None
    assert "SCOPE_CONDITION" in result["triggered_rules"]
    assert review["knowledge_context"]["results"][0]["condition_id"] == "gastroenteritis"
    assert review["latest"]["decision_record"]["preliminary_estimates"] == []


def test_scope_text_match_displays_explicit_denial_of_gi_symptoms(app_factory):
    with TestClient(app_factory(predictor=RecordingPredictor(condition="out_of_scope"))) as client:
        session_id = start(client)
        for answer in ("stomach ache", "upper", "24", "no", "none", "none", "2 days, mild, stable"):
            turn(client, session_id, answer)
        review = client.get(f"/api/conversations/{session_id}/review").json()

    gastro = next(
        result for result in review["knowledge_context"]["results"]
        if result["condition_id"] == "gastroenteritis"
    )
    assert {name.lower() for name in gastro["reported_evidence"]["absent"]} >= {
        "diarrhoea", "vomiting", "nausea"
    }
