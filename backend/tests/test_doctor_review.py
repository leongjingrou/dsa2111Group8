"""The clinician view must reflect the persisted, guarded decision path."""

from backend.tests.conftest import start, turn


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
