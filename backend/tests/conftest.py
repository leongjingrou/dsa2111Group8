import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from llm_service.app.keyword_llm import KeywordLLM
from llm_service.session_store import SessionStore
from safety_layer import SafetyEngine


class ConstantStopModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, rows):
        return np.tile([0.05, 0.95], (len(rows), 1))


class RecordingPredictor:
    name = "acceptance-test-double"

    def __init__(self, condition="common_cold", escalation="routine"):
        self.condition, self.escalation = condition, escalation
        self.calls = []

    def predict(self, turns, evidence, answer_number):
        self.calls.append((turns, evidence, answer_number))
        return {
            "condition": {
                "choice": self.condition,
                "probabilities": {self.condition: 0.9, "influenza": 0.1},
                "answer_confidence": 0.9,
            },
            "escalation": {
                "choice": self.escalation,
                "probabilities": {self.escalation: 0.9},
                "answer_confidence": 0.9,
            },
        }


@pytest.fixture
def app_factory(tmp_path):
    def make(*, predictor=None, llm=None, store=None, audit=None):
        return create_app(
            predictor if predictor is not None else RecordingPredictor(),
            {"model": ConstantStopModel(), "threshold": 0.9},
            llm if llm is not None else KeywordLLM(),
            safety_engine=SafetyEngine(audit_log=audit or tmp_path / "audit.jsonl"),
            session_store=store
            if store is not None
            else SessionStore(tmp_path / "sessions.sqlite3"),
        )

    return make


@pytest.fixture
def client(app_factory):
    with TestClient(app_factory()) as client:
        yield client


def start(client):
    response = client.post("/api/conversations", json={})
    assert response.status_code == 200, response.text
    return response.json()["session_id"]


def turn(client, session_id, message, **kwargs):
    response = client.post(
        f"/api/conversations/{session_id}/turn",
        json={"patient_message": message, **kwargs},
    )
    assert response.status_code == 200, response.text
    return response.json()


def finish_screen(
    client, session_id, result, *, maximum=40, pregnancy_answer="not pregnant"
):
    seen = []
    for _ in range(maximum):
        if result["stop"]:
            return result, seen
        target = result["question_target"]
        seen.extend(target.get("targets") or [target["target"]])
        answer = (
            "30"
            if target["target"] == "age"
            else pregnancy_answer
            if target["target"] == "pregnancy_status"
            else "no"
        )
        result = turn(client, session_id, answer)
    raise AssertionError("Screen did not reach a terminal decision")
