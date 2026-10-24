from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from serving.app import Counters, TurnRequest, create_app
from serving.load_test import summarize


class FakeBackend:
    name = "fake"

    def predict(self, scenario_id, patient_answers):
        if scenario_id == "missing":
            raise KeyError(scenario_id)
        return {
            "scenario_id": scenario_id,
            "backend": self.name,
            "patient_answers": patient_answers,
            "answers": {
                "condition": {"choice": "common_cold", "answer_confidence": 0.8},
                "escalation": {"choice": "routine", "answer_confidence": 0.9},
            },
            "stop_probability": 0.8,
            "stop_threshold": 0.75,
            "should_stop": patient_answers >= 2,
            "forced_final_stop": patient_answers == 5,
            "next_question": None if patient_answers >= 2 else "How long?",
        }


def test_api_contract_is_stateless_and_replica_identified(monkeypatch):
    monkeypatch.setenv("HOSTNAME", "replica-test")
    app = create_app(FakeBackend())
    paths = {route.path for route in app.routes}
    assert app.state.backend.name == "fake"
    assert app.state.replica == "replica-test"
    assert "/v1/simulations/consultations/turn" in paths
    assert {"/healthz", "/readyz", "/metrics"} <= paths


def test_request_contract_rejects_invalid_answer_count():
    with pytest.raises(ValidationError):
        TurnRequest(
            client_id="d1", request_id="r1", scenario_id="SYN-00001",
            patient_answers=6,
        )


def test_metrics_counter_is_thread_safe():
    counters = Counters()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda index: counters.observe(0.01, error=index % 5 == 0), range(100)))
    rendered = counters.render()
    assert "consultation_requests_total 100" in rendered
    assert "consultation_errors_total 20" in rendered


def test_load_summary_reports_tail_latency_replicas_devices_and_stop_turns():
    sessions = [
        {
            "device_type": "mobile",
            "stopped_at": 2,
            "requests": [
                {"latency_ms": 10.0, "error": None, "replica": "pod-a"},
                {"latency_ms": 30.0, "error": None, "replica": "pod-b"},
            ],
        },
        {
            "device_type": "desktop",
            "stopped_at": 3,
            "requests": [
                {"latency_ms": 20.0, "error": None, "replica": "pod-a"},
                {"latency_ms": 40.0, "error": "timeout", "replica": None},
            ],
        },
    ]
    result = summarize(8, sessions, elapsed=1.0)
    assert result["latency_ms"] == {"p50": 20.0, "p95": 30.0, "p99": 30.0, "max": 30.0}
    assert result["error_rate"] == 0.25
    assert result["replica_requests"] == {"pod-a": 2, "pod-b": 1}
    assert result["device_sessions"] == {"desktop": 1, "mobile": 1}
    assert result["stop_turns"] == {"2": 1, "3": 1}
