"""Real TCP HTTP tests: actual app startup, fitted EBM, safety, KB and language.

The external paid LLM is replaced by the repo's explicit KeywordLLM adapter.
The diagnostic and stopping models are trained by the real project pipeline.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
import pytest

from diagnostic_model.sequential_ebm import train_sequential_ebm

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def trained_models(tmp_path_factory):
    data = pd.read_csv(ROOT / "data/generated/synthetic_scenarios.csv")
    # Every condition/escalation stratum is represented. Training is not mocked.
    data = (
        data.groupby(["condition_id", "escalation"], sort=False)
        .head(6)
        .reset_index(drop=True)
    )
    output = tmp_path_factory.mktemp("real-models")
    train_sequential_ebm(data, output)
    return output


@pytest.fixture(scope="module")
def live_server(trained_models, tmp_path_factory):
    output = tmp_path_factory.mktemp("real-http")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {
        **os.environ,
        "MODEL_BACKEND": "ebm",
        "LLM_BACKEND": "keyword",
        "EBM_LIVE_MODEL": str(trained_models / "ebm_sequential_models.joblib"),
        "STOPPING_MODEL": str(trained_models / "ebm_stopping_controller.joblib"),
        "CONSULTATION_DB": str(output / "sessions.sqlite3"),
        "SAFETY_AUDIT_LOG": str(output / "audit.jsonl"),
    }
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", timeout=30, trust_env=False
            ) as client:
                deadline = time.monotonic() + 25
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        pytest.fail((output / "server.log").read_text())
                    try:
                        if client.get("/readyz").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.1)
                else:
                    pytest.fail(
                        "Backend did not become ready: "
                        + (output / "server.log").read_text()
                    )
                yield client, output
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def test_real_http_emergency_from_start_to_terminal_decision(live_server):
    client, _ = live_server
    session_id = client.post("/api/conversations", json={}).json()["session_id"]
    response = client.post(
        f"/api/conversations/{session_id}/turn",
        json={"patient_message": "I have chest pain"},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["backend"] == "ebm"
    assert result["safety"]["disposition"] == "emergency"
    assert result["diagnosis"] is None
    assert "995" in result["doctor_response"]


def test_real_http_complete_screen_runs_trained_model(live_server):
    client, _ = live_server
    session_id = client.post("/api/conversations", json={}).json()["session_id"]
    message = "I have a mild runny nose for 2 days and it is stable"
    for _ in range(40):
        response = client.post(
            f"/api/conversations/{session_id}/turn", json={"patient_message": message}
        )
        assert response.status_code == 200, response.text
        result = response.json()
        if result["stop"]:
            break
        target = result["question_target"]["target"]
        message = (
            "30"
            if target == "age"
            else "not pregnant"
            if target == "pregnancy_status"
            else "no"
        )
    else:
        pytest.fail("Consultation did not complete")
    assert result["backend"] == "ebm"
    assert result["safety"]["phase"] == "post_model"
    assert result["decision_record"] is not None
    assert result["patient_answer"] >= 4
    # Synthetic model labels are not clinical ground truth. A safe model-based
    # referral is valid; the EBM is not forced to return a routine diagnosis.
    assert result["status"] in {"complete", "urgent"}


def test_live_api_evaluation_report(live_server):
    client, output = live_server
    from scripts.evaluate_safety_api import evaluate

    report = evaluate(client, output / "evaluation.json")
    assert report["passed"] == report["cases"]
