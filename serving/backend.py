"""Inference backends for the synthetic multi-user serving experiment."""

import json
import os
from pathlib import Path

import joblib

from diagnostic_model.laya_model import (
    confidence_traces,
    conversation_turns,
)
from diagnostic_model.scenario_data import load_scenarios
from diagnostic_model.sequential_ebm import ebm_confidence_traces
from diagnostic_model.stopping import stopping_probabilities


MAX_PATIENT_ANSWERS = 5


class ScenarioStore:
    """Read-only synthetic cases used by the reproducible load simulation."""

    def __init__(self, data_path):
        data = load_scenarios(data_path).set_index("scenario_id", drop=False)
        if not data.index.is_unique:
            raise ValueError("scenario_id must be unique")
        self.data = data

    def row(self, scenario_id):
        if scenario_id not in self.data.index:
            raise KeyError(scenario_id)
        return self.data.loc[[scenario_id]].reset_index(drop=True)

    def scenario_ids(self):
        return self.data.index.astype(str).tolist()


def _response(scenario_id, patient_answers, trace, stop_scores, threshold, backend):
    checkpoint = trace[patient_answers - 1]
    forced = patient_answers == MAX_PATIENT_ANSWERS
    policy_stop = patient_answers >= 2 and stop_scores[patient_answers - 1] >= threshold
    turns = trace[patient_answers - 1].get("turns")
    return {
        "scenario_id": scenario_id,
        "backend": backend,
        "patient_answers": patient_answers,
        "latest_response": checkpoint["latest_response"],
        "answers": checkpoint["answers"],
        "stop_probability": round(float(stop_scores[patient_answers - 1]), 4),
        "stop_threshold": float(threshold),
        "should_stop": bool(forced or policy_stop),
        "forced_final_stop": forced,
        "next_question": None if forced or policy_stop else turns[patient_answers * 2]["content"],
    }


class EBMBackend:
    name = "ebm"

    def __init__(self, artifact_dir, data_path):
        artifact_dir = Path(artifact_dir)
        self.models = joblib.load(artifact_dir / "ebm_sequential_models.joblib")
        stop_bundle = joblib.load(artifact_dir / "ebm_stopping_controller.joblib")
        self.stop_model = stop_bundle["model"]
        self.threshold = float(stop_bundle["threshold"])
        self.store = ScenarioStore(data_path)

    def predict(self, scenario_id, patient_answers):
        rows = self.store.row(scenario_id)
        trace = ebm_confidence_traces(rows, self.models)[0]
        turns = conversation_turns(rows.iloc[0])
        for checkpoint in trace:
            checkpoint["turns"] = turns
        scores = stopping_probabilities(self.stop_model, [trace])[0]
        return _response(
            scenario_id, patient_answers, trace, scores, self.threshold, self.name
        )


class LayaBackend:
    name = "laya"

    def __init__(self, model_dir, artifact_dir, data_path, device="cuda"):
        import laya

        artifact_dir = Path(artifact_dir)
        self.agent = laya.load(str(model_dir), device=device)
        stop_bundle = joblib.load(artifact_dir / "laya_stopping_controller.joblib")
        self.stop_model = stop_bundle["model"]
        self.threshold = float(stop_bundle["threshold"])
        self.store = ScenarioStore(data_path)

    def predict(self, scenario_id, patient_answers):
        rows = self.store.row(scenario_id)
        trace = confidence_traces(rows, self.agent, batch_size=5)[0]
        turns = conversation_turns(rows.iloc[0])
        for checkpoint in trace:
            checkpoint["turns"] = turns
        scores = stopping_probabilities(self.stop_model, [trace])[0]
        return _response(
            scenario_id, patient_answers, trace, scores, self.threshold, self.name
        )


def backend_from_environment():
    backend = os.getenv("MODEL_BACKEND", "ebm").lower()
    data_path = os.getenv("SCENARIO_DATA", "data/generated/synthetic_scenarios.csv")
    artifact_dir = os.getenv("MODEL_ARTIFACT_DIR", "/models")
    if backend == "ebm":
        return EBMBackend(artifact_dir, data_path)
    if backend == "laya":
        return LayaBackend(
            os.environ["LAYA_MODEL_DIR"],
            artifact_dir,
            data_path,
            device=os.getenv("LAYA_DEVICE", "cuda"),
        )
    raise ValueError(f"Unsupported MODEL_BACKEND: {backend}")
