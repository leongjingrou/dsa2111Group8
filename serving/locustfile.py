"""Locust workload for gradual multi-device conversational inference tests."""

import csv
import itertools
import os
import random
import uuid
from pathlib import Path

from locust import HttpUser, LoadTestShape, between, task


MAX_PATIENT_ANSWERS = 5
DEVICES = ("mobile",) * 7 + ("desktop",) * 2 + ("tablet",)
DEFAULT_STAGES = (
    (45, 8, 2),
    (105, 32, 4),
    (165, 128, 16),
    (225, 256, 32),
    (285, 512, 64),
    (405, 1024, 128),
)


def load_scenario_ids(path=None):
    source_path = Path(path or os.getenv(
        "SCENARIO_DATA", "data/generated/synthetic_scenarios.csv"
    ))
    with source_path.open(newline="", encoding="utf-8") as source:
        return [row["scenario_id"] for row in csv.DictReader(source)]


def parse_stages(value=None):
    """Parse seconds:users:spawn_rate triples separated by commas."""
    if not value:
        return DEFAULT_STAGES
    stages = []
    for item in value.split(","):
        seconds, users, rate = (int(part) for part in item.split(":"))
        stages.append((seconds, users, rate))
    if not stages or any(a >= b for (a, *_), (b, *__) in zip(stages, stages[1:])):
        raise ValueError("stage end times must increase")
    return tuple(stages)


class ConsultationUser(HttpUser):
    # Human reading/typing delay between successive replies.
    wait_time = between(0.6, 2.4)

    def on_start(self):
        if not hasattr(self.environment, "scenario_cycle"):
            ids = load_scenario_ids()
            random.Random(2111).shuffle(ids)
            self.environment.scenario_cycle = itertools.cycle(ids)
        self.device_type = random.choice(DEVICES)
        self._start_consultation()

    def _start_consultation(self):
        self.client_id = f"{self.device_type}-{uuid.uuid4()}"
        self.scenario_id = next(self.environment.scenario_cycle)
        self.patient_answers = 1

    @task
    def consultation_turn(self):
        payload = {
            "client_id": self.client_id,
            "request_id": str(uuid.uuid4()),
            "scenario_id": self.scenario_id,
            "patient_answers": self.patient_answers,
            "device_type": self.device_type,
        }
        with self.client.post(
            "/v1/simulations/consultations/turn",
            json=payload,
            name="consultation turn",
            timeout=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                self._start_consultation()
                return
            try:
                body = response.json()
            except ValueError:
                response.failure("invalid JSON")
                self._start_consultation()
                return
            if body.get("should_stop") or self.patient_answers >= MAX_PATIENT_ANSWERS:
                self._start_consultation()
            else:
                self.patient_answers += 1


class RampToOneThousandTwentyFour(LoadTestShape):
    """Six-stage, 6m45s ramp with a two-minute 1,024-user observation."""

    stages = parse_stages(os.getenv("LOCUST_STAGES"))

    def tick(self):
        elapsed = self.get_run_time()
        for end_time, users, spawn_rate in self.stages:
            if elapsed < end_time:
                return users, spawn_rate
        return None
