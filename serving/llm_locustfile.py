"""Locust workload for the stateful LLM-mediated consultation endpoint."""

import json
import os
import random
from pathlib import Path

from locust import HttpUser, LoadTestShape, between, task

from serving.locustfile import DEFAULT_STAGES, DEVICES, parse_stages


def load_replies():
    path = Path(os.getenv("SCENARIO_DATA", "data/generated/synthetic_scenarios.csv"))
    import csv

    rows = []
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            conversation = json.loads(row["conversation"])
            replies = [
                turn["content"] for turn in conversation if turn["role"] == "patient"
            ]
            rows.append(replies)
    return rows


REPLIES = load_replies()


class LLMConsultationUser(HttpUser):
    wait_time = between(0.2, 1.0)

    def on_start(self):
        self.device_type = random.choice(DEVICES)
        self._start()

    def _start(self):
        if getattr(self, "session_id", None):
            self.client.delete(
                f"/api/conversations/{self.session_id}", name="discard conversation"
            )
        self.replies = random.choice(REPLIES)
        self.reply_index = 0
        with self.client.post(
            "/api/conversations",
            json={},
            name="start conversation",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                self.session_id = None
                return
            self.session_id = response.json()["session_id"]

    @task
    def turn(self):
        if not self.session_id:
            self._start()
            return
        reply = self.replies[min(self.reply_index, len(self.replies) - 1)]
        with self.client.post(
            f"/api/conversations/{self.session_id}/turn",
            json={"patient_message": reply},
            name="conversation turn",
            timeout=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                self._start()
                return
            body = response.json()
            self.reply_index += 1
            if body.get("stop") or self.reply_index >= len(self.replies):
                self._start()


class ConversationRamp(LoadTestShape):
    stages = parse_stages(os.getenv("LOCUST_STAGES")) or DEFAULT_STAGES

    def tick(self):
        elapsed = self.get_run_time()
        for end_time, users, spawn_rate in self.stages:
            if elapsed < end_time:
                return users, spawn_rate
        return None
