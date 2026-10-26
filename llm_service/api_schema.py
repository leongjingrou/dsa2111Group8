"""Shared response contract for the conversation and diagnose routes."""

from typing import Literal

from pydantic import BaseModel

from llm_service.app.schema import DecisionRecord, ExtractionResult
from safety_layer import SafetyDecision


class ModelAnswer(BaseModel):
    choice: str
    probabilities: dict[str, float]
    answer_confidence: float


class QuestionTarget(BaseModel):
    kind: Literal["safety", "symptom", "context"]
    target: str
    focus: Literal["presence", "duration", "severity", "progression"] | None


class ConsultationResponse(BaseModel):
    session_id: str
    conversation_id: str
    status: Literal["asking", "complete", "urgent"]
    reply: str
    doctor_response: str
    backend: str
    patient_answer: int
    extraction: ExtractionResult
    diagnosis: ModelAnswer | None
    escalation: ModelAnswer | None
    decision_record: DecisionRecord | None
    knowledge_base: dict | None
    stop_probability: float | None
    stop_threshold: float
    stop: bool
    question_target: QuestionTarget | None
    question_fallback_used: bool
    safety_profile: Literal["rules_v1"]
    safety: SafetyDecision
    safety_override_applied: bool
    triggered_rules: list[str]
