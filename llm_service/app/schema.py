"""app/schema.py

All data models for llm_service.

Design rule: nothing the LLM returns can express a diagnosis or an urgency
level. ExtractionResult only has symptom facts, and the explainer only
receives a recommended_action decided elsewhere (Role 3's safety layer).
"""
from typing import Literal

from pydantic import BaseModel, Field

Presence = Literal["present", "absent", "uncertain"]
Severity = Literal["mild", "moderate", "severe"]
Progression = Literal["improving", "stable", "worsening"]
Focus = Literal["presence", "severity", "duration", "progression"]
Audience = Literal["patient", "clinician"]
Action = Literal[
    "self_care",
    "routine_teleconsultation",
    "specialist_review",
    "in_person_assessment",
    "urgent_medical_review",
]


# ---------------------------------------------------------------------------
# LLM output models. No defaults: every field is required. Use null / [] when nothing applies.
# ---------------------------------------------------------------------------
class ExtractedSymptom(BaseModel):
    symptom_id: str
    presence: Presence
    severity: Severity | None
    duration_days: float | None
    progression: Progression | None
    evidence: str  # short phrase copied from the patient's message


class ExtractionResult(BaseModel):
    symptoms: list[ExtractedSymptom]
    unmapped_mentions: list[str]  # patient's own words for symptoms not in the vocab
    medical_history: list[str]
    allergies: list[str]
    medications: list[str]


# ---------------------------------------------------------------------------
# Consultation profile (built by merge.py)
# ---------------------------------------------------------------------------
class SymptomState(BaseModel):
    symptom_id: str
    presence: Presence
    severity: Severity | None = None
    duration_days: float | None = None
    progression: Progression | None = None
    evidence: str | None = None


class Contradiction(BaseModel):
    symptom_id: str
    previous: Presence
    new: Presence


class Profile(BaseModel):
    symptoms: dict[str, SymptomState] = Field(default_factory=dict)
    medical_history: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    unmapped_mentions: list[str] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    # Items look like "fever:duration" or "presenting_symptoms". The part after
    # the colon matches the Focus values used by /next-question.
    missing_info: list[str] = Field(default_factory=list)
    turn_count: int = 0


# ---------------------------------------------------------------------------
# Decision record (produced by Role 3 / orchestrator, explained by us)
# AGREE THIS SHAPE WITH ROLE 3.
# ---------------------------------------------------------------------------
class ConditionEstimate(BaseModel):
    condition: str
    score: float | None = None


class DecisionRecord(BaseModel):
    recommended_action: Action
    responses: list[str] = Field(default_factory=list)
    preliminary_estimates: list[ConditionEstimate] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    triggered_rules: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------
class ExtractRequest(BaseModel):
    patient_message: str = Field(min_length=1, max_length=2000)
    current_profile: Profile = Field(default_factory=Profile)


class ExtractResponse(BaseModel):
    profile: Profile
    extraction: ExtractionResult


class NextQuestionRequest(BaseModel):
    symptom_id: str
    focus: Focus = "presence"


class NextQuestionResponse(BaseModel):
    question: str
    fallback_used: bool  # True if a fixed template was used instead of the LLM


class ExplainRequest(BaseModel):
    record: DecisionRecord
    audience: Audience = "patient"


class ExplainResponse(BaseModel):
    headline: str       # fixed text chosen by code from recommended_action
    explanation: str    # LLM-written (or fallback) body
    disclaimer: str     # fixed text
    fallback_used: bool
