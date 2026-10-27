"""Gate model calls and build records compatible with llm_service DecisionRecord."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .engine import SafetyEngine
from .facts import as_mapping
from .models import SafetyDecision


@dataclass(frozen=True)
class GuardedResult:
    safety: SafetyDecision
    record: dict[str, Any] | None

    @property
    def patient_message(self) -> str:
        return self.safety.patient_message

    def to_dict(self) -> dict:
        return {
            "safety": self.safety.to_dict(),
            "record": self.record,
            "patient_message": self.patient_message,
        }


def result_from_safety(
    safety: SafetyDecision,
    proposed: Mapping | None = None,
    source_urls: Mapping | None = None,
) -> GuardedResult:
    """Build the permitted record or fixed referral from an evaluated screen."""
    if safety.disposition == "needs_clarification":
        return GuardedResult(safety=safety, record=None)
    if safety.can_continue:
        return GuardedResult(
            safety=safety, record=dict(proposed) if proposed is not None else None
        )
    # Restrictive results contain no model diagnosis or speculative explanation.
    source_ids = sorted(
        {source for hit in safety.triggered_rules for source in hit.source_ids}
    )
    return GuardedResult(
        safety=safety,
        record={
            "recommended_action": safety.recommended_action,
            "responses": [],
            "preliminary_estimates": [],
            "risk_factors": [hit.reason for hit in safety.triggered_rules],
            "triggered_rules": [hit.rule_id for hit in safety.triggered_rules],
            "missing_information": list(safety.missing_information),
            "limitations": [
                "Research prototype; clinical review of policy is pending."
            ],
            "sources": [
                (source_urls or {}).get(source, source) for source in source_ids
            ],
        },
    )


def guard_decision(
    profile: Any,
    proposed_record: Any,
    *,
    candidate_condition: str | None = None,
    evidence: Any = None,
    engine: SafetyEngine | None = None,
    preserve_model_referral: bool = True,
) -> GuardedResult:
    """Mandatory post-model check. Use the returned record, never the original.

    On a restrictive result, render `patient_message` directly: do not let an
    LLM paraphrase away an emergency instruction. On clarification, collect the
    named missing information and re-evaluate instead of explaining a diagnosis.
    """
    engine = engine or SafetyEngine()
    record = as_mapping(proposed_record)
    action = record.get("recommended_action", "") if record is not None else ""
    # An in-scope top label must not hide an out-of-scope alternative in the
    # record that the patient-facing LLM will see.
    estimates = record.get("preliminary_estimates", []) if record is not None else []
    if not isinstance(estimates, list):
        candidate_condition = "invalid_model_estimates"
    else:
        for estimate in estimates:
            item = as_mapping(estimate)
            condition = item.get("condition") if item is not None else None
            if not engine.condition_is_in_scope(condition):
                candidate_condition = (
                    condition
                    if isinstance(condition, str)
                    else "invalid_model_estimates"
                )
                break
    decision = engine.evaluate(
        profile,
        evidence=evidence,
        candidate_condition=candidate_condition,
        proposed_action=action,
        preserve_model_referral=preserve_model_referral,
        phase="post_model",
    )
    return result_from_safety(decision, record, engine.source_urls)


def guard_model_call(
    profile: Any,
    predict: Callable[[], Mapping],
    *,
    evidence: Any = None,
    engine: SafetyEngine | None = None,
) -> GuardedResult:
    """Run a zero-argument model callback only after a completed safety screen.

    The callback returns {"condition_id": "...", "record": DecisionRecord/dict}.
    This synchronous adapter requires no model packages, network, or API keys.
    """
    engine = engine or SafetyEngine()
    before = engine.evaluate(profile, evidence=evidence)
    if not before.can_continue:
        return result_from_safety(before, source_urls=engine.source_urls)
    proposal = predict()
    if not isinstance(proposal, Mapping):
        proposal = {}
    return guard_decision(
        profile,
        proposal.get("record"),
        candidate_condition=proposal.get("condition_id"),
        evidence=evidence,
        engine=engine,
    )
