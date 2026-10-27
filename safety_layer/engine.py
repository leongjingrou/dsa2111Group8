"""Apply clinical, scope and data-quality rules to structured patient facts."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .facts import PatientFacts
from .models import DISPOSITION_PRIORITY, RuleHit, SafetyDecision
from .policy import DEFAULT_POLICY, PolicyError, validate_policy

SUPPORTED_ACTIONS = {
    "self_care",
    "routine_teleconsultation",
    "specialist_review",
    "in_person_assessment",
    "urgent_medical_review",
}
SCOPE_FIELDS = {"age", "pregnancy_status", "candidate_condition"}
_AUDIT_LOCK = threading.Lock()


class AuditLogError(RuntimeError):
    """Audit storage failed; the evaluated decision still carries urgent advice."""

    def __init__(self, decision: SafetyDecision):
        super().__init__("Safety evaluation could not be written to the audit log")
        self.decision = decision


@dataclass
class _ScopeAssessment:
    rule_hits: list[RuleHit] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    is_out_of_scope: bool = False


class SafetyEngine:
    """Load one policy and evaluate facts without consulting a diagnostic model."""

    def __init__(
        self,
        policy_path: str | Path = DEFAULT_POLICY,
        audit_log: str | Path | None = None,
    ):
        try:
            policy_bytes = Path(policy_path).read_bytes()
            policy = json.loads(policy_bytes)
        except (OSError, ValueError) as error:
            raise PolicyError(f"Cannot load safety policy: {error}") from error
        validate_policy(policy)
        self._policy = policy
        self.policy_sha256 = hashlib.sha256(policy_bytes).hexdigest()
        self.audit_log = Path(audit_log) if audit_log is not None else None

    @property
    def version(self) -> str:
        return self._policy["version"]

    @property
    def required_screening(self) -> tuple[str, ...]:
        return tuple(self._policy["required_screening"])

    @property
    def clinical_rule_ids(self) -> tuple[str, ...]:
        return tuple(rule["id"] for rule in self._policy["rules"])

    @property
    def source_urls(self) -> dict[str, str]:
        return {
            source_id: source["url"]
            for source_id, source in self._policy["sources"].items()
        }

    def condition_is_in_scope(self, condition: Any) -> bool:
        if not isinstance(condition, str):
            return False
        condition_id = re.sub(r"[^a-z0-9]+", "_", condition.lower()).strip("_")
        return condition_id in self._policy["scope"]["condition_ids"]

    def evaluate(
        self,
        profile: Any,
        *,
        evidence: Any = None,
        candidate_condition: str | None = None,
        proposed_action: str | None = None,
        preserve_model_referral: bool = True,
        phase: str = "pre_model",
    ) -> SafetyDecision:
        """Normalize facts, collect every rule hit, then choose the highest urgency.

        Inputs may be mappings, Pydantic Profiles or EvidenceState dataclasses.
        Missing/uncertain findings never count as negative screening answers.
        """
        if phase not in {"pre_model", "post_model"}:
            raise ValueError("phase must be pre_model or post_model")
        facts = PatientFacts.from_inputs(
            profile, evidence, set(self._policy["symptom_ids"])
        )
        scope = self._assess_scope(facts, candidate_condition, phase)
        rule_hits = self._match_clinical_rules(facts)
        rule_hits.extend(scope.rule_hits)
        rule_hits.extend(self._check_data_quality(facts, scope.missing_fields))
        if preserve_model_referral:
            rule_hits.extend(self._preserve_model_referral(proposed_action))
        decision = self._build_decision(facts, scope, rule_hits, proposed_action, phase)
        if self.audit_log is not None:
            try:
                self._write_audit(decision)
            except OSError as error:
                raise AuditLogError(decision) from error
        return decision

    def _match_clinical_rules(self, facts: PatientFacts) -> list[RuleHit]:
        rule_hits = []
        for rule in self._policy["rules"]:
            any_required = set(rule.get("any_of", []))
            all_required = set(rule.get("all_of", []))
            matches_any = not any_required or bool(
                any_required & facts.positive_symptoms
            )
            matches_all = all_required <= facts.positive_symptoms
            if matches_any and matches_all:
                matching_fields = sorted(
                    (any_required & facts.positive_symptoms) | all_required
                )
                rule_hits.append(
                    RuleHit(
                        rule["id"],
                        rule["disposition"],
                        rule["reason"],
                        tuple(matching_fields),
                        tuple(rule["source_ids"]),
                    )
                )
        return rule_hits

    def _assess_scope(self, facts, candidate_condition, phase):
        assessment = _ScopeAssessment()
        scope_policy = self._policy["scope"]
        if not facts.ages:
            assessment.missing_fields.append("age")
        elif any(
            not scope_policy["min_age"] <= age < scope_policy["max_age_exclusive"]
            for age in facts.ages
        ):
            assessment.is_out_of_scope = True
            assessment.rule_hits.append(
                RuleHit(
                    "SCOPE_AGE",
                    "clinician_review",
                    "Age is outside the configured prototype scope.",
                    ("age",),
                )
            )
        if "pregnant" in facts.pregnancy_states:
            assessment.is_out_of_scope = True
            assessment.rule_hits.append(
                RuleHit(
                    "SCOPE_PREGNANCY",
                    "clinician_review",
                    "Pregnancy requires care outside this prototype.",
                    ("pregnancy_status",),
                )
            )
        elif not facts.pregnancy_states or facts.pregnancy_uncertain:
            assessment.missing_fields.append("pregnancy_status")
        if candidate_condition is not None:
            if not self.condition_is_in_scope(candidate_condition):
                assessment.is_out_of_scope = True
                assessment.rule_hits.append(
                    RuleHit(
                        "SCOPE_CONDITION",
                        "clinician_review",
                        "The proposed condition is outside the six-condition knowledge base.",
                        ("candidate_condition",),
                    )
                )
        elif phase == "post_model":
            assessment.missing_fields.append("candidate_condition")
        if facts.has_unmapped_findings:
            assessment.is_out_of_scope = True
            assessment.rule_hits.append(
                RuleHit(
                    "SCOPE_UNMAPPED",
                    "clinician_review",
                    "Unmapped findings require clinician assessment; they cannot be silently discarded.",
                    ("unmapped_findings",),
                )
            )
        return assessment

    def _check_data_quality(self, facts, missing_fields):
        rule_hits = []
        if facts.invalid_fields:
            rule_hits.append(
                RuleHit(
                    "INPUT_INVALID",
                    "needs_clarification",
                    "Some structured fields are invalid and must be corrected.",
                    tuple(sorted(set(facts.invalid_fields))),
                )
            )
        if facts.conflicting_fields:
            rule_hits.append(
                RuleHit(
                    "DATA_CONTRADICTION",
                    "needs_clarification",
                    "Conflicting findings need clarification; prior positive red flags remain active.",
                    tuple(sorted(facts.conflicting_fields)),
                )
            )
        if not facts.positive_symptoms:
            missing_fields.append("presenting_symptoms")
        for symptom_id in self.required_screening:
            is_negative = facts.symptom_states.get(symptom_id) == "absent"
            has_positive_history = symptom_id in facts.positive_symptoms
            if not is_negative and not has_positive_history:
                missing_fields.append(f"symptoms.{symptom_id}")
        if missing_fields:
            rule_hits.append(
                RuleHit(
                    "SCREEN_INCOMPLETE",
                    "needs_clarification",
                    "Required scope or red-flag answers are missing or uncertain.",
                    tuple(sorted(set(missing_fields))),
                )
            )
        return rule_hits

    @staticmethod
    def _preserve_model_referral(proposed_action):
        if proposed_action is None:
            return []
        if (
            not isinstance(proposed_action, str)
            or proposed_action not in SUPPORTED_ACTIONS
        ):
            return [
                RuleHit(
                    "MODEL_ACTION_INVALID",
                    "needs_clarification",
                    "The proposed action is not a supported action.",
                )
            ]
        if proposed_action == "urgent_medical_review":
            return [
                RuleHit(
                    "MODEL_URGENT",
                    "urgent_review",
                    "Preserve the model's urgent referral; safety screening cannot downgrade it.",
                )
            ]
        if proposed_action in {"in_person_assessment", "specialist_review"}:
            return [
                RuleHit(
                    "MODEL_REFERRAL",
                    "clinician_review",
                    "Preserve the model's clinician referral.",
                )
            ]
        return []

    @staticmethod
    def _recommended_action(disposition, proposed_action):
        if DISPOSITION_PRIORITY[disposition] >= DISPOSITION_PRIORITY["urgent_review"]:
            return "urgent_medical_review"
        if disposition == "clinician_review":
            if isinstance(proposed_action, str) and proposed_action in {
                "in_person_assessment",
                "specialist_review",
            }:
                return proposed_action
            return "in_person_assessment"
        if disposition == "continue_assessment":
            return proposed_action
        return None

    def _build_decision(self, facts, scope, rule_hits, proposed_action, phase):
        disposition = max(
            (hit.disposition for hit in rule_hits),
            key=DISPOSITION_PRIORITY.get,
            default="continue_assessment",
        )
        if scope.is_out_of_scope:
            scope_status = "out_of_scope"
        elif (
            facts.invalid_fields
            or facts.conflicting_fields
            or SCOPE_FIELDS.intersection(scope.missing_fields)
        ):
            scope_status = "unknown"
        else:
            scope_status = "in_scope"
        return SafetyDecision(
            event_id=str(uuid.uuid4()),
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            policy_version=self.version,
            policy_sha256=self.policy_sha256,
            phase=phase,
            disposition=disposition,
            recommended_action=self._recommended_action(disposition, proposed_action),
            can_continue=disposition == "continue_assessment",
            scope_status=scope_status,
            triggered_rules=tuple(rule_hits),
            missing_information=tuple(sorted(set(scope.missing_fields))),
            patient_message=self._policy["messages"][disposition],
        )

    def _write_audit(self, decision: SafetyDecision) -> None:
        """Append before returning; preserve advice if storage fails.

        Threads in one process are serialized. Patient messages, identities and
        free-text findings are intentionally excluded from the event.
        """
        event = decision.to_dict()
        event.pop("patient_message")
        line = json.dumps(event, separators=(",", ":")) + "\n"
        with _AUDIT_LOCK:
            self.audit_log.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_log.open("a", encoding="utf-8") as stream:
                stream.write(line)
