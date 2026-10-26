"""Public safety results shared by the engine, audit writer and HTTP contract."""

import json
from dataclasses import asdict, dataclass
from typing import Any

DISPOSITION_PRIORITY = {
    "continue_assessment": 0,
    "needs_clarification": 1,
    "clinician_review": 2,
    "urgent_review": 3,
    "emergency": 4,
}


@dataclass(frozen=True)
class RuleHit:
    rule_id: str
    disposition: str
    reason: str
    fields: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SafetyDecision:
    event_id: str
    evaluated_at: str
    policy_version: str
    policy_sha256: str
    phase: str
    disposition: str
    recommended_action: str | None
    can_continue: bool
    scope_status: str
    triggered_rules: tuple[RuleHit, ...]
    missing_information: tuple[str, ...]
    patient_message: str

    def to_dict(self) -> dict[str, Any]:
        # JSON round-trip also turns immutable tuples into JSON arrays.
        return json.loads(json.dumps(asdict(self)))
