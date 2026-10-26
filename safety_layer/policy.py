"""Load-time validation for the versioned, data-only safety policy."""

import re
from pathlib import Path
from typing import Any

from .facts import is_finite_number
from .models import DISPOSITION_PRIORITY

DEFAULT_POLICY = Path(__file__).parent / "config" / "rules.v1.json"


class PolicyError(ValueError):
    """A policy cannot be loaded safely. Callers must not bypass this error."""


def validate_policy(policy: Any) -> None:
    """Reject malformed rules rather than silently skipping them."""
    try:
        if (
            not isinstance(policy, dict)
            or type(policy["schema_version"]) is not int
            or policy["schema_version"] != 1
        ):
            raise ValueError("Unsupported policy schema")
        if not re.fullmatch(r"\d+\.\d+\.\d+", policy["version"]):
            raise ValueError("Policy version must be semantic")
        scope = policy["scope"]
        if not (
            is_finite_number(scope["min_age"])
            and is_finite_number(scope["max_age_exclusive"])
            and 0 <= scope["min_age"] < scope["max_age_exclusive"] <= 121
        ):
            raise ValueError("Invalid age limits")
        conditions = scope["condition_ids"]
        if (
            not isinstance(conditions, list)
            or not conditions
            or len(set(conditions)) != len(conditions)
        ):
            raise ValueError("Invalid condition list")
        vocab = policy["symptom_ids"]
        screening = policy["required_screening"]
        if not isinstance(vocab, list) or not vocab or len(vocab) != len(set(vocab)):
            raise ValueError("Invalid symptom vocabulary")
        if (
            not isinstance(screening, list)
            or not screening
            or len(screening) != len(set(screening))
        ):
            raise ValueError("Invalid screening list")
        if not set(screening) <= set(vocab):
            raise ValueError("Screening has unknown symptoms")
        identifiers = conditions + vocab + list(policy["sources"])
        if not all(
            isinstance(x, str) and re.fullmatch(r"[a-z][a-z0-9_]*", x)
            for x in identifiers
        ):
            raise ValueError("Invalid identifier")
        for source in policy["sources"].values():
            if not source["url"].startswith("https://") or not source["accessed_date"]:
                raise ValueError("Missing source provenance")
        if set(policy["messages"]) != set(DISPOSITION_PRIORITY) or not all(
            policy["messages"].values()
        ):
            raise ValueError("Missing disposition messages")
        rule_ids, covered = set(), set()
        for rule in policy["rules"]:
            if set(rule) - {
                "id",
                "disposition",
                "reason",
                "any_of",
                "all_of",
                "source_ids",
            }:
                raise ValueError("Unknown rule field")
            if not re.fullmatch(r"RF_[A-Z0-9_]+", rule["id"]) or rule["id"] in rule_ids:
                raise ValueError("Invalid or duplicate rule ID")
            rule_ids.add(rule["id"])
            if rule["disposition"] not in {
                "clinician_review",
                "urgent_review",
                "emergency",
            }:
                raise ValueError("Unsafe red-flag disposition")
            if not isinstance(rule["reason"], str) or not rule["reason"]:
                raise ValueError("Missing rule reason")
            if not rule.get("any_of") and not rule.get("all_of"):
                raise ValueError("Empty rule predicate")
            for key in ("any_of", "all_of"):
                values = rule.get(key, [])
                if not isinstance(values, list) or not set(values) <= set(vocab):
                    raise ValueError("Unknown symptom in rule")
            if not rule["source_ids"] or not set(rule["source_ids"]) <= set(
                policy["sources"]
            ):
                raise ValueError("Unknown rule source")
            # Every screened flag needs a standalone positive rule.
            if not rule.get("all_of"):
                covered.update(rule.get("any_of", []))
        if not set(screening) <= covered:
            raise ValueError("Screened symptoms lack positive rules")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise PolicyError(f"Invalid safety policy: {exc}") from exc
