"""Curated symptom questions shared by the six supported-condition workflows."""

import json
from functools import lru_cache
from pathlib import Path

from knowledge_base.kb_service import retrieve_knowledge
from knowledge_base.retrieval import load_json
from llm_service.app.vocab import get_vocab

CONFIG_PATH = Path(__file__).resolve().parent / "data" / "condition_followups.json"


@lru_cache
def followup_config():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    supported = {item["id"] for item in load_json("conditions.json")["conditions"]}
    if set(config) != supported:
        raise ValueError("Follow-up configuration must cover every supported condition")
    for condition_id, plan in config.items():
        for symptom_id in plan["anchors"] + plan["followups"]:
            if not get_vocab().is_valid(symptom_id):
                raise ValueError(f"Unknown follow-up symptom for {condition_id}: {symptom_id}")
    return config


def positive_symptoms(profile):
    return {
        symptom_id
        for symptom_id, symptom in profile.symptoms.items()
        if symptom.presence == "present"
    }


def knowledge_query(profile):
    return " ".join(
        get_vocab().name(symptom_id)
        for symptom_id, symptom in profile.symptoms.items()
        if symptom.presence == "present" and get_vocab().is_valid(symptom_id)
    )


def next_followup(profile, trace):
    """Choose one evidence question; retrieval ranks topics, never diagnoses."""
    if any(
        (item.get("question_target") or {}).get("target") == "condition_symptoms"
        for item in trace
    ):
        return None
    query = knowledge_query(profile)
    if not query:
        return None
    present = positive_symptoms(profile)
    config = followup_config()
    for result in retrieve_knowledge(query, top_k=len(config))["results"]:
        condition_id = result["condition_id"]
        plan = config.get(condition_id)
        if not plan or not present.intersection(plan["anchors"]):
            continue
        missing = [
            symptom_id
            for symptom_id in plan["followups"]
            if symptom_id not in profile.symptoms
        ]
        if missing:
            return condition_id, missing
    return None


def matched_condition_evidence(result, symptoms):
    """Expose checked facts alongside each text match, including denials."""
    plan = followup_config().get(result["condition_id"])
    if plan is None:
        return {"present": [], "absent": []}
    relevant = set(plan["anchors"] + plan["followups"])
    return {
        presence: [
            get_vocab().name(symptom_id)
            for symptom_id, symptom in symptoms.items()
            if symptom["presence"] == presence and symptom_id in relevant
        ]
        for presence in ("present", "absent")
    }


def supported_condition_matches(profile):
    """Return text matches backed by an anchor and a second reported symptom.

    A retrieval score alone can rank a condition for a shared symptom such as
    nausea. Requiring two positive facts, including a condition anchor, keeps
    that weak overlap out of patient-facing possibilities.
    """
    query = knowledge_query(profile)
    if not query:
        return None
    matches = retrieve_knowledge(query, top_k=len(followup_config()))
    present = positive_symptoms(profile)
    supported = []
    for result in matches["results"]:
        plan = followup_config().get(result["condition_id"])
        if plan is None or not present.intersection(plan["anchors"]):
            continue
        result["reported_evidence"] = matched_condition_evidence(
            result, {key: value.model_dump() for key, value in profile.symptoms.items()}
        )
        if len(result["reported_evidence"]["present"]) >= 2:
            supported.append(result)
    matches["results"] = supported
    return matches
