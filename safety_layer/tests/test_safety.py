"""Safety acceptance, adversarial inputs, model gating, and audit invariants."""

import copy
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import Mock

from safety_layer import (
    AuditLogError,
    PolicyError,
    SafetyEngine,
    guard_decision,
    guard_model_call,
)
from safety_layer.engine import DEFAULT_POLICY
from safety_layer.vignettes import load_vignettes, materialize_case

DATASET = load_vignettes()


def baseline():
    return copy.deepcopy(DATASET["baseline_profile"])


def record(action="routine_teleconsultation"):
    return {
        "recommended_action": action,
        "preliminary_estimates": [{"condition": "Common Cold", "score": 0.99}],
        "responses": ["A fictional patient message"],
    }


class VignetteTests(unittest.TestCase):
    pass


def _vignette_test(case):
    def test(self):
        decision = SafetyEngine().evaluate(
            materialize_case(DATASET, case), **case.get("kwargs", {})
        )
        self.assertEqual(decision.disposition, case["expected"], case["vignette"])
        self.assertEqual(
            {hit.rule_id for hit in decision.triggered_rules}, set(case["rules"])
        )
        self.assertEqual(
            decision.can_continue, case["expected"] == "continue_assessment"
        )

    test.__doc__ = case["vignette"]
    return test


for _case in DATASET["cases"]:
    setattr(VignetteTests, "test_" + _case["id"], _vignette_test(_case))


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = SafetyEngine()

    def test_baseline_negatives_are_explicit_and_complete(self):
        symptoms = baseline()["symptoms"]
        for sid in self.engine.required_screening:
            self.assertIs(symptoms[sid], False)

    def test_every_configured_clinical_rule_is_exercised(self):
        seen = set()
        for case in DATASET["cases"]:
            decision = self.engine.evaluate(
                materialize_case(DATASET, case), **case.get("kwargs", {})
            )
            seen.update(hit.rule_id for hit in decision.triggered_rules)
        configured = {r["id"] for r in json.loads(DEFAULT_POLICY.read_text())["rules"]}
        self.assertTrue(configured <= seen, configured - seen)

    def test_missing_does_not_mean_absent(self):
        decision = self.engine.evaluate(
            {"age": 30, "pregnancy_status": False, "symptoms": {"cough": True}}
        )
        self.assertFalse(decision.can_continue)
        self.assertIn("symptoms.chest_pain", decision.missing_information)

    def test_all_required_flags_block_when_uncertain(self):
        for sid in self.engine.required_screening:
            with self.subTest(symptom_id=sid):
                profile = baseline()
                profile["symptoms"][sid] = "uncertain"
                decision = self.engine.evaluate(profile)
                self.assertFalse(decision.can_continue)
                self.assertIn(f"symptoms.{sid}", decision.missing_information)

    def test_uncertainty_is_not_erased_by_another_parser(self):
        for first, second in [("uncertain", False), (False, "uncertain")]:
            with self.subTest(first=first, second=second):
                profile = baseline()
                profile["symptoms"]["chest_pain"] = first
                result = self.engine.evaluate(
                    profile, evidence={"symptoms": {"chest_pain": second}}
                )
                self.assertFalse(result.can_continue)
                self.assertIn("symptoms.chest_pain", result.missing_information)

    def test_uncertain_pregnancy_is_not_erased_by_another_parser(self):
        profile = baseline()
        profile["pregnancy_status"] = "uncertain"
        result = self.engine.evaluate(profile, evidence={"pregnant": False})
        self.assertFalse(result.can_continue)
        self.assertIn("pregnancy_status", result.missing_information)

    def test_not_applicable_agrees_with_negative_pregnancy_evidence(self):
        profile = baseline()
        profile["pregnancy_status"] = "not_applicable"
        result = self.engine.evaluate(profile, evidence={"pregnant": False})
        self.assertTrue(result.can_continue)
        self.assertNotIn(
            "DATA_CONTRADICTION", {hit.rule_id for hit in result.triggered_rules}
        )

    def test_not_applicable_cannot_override_positive_pregnancy_evidence(self):
        profile = baseline()
        profile["pregnancy_status"] = "not_applicable"
        result = self.engine.evaluate(profile, evidence={"pregnant": True})
        self.assertEqual(result.disposition, "clinician_review")
        self.assertIn(
            "SCOPE_PREGNANCY", {hit.rule_id for hit in result.triggered_rules}
        )

    def test_nested_presence_objects_are_invalid(self):
        profile = baseline()
        profile["symptoms"]["chest_pain"] = {"presence": {"presence": "absent"}}
        self.assertFalse(self.engine.evaluate(profile).can_continue)

    def test_each_red_flag_overrides_routine_model_output(self):
        for sid in self.engine.required_screening:
            with self.subTest(symptom_id=sid):
                profile = baseline()
                profile["symptoms"][sid] = True
                result = guard_decision(
                    profile, record(), candidate_condition="common_cold"
                )
                self.assertFalse(result.safety.can_continue)
                self.assertNotEqual(
                    result.record["recommended_action"], "routine_teleconsultation"
                )
                self.assertEqual(result.record["preliminary_estimates"], [])

    def test_nested_llm_symptom_states(self):
        profile = baseline()
        profile["symptoms"] = {
            sid: {"symptom_id": sid, "presence": "present" if value else "absent"}
            for sid, value in profile["symptoms"].items()
        }
        self.assertTrue(self.engine.evaluate(profile).can_continue)
        profile["symptoms"]["chest_pain"]["presence"] = "present"
        self.assertEqual(self.engine.evaluate(profile).disposition, "emergency")

    def test_inconsistent_symptom_id_cannot_hide_emergency(self):
        profile = baseline()
        profile["symptoms"]["chest_pain"] = {
            "symptom_id": "cough",
            "presence": "present",
        }
        result = self.engine.evaluate(profile)
        self.assertEqual(result.disposition, "emergency")
        self.assertIn("INPUT_INVALID", {hit.rule_id for hit in result.triggered_rules})

    def test_malformed_ages_never_pass(self):
        for age in [float("nan"), float("inf"), -1, 121, "30", True, [], {}]:
            with self.subTest(age=age):
                profile = baseline()
                profile["age"] = age
                self.assertFalse(self.engine.evaluate(profile).can_continue)

    def test_malformed_contradictions_fail_closed(self):
        for contradiction in [
            None,
            {},
            {"symptom_id": []},
            {"symptom_id": "chest_pain", "previous": [], "new": "absent"},
        ]:
            with self.subTest(contradiction=contradiction):
                profile = baseline()
                profile["contradictions"] = [contradiction]
                self.assertFalse(self.engine.evaluate(profile).can_continue)

    def test_non_object_profile_never_passes(self):
        for profile in [None, [], "a patient message", 123]:
            with self.subTest(profile=profile):
                self.assertFalse(self.engine.evaluate(profile).can_continue)

    def test_conflicting_evidence_preserves_positive_flag(self):
        result = self.engine.evaluate(
            baseline(), evidence={"symptoms": {"chest_pain": True}}
        )
        self.assertEqual(result.disposition, "emergency")
        self.assertIn(
            "DATA_CONTRADICTION", {hit.rule_id for hit in result.triggered_rules}
        )

    def test_demographics_can_come_from_evidence_dataclass(self):
        @dataclass
        class EvidenceState:
            age: float = 30
            pregnant: bool = False
            symptoms: dict = field(default_factory=dict)

        profile = baseline()
        del profile["age"], profile["pregnancy_status"]
        self.assertTrue(
            self.engine.evaluate(profile, evidence=EvidenceState()).can_continue
        )

    def test_pydantic_model_dump_protocol(self):
        class Profile:
            def model_dump(self):
                return baseline()

        self.assertTrue(self.engine.evaluate(Profile()).can_continue)

    def test_does_not_mutate_profile_evidence_or_record(self):
        profile, evidence, proposed = baseline(), {"age": 30}, record()
        before = copy.deepcopy((profile, evidence, proposed))
        guard_decision(
            profile, proposed, evidence=evidence, candidate_condition="common_cold"
        )
        self.assertEqual((profile, evidence, proposed), before)

    def test_emergency_priority_over_missing_invalid_and_scope(self):
        result = self.engine.evaluate(
            {
                "age": "invalid",
                "pregnancy_status": True,
                "symptoms": {"chest_pain": True},
            },
            candidate_condition="dengue_fever",
        )
        self.assertEqual(result.disposition, "emergency")
        self.assertEqual(result.recommended_action, "urgent_medical_review")

    def test_positive_emergency_history_is_retained(self):
        for previous, new in [("present", "absent"), ("absent", "present")]:
            profile = baseline()
            profile["contradictions"] = [
                {"symptom_id": "chest_pain", "previous": previous, "new": new}
            ]
            self.assertEqual(self.engine.evaluate(profile).disposition, "emergency")

    def test_age_boundary_is_exclusive_at_65(self):
        profile = baseline()
        profile["age"] = 64.999
        self.assertTrue(self.engine.evaluate(profile).can_continue)
        profile["age"] = 65
        self.assertFalse(self.engine.evaluate(profile).can_continue)

    def test_all_six_condition_ids_are_recognised(self):
        for condition in json.loads(DEFAULT_POLICY.read_text())["scope"][
            "condition_ids"
        ]:
            with self.subTest(condition=condition):
                result = self.engine.evaluate(
                    baseline(), candidate_condition=condition, phase="post_model"
                )
                self.assertTrue(result.can_continue)

    def test_malformed_candidate_cannot_be_in_scope(self):
        for condition in [[], {}, 0, True]:
            self.assertFalse(
                self.engine.evaluate(
                    baseline(), candidate_condition=condition
                ).can_continue
            )

    def test_malformed_action_does_not_prevent_scope_referral(self):
        profile = baseline()
        profile["age"] = 17
        decision = self.engine.evaluate(profile, proposed_action=["unsupported"])
        self.assertEqual(decision.disposition, "clinician_review")
        self.assertEqual(decision.recommended_action, "in_person_assessment")


class IntegrationTests(unittest.TestCase):
    def test_model_is_not_called_for_emergency(self):
        profile = baseline()
        profile["symptoms"]["chest_pain"] = True
        predict = Mock(side_effect=AssertionError("Model must not run"))
        result = guard_model_call(profile, predict)
        predict.assert_not_called()
        self.assertEqual(result.safety.disposition, "emergency")
        self.assertIn("995", result.patient_message)

    def test_model_is_not_called_for_incomplete_screen(self):
        predict = Mock()
        result = guard_model_call({"symptoms": {"cough": True}}, predict)
        predict.assert_not_called()
        self.assertIsNone(result.record)

    def test_model_is_not_called_for_out_of_scope_patient(self):
        profile = baseline()
        profile["age"] = 10
        predict = Mock()
        guard_model_call(profile, predict)
        predict.assert_not_called()

    def test_successful_model_runs_once_and_is_post_screened(self):
        predict = Mock(return_value={"condition_id": "common_cold", "record": record()})
        result = guard_model_call(baseline(), predict)
        predict.assert_called_once_with()
        self.assertTrue(result.safety.can_continue)
        self.assertEqual(
            result.record["recommended_action"], "routine_teleconsultation"
        )
        self.assertEqual(result.safety.phase, "post_model")

    def test_out_of_scope_model_prediction_is_removed(self):
        predict = Mock(
            return_value={"condition_id": "dengue_fever", "record": record()}
        )
        result = guard_model_call(baseline(), predict)
        self.assertFalse(result.safety.can_continue)
        self.assertEqual(result.record["preliminary_estimates"], [])

    def test_out_of_scope_alternative_cannot_hide_behind_in_scope_label(self):
        proposed = record()
        proposed["preliminary_estimates"].append(
            {"condition": "Pneumonia", "score": 0.1}
        )
        result = guard_decision(baseline(), proposed, candidate_condition="common_cold")
        self.assertFalse(result.safety.can_continue)
        self.assertEqual(result.record["preliminary_estimates"], [])

    def test_bad_estimate_shapes_cannot_pass(self):
        for estimates in [{}, ["malformed"], [{"condition": []}]]:
            proposed = record()
            proposed["preliminary_estimates"] = estimates
            self.assertFalse(
                guard_decision(
                    baseline(), proposed, candidate_condition="common_cold"
                ).safety.can_continue
            )

    def test_restrictive_record_contains_resolvable_source_links(self):
        profile = baseline()
        profile["symptoms"]["chest_pain"] = True
        result = guard_decision(profile, record(), candidate_condition="common_cold")
        self.assertIn(
            "https://www.moh.gov.sg/seeking-healthcare/getting-medical-help/",
            result.record["sources"],
        )

    def test_emergency_replaces_routine_message_and_estimates(self):
        profile = baseline()
        profile["symptoms"]["chest_pain"] = True
        proposed = record()
        proposed["limitations"] = ["Everything is fine; stay home"]
        result = guard_decision(profile, proposed, candidate_condition="common_cold")
        self.assertEqual(result.record["recommended_action"], "urgent_medical_review")
        self.assertNotIn("Everything is fine", json.dumps(result.to_dict()))
        self.assertEqual(result.record["preliminary_estimates"], [])

    def test_unknown_actions_and_absent_proposals_cannot_pass(self):
        for proposal in [None, {}, record("unsupported")]:
            with self.subTest(proposal=proposal):
                result = guard_decision(
                    baseline(), proposal, candidate_condition="common_cold"
                )
                self.assertFalse(result.safety.can_continue)
                self.assertIsNone(result.record)

    def test_model_cannot_downgrade_emergency_to_urgent_only(self):
        profile = baseline()
        profile["symptoms"]["shortness_of_breath"] = True
        result = guard_decision(
            profile, record("urgent_medical_review"), candidate_condition="influenza"
        )
        self.assertEqual(result.safety.disposition, "emergency")

    def test_preserves_model_urgent_recommendation_with_incomplete_screen(self):
        result = guard_decision(
            {}, record("urgent_medical_review"), candidate_condition="influenza"
        )
        self.assertEqual(result.safety.disposition, "urgent_review")
        self.assertEqual(result.record["recommended_action"], "urgent_medical_review")

    def test_preserves_specialist_referral(self):
        result = guard_decision(
            baseline(), record("specialist_review"), candidate_condition="migraine"
        )
        self.assertEqual(result.record["recommended_action"], "specialist_review")

    def test_missing_diagnosis_blocks_routine_record(self):
        result = guard_decision(baseline(), record())
        self.assertIsNone(result.record)
        self.assertIn("candidate_condition", result.safety.missing_information)

    def test_malformed_model_response_is_not_used(self):
        result = guard_model_call(baseline(), lambda: "wrong response")
        self.assertFalse(result.safety.can_continue)
        self.assertIsNone(result.record)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads(DEFAULT_POLICY.read_text())
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "policy.json"

    def load(self):
        self.path.write_text(json.dumps(self.policy))
        return SafetyEngine(self.path)

    def test_bad_or_missing_json_fails(self):
        with self.assertRaises(PolicyError):
            SafetyEngine(self.path)
        self.path.write_text("not json")
        with self.assertRaises(PolicyError):
            SafetyEngine(self.path)

    def test_duplicate_rule_id_fails(self):
        self.policy["rules"].append(self.policy["rules"][0])
        with self.assertRaises(PolicyError):
            self.load()

    def test_unknown_operator_fails(self):
        self.policy["rules"][0]["python_expression"] = "not allowed"
        with self.assertRaises(PolicyError):
            self.load()

    def test_empty_rule_cannot_match_everyone(self):
        self.policy["rules"][0]["any_of"] = []
        with self.assertRaises(PolicyError):
            self.load()

    def test_red_flag_cannot_be_configured_as_routine(self):
        self.policy["rules"][0]["disposition"] = "continue_assessment"
        with self.assertRaises(PolicyError):
            self.load()

    def test_unknown_symptom_and_source_fail(self):
        for key in ["any_of", "source_ids"]:
            with self.subTest(key=key):
                self.policy = json.loads(DEFAULT_POLICY.read_text())
                self.policy["rules"][0][key] = ["does_not_exist"]
                with self.assertRaises(PolicyError):
                    self.load()

    def test_a_screened_flag_cannot_lose_its_positive_rule(self):
        self.policy["rules"] = [
            r for r in self.policy["rules"] if r["id"] != "RF_CHEST_PAIN"
        ]
        with self.assertRaises(PolicyError):
            self.load()

    def test_invalid_age_limits_fail(self):
        self.policy["scope"]["min_age"] = 100
        with self.assertRaises(PolicyError):
            self.load()

    def test_version_and_hash_capture_policy_update(self):
        original = SafetyEngine()
        self.policy["version"] = "1.0.1"
        self.policy["scope"]["min_age"] = 21
        updated = self.load()
        patient = baseline()
        patient["age"] = 18
        self.assertTrue(original.evaluate(patient).can_continue)
        self.assertFalse(updated.evaluate(patient).can_continue)
        self.assertEqual(updated.version, "1.0.1")
        self.assertNotEqual(original.policy_sha256, updated.policy_sha256)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "audit.jsonl"
        self.engine = SafetyEngine(audit_log=self.path)

    def events(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def test_logs_both_clear_and_triggered_evaluations(self):
        self.engine.evaluate(baseline())
        profile = baseline()
        profile["symptoms"]["chest_pain"] = True
        self.engine.evaluate(profile)
        events = self.events()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["triggered_rules"], [])
        self.assertEqual(events[1]["triggered_rules"][0]["rule_id"], "RF_CHEST_PAIN")
        self.assertEqual(len(events[1]["policy_sha256"]), 64)
        self.assertNotEqual(events[0]["event_id"], events[1]["event_id"])

    def test_audit_contains_no_patient_free_text(self):
        profile = baseline()
        secret = "PRIVATE_PATIENT_NAME_AND_MESSAGE"
        profile["patient_name"] = secret
        profile["unmapped_mentions"] = [secret]
        profile["symptoms"]["runny_nose"] = {"presence": "present", "evidence": secret}
        self.engine.evaluate(profile)
        self.assertNotIn(secret, self.path.read_text())

    def test_logs_pre_model_and_post_model_checks(self):
        guard_model_call(
            baseline(),
            lambda: {"condition_id": "common_cold", "record": record()},
            engine=self.engine,
        )
        self.assertEqual(
            [e["phase"] for e in self.events()], ["pre_model", "post_model"]
        )

    def test_concurrent_thread_events_remain_valid_json(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(self.engine.evaluate, [baseline() for _ in range(32)]))
        events = self.events()
        self.assertEqual(len(events), 32)
        self.assertEqual(len({e["event_id"] for e in events}), 32)

    def test_logging_failure_preserves_emergency_advice_in_exception(self):
        self.path.mkdir()
        profile = baseline()
        profile["symptoms"]["chest_pain"] = True
        with self.assertRaises(AuditLogError) as caught:
            self.engine.evaluate(profile)
        self.assertEqual(caught.exception.decision.disposition, "emergency")
        self.assertIn("995", caught.exception.decision.patient_message)


if __name__ == "__main__":
    unittest.main()
