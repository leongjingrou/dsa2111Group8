import asyncio

import numpy as np

from diagnostic_model.llm_live_conversation import LLMDrivenConsultation
from llm_service.app.schema import ExtractedSymptom, ExtractionResult
from safety_layer import SafetyEngine


class ConstantStopModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, rows):
        return np.tile([0.05, 0.95], (len(rows), 1))


class StubPredictor:
    name = "ebm"

    def predict(self, turns, evidence, answer_number):
        return {
            "condition": {
                "choice": "influenza",
                "probabilities": {"influenza": 0.8, "common_cold": 0.2},
                "answer_confidence": 0.8,
            },
            "escalation": {
                "choice": "routine",
                "probabilities": {"routine": 0.9, "urgent": 0.1},
                "answer_confidence": 0.9,
            },
        }


class SequenceLLM:
    def __init__(self):
        self.extractions = [
            ExtractionResult(
                symptoms=[
                    ExtractedSymptom(
                        symptom_id="fever",
                        presence="present",
                        severity=None,
                        duration_days=None,
                        progression=None,
                        evidence="fever",
                    )
                ],
                unmapped_mentions=[],
                medical_history=[],
                allergies=[],
                medications=[],
            ),
            ExtractionResult(
                symptoms=[
                    ExtractedSymptom(
                        symptom_id="fever",
                        presence="present",
                        severity=None,
                        duration_days=2,
                        progression=None,
                        evidence="two days",
                    )
                ],
                unmapped_mentions=[],
                medical_history=[],
                allergies=[],
                medications=[],
            ),
        ]
        self.calls = []

    async def complete_structured(
        self, prompt_name, variables, output_model, temperature=0
    ):
        self.calls.append(prompt_name)
        return self.extractions.pop(0)

    async def complete_text(self, prompt_name, variables, temperature=0.3):
        self.calls.append(prompt_name)
        if prompt_name == "ask":
            return "How long have you had the fever?"
        if prompt_name == "explain":
            return "The model has enough information for a preliminary assessment."
        raise AssertionError(prompt_name)


def test_llm_phrases_questions_but_predictor_owns_decisions():
    async def run():
        llm = SequenceLLM()
        session = LLMDrivenConsultation(
            StubPredictor(), {"model": ConstantStopModel(), "threshold": 0.9}, llm
        )
        # This unit isolates the post-screen language/model contract. HTTP
        # acceptance tests exercise collection of these answers from scratch.
        from llm_service.app.schema import SymptomState

        session.profile.age = session.evidence.age = 30
        session.profile.pregnancy_status = "not_pregnant"
        session.evidence.pregnant = False
        session.profile.symptoms = {
            sid: SymptomState(symptom_id=sid, presence="absent")
            for sid in SafetyEngine().required_screening
        }
        # Isolate the model/LLM contract after the bounded intake questions.
        session.trace = [
            {"question_target": {"target": "condition_symptoms"}, "answers": None},
            {"question_target": {"target": "presenting_details"}, "answers": None},
        ]
        first = await session.receive("I have a fever")
        assert first["stop"] is False
        assert first["doctor_response"] == "How long have you had the fever?"
        assert first["question_target"] == {
            "kind": "symptom",
            "target": "fever",
            "focus": "duration",
        }
        assert first["answers"]["condition"]["choice"] == "influenza"

        second = await session.receive("For two days")
        assert second["stop"] is True
        assert second["diagnostic_backend"] == "ebm"
        assert second["answers"]["condition"]["choice"] == "influenza"
        assert second["profile"]["symptoms"]["fever"]["duration_days"] == 2
        assert "automated pre-screening" in second["doctor_response"].lower()
        # The contextual binder understands "For two days" without spending a
        # second LLM call; the model still owns the decision and explanation.
        assert llm.calls == ["extract", "ask", "explain"]

    asyncio.run(run())
