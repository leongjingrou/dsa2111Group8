"""Shared application workflow for both consultation HTTP entry points."""

from diagnostic_model.llm_live_conversation import LLMDrivenConsultation


class ConsultationFinished(RuntimeError):
    """A new answer was submitted after a final decision."""


class RequestReplayConflict(RuntimeError):
    """An idempotency key was reused with a different patient answer."""


def consultation_response(session_id, checkpoint):
    """Expose the guarded decision, with null model outputs when safety blocked it."""
    model_answers = checkpoint["answers"] or {}
    disposition = checkpoint["safety"]["disposition"]
    if disposition in {"emergency", "urgent_review"}:
        status = "urgent"
    elif checkpoint["stop"]:
        status = "complete"
    else:
        status = "asking"
    return {
        "session_id": session_id,
        "conversation_id": session_id,
        "status": status,
        "reply": checkpoint["doctor_response"],
        "doctor_response": checkpoint["doctor_response"],
        "backend": checkpoint["diagnostic_backend"],
        "patient_answer": checkpoint["patient_answer"],
        "extraction": checkpoint["extraction"],
        "diagnosis": model_answers.get("condition"),
        "escalation": model_answers.get("escalation"),
        "decision_record": checkpoint["decision_record"],
        "knowledge_base": checkpoint["knowledge_base"],
        "stop_probability": checkpoint["stop_probability"],
        "stop_threshold": checkpoint["stop_threshold"],
        "stop": checkpoint["stop"],
        "question_target": checkpoint["question_target"],
        "question_fallback_used": checkpoint["question_fallback_used"],
        "safety_profile": checkpoint["safety_profile"],
        "safety": checkpoint["safety"],
        "safety_override_applied": checkpoint["safety_override_applied"],
        "triggered_rules": [
            hit["rule_id"] for hit in checkpoint["safety"]["triggered_rules"]
        ],
    }


class ConsultationService:
    """Reload server-owned state, process one answer, then save atomically."""

    def __init__(self, predictor, stopping, llm, safety_engine, store):
        self.predictor = predictor
        self.stopping = stopping
        self.llm = llm
        self.safety_engine = safety_engine
        self.store = store

    def new_session(self):
        return LLMDrivenConsultation(
            self.predictor, self.stopping, self.llm, safety_engine=self.safety_engine
        )

    def start(self):
        return self.store.create(self.new_session().snapshot())

    async def turn(self, session_id, message, request_id=None):
        version, saved_state = self.store.load(session_id)
        request_history = saved_state.get("requests", {})
        if request_id is not None and request_id in request_history:
            previous_request = request_history[request_id]
            if previous_request["message"] != message:
                raise RequestReplayConflict(
                    "A request ID cannot be reused for a different answer"
                )
            return previous_request["response"]

        consultation = self.new_session()
        consultation.restore(saved_state)
        if consultation.finished:
            raise ConsultationFinished("Consultation already finalised")
        checkpoint = await consultation.receive(message)
        response = consultation_response(session_id, checkpoint)

        updated_state = consultation.snapshot()
        if request_id is not None:
            request_history[request_id] = {"message": message, "response": response}
        updated_state["requests"] = request_history
        # Compare-and-save prevents concurrent answers from overwriting each
        # other. Saved responses also make retries safe after a lost HTTP reply.
        self.store.save(session_id, version, updated_state)
        return response
