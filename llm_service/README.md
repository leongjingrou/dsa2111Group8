# LLM-mediated EBM/Laya consultation

This service incorporates Jingrou's `feature/LLM` layer without giving the LLM
diagnostic authority.

1. The LLM extracts a validated symptom profile from each patient message.
2. A deterministic controller selects the information target for the next turn.
3. The LLM phrases that target as a natural patient-facing question.
4. EBM or Laya produces the diagnosis and escalation distributions.
5. The trained stopping controller decides whether to ask again or finalise.
6. At finalisation, the LLM may explain the fixed result but cannot change its
   diagnosis, escalation, or urgency headline.

The conversation contract is documented in
[`prompts/CONVERSATION_STYLE.md`](prompts/CONVERSATION_STYLE.md). Runtime prompt
versions remain in `prompts/prompts.yaml`. The pending server-selected question
is supplied to extraction, so fragments such as `37`, `about two weeks`, or
`not that I've noticed` are interpreted in context without requiring exact
commands. Common replies are handled locally; a configured OpenAI-compatible
lightweight model handles broader colloquial free text while remaining outside
the diagnosis, escalation and stopping decisions.

Run the combined HTTP service with the relevant model paths and provider
settings:

```bash
MODEL_BACKEND=ebm \
EBM_LIVE_MODEL=/models/ebm_sequential_models.joblib \
STOPPING_MODEL=/models/ebm_stopping_controller.joblib \
LLM_API_KEY=... LLM_MODEL=... \
uvicorn llm_service.orchestrated_main:app --host 0.0.0.0 --port 8080
```

For Laya, set `MODEL_BACKEND=laya`, `LAYA_MODEL_DIR`, and `LAYA_DEVICE=cuda`.
Start a session with `POST /api/conversations`, then send each patient reply to
`POST /api/conversations/{session_id}/turn`.

The integrated backend persists versioned consultation snapshots in SQLite on a
shared Docker volume, so the supplied multi-replica Compose stack can continue a
conversation across backend replicas.
`LLM_BACKEND=keyword` selects the deterministic language adapter used by server
integration and Kubernetes load tests. It removes external provider latency,
cost, and rate limits; it is not a clinical model or evidence of real-LLM
quality. Production defaults to `LLM_BACKEND=remote`. The `deploy/k8s-llm`
overlay enables client-IP affinity while sessions remain process-local.
The standalone language-service entry point retains its configurable safety
profile. The patient-facing integrated backend always applies the mandatory
versioned safety rules; language-model output never alters model confidence or
the safety disposition.
