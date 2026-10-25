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

The service keeps session state in one process. A multi-replica deployment must
use sticky routing or replace the in-memory session store with shared storage.
The Singapore safety profile remains disabled by default. Setting
`safety_profile` to `sg_moh` enables the separately logged jurisdictional
emergency-phrase override; it does not alter model confidence.
