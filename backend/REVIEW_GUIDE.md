# Human review guide

Start with the workflow below, then review policy and failure behavior. Names
refer to patient facts, safety decisions, model records and consultation state.
HTTP response keys and existing public class names are retained for compatibility.

## Suggested reading order

| Order | File | What to review |
| --- | --- | --- |
| 1 | [`../safety_layer/config/rules.v1.json`](../safety_layer/config/rules.v1.json) | The 22 clinical rules, scope assumptions, required answers and fixed advice |
| 2 | [`../safety_layer/models.py`](../safety_layer/models.py) | `RuleHit`, `SafetyDecision` and disposition priority |
| 3 | [`../safety_layer/facts.py`](../safety_layer/facts.py) | `PatientFacts`: current answers, positive history, missing/invalid/conflicting facts |
| 4 | [`../safety_layer/policy.py`](../safety_layer/policy.py) | Reject malformed configuration at load time |
| 5 | [`../safety_layer/engine.py`](../safety_layer/engine.py) | `evaluate`: collect rule hits, choose urgency, build the result and audit it |
| 6 | [`../safety_layer/integration.py`](../safety_layer/integration.py) | `guard_decision`: filter unsafe model output before explanation |
| 7 | [`../diagnostic_model/llm_live_conversation.py`](../diagnostic_model/llm_live_conversation.py) | `receive`: patient answer → emergency check → extraction → screening → prediction → guard → response |
| 8 | [`../llm_service/intake.py`](../llm_service/intake.py) | Bind one short reply to one server-owned question |
| 9 | [`../llm_service/consultation_service.py`](../llm_service/consultation_service.py) | One shared turn handler, retry replay and snapshot saving |
| 10 | [`../llm_service/session_store.py`](../llm_service/session_store.py) | SQLite transactions, expiration, active capacity and stale-write rejection |
| 11 | [`../llm_service/orchestrated_main.py`](../llm_service/orchestrated_main.py), [`app/api/diagnose.py`](app/api/diagnose.py) | Startup, routes and HTTP error translation |
| 12 | [`../llm_service/api_schema.py`](../llm_service/api_schema.py) | The shared typed response consumed by the future frontend |

## Terms used in the code

- **Patient facts:** structured observations supplied by extraction and evidence.
  Missing, uncertain, absent and present are distinct states.
- **Safety decision:** a rule-based disposition, matching rules, missing fields
  and fixed patient advice. This does not select a diagnosis.
- **Model record:** a proposed diagnostic estimate and action. Only a guarded
  record may reach the explanation step.
- **Checkpoint:** one turn's extraction, safety/model results and response. It is
  included in the persisted consultation trace.
- **Snapshot:** the consultation state needed to resume on another backend worker.
- **Request history:** accepted request IDs and their original responses, allowing
  retries without processing an answer twice.

The workflow uses small methods with one responsibility and explicit branches.
Comments explain why a safety restriction or persistence step exists. Policy
values stay in JSON; model, language and HTTP concerns stay outside the engine.

## Tests to read alongside the implementation

[`tests/test_consultation_e2e.py`](tests/test_consultation_e2e.py) walks the complete
API workflow using controlled model responses, including failure cases.
[`tests/test_real_http_e2e.py`](tests/test_real_http_e2e.py) fits real model artifacts,
starts the actual backend over TCP and exercises intake through a final decision.
The safety package's tests cover individual predicates, all 64 fictional
vignettes, configuration rejection, data quality and audit behavior.

Run the full suite with `python -m pytest` in the environment described in the
[root README](../README.md). Formatting follows Ruff, Python 3.13, 88 columns.
The [validation report](evaluation/REPORT.md) records the actual tools and results.
No frontend implementation is required for these tests.
