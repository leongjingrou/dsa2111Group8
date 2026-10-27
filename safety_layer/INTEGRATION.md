# Task 4 integration

The safety layer is integrated into the complete team repository on
`safety-integration`, based on the team `main` at `9ee31c2`.
Both `/api/conversations/{session_id}/turn` and `/api/diagnose` use the same
consultation service, persisted state, safety rules and typed response.
See [the review guide](../backend/REVIEW_GUIDE.md) for a file-by-file reading order
and [the API contract](../backend/README.md) for requests, responses and errors.

## Implemented conversation flow

```text
HTTP patient answer
    -> load server-owned session and check retry ID
    -> bind a short reply to the server's pending question
    -> update deterministic evidence
    -> immediately act on recognized emergency evidence
    -> extract and merge patient facts, retaining uncertainty and positive history
    -> pre-model safety evaluation and audit
         -> emergency/referral: fixed message and terminal guarded record
         -> missing/conflicting facts: deterministic follow-up question
         -> complete screen: diagnostic predictor and knowledge-base retrieval
              -> post-model safety evaluation and audit
                   -> unsafe/out-of-scope proposal: fixed referral without estimates
                   -> permitted proposal: stopping controller and LLM explanation
    -> save the versioned snapshot and retry response
    -> return the common ConsultationResponse
```

`LLMDrivenConsultation.receive` owns this order. The LLM extracts facts and phrases
permitted questions/explanations. It cannot choose a clinical rule, disable the
screen, change its urgency, or explain an unguarded model proposal.

## Responsibilities and interfaces

| Component | Responsibility |
| --- | --- |
| `safety_layer/config/rules.v1.json` | Versioned predicates, scope assumptions, source registry and fixed messages |
| `safety_layer/facts.py` | Normalize mappings, `Profile` and `EvidenceState`; retain positive history and data-quality problems |
| `safety_layer/policy.py` | Validate the full policy before the service becomes ready |
| `safety_layer/engine.py` | Evaluate all applicable rules, select the highest urgency and append an audit event |
| `safety_layer/integration.py` | Check the selected condition and every alternative estimate; replace restrictive records |
| `llm_service/intake.py` | Select missing/conflicting safety questions and bind explicit short answers |
| `diagnostic_model/llm_live_conversation.py` | Run the pre-model and post-model checks around prediction |
| `llm_service/consultation_service.py` | Share the workflow across both API routes and handle idempotent retries |
| `llm_service/session_store.py` | Persist profile, evidence, pending question, trace and retry history with version checks |

Create one `SafetyEngine(audit_log=...)` per worker. Startup validates the policy
and audit destination before loading the shared consultation service. Production
entry points enable `rules_v1`; `sg_moh` is a compatibility alias. `none` and
client-supplied rule overrides are rejected. Historical raw-model research tools
remain separate from these patient API routes.

## Required facts and invariants

- The integrated `Profile` has explicit age and pregnancy fields. The configured
  age scope is 18 to 64; pregnancy requires referral. These are project
  assumptions, not MOH eligibility rules.
- `engine.required_screening` is the compact mandatory emergency screen. Other
  red flags remain active whenever they are reported and are explored by the
  adaptive consultation when relevant.
  All seven supplemental safety symptoms are included in the vocabulary and
  questionnaire. No negative answer is supplied by default.
- Missing or uncertain findings block model assessment. Conflicting ordinary
  symptoms can be explicitly clarified; earlier positive red flags remain active.
- The learned five-answer stopping policy counts model-guided evidence turns,
  not demographic or mandatory safety-intake turns. Model features retain their
  trained five-answer range, so a consultation may legitimately contain more
  than five patient replies. At 64
  unresolved intake answers the patient is referred without a model diagnosis.
- A post-model check validates the actual selected condition and all displayed
  alternatives against the six-condition KB. It preserves urgent/specialist
  referrals. Restrictive records have no speculative diagnostic estimates.
- Every evaluation has an event ID, policy version/hash and complete rule hits.
  A write failure blocks routine output; its exception still carries the safety
  decision so emergency advice can be displayed in the 503 response.

For other synchronous callers, `guard_model_call` wraps both checks around a
zero-argument predictor callback returning `{"condition_id": ..., "record": ...}`.
For asynchronous callers, use `evaluate` before inference and `guard_decision`
before explanation. Always render the returned guarded record or fixed message.

## Boundaries and remaining review

The source screenshot supplies assignment requirements; it does not authorize
other team tasks. This implementation covers item 4 and its backend integration.
Frontend source has not been changed. The frontend team can consume the shared
OpenAPI response, retain the session/request IDs, render the returned question,
and display fixed safety advice on terminal results or audit-error responses.

Software tests cover rule execution and full API flows, including a real HTTP
server with fitted EBM/stopping models and the explicit `KeywordLLM` test adapter.
See [integration evidence](../backend/evaluation/REPORT.md). External paid LLMs,
GPU Laya inference and deployed containers are outside that test result.

Clinical review of predicates, scope, language recognition and fictional vignette
expectations remains pending. In particular, bundled or broad findings do not
establish severity/onset automatically. Persistent vomiting is not equivalent to
continuous vomiting; generic weakness/numbness is not an established acute focal
deficit. The knowledge-base entries' empty red-flag arrays are not negative
patient screening answers. Software acceptance is not clinical validation.
