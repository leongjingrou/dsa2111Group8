# Integrated consultation backend

`backend.app.main:app` includes the existing conversation endpoints, the
frontend-compatible `/api/diagnose` endpoint, and optional speech transcription.
Both consultation routes use the same persisted state and mandatory safety
pipeline. The frontend source is unchanged and is not required to run these tests.

For a guided code review, start with [REVIEW_GUIDE.md](REVIEW_GUIDE.md).
Recorded test results are in [evaluation/REPORT.md](evaluation/REPORT.md).

## Start a consultation

```http
POST /api/conversations
Content-Type: application/json

{}
```

The response contains `session_id`, the greeting, backend name and safety policy
version. `rules_v1` is mandatory. The legacy name `sg_moh` is accepted as an alias
for the full rules; `none` and client-supplied safety overrides are rejected.

## Submit each answer

```http
POST /api/conversations/{session_id}/turn
Content-Type: application/json

{"patient_message":"I have a mild runny nose for 2 days and it is stable","request_id":"a-unique-turn-id"}
```

`request_id` is optional. Reusing an accepted ID with the same message returns its
original response, even after later turns. Reusing it for different text returns
409. Concurrent writes use a stored version check; a stale write returns 409.

The response includes:

| Field | Meaning |
| --- | --- |
| `status` | `asking`, `complete`, or `urgent` |
| `reply` / `doctor_response` | Text to display |
| `stop` | Whether this consultation is terminal |
| `question_target` | Server-selected subject of the next question, or null |
| `safety` | Disposition, scope, missing information, rule hits, version/hash and audit event ID |
| `triggered_rules` | Rule IDs for convenient rendering/inspection |
| `diagnosis` / `escalation` | Model output only after safety permits it; null on safety-only paths |
| `decision_record` | Guarded recommendation; null while clarification is needed |
| `knowledge_base` | Retrieved reference entries and KB version, when a model assessment is made |

Repeat until `stop` is true. Ask the actual returned question. A short yes/no/
not-sure answer is bound to that server-selected question, not to a question
submitted by the client. Age and pregnancy answers are collected explicitly.
There are no blanket negative defaults. The seven supplemental safety symptoms
are now in the extraction vocabulary and deterministic questionnaire.

The five-answer model stopping policy counts model-guided evidence turns rather
than demographic or mandatory safety-intake turns, so a consultation may ask
more than five questions. It cannot terminate an incomplete safety screen. The
mandatory screen is a compact grouped emergency check; other red flags remain
active when reported and may be explored adaptively. After 64 unresolved intake
answers, the backend refers to a clinician without producing a diagnosis.
Ordinary symptom contradictions can be explicitly clarified; positive safety
findings remain active and cause referral.

Grouped negative emergency answers remain available to the safety engine and
audit trail but are not injected as a batch of diagnostic-model features. This
keeps the compact checklist from distorting the sequential model's first
checkpoint. A model-only urgent label is provisional until the stopping policy
has enough model evidence; explicit rule-based danger signs remain immediate.

## Frontend-compatible route

```http
POST /api/diagnose
Content-Type: application/json

{"patient_message":"no","conversation_id":"the-returned-session-id","request_id":"another-unique-turn-id"}
```

The initial request may omit `conversation_id`. Every subsequent request must
return it. The old `messages` array is accepted instead of `patient_message`,
but only its last patient message is processed. Bot messages are never trusted as
screening state. Initial multi-turn histories without an ID are rejected.

Both routes publish the same typed `ConsultationResponse` in `/openapi.json`.
The frontend team should keep the returned consultation ID, preserve request IDs
when retrying, stop routine input on terminal results and render the fixed safety
message. It should handle safety advice in a 503 error as described below.

## Errors and lifecycle

- 404: unknown, deleted or expired session.
- 409: finished session, concurrent stale write, or conflicting request-ID reuse.
- 422: invalid/empty/oversized input or unsupported client safety fields.
- 503: provider, audit storage or active-session capacity failure. An audit
  failure includes `detail.safety` and `detail.reply`; show any emergency advice
  immediately even though the HTTP request failed. Never replace it with a generic
  retry message or a routine assessment.

`DELETE /api/conversations/{session_id}` discards the session. `/healthz` and
`/readyz` expose process readiness and the loaded policy fingerprint. A broken
policy or unusable audit destination prevents startup.

## Persistence and replicas

`CONSULTATION_DB` defaults to `runtime/consultations.sqlite3`. It stores profile,
demographics, pending question, transcript, decision trace and retry responses.
The default TTL is one hour (`CONSULTATION_TTL_SECONDS`); expired sessions are
unreadable and are purged when a new session is created. Completed sessions do not
consume `MAX_ACTIVE_SESSIONS` capacity and remain retryable until expiry/deletion.

`SAFETY_AUDIT_LOG` overrides the default per-host/process JSONL path. Audit events
exclude patient free text; the session database **does contain patient text**.
Neither is encrypted by this prototype. Deployment must control access and
retention. The audit writer supports threads in a single process, so use separate
audit paths for worker processes or an external audit sink.

Compose shares a named local volume between its three backend replicas.
The Kubernetes LLM overlay now supplies a ReadWriteOnce PVC and writable runtime
mount, preserving the read-only container root. SQLite WAL requires a compatible
local filesystem; the PVC keeps replicas on the attached node. A multi-node
production service needs an appropriate shared database instead of network-file
SQLite. No Kubernetes or container deployment was performed for this change.

To use Compose, first train the model files into `runtime/models` as described in
the root README and copy `backend/.env_template` to `backend/.env`, filling in
provider settings (or deliberately choosing `LLM_BACKEND=keyword` for a synthetic
demo). The backend image now builds from the repository root so it contains the
safety, model, KB and language packages.

## End-to-end tests

```sh
python -m pytest backend/tests
python -m pytest
```

Tests cover the full routine path, emergency interruption, age/pregnancy scope,
unknown answers, contradictions, model out-of-scope output, provider failure,
audit failure, retries, lifecycle, capacity and switching replicas. Real TCP HTTP
tests train EBM/stopping artifacts and launch the actual backend with
`KeywordLLM`. Controlled predictor doubles establish exact routing expectations.
Paid external LLM/Groq requests and GPU Laya inference are not exercised.

The evaluation script saves complete fictional API exchanges against any running
backend. Historical five-turn synthetic replay scripts now retain missing
diagnoses and incomplete screens explicitly; they do not count safety referrals
as fabricated diagnostic predictions.
