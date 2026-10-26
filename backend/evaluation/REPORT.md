# Backend integration validation

Validated locally on 26 October 2026 with Python 3.13.7, on
`codex/safety-integration` based on team commit `9ee31c2`.
**214 tests passed, with zero failures, errors or skips.** The frontend source
was not modified. These results were recorded locally before publication.

## Recorded evidence

| Check | Result |
| --- | --- |
| Full repository pytest suite | 214 passed in 18.522 seconds |
| Safety package | 118 tests, including 64 fictional vignettes and all 22 clinical rules |
| Backend acceptance and real HTTP tests | 35 passed |
| Existing diagnostic / KB / language / serving tests | 29 / 18 / 9 / 5 passed |
| Fictional flows against the actual HTTP server | 4/4 passed |
| Ruff formatting | 40 affected Python files conform |
| Ruff checks | No import-order, unused-import, undefined-name or selected syntax/style errors |
| Git whitespace check | Passed |
| Compose / Kubernetes configuration | Compose configuration parsed; LLM Kustomize overlay rendered |

[`test-summary.json`](test-summary.json) records the counts, time and policy
fingerprint. [`api-flows.json`](api-flows.json) records the complete fictional
requests and responses for the four HTTP flows. The full local JUnit output is
`runtime/full-integration.xml` and is ignored by Git.

| Actual HTTP flow | Patient turns | Final disposition | Outcome |
| --- | --- | --- | --- |
| Chest-pain emergency | 1 | `emergency` | Immediate fixed advice, no diagnosis |
| Age outside configured scope | 1 | `clinician_review` | Referral before prediction |
| Pregnancy outside configured scope | 1 | `clinician_review` | Referral before prediction |
| Complete explicit safety screen | 31 | `urgent_review` | Real model ran; its urgent referral was preserved |

The fully screened case is not forced to produce a routine diagnosis. Controlled
predictor tests separately demonstrate the complete routine path through intake,
all required screening questions, prediction, KB retrieval and explanation.
They cover both explicit `not pregnant` and `not applicable` answers.

## What ran

The real HTTP fixture fits sequential EBM and stopping-controller artifacts using
the project's training pipeline on a small synthetic subset containing every
condition/escalation stratum. It starts `backend.app.main:app` in a Uvicorn
subprocess, waits for readiness, submits requests through TCP, and shuts it down.
The diagnostic and stopping artifacts are fitted, not mocked. Language handling
uses the repository's explicit `KeywordLLM` adapter.

Other backend tests use controlled diagnostic/stopping doubles to establish exact
routing expectations. They cover emergencies interrupting intake, negation,
scope limits, uncertainty, contradictory facts, out-of-scope model estimates,
provider and audit failures, server-owned screening state, retries, finished and
deleted sessions, active capacity, replica changes and stale-write rejection.

The readability refactor separates result models, fact normalization, policy
validation, evaluation, intake, persistence and HTTP handling. The clinical JSON
policy remains version 1.0.0 with SHA-256:

```text
4c6a8334fc3f420f1b1640608fee5d2d2ce7978bd9a6a58ff7ef0afc01cd24fa
```

## Reproduce

From the repository root, in a clean environment with installed test, serving
and speech dependencies:

```sh
python -m pytest --junitxml=runtime/full-integration.xml
python -m unittest discover -s safety_layer/tests
```

The real HTTP test requires permission to bind a loopback socket. No external
LLM/API key is required. On this Mac the verified environment is `.venv-e2e`,
created with `/opt/homebrew/bin/python3.13`.

See [the review guide](../REVIEW_GUIDE.md) for the code-reading order and
[the backend contract](../README.md) for runtime settings and HTTP behavior.

## Limits of this result

This verifies software behavior on fictional data. Clinical review of the rules,
scope assumptions and language recognition is pending. Paid external LLM/Groq
requests, GPU Laya inference, frontend behavior and a live container/Kubernetes
deployment were not tested. The small training subset does not establish
clinical diagnostic accuracy.

The test run emitted one dependency deprecation warning: the installed Starlette
TestClient deprecates its `httpx` adapter. No test failed or was skipped because
of it. The actual HTTP tests also use a separate network client and live server.
