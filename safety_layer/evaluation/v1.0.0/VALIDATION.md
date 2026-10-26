# Task-4 validation record

Validated on 26 October 2026 using Python 3.13.5.

| Check | Result |
| --- | --- |
| `python3 -m unittest discover -s safety_layer/tests` | 115 tests passed |
| `python3 -m safety_layer demo --output-dir safety_layer/evaluation/v1.0.0` | 64/64 fictional vignette expectations passed |
| Configured clinical rule execution | 22/22 rules exercised by the vignette suite |
| `python3 -m safety_layer evaluate safety_layer/examples/emergency.json --audit-log safety_layer/output/cli-smoke.jsonl` | Emergency advice returned immediately despite missing demographics and remaining screening answers |
| Actual team Pydantic `Profile` | Accepted with EvidenceState-shaped demographic facts |
| Actual team routine `DecisionRecord` | Preserved after a complete, negative screen |
| Actual team emergency replacement `DecisionRecord` | Validated against the team's schema with `urgent_medical_review` and no diagnostic estimates |

The compatibility check loaded the actual `llm_service/app/schema.py` from team
commit `d21c92e`. Pydantic was used for that check only; the safety package and its
unit tests require only Python's standard library. Neither diagnostic model
weights nor a live LLM/API server were used.

See [`evaluation.md`](evaluation.md) for case outcomes,
[`vignette_results.json`](vignette_results.json) for expected and actual rule IDs,
and [`audit.jsonl`](audit.jsonl) for the corresponding 64 audit events. Event IDs
join the result entries to the audit records. The logged policy SHA-256 identifies
the exact configuration evaluated.

This verifies software behavior, not clinical safety or diagnostic accuracy.
Clinical review remains pending. This is the initial standalone validation
record. The service has since been integrated; see the current
[integration guide](../../INTEGRATION.md) and
[backend validation report](../../../backend/evaluation/REPORT.md).
