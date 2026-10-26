# Health project — team roles

**Your assigned item: 4. Safety layer lead.**

Repository: [leongjingrou/dsa2111Group8](https://github.com/leongjingrou/dsa2111Group8).

The role breakdown below was transcribed from the [provided screenshot](team-role-breakdown.png) on 26 October 2026. The screenshot's column headings and the end of the last row are cropped. The last column is recorded as “Related roles” because its full heading is not visible.

## Your responsibility: Safety layer lead

**Focus:** Red-flag rules and escalation.

Deliverables listed in the screenshot:

- Versioned rule config.
- Scope limits: age, pregnancy, out-of-scope conditions, and contradictions.
- Rule tests against vignettes.
- A log of which rule fired.

**Related role:** Role 2.

## Full visible role breakdown

| Item | Lead | Focus | Deliverables visible in the screenshot | Related roles |
| --- | --- | --- | --- | --- |
| 1 | Data and clinical scenarios lead | Training data and the evaluation set | Condition list (5 to 8 in-scope plus dangerous mimics), synthetic or public training data, 100 to 200 hand-written vignettes with ground-truth labels, a clinician or med student to review them | Role 6 |
| 2 | Knowledge base lead | Medical sources and versioning | Curated sources (ideally MOH/Singapore guidelines), git-versioned knowledge base, index or RAG lookup, one demonstrated update-and-retest cycle | Role 4 |
| 3 | Diagnostic model lead | GAM/EBM model | Trained and tuned EBM, feature contribution plots, calibration, comparison with a simple baseline (e.g., logistic regression) | Role 1 |
| **4 — yours** | **Safety layer lead** | **Red-flag rules and escalation** | **Versioned rule config, scope limits (age, pregnancy, out-of-scope conditions, contradictions), rule tests against vignettes, a log of which rule fired** | **Role 2** |
| 5 | LLM and conversation lead | Patient-facing layer | Symptom extraction into the structured profile, follow-up question phrasing, decision record template, plain-language and clinician summaries, guardrails so the LLM only explains the record | Role 6 |
| 6 | Integration, evaluation and UI lead | Pipeline, metrics and demo | Repo and task board setup, end-to-end baseline pipeline, question-selection controller (or shared with Role 3), test harness, metrics and ablations, … [remainder cropped] | Roles 1 and 5 |

This records the assignment and source material. The task-4 implementation is documented in [the safety-layer README](../safety_layer/README.md).
