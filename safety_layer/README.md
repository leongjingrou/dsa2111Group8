# Task 4 — Safety layer

This component screens structured patient facts before a diagnostic model runs,
then checks its proposed decision before it reaches the patient. It is independent
of the EBM, Laya, and LLM, and has no third-party dependencies.

| Assignment deliverable | Implementation |
| --- | --- |
| Versioned rule config | [`config/rules.v1.json`](config/rules.v1.json): semantic version, source registry, scope, 22 clinical rules and fixed messages |
| Scope limits | Age, pregnancy, six supported conditions, unmapped findings, incomplete screening, and conflicting answers |
| Rule tests against vignettes | [`tests/vignettes.json`](tests/vignettes.json): 64 fictional acceptance cases; additional engine, integration, configuration and logging tests |
| Log of which rule fired | JSONL audit events with rule IDs, reasons, structured field names, source IDs, policy version and SHA-256 |

## Run it

Use Python 3.10 or newer for this standalone package; the integrated backend
requires Python 3.13. Current verification uses Python 3.13.7. No API
key, model checkpoint, GPU, server or package installation is needed.

Run from the repository root:

```sh
python3 -m unittest discover -s safety_layer/tests
python3 -m safety_layer demo
python3 -m safety_layer evaluate safety_layer/examples/emergency.json --audit-log safety_layer/output/audit.jsonl
```

The demo writes `evaluation.md`, `vignette_results.json`, and `audit.jsonl` under
`safety_layer/output/`. It exits nonzero if a case's disposition or exact set of
triggered rules differs from its acceptance expectation. Repeated runs append to
the audit log; they replace the two summary files. Runtime output is git-ignored.

The [saved v1.0.0 report](evaluation/v1.0.0/evaluation.md) and
[audit events](evaluation/v1.0.0/audit.jsonl) are fictional demonstration artifacts
for the assignment. That initial record contains 115 tests. The current standalone
suite passes 118 tests, including the same 64 vignette cases and regressions for
malformed model actions and consistent pregnancy-status normalization. Every one of the 22 configured clinical rules is exercised. These are software
acceptance results, not estimates of clinical sensitivity or specificity.

## Scope and policy assumptions

The six conditions match the team's knowledge base v1.2.0:

- `common_cold`
- `influenza`
- `gastroenteritis`
- `migraine`
- `allergic_rhinitis`
- `urinary_tract_infection`

The configured age range is **18 <= age < 65**. A pregnant patient is referred to
a clinician. Negative pregnancy status or explicit `not_applicable` must be
confirmed; the engine never infers it from sex. These age limits and exclusions
are conservative **project assumptions**, configurable in the policy. They are
not MOH telemedicine eligibility criteria. Younger and older patients still
receive emergency advice if an emergency rule fires.

The screen does not diagnose these six conditions. Before model inference,
`candidate_condition` is optional. After model inference, it is required and must
be in scope. A human-readable label such as `Common Cold` is normalized to the
same identifier. Unknown or out-of-scope candidate labels, alternative estimates,
or symptom findings are referred for clinician review.

## Rule behavior

All matching rules are retained. The highest-priority disposition wins:

| Priority | Disposition | Behavior |
| --- | --- | --- |
| 4 | `emergency` | Stop the routine path and render the fixed emergency message immediately |
| 3 | `urgent_review` | Refer for urgent assessment; remove speculative diagnostic estimates |
| 2 | `clinician_review` | Refer outside the routine prototype pathway |
| 1 | `needs_clarification` | Ask for missing/conflicting information; no diagnostic record is returned |
| 0 | `continue_assessment` | Permit assessment to continue; this is not a clinical all-clear |

Examples include emergency rules for chest pain, breathlessness, significant
bleeding, severe abdominal pain, acute neurological findings and airway threats.
These mappings use [MOH's emergency guidance](https://www.moh.gov.sg/seeking-healthcare/getting-medical-help/),
[SCDF's emergency guidance](https://www.scdf.gov.sg/home/about-scdf/emergency-medical-services)
and the rule-specific sources in the configuration.

Combination rules cover headache with neurological findings or neck stiffness,
red eyes with blurred vision, and fever with mucosal bleeding. Source-backed
concerns are separated from implementation choices: for example, a bundled
confusion/drowsiness finding is treated conservatively as an emergency, whereas
the less-specific `persistent_vomiting` field triggers urgent review. It does not
distinguish continuous vomiting, which MOH lists as an emergency. A source does
not imply clinical approval of the exact predicate or routing threshold.

The seven supplemental safety IDs below extend the team's existing symptom
vocabulary. They are now included in the LLM vocabulary and backend questionnaire;
the server collects explicit answers for them:

```text
loss_of_consciousness       uncontrolled_bleeding
facial_droop               speech_difficulty
sudden_one_sided_weakness   swollen_tongue_or_throat
seizure
```

There are 28 required screening findings in total. Every one must be explicitly
absent before the routine path can proceed. A positive rule is acted on
immediately even if the remaining screen or demographics are incomplete. Missing
and uncertain answers stay unknown; no checklist is prefilled with negative
answers. The negative baseline in the fictional vignette file describes explicit
test-patient answers and must never be used to initialize a real consultation.

When a Profile and an EvidenceState disagree, positive symptoms are retained.
An uncertain finding cannot be erased by another parser's negative answer.
Contradictions need clarification, and prior positive red flags remain active
for the consultation. The upstream merge history is not assumed to indicate a
clinician-resolved contradiction.

## Python interface

`evaluate` accepts a mapping, the team's Pydantic `Profile`, or an `EvidenceState`
dataclass. The integrated Profile includes age and pregnancy status; optional
`evidence` supplies additional parser facts. See [integration instructions](INTEGRATION.md)
for the implemented conversation flow and HTTP contract.

```python
from safety_layer import SafetyEngine, guard_decision, guard_model_call

engine = SafetyEngine(audit_log="safety_layer/output/audit.jsonl")

emergency = engine.evaluate({"symptoms": {"chest_pain": True}})
assert emergency.disposition == "emergency"
print(emergency.patient_message)


# For a fully screened profile, gate a synchronous model callback:
def propose_decision():
    return {
        "condition_id": "common_cold",  # actual model choice
        "record": {
            "recommended_action": "routine_teleconsultation",
            "preliminary_estimates": [{"condition": "Common Cold", "score": 0.7}],
        },
    }


# result = guard_model_call(profile, propose_decision, evidence=evidence, engine=engine)
# Or check an already-created DecisionRecord:
# result = guard_decision(profile, record, candidate_condition=model_choice,
#                         evidence=evidence, engine=engine)
```

Symptoms may be booleans, `present` / `absent` / `uncertain`, or objects with a
`presence` field. Age is a finite number of years. Pregnancy status accepts a
boolean or `pregnant`, `not_pregnant`, `not_applicable`, `unknown`, `uncertain`.
The alias `pregnant` supports EvidenceState. Numeric strings and the strings
`"true"` / `"false"` are rejected rather than silently coerced. Parse CSV columns
into these types at the data boundary.

For restrictive results, use `result.patient_message` directly. The replacement
record contains the minimum permitted action, triggered rule IDs, resolvable
source URLs and no diagnostic estimates. For clarification, `result.record` is
`None`. Only a result with `safety.can_continue` may enter the ordinary diagnosis
and explanation flow. Never send the original unsafe record to the explainer.
An urgent or specialist referral proposed by a model is preserved.

## Audit and failure handling

Pass `audit_log` when constructing the engine to enable persistence. Every
evaluation, including a clear screen, produces an event with a unique ID, UTC
timestamp, evaluation phase, policy version/hash, disposition and all rule hits.
Pre-model and post-model evaluations produce separate events. No patient name,
message, symptom evidence quote, unmapped free text or diagnosis score is logged.

The append writer supports concurrent threads in one process. Configure one
writer process per file; distributed workers should send events to their own
files or a centralized audit store. The log is not encrypted, signed or tamper
proof. Storage permissions, retention and access control belong to deployment.

An invalid or unavailable policy raises `PolicyError`; callers must not fall back
to ungated routine output. A write failure raises `AuditLogError`, which includes
the evaluated `decision`. Render emergency advice from that decision immediately,
then surface the storage failure and block routine continuation. The CLI returns
exit code 2 and retains that advice in its JSON error response.

## Clinical review and updates

Clinical review is **pending**. The source registry was checked on 26 October
2026 and includes primary Singapore sources plus NHS guidance for selected
findings. The software is an assignment prototype, with no validated free-text
recognition, exhaustive red-flag coverage, treatment recommendations or evidence
of real-patient safety. Clinician review of the exact mappings, age scope,
pregnancy handling, questionnaire and vignette expectations is required before
patient use. It is separate from the completed software acceptance work.

To update the policy:

1. Check the primary source and record its URL and access date.
2. Change the rule or scope and increment its semantic version; keep a new
   versioned config file when releasing a changed policy.
3. Add or revise fictional cases covering the boundary and adverse behavior.
4. Run the tests and demo with the intended policy file.
5. Save the evaluation and audit events, and record the review in `CHANGELOG.md`.

Tests demonstrate that a separately versioned age-scope update changes behavior
and changes the policy hash. Invalid rules, unknown operators, missing source
IDs, duplicate rules and uncovered required flags prevent the engine from loading.
