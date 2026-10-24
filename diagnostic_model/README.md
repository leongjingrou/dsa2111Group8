# Low-level EBM model

Part 3 contains:

- an EBM with main effects only;
- a logistic-regression baseline;
- accuracy, macro-F1, log loss and Brier score;
- feature-importance and calibration plots; and
- per-prediction feature contributions.

Use Python 3.13:

```bash
python -m pip install -e ".[test]"
python -m diagnostic_model.demo_data
python -m diagnostic_model.training diagnostic_model/demo_data.csv
```

The CSV needs a `condition_id` label and numeric feature columns. `vignette_id`
and `vignette_text` are ignored if present.

Generated files go to `diagnostic_model/artifacts/`. The demo data is synthetic
and cannot be used to claim clinical performance. Safety escalation remains a
separate component.

## Scenario dataset and matched EBM/Laya evaluation

The v0.3 scenario dataset is handled as three separate decisions:

1. whether the case is within the six-condition knowledge-base scope;
2. which supported condition best matches an in-scope case; and
3. whether urgent escalation is required.

This avoids treating the eleven unsupported diagnoses as though they were
already grounded in the knowledge base. The adapter expands each JSON symptom
record into `asked` and `present` features, so an unasked symptom is not confused
with an explicit negative answer. Diagnosis labels, scope labels, identifiers,
dataset metadata, height and weight are excluded from the default feature set.

```bash
python -m diagnostic_model.training data/generated/synthetic_scenarios.csv \
  --scenario-schema --generate-scenarios --no-tune
python -m diagnostic_model.laya_model data/generated/synthetic_scenarios.csv
python -m diagnostic_model.laya_model data/generated/synthetic_scenarios.csv \
  --trace-conversation
python -m diagnostic_model.laya_model data/generated/synthetic_scenarios.csv \
  --evidence-mode prioritized --trace-conversation
```

`--evidence-mode prioritized` is an alternate, label-free path that supplies
Laya with a structured evidence list alongside the original conversation. It
uses the same observed fields available to EBM and orders them from P1 to P5:
present red flags first; then severe/worsening/high-temperature or pregnancy
risk; other present symptoms; general context; and ordinary negative findings.
During turn-level evaluation it includes only evidence already revealed by that
patient answer, preventing future-turn leakage.

With `--trace-conversation`, the metrics report contains a `pathway_map` of
decision nodes and transitions. Each node records patient-answer number,
condition choice, escalation choice, case count and mean confidences; each edge
records how many cases moved between consecutive decisions. This supports
Sankey, alluvial or directed-flow visual analysis without exposing labels to
the model state.

For a smaller visualization-ready report with human-readable labels and
uncertainty estimates:

```bash
python -m diagnostic_model.decision_analysis \
  diagnostic_model/artifacts/laya_decisions.jsonl \
  --output diagnostic_model/artifacts/decision_analysis.json
```

This exports separate condition and escalation transition graphs plus accuracy
after every patient answer with Wilson 95% confidence intervals. The compact
report is intended for readable flow diagrams; the original JSONL remains the
complete audit log.

Export that report as human-readable PNGs (requires Pillow):

```bash
python -m diagnostic_model.export_decision_visuals \
  diagnostic_model/artifacts/decision_analysis.json \
  --output-dir diagnostic_model/visualizations --name conversation
```

For a matched 50/50 experiment, pass `--test-size 0.5` to EBM training,
Laya evaluation, and Laya fine-tuning. The split remains stratified by condition
and escalation, so every diagnosis/risk combination is represented on both
sides.

`--generate-scenarios` explicitly invokes the deterministic `generate()`
function in `data/generate.py` before validation and training. That generator is
the scenario-schema evolution of the repository's `data` branch generator: it
retains the fixed seed, symptom vocabulary and disease-profile logic, while
adding scope labels, knowledge-base condition IDs, dataset versioning and
vignette text needed by the matched EBM/Laya pipeline. The same flag is available
on the Laya evaluation and fine-tuning entry points when they are run directly.

The EBM command writes separate diagnosis, scope and escalation artifacts. Laya
uses the identical holdout rows, consumes the chained doctor-patient transcript,
and returns a supported condition or `out_of_scope` together with a `routine` or
`urgent` decision. `--trace-conversation` also records those probabilities after
every patient response, making it possible to analyse how confidence changes as
the consultation gathers evidence. Escalation is only a synthetic near-term risk
proxy; it is not a clinical prognosis or longitudinal outcome estimate.

The fixed-seed scenarios are rule-generated and balanced by construction. Their
holdout metrics measure interpolation within generator version 0.3.0 only; they
are not clinical performance or deployable probability calibration.

The matched server results and the accepted/rejected Laya design comparison are
recorded in [SCENARIO_RESULTS.md](SCENARIO_RESULTS.md).

## V2: fixed Q&A with learned stopping

The v2 pipeline keeps every consultation at a maximum of five doctor/patient
Q&A pairs. Both approaches are trained and evaluated at every partial state:

- the sequential EBM masks evidence that has not yet been asked and trains its
  diagnosis, scope and escalation models on answers 1 through 5;
- the Laya fine-tune export contains the same five partial conversational
  states rather than only the complete conversation;
- a separate stop/continue controller learns from live diagnosis/escalation
  confidence, probability margins, prediction stability and answer number;
- the stop threshold is chosen on validation data with an explicit penalty for
  urgent false negatives, then reported once on the untouched test split.

Run the local EBM pipeline and export its overview plus individual maps:

```bash
python -m diagnostic_model.sequential_ebm data/generated/synthetic_scenarios.csv \
  --output-dir diagnostic_model/artifacts_v2/ebm
python -m diagnostic_model.export_stopping_visuals \
  diagnostic_model/artifacts_v2/ebm/ebm_sequential_metrics.json \
  diagnostic_model/artifacts_v2/ebm/ebm_sequential_decisions.jsonl \
  --output-dir diagnostic_model/visualizations_v2 --approach EBM
```

`write_sequential_rows` in `laya_finetune.py` exports stratified train,
validation and test JSONL files for Laya. After fine-tuning, run:

```bash
python -m diagnostic_model.laya_sequential data/generated/synthetic_scenarios.csv \
  --model-dir /path/to/laya-v2 --output-dir diagnostic_model/artifacts_v2/laya
```

The images distinguish model confidence from the learned stop probability and
show the forced fifth-answer fallback. These outputs remain synthetic pipeline
checks, not evidence of clinical safety or effectiveness.
