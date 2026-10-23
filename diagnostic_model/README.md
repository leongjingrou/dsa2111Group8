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

The v0.2 scenario dataset is handled as three separate decisions:

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
  --scenario-schema --no-tune
python -m diagnostic_model.laya_model data/generated/synthetic_scenarios.csv
```

The EBM command writes separate diagnosis, scope and escalation artifacts. Laya
uses the identical holdout rows and returns a supported condition or
`out_of_scope` together with a `routine` or `urgent` decision.

The fixed-seed scenarios are rule-generated and balanced by construction. Their
holdout metrics measure interpolation within generator version 0.2.0 only; they
are not clinical performance or deployable probability calibration.
