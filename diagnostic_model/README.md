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
