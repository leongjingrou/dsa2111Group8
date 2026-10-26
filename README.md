# DSA2111 Group 8 — EBM contribution

This branch adds the Part 3 diagnostic model prototype:

- main-effects Explainable Boosting Machine with light tuning;
- logistic-regression baseline using the same split;
- accuracy, macro-F1, log loss and Brier score;
- feature-importance, calibration and local-contribution outputs;
- Python 3.13 project setup and automated tests.

Run with:

```bash
python -m pip install -e ".[test]"
python -m pytest
python -m diagnostic_model.demo_data
python -m diagnostic_model.training diagnostic_model/demo_data.csv
```

The demo data is synthetic. Results are not clinical validation, and safety escalation remains a separate component.
