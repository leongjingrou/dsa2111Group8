# DSA2111 Group 8 — Telemedicine Project

Prototype for retrieving medical references and comparing an explainable
diagnostic model with a simple baseline.

## Components

```text
knowledge_base/    condition data, sources, keyword and TF-IDF retrieval
diagnostic_model/  EBM training, baseline comparison and explanations
```

The EBM uses main effects only. It produces:

- accuracy, macro-F1, log loss and Brier score;
- a logistic-regression comparison;
- feature-importance and calibration plots; and
- local feature contributions for each prediction.

## Setup

Python 3.13 is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

On Windows, activate with `.venv\Scripts\activate`.

## Run

```bash
python -m pytest
python -m diagnostic_model.demo_data
python -m diagnostic_model.training diagnostic_model/demo_data.csv
```

Model outputs are saved in `diagnostic_model/artifacts/`.

## Data

Training data is a CSV with a `condition_id` label and numeric feature columns.
`vignette_id` and `vignette_text` are ignored if present. The demo generator
uses the knowledge base vocabulary to create synthetic data for testing only.

## Safety

This prototype is not validated for diagnosis or clinical triage. Synthetic
metrics are not evidence of clinical performance. Red-flag escalation must be
handled by the independent safety layer.
