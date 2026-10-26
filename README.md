# DSA2111 Group 8 — Telemedicine prototype

The complete team repository is integrated with task 4's versioned safety layer.
The backend runs intake, safety screening, EBM/Laya decision interfaces,
knowledge-base retrieval, and guarded explanation. The frontend is still in
progress; backend testing does not depend on it.

## Install and test

Use a clean Python 3.13 environment:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -e ".[test,serve,speech]"
.venv/bin/python -m pytest
```

The suite includes real local HTTP tests, which train small synthetic EBM and
stopping models and launch the actual backend in a subprocess. Internet/API keys
are unnecessary after dependencies are installed. Those tests use the repo's
explicit `KeywordLLM` language adapter; the diagnostic models are actually fitted.
Controlled predictor doubles separately verify exact routine/referral behavior.

On the current Mac, the verified interpreter is `/opt/homebrew/bin/python3.13`
and the clean environment is `.venv-e2e`. The pre-existing Anaconda interpreter
crashed on pytest's debugger import; it is not used for the final test results.

## Run the backend

```sh
python -m diagnostic_model.sequential_ebm data/generated/synthetic_scenarios.csv --output-dir runtime/models
```

Then, in the same activated environment:

```sh
MODEL_BACKEND=ebm LLM_BACKEND=keyword \
EBM_LIVE_MODEL=runtime/models/ebm_sequential_models.joblib \
STOPPING_MODEL=runtime/models/ebm_stopping_controller.joblib \
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Open the API documentation at `http://127.0.0.1:8000/docs`. `keyword` is an offline
testing adapter, not a clinical language model. For a configured external
provider, use `LLM_BACKEND=remote` and supply `LLM_API_KEY` or `GROQ_API_KEY`,
`LLM_MODEL`, and the appropriate `LLM_BASE_URL`.

```sh
python scripts/evaluate_safety_api.py --base-url http://127.0.0.1:8000 --output runtime/api-evaluation.json
```

## Components

| Component | Documentation |
| --- | --- |
| Patient-facing API and end-to-end contract | [backend/README.md](backend/README.md) |
| Human review order and code map | [backend/REVIEW_GUIDE.md](backend/REVIEW_GUIDE.md) |
| Safety rules, scope and audit log | [safety_layer/README.md](safety_layer/README.md) |
| Safety integration points | [safety_layer/INTEGRATION.md](safety_layer/INTEGRATION.md) |
| Diagnostic model, calibration and comparisons | [diagnostic_model/README.md](diagnostic_model/README.md) |
| Medical knowledge base | [knowledge_base/README.md](knowledge_base/README.md) |
| Language service | [llm_service/README.md](llm_service/README.md) |
| Integration validation evidence | [backend/evaluation/REPORT.md](backend/evaluation/REPORT.md) |

The consultation API always enables the versioned safety layer. Historical raw
model/simulation benchmarks remain separate research tools. Synthetic test
results are not clinical validation; clinical review of rules, scope assumptions,
language recognition and vignette expectations is pending.
