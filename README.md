# DSA2111 Group 8 — Telemedicine prototype

The complete team repository includes the browser frontend, persisted multi-turn
intake, versioned safety screening, EBM/Laya decision interfaces, knowledge-base
retrieval, and guarded explanation. The Docker stack runs the frontend with three
backend replicas sharing consultation state.

The latest browser, Docker, safety and load-test handoff is recorded in
[TESTING_NOTES.md](TESTING_NOTES.md).

## Recommended: run with Docker

Docker is the preferred local installation and runtime. It keeps the browser,
Python dependencies, model artifacts, API, and three backend replicas in one
repeatable stack. Install Docker Desktop on Windows/macOS or Docker Engine with
the Compose plugin on Linux, clone the repository, and run from its root:

```sh
docker compose up -d --build
docker compose ps
```

That single build command installs the container dependencies and trains/packages
the local EBM artifacts automatically. No host Python environment, copied `.env`
file, or manual model-training command is required for the offline demonstration.
The same commands work in PowerShell, Command Prompt, macOS Terminal and common
Linux shells. Wait until `docker compose ps` reports the frontend and all backend
replicas as healthy, then open `http://localhost:8080`. Readiness is also exposed
at `http://localhost:8080/readyz`.

The browser sends `/api` requests to the frontend
container, whose Nginx proxy forwards them to the replicated backend service.
Only the server loads the diagnostic model, safety rules, consultation store and
optional language model; the browser never downloads or runs those models.

The local clinician view is available at `http://localhost:8081`. After the
first patient answer, the chat's **Doctor view** button opens the matching
consultation there. The page refreshes every three seconds and shows the actual
question sequence, patient answers, safety rule hits, guarded recommendation,
model probabilities (when permitted), and stopping score. It reads the saved
server trace and does not expose hidden language-model reasoning. The clinician
view contains patient text and is a local demonstration feature; add access
control before exposing either port to other users or networks.

The intake accepts brief replies against the current question. A short reply
that explicitly names a symptom from one of the last three questions is
applied to that earlier question, and the review shows which question was used.
Bare answers are never reassigned to a past question. Duration replies need
a time unit (for example, `2d`, `3 weeks`, or `yesterday`); a bare number is not
silently treated as days. The intake asks for the unit and accepts a brief
follow-up such as `days`.

`localhost` is reachable only from this computer. Testing from another device
requires publishing the service on a reachable host and configuring its firewall,
TLS and access controls. Stop the local stack with `docker compose down`; the
named consultation-state volume is preserved unless it is explicitly removed.

For a clean rebuild after source changes, rerun `docker compose up -d --build`.
Useful checks are `docker compose logs -f` and `docker compose ps`; press Ctrl+C
to leave the log view without stopping the containers.

The default Compose profile is the validated CPU EBM path. GPU inference is a
server deployment choice, not a frontend setting. For Laya on an NVIDIA host,
use the CUDA image and runtime described in [deploy/README.md](deploy/README.md),
mount the trained checkpoint, and set `MODEL_BACKEND=laya`,
`LAYA_MODEL_DIR=/models/laya`, `LAYA_DEVICE=cuda`, and the matching
`STOPPING_MODEL`. Confirm `nvidia-smi` works inside a container before relying on
GPU inference. Do not switch the ordinary Compose service to Laya without those
artifacts and the NVIDIA container runtime.

## Optional native development and tests

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
| Latest browser and server test notes | [TESTING_NOTES.md](TESTING_NOTES.md) |

The consultation API always enables the versioned safety layer. Historical raw
model/simulation benchmarks remain separate research tools. Synthetic test
results are not clinical validation; clinical review of rules, scope assumptions,
language recognition and vignette expectations is pending.
