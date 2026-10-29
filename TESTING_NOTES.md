# Browser and server test notes

Verified on 29 October 2026 using the Docker Compose stack at
`http://localhost:8080` with one frontend and three backend replicas.
This device ran the CPU EBM path; Docker did not expose an NVIDIA adapter, so
Laya/CUDA inference was documented but not claimed as locally verified.

## Delivered behavior

- The browser preserves the consultation ID and stops accepting input after a
  terminal response.
- Intake accepts open-ended sentences, fragments, typos and common Singlish such
  as `ytd`, `abit`, `bo`, `lah`, `leh`, `lor`, `jialat` and `nvr measure`.
- The offline adapter uses exact aliases first and a conservative local TF-IDF
  text-embedding/cosine-similarity fallback second. Low-confidence language is
  preserved as an unmapped finding for clinician review instead of being dropped
  or causing the opening question to loop.
- Mandatory emergency coverage is grouped briefly; adaptive clinical questions
  may continue beyond five total turns when information is still missing.
- Recognized answers are not asked again. Temperature, sex and symptom-detail
  questions remember both supplied and declined answers.
- Completion presents a preliminary model match and routes the patient to doctor
  review. It does not claim a confirmed diagnosis.

## Verification summary

- Full project test suite: passed.
- Safety regression suite: 118/118 passed.
- Live API safety evaluation: 4/4 flows passed.
- Browser flow: completed with typo-heavy and Singlish replies, producing a
  routine teleconsultation recommendation and preliminary common-cold match.
- Load run up to 64 users: 1/1,594 failures (0.06%), p50 330 ms, p95 2.3 s.
- Saturation run at 128 users: 113/2,260 failures (5.0%), p95 4.0 s.

## Run locally

```sh
docker compose up -d --build
docker compose ps
```

The image build installs dependencies and produces the EBM model files itself;
there is no separate host-Python, `.env`, or model-generation step for this
offline path. These commands are shell-neutral and work from the repository root
on Docker Desktop (Windows/macOS) and Docker Engine plus Compose (Linux). Open
`http://localhost:8080` after `docker compose ps` shows the frontend and all three
backend replicas as healthy. Check `http://localhost:8080/readyz` for the active
backend and safety-policy version. Stop with `docker compose down`.

The Docker path is preferred. Nginx serves the browser and proxies `/api` to the
backend service; the backend owns consultation state, safety screening, model
inference, knowledge retrieval and the final clinician-handoff recommendation.
The browser alone is not a standalone diagnostic application, so the backend
containers (or an equivalent remote deployment) must be running.

## Repeatable browser flow

Use this controlled in-scope fixture when demonstrating the complete browser-to-
server path. It guarantees the test route, not a diagnosis or clinical outcome:

1. `Age 31; not pregnant. Mild runny nose for 2 days; stable, no worsening.`
2. Answer the compact emergency screen with `none of those`.
3. Answer `No wheezing.`
4. Answer `No, drinking normally and urine is normal.`

On 29 October 2026 this completed through the live Docker stack with a routine
teleconsultation recommendation, a preliminary common-cold model match, and the
doctor-review disclaimer. A dense all-in-one symptom list is intentionally not
treated as a guaranteed route: if it falls outside the model's supported evidence
distribution, the server refers for clinician assessment instead of forcing an
in-scope label.

The checked browser path uses the deterministic local language adapter. A remote
OpenAI-compatible lightweight model can broaden free-text handling when configured,
but it remains outside diagnosis, stopping and safety decisions. The system creates
a clinician-ready recommendation; sending or booking it with a real provider still
requires a separate healthcare integration.
