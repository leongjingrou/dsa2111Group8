# Browser and server test notes

Verified on 27 October 2026 using the Docker Compose stack at
`http://localhost:8080` with one frontend and three backend replicas.

## Delivered behavior

- The browser preserves the consultation ID and stops accepting input after a
  terminal response.
- Intake accepts open-ended sentences, fragments, typos and common Singlish such
  as `ytd`, `abit`, `bo`, `lah`, `leh`, `lor`, `jialat` and `nvr measure`.
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
```

Open `http://localhost:8080`. Use `docker compose ps` to confirm the frontend and
all three backend replicas are running.

The checked browser path uses the deterministic local language adapter. A remote
OpenAI-compatible lightweight model can broaden free-text handling when configured,
but it remains outside diagnosis, stopping and safety decisions. The system creates
a clinician-ready recommendation; sending or booking it with a real provider still
requires a separate healthcare integration.
