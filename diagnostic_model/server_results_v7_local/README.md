# Local Docker, browser, model and load validation

Recorded on 2026-11-01 at commit `903a0a8e12cff4049757101ce86bcb3ec32cedd9` after a fast-forward pull from `origin/main`.

## Environment

- Docker Compose rebuilt the backend image (`782eb1e1230f`, 235 MB).
- Three backend replicas and the frontend reverse proxy were healthy.
- Patient UI: `http://localhost:8080/chat`.
- Clinician UI: `http://localhost:8081/doctor/<consultation-id>`.
- The frontend logged temporary 502 responses while all three backends were being recreated. It recovered when the health checks passed, and the final five-minute error scan was empty.

## Automated checks

- Full Pytest suite: passed in a clean container made from the rebuilt backend image.
- Safety unit suite: 118/118 passed.
- Live safety API evaluation: 4/4 flows passed.
- Docker Compose configuration validation: passed.
- All four final service containers reported healthy.

The older orchestrated replay script ran 100 consultations and 241 turn requests with p95 client latency of 55.961 ms, but it reported zero model decisions and five incomplete screens. Its fixed historical reply sequence no longer follows the current mandatory-coverage and question-target contract. That output is a harness-compatibility failure and must not be presented as a model-accuracy result.

## Browser end-to-end checks

Two consultations were driven through the rendered browser, not directly through the API.

1. Structured routine flow: a runny nose, mild cough, sneezing, two-day stable duration and explicit red-flag denial reached a completed routine teleconsultation recommendation. The preliminary EBM result was common cold at 94.9%. The separate clinician page showed all seven patient answers, extracted present/absent facts, safety rules, probabilities, stop threshold, audit event and final action.
2. Vague/error-prone flow: `little sick leh` triggered an open clarification. `mmmm lower back pain n cramps a bit since ytd, can tahan, not worse` did not loop. It completed as an out-of-scope clinician review, preserved the unresolved wording verbatim, withheld unsupported model probabilities and recommended in-person assessment.

The Singlish answer `Sneezing quite a bit, no sore throat lah` was interpreted as sneezing present and sore throat absent.

Observed conversational quality defect: the first structured answer already stated a two-day duration, but the intake later asked for the runny-nose duration again. History is retained in the clinician record, yet question selection is still somewhat redundant.

## Fresh EBM and logistic-regression comparison

The deterministic 180-row demo set was regenerated inside the rebuilt container. EBM and logistic regression used the same stratified 45-row holdout.

| Model | Accuracy | Macro F1 | Log loss | Multiclass Brier |
| --- | ---: | ---: | ---: | ---: |
| EBM, fresh | 0.9778 | 0.9774 | 0.0597 | 0.0296 |
| Logistic regression, fresh | 0.9778 | 0.9774 | 0.1051 | 0.0392 |
| Laya full tuning, recorded matched artifact | 0.8667 | 0.8661 | 0.4057 | 0.2057 |

The two fresh structured models made 44/45 correct predictions. EBM had the better probability scores despite identical accuracy. The Laya row comes from the repository's matched evaluation audit; it was not rerun on this device.

## Laya/GPU status

The local Docker installation advertises the NVIDIA runtime, but a real allocation failed with `WSL environment detected but no adapters were found`. The fine-tuned Laya checkpoint is also intentionally absent from Git. A fresh Laya run therefore requires the server-side checkpoint plus a Docker host whose NVIDIA adapter is visible to WSL.

The repository's recorded server artifacts remain useful for comparison: the matched Laya evaluation produced 39/45 correct predictions, and the Tesla T4 integrated smoke record reports a 5.812 s model load, about 2.34 GiB peak GPU memory, and warm turns around 61-69 ms after a 1.093 s first turn. These are historical server results, not measurements from this local run.

## Fresh bounded chat load

The current history-aware conversation API was exercised through the frontend proxy with eight concurrent workers and 20 complete consultations.

- 20/20 consultations completed.
- 160 requests, zero errors.
- 34.116 requests/second over 4.690 seconds.
- Mean 188.330 ms; p50 176.801 ms; p95 410.792 ms; p99 479.102 ms; maximum 684.437 ms.

This is a deliberately bounded developer test, not a production capacity claim. The older `serving/load_test.py` harness returned 100% errors because it targets the retired `/v1/simulations/consultations/turn` endpoint. `scripts/load_test_chat_api.py` is the current bounded replacement and follows each returned question target.

## Records

- `bounded-chat-load.json`: fresh current-stack load measurements.
- `fresh-model-comparison.json`: fresh EBM/logistic metrics and clearly labelled recorded Laya values.
- `../evaluation_audit.json`: matched historical audit for all three model families.
- `../server_results_v6_laya/`: historical GPU smoke, evaluation and load artifacts.

All scenarios and labels are synthetic. None of these results establish clinical safety, diagnostic effectiveness or a deployment SLA.
