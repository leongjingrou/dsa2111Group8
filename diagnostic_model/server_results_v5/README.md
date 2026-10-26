# LLM-mediated EBM server and Kubernetes results (v5)

Run date: 2026-10-25 (UTC)  
Application revision: `c80584f`  
Load overlay revision: `6284cfd`  
Report parser revision: `56f22e0`

## What was measured

The service used the language layer to extract patient facts and phrase the
next question, EBM to produce diagnosis/escalation probabilities, and the
trained stopping controller to decide when to finalise. The Singapore override
was disabled. There was no external LLM credential on the server, so these
tests explicitly selected the deterministic `keyword` language adapter. This
isolates local orchestration and model processing time; it does **not** measure
the quality or network latency of an OpenAI-compatible provider.

The image was rebuilt on the server and the complete container test suite
passed (61 tests). An in-cluster smoke request took 9.492 ms to create a
session and 35.387 ms for the first full extraction/prediction/stopping/question
turn.

## Labelled replay

One hundred held-out synthetic consultations were replayed through the actual
Kubernetes HTTP service. Labels are synthetic source labels, not diagnoses
confirmed by clinicians.

| Metric | Result |
|---|---:|
| Consultations | 100 |
| Conversation turns | 321 |
| Diagnosis accuracy | 94% |
| Escalation accuracy | 96% |
| Mean turn time | 25.932 ms |
| Turn p50 | 25.434 ms |
| Turn p95 | 29.890 ms |
| Turn p99 | 35.910 ms |

`evaluation/orchestrated-evaluation.json` preserves full raw response traces,
including extracted facts, every probability distribution, stop probability,
LLM-worded doctor response, truth label, and correctness flag. Examples include:

| State | Sample | Actual → final | Final confidence | Answers | Stop probability |
|---|---|---|---:|---:|---:|
| Correct | SYN-01361 | out_of_scope → out_of_scope | 1.00 | 3 | 0.99 |
| Correct | SYN-01362 | out_of_scope → out_of_scope | 0.99 | 4 | 0.94 |
| Wrong | SYN-01363 | influenza → out_of_scope | 0.85 | 2 | 0.90 |
| Wrong | SYN-01372 | urinary_tract_infection → gastroenteritis | 0.32 | 5 | 0.85 |

The confidently wrong SYN-01363 result is especially important: stopping
confidence is not the same as diagnostic correctness, so calibration and an
abstention/review policy remain necessary.

## Distributed Kubernetes/Locust ramp

Environment: three k3d/K3s nodes on one EC2 host, four Locust workers, two
initial API replicas, HPA range 2–12, and a mixed simulated device population.
The ramp was 8 → 32 → 128 → 256 → 512 → 1,024 users over 165 seconds. Because
conversation state is process-local, the Service used `ClientIP` affinity.

| Users | Mean request rate | Peak request rate | Mean failure rate | Median p95 | Max p99 |
|---:|---:|---:|---:|---:|---:|
| 8 | 12.5/s | 16.6/s | 0.0/s | 44 ms | 170 ms |
| 32 | 45.0/s | 63.7/s | 0.0/s | 140 ms | 230 ms |
| 128 | 58.1/s | 81.7/s | 0.0/s | 3,900 ms | 6,600 ms |
| 256 | 128.7/s | 166.3/s | 40.2/s | 3,800 ms | 7,400 ms |
| 512 | 84.0/s | 112.0/s | 8.1/s | 9,100 ms | 13,000 ms |
| 1,024 | 104.4/s | 138.8/s | 18.8/s | 18,000 ms | 22,000 ms |

Aggregate: 13,888 requests, 11,637 successful, 2,251 failed (16.21%),
84.26 requests/s, 4,000.7 ms mean, 2,200 ms p50, 14,000 ms p95, and
21,000 ms p99. Failures comprised 2,129 transport/timeout failures and 122
HTTP 404 responses.

This run does **not** demonstrate support for 1,024 concurrent users. The
service was comfortable at 32 users, showed material queuing at 128, and
crossed its failure boundary at 256. HPA reached eight replicas, but four
worker source IPs remained pinned by session affinity. Under pressure the
one-second health probes timed out, Kubernetes restarted API processes
(including exit code 137), and their in-memory sessions disappeared. New HPA
pods could not adopt those sessions, which explains the 404s and why adding
replicas did not linearly improve the run.

Before another capacity claim, move sessions to Redis/shared storage, remove
client-IP affinity, use a separate lightweight health listener or longer probe
timeouts, add bounded queues/backpressure, and repeat with physically separate
load-generator hosts and a longer steady-state window.

## GPU status

The host exposes a Tesla T4 (15,360 MiB) through `nvidia-smi`. The current
Docker daemon, however, advertises only the `runc` runtime; the integrated Laya
smoke attempt failed before model startup with `no known GPU vendor found`.
`laya-gpu-smoke-stderr.log` records the failure. Therefore this v5 run makes no
new Laya/GPU timing claim. The earlier v4 artifact remains the last successful
T4 result (100% peak utilisation, 9,327 MiB), but it tested the prior serving
path and must not be presented as a v5 integrated result.

## Artifacts

- `evaluation/orchestrated-evaluation.json`: raw correct/wrong consultation traces.
- `load/locust-summary.json`: stage-level time series and summary.
- `load/locust_stats.csv`: final aggregate and endpoint statistics.
- `load/locust_stats_history.csv`: raw per-second Locust history.
- `load/locust_failures.csv`: failure classes and counts.
- `load/locust-timeseries.png`: users, throughput, and p95 latency.
- `load/events-tail.txt`: Kubernetes probe failures and restarts.
- `nodes.txt`, `kubernetes-resources.txt`, `gpu.txt`: environment evidence.

These results are engineering measurements on synthetic data. They are not
clinical validation, medical advice, or a production service-level guarantee.
