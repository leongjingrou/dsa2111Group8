# Server-side Kubernetes concurrency result

This run used branch revision `c7e9f1b`, Docker 29.8.1, K3d 5.9.0, and K3s
1.35.5 on the project server. A two-agent K3d cluster started with two EBM API
replicas. The CPU HPA had a 60% target and a range of two to six replicas.

The load driver ran inside Kubernetes and addressed the ClusterIP service. Each
of four experiments simulated 100 independent consultations, with device labels
balanced across mobile, tablet, and desktop. A consultation issued successive
answer-prefix requests until the learned policy stopped or the fifth-answer
fallback fired.

| Concurrent devices | Requests | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Errors |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 318 | 5.512 | 171.271 | 271.094 | 318.242 | 0 |
| 8 | 310 | 9.920 | 606.560 | 1,809.572 | 2,098.440 | 0 |
| 32 | 300 | 9.812 | 2,343.852 | 6,592.898 | 6,959.567 | 0 |
| 64 | 330 | 9.664 | 4,766.659 | 13,786.840 | 15,186.873 | 0 |

The HPA expanded the deployment from two to its six-replica ceiling. The saved
request counts show traffic reaching all six replicas. Throughput flattened at
about 9.7-9.9 requests per second from concurrency 8 onward while tail latency
rose sharply, indicating saturation rather than additional useful capacity.
At concurrency 64, p95 latency was 13.79 seconds despite a zero error rate.

`load-test.json` is the machine-readable result, `load-test.png` is the compact
comparison, `pods.txt` and `hpa.txt` record final cluster state, and
`pod-logs.jsonl` contains structured per-request server logs from both the
accepted run and an earlier, discarded single-pod routing diagnostic. It is
retained for operational comparison but must not be used to recompute the
accepted workload totals. `launcher.log` also retains the first failed post-run
copy attempt; the measurements themselves completed successfully and are
preserved verbatim in `load-test.json`.

These are synthetic patient and load data from one server run, not a clinical
validation or production capacity service-level agreement.
