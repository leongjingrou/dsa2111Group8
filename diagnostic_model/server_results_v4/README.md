# Server-side EBM/Laya results (v4)

Revision: `1551860420da282cfdcbeb7ac69bf5cd5bcc5719`

This directory contains compact artifacts copied from the server run. The full
datasets and logs remain under `/home/ubuntu/dsa2111-persistent/results` on the
server.

## Model comparison

| Model | Test diagnosis accuracy | Test escalation accuracy | Test joint accuracy | Urgent recall | Mean answers | Early stop rate |
|---|---:|---:|---:|---:|---:|---:|
| EBM | 0.9353 | 0.9559 | 0.8912 | 0.9412 | 3.212 | 0.8824 |
| Laya | 0.7824 | 0.8441 | 0.6618 | 0.8235 | 3.938 | 0.5412 |

`selected_outputs.json` contains one correct and one incorrect final decision
for each approach, including the actual source label, full final probability
distribution, escalation result, stopping point, and confidence trajectory.

The visualizations use green for a diagnosis matching the source label and red
for a mismatch. Stop decisions remain blue and urgent escalation remains orange
so correctness is not confused with policy state.

## Grounded data

The server generated 10,000 train, 1,000 validation, and 1,000 test
conversations. Medical content and differentials are grounded in the official
DDXPlus English v2 release. Two hundred public Vietnamese and Indonesian seed
records provide provenance-bearing Southeast Asian language/style metadata;
they are not treated as medical evidence. Split pairing is same-split and
deterministic.

DDXPlus: https://github.com/mila-iqia/ddxplus  
Dataset release: https://doi.org/10.6084/m9.figshare.22687585.v2

## GPU/Kubernetes load result

Four Locust worker pods ramped a single Tesla T4 Laya replica to 1,024 simulated
users. The GPU reached 100% utilization and 9,327 MiB. At the end of the short
ramp, 674 requests had completed: 116 succeeded and 558 failed, with 27.653 s
mean latency and a 30 s median/timeout ceiling. The failure trace shows file
descriptor exhaustion in the single API process. This is a capacity result,
not a claim that one T4 supports 1,024 concurrent medical conversations.

The knowledge-base loader is now process-cached to remove repeated JSON file
opens discovered by this test. Production scaling still needs request
backpressure, inference batching, raised descriptor limits, and multiple GPU
replicas before repeating a sustained 1,024-user test.

All medical metrics here evaluate synthetic records against their source labels;
they are not clinical validation or medical advice.
