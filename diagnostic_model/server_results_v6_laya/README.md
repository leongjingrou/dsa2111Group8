# Laya GPU server results (v6)

Measured on the project server with one NVIDIA Tesla T4. The runtime used
PyTorch 2.14.1+cu130 and the `diagnosis-laya-api:orchestrated` image.

## Functional and labelled replay

- CUDA model load: 5,812 ms.
- Peak allocated GPU memory in the live smoke test: 2,339.8 MiB.
- Warm live turns after first inference: 61-69 ms.
- 100-consultation labelled test replay: 77% diagnosis accuracy and 89%
  escalation accuracy across 429 turn requests.
- Replay latency: 73.7 ms mean, 72.5 ms p50, 91.9 ms p95, 95.8 ms p99.

`integrated-smoke.json` is the raw live conversation. It stopped after three
answers and selected urgent escalation, but finalized `out_of_scope` rather
than the expected influenza-like condition. `laya-eval-100.json` retains raw
traces for both correct and wrong final states, including probabilities and
stop decisions.

## Staged concurrency test

Locust ramped one Laya GPU replica through 8, 32, 128, 256, 512, and 1,024
virtual patients. At 8 users, median p95 was 240 ms with no failures. At 32 it
was 1.6 s; at 128, 5.7 s; at 256, 8.0 s; and at 512, 15 s. At 1,024 users the
median p95 was 22 s and timeout failures appeared. The final aggregate CSV
contains 2,947 completed requests, 212 failures (7.19%), 8.1 s p50 and 23 s
p95. This is a saturation result for one physical T4, not a capacity claim.

Raw CSV, HTML, JSON and the generated time-series plot are under `load/`.

## Reproduce the Docker setup

On Ubuntu with a working NVIDIA driver:

```bash
./deploy/install_nvidia_container_toolkit_ubuntu.sh
docker build -f deploy/Dockerfile.laya -t diagnosis-laya-api:local .
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -p 8080:8080 \
  -e MODEL_BACKEND=laya -e LAYA_DEVICE=cuda \
  -e LAYA_MODEL_DIR=/models/laya \
  -e STOPPING_MODEL=/models/results/laya_stopping_controller.joblib \
  -v /absolute/path/to/laya-checkpoint:/models/laya:ro \
  -v /absolute/path/to/results:/models/results:ro \
  diagnosis-laya-api:local
```

The matching user ID lets the read-only container access private model weights
without making them world-readable. Verify with `curl http://localhost:8080/readyz`.
