# Multi-user Kubernetes simulation

This branch turns the fixed five-answer diagnostic experiment into a stateless
HTTP workload that can be replicated across K3s, K3d, MicroK8s, or a managed
Kubernetes service. It is a synthetic systems experiment, not a clinical API.

## What is simulated

Each virtual device is assigned a synthetic scenario and submits one request per
patient answer until the learned policy stops or answer five forces a stop. A
request includes the complete simulation key (`scenario_id`) and answer count,
so any replica can process it; no conversation session is held in pod memory.
This is what makes load balancing, rolling replacement, and horizontal scaling
safe for the experiment.

The load driver cycles through mobile, tablet, and desktop device labels and
runs the same number of consultations at several concurrency levels. Its output
records:

- end-to-end p50, p95, p99, and maximum request latency;
- requests per second and error rate;
- requests handled by each replica;
- sessions by device type; and
- the distribution of stop answers.

The service also emits one structured JSON log per request and exposes minimal
Prometheus counters at `/metrics`. Client and request IDs must be synthetic or
pseudonymous; the logs are not designed to hold patient-identifying data.

## EBM: runnable free-cluster path

`deploy/Dockerfile.ebm` trains and packages the evaluated sequential EBM and its
learned stop controller. The base manifests start two CPU replicas, add health
checks, resource bounds, topology spreading, a disruption budget, and a CPU HPA
that can scale from two to six replicas. K3s includes metrics-server by default;
on MicroK8s, enable DNS and metrics-server first.

With Docker, K3d, kubectl, and Python available:

```bash
scripts/run_k3d_load_test.sh
```

For MicroK8s, build/import `diagnosis-api:local`, then run:

```bash
microk8s kubectl apply -k deploy/k8s
microk8s kubectl -n diagnosis-sim rollout status deployment/diagnosis-api
```

The scripted K3d run creates a two-agent cluster, imports the local image,
deploys the workload, forwards the service, runs concurrent experiments at
1/4/16/32 devices, and saves results plus pod/HPA/log evidence.

The recorded server run under `diagnostic_model/k8s_results_v2/` used 100
consultations at 1/8/32/64 concurrent devices. The HPA expanded from two to six
replicas and all experiments completed without request errors. Throughput
flattened near 9.8 requests/second after concurrency 8 while p95 latency rose
from 1.81 seconds at concurrency 8 to 13.79 seconds at concurrency 64, marking
the tested deployment's saturation region.

## Laya: GPU overlay

The same API has a `LayaBackend`. Set `MODEL_BACKEND=laya`, mount the trained
checkpoint plus `laya_stopping_controller.joblib`, and use the overlay in
`deploy/k8s-laya`. It requests one NVIDIA GPU per pod and intentionally starts
with one replica. Multiple Laya replicas require one schedulable GPU apiece (or
an explicitly configured GPU-sharing runtime), a CUDA/Laya image matching the
checkpoint environment, and a ReadOnlyMany `laya-models` volume.

The project server has one T4, so it can validate concurrent queries to a single
Laya replica but cannot honestly demonstrate multi-GPU scaling. The CPU EBM
deployment is therefore the free, multi-replica reference simulation; the Laya
overlay documents and enforces the additional hardware contract rather than
pretending that a single GPU is multiple devices.

## Interpretation

Load results are specific to the selected node sizes, replica count, image,
request mix, and synthetic model. Record pod placement and HPA state alongside
every run. Do not generalize these measurements to real patient traffic or use
the endpoint for diagnosis.
