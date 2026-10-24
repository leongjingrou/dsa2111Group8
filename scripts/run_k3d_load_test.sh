#!/usr/bin/env bash
set -euo pipefail

K3D_BIN="${K3D_BIN:-k3d}"
KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CLUSTER_NAME="${CLUSTER_NAME:-diagnosis-sim}"
IMAGE_NAME="${IMAGE_NAME:-diagnosis-api:local}"
RESULTS_DIR="${RESULTS_DIR:-diagnostic_model/k8s_results_v2}"

if ! "${K3D_BIN}" cluster get "${CLUSTER_NAME}" >/dev/null 2>&1; then
  "${K3D_BIN}" cluster create "${CLUSTER_NAME}" --agents 2 --wait
fi

docker build -f deploy/Dockerfile.ebm -t "${IMAGE_NAME}" .
"${K3D_BIN}" image import "${IMAGE_NAME}" -c "${CLUSTER_NAME}"
"${KUBECTL_BIN}" apply -k deploy/k8s
"${KUBECTL_BIN}" -n diagnosis-sim rollout status deployment/diagnosis-api --timeout=300s

"${KUBECTL_BIN}" -n diagnosis-sim port-forward service/diagnosis-api 18080:80 >"${RESULTS_DIR}.port-forward.log" 2>&1 &
forward_pid=$!
trap 'kill "${forward_pid}" 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
  if curl --fail --silent http://127.0.0.1:18080/readyz >/dev/null; then
    break
  fi
  sleep 1
done

"${PYTHON_BIN}" -m serving.load_test \
  --base-url http://127.0.0.1:18080 \
  --concurrency "${CONCURRENCY:-1,4,16,32}" \
  --sessions "${SESSIONS:-60}" \
  --output-dir "${RESULTS_DIR}"

"${KUBECTL_BIN}" -n diagnosis-sim get pods -o wide >"${RESULTS_DIR}/pods.txt"
"${KUBECTL_BIN}" -n diagnosis-sim get hpa >"${RESULTS_DIR}/hpa.txt"
"${KUBECTL_BIN}" -n diagnosis-sim logs -l app.kubernetes.io/name=diagnosis-api --prefix --tail=2000 >"${RESULTS_DIR}/pod-logs.jsonl"
