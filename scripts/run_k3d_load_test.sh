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

mkdir -p "${RESULTS_DIR}"
"${KUBECTL_BIN}" -n diagnosis-sim delete pod load-driver --ignore-not-found
"${KUBECTL_BIN}" -n diagnosis-sim run load-driver \
  --image "${IMAGE_NAME}" \
  --image-pull-policy IfNotPresent \
  --restart Never \
  --command -- \
  python -m serving.load_test \
  --base-url http://diagnosis-api \
  --concurrency "${CONCURRENCY:-1,4,16,32}" \
  --sessions "${SESSIONS:-60}" \
  --output-dir /tmp/results

if ! "${KUBECTL_BIN}" -n diagnosis-sim wait \
  --for=jsonpath='{.status.phase}'=Succeeded pod/load-driver --timeout=900s; then
  "${KUBECTL_BIN}" -n diagnosis-sim logs load-driver || true
  exit 1
fi
"${KUBECTL_BIN}" -n diagnosis-sim logs load-driver >"${RESULTS_DIR}/load-driver.log"
"${KUBECTL_BIN}" cp \
  diagnosis-sim/load-driver:/tmp/results/. "${RESULTS_DIR}"

"${KUBECTL_BIN}" -n diagnosis-sim get pods -o wide >"${RESULTS_DIR}/pods.txt"
"${KUBECTL_BIN}" -n diagnosis-sim get hpa >"${RESULTS_DIR}/hpa.txt"
"${KUBECTL_BIN}" -n diagnosis-sim logs -l app.kubernetes.io/name=diagnosis-api --prefix --tail=2000 >"${RESULTS_DIR}/pod-logs.jsonl"
