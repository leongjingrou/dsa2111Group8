#!/usr/bin/env bash
set -euo pipefail

K3D_BIN="${K3D_BIN:-k3d}"
KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CLUSTER_NAME="${CLUSTER_NAME:-diagnosis-sim}"
IMAGE_NAME="${IMAGE_NAME:-diagnosis-api:locust}"
RESULTS_DIR="${RESULTS_DIR:-diagnostic_model/k8s_results_v4/ebm}"
RUN_TIME="${RUN_TIME:-7m15s}"

if ! "${K3D_BIN}" cluster get "${CLUSTER_NAME}" >/dev/null 2>&1; then
  "${K3D_BIN}" cluster create "${CLUSTER_NAME}" --agents 2 --wait
fi

docker build -f deploy/Dockerfile.ebm -t "${IMAGE_NAME}" .
"${K3D_BIN}" image import "${IMAGE_NAME}" -c "${CLUSTER_NAME}"
"${KUBECTL_BIN}" apply -k deploy/k8s
"${KUBECTL_BIN}" -n diagnosis-sim rollout status deployment/diagnosis-api --timeout=300s

mkdir -p "${RESULTS_DIR}"
"${KUBECTL_BIN}" -n diagnosis-sim delete pod locust-driver --ignore-not-found
"${KUBECTL_BIN}" -n diagnosis-sim run locust-driver \
  --image "${IMAGE_NAME}" \
  --image-pull-policy IfNotPresent \
  --restart Never \
  --command -- \
  locust -f /app/serving/locustfile.py \
  --headless --host http://diagnosis-api \
  --run-time "${RUN_TIME}" --stop-timeout 30 \
  --csv /tmp/results/locust --csv-full-history \
  --html /tmp/results/locust-report.html

if ! "${KUBECTL_BIN}" -n diagnosis-sim wait \
  --for=jsonpath='{.status.phase}'=Succeeded pod/locust-driver --timeout=900s; then
  "${KUBECTL_BIN}" -n diagnosis-sim logs locust-driver >"${RESULTS_DIR}/locust-driver.log" || true
  exit 1
fi
"${KUBECTL_BIN}" -n diagnosis-sim logs locust-driver >"${RESULTS_DIR}/locust-driver.log"
"${KUBECTL_BIN}" -n diagnosis-sim cp locust-driver:/tmp/results "${RESULTS_DIR}/raw"
"${PYTHON_BIN}" -m serving.locust_report \
  "${RESULTS_DIR}/raw/locust_stats_history.csv" --output-dir "${RESULTS_DIR}"
"${KUBECTL_BIN}" -n diagnosis-sim get pods -o wide >"${RESULTS_DIR}/pods.txt"
"${KUBECTL_BIN}" -n diagnosis-sim get hpa >"${RESULTS_DIR}/hpa.txt"
"${KUBECTL_BIN}" get nodes -o wide >"${RESULTS_DIR}/nodes.txt"
"${KUBECTL_BIN}" -n diagnosis-sim logs -l app.kubernetes.io/name=diagnosis-api \
  --prefix --tail=5000 >"${RESULTS_DIR}/pod-logs.jsonl"
