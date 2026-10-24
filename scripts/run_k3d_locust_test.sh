#!/usr/bin/env bash
set -euo pipefail

K3D_BIN="${K3D_BIN:-k3d}"
KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CLUSTER_NAME="${CLUSTER_NAME:-diagnosis-sim}"
IMAGE_NAME="${IMAGE_NAME:-diagnosis-api:locust}"
RESULTS_DIR="${RESULTS_DIR:-diagnostic_model/k8s_results_v4/ebm}"
RUN_TIME="${RUN_TIME:-7m15s}"
LOCUST_WORKERS="${LOCUST_WORKERS:-4}"
MONITOR_PID=""

cleanup() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if ! "${K3D_BIN}" cluster get "${CLUSTER_NAME}" >/dev/null 2>&1; then
  "${K3D_BIN}" cluster create "${CLUSTER_NAME}" --agents 2 --wait
fi

docker build -f deploy/Dockerfile.ebm -t "${IMAGE_NAME}" .
"${K3D_BIN}" image import "${IMAGE_NAME}" -c "${CLUSTER_NAME}"
"${KUBECTL_BIN}" apply -k deploy/k8s
"${KUBECTL_BIN}" -n diagnosis-sim rollout status deployment/diagnosis-api --timeout=300s

mkdir -p "${RESULTS_DIR}"
"${KUBECTL_BIN}" -n diagnosis-sim delete pod locust-driver --ignore-not-found
"${KUBECTL_BIN}" -n diagnosis-sim delete pod locust-master --ignore-not-found
"${KUBECTL_BIN}" apply -k deploy/k8s-load
"${KUBECTL_BIN}" -n diagnosis-sim scale deployment/locust-worker --replicas="${LOCUST_WORKERS}"
"${KUBECTL_BIN}" -n diagnosis-sim set image deployment/locust-worker worker="${IMAGE_NAME}"
"${KUBECTL_BIN}" -n diagnosis-sim rollout status deployment/locust-worker --timeout=300s
"${KUBECTL_BIN}" -n diagnosis-sim run locust-master \
  --image "${IMAGE_NAME}" \
  --image-pull-policy IfNotPresent \
  --restart Never \
  --labels app.kubernetes.io/name=locust-master \
  --command -- \
  locust -f /app/serving/locustfile.py \
  --master --expect-workers "${LOCUST_WORKERS}" \
  --headless --host http://diagnosis-api \
  --run-time "${RUN_TIME}" --stop-timeout 30 \
  --csv /tmp/results/locust --csv-full-history \
  --html /tmp/results/locust-report.html

(
  while "${KUBECTL_BIN}" -n diagnosis-sim get pod locust-master >/dev/null 2>&1; do
    phase="$("${KUBECTL_BIN}" -n diagnosis-sim get pod locust-master -o jsonpath='{.status.phase}')"
    if [[ "${phase}" == "Succeeded" ]] || [[ "${phase}" == "Failed" ]]; then
      break
    fi
    date --iso-8601=seconds
    "${KUBECTL_BIN}" -n diagnosis-sim get hpa diagnosis-api
    "${KUBECTL_BIN}" -n diagnosis-sim get deployment diagnosis-api locust-worker
    "${KUBECTL_BIN}" -n diagnosis-sim top pods 2>/dev/null || true
    sleep 5
  done
) >"${RESULTS_DIR}/cluster-timeseries.log" &
MONITOR_PID=$!

if ! "${KUBECTL_BIN}" -n diagnosis-sim wait \
  --for=jsonpath='{.status.phase}'=Succeeded pod/locust-master --timeout=900s; then
  "${KUBECTL_BIN}" -n diagnosis-sim logs locust-master >"${RESULTS_DIR}/locust-master.log" || true
  exit 1
fi
wait "${MONITOR_PID}" || true
MONITOR_PID=""
"${KUBECTL_BIN}" -n diagnosis-sim logs locust-master >"${RESULTS_DIR}/locust-master.log"
"${KUBECTL_BIN}" -n diagnosis-sim logs deployment/locust-worker --prefix \
  >"${RESULTS_DIR}/locust-workers.log" || true
"${KUBECTL_BIN}" -n diagnosis-sim cp locust-master:/tmp/results "${RESULTS_DIR}/raw"
"${PYTHON_BIN}" -m serving.locust_report \
  "${RESULTS_DIR}/raw/locust_stats_history.csv" --output-dir "${RESULTS_DIR}" \
  --workers "${LOCUST_WORKERS}" --api-replica-ceiling 12
"${KUBECTL_BIN}" -n diagnosis-sim get pods -o wide >"${RESULTS_DIR}/pods.txt"
"${KUBECTL_BIN}" -n diagnosis-sim get deployment locust-worker -o wide \
  >"${RESULTS_DIR}/locust-worker-deployment.txt"
"${KUBECTL_BIN}" -n diagnosis-sim get hpa >"${RESULTS_DIR}/hpa.txt"
"${KUBECTL_BIN}" get nodes -o wide >"${RESULTS_DIR}/nodes.txt"
"${KUBECTL_BIN}" -n diagnosis-sim logs -l app.kubernetes.io/name=diagnosis-api \
  --prefix --tail=5000 >"${RESULTS_DIR}/pod-logs.jsonl"
