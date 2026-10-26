#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on the existing CUDA server.
PYTHON_BIN="${PYTHON_BIN:-python}"
PERSISTENT_ROOT="${PERSISTENT_ROOT:-/home/ubuntu/dsa2111-persistent}"
RESULTS_DIR="${RESULTS_DIR:-${PERSISTENT_ROOT}/results/conversation-v0.3-v2}"
MODEL_DIR="${MODEL_DIR:-${PERSISTENT_ROOT}/models/laya-head-v0.3-v2}"
EBM_DIR="${RESULTS_DIR}/ebm"
LAYA_DIR="${RESULTS_DIR}/laya"
COMPARISON_DIR="${RESULTS_DIR}/comparison"

export HF_HOME="${HF_HOME:-${PERSISTENT_ROOT}/cache/huggingface}"
mkdir -p "${EBM_DIR}" "${LAYA_DIR}" "${COMPARISON_DIR}" "${MODEL_DIR}" "${HF_HOME}"
exec > >(tee "${RESULTS_DIR}/run.log") 2>&1

printf 'Started: %s\n' "$(date --iso-8601=seconds)"
printf 'Revision: %s\n' "$(git rev-parse HEAD)"

"${PYTHON_BIN}" -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA device is required'; print(torch.cuda.get_device_name(0))"

"${PYTHON_BIN}" -m diagnostic_model.sequential_ebm \
  data/generated/synthetic_scenarios.csv \
  --output-dir "${EBM_DIR}" \
  --validation-size 0.2 \
  --test-size 0.2

"${PYTHON_BIN}" -m diagnostic_model.export_stopping_visuals \
  "${EBM_DIR}/ebm_sequential_metrics.json" \
  "${EBM_DIR}/ebm_sequential_decisions.jsonl" \
  --output-dir "${EBM_DIR}/visualizations" \
  --approach EBM

"${PYTHON_BIN}" -m diagnostic_model.confidence_narrowing \
  "${EBM_DIR}/ebm_sequential_decisions.jsonl" \
  --output-dir "${EBM_DIR}/visualizations" \
  --approach EBM

"${PYTHON_BIN}" -m diagnostic_model.laya_finetune \
  data/generated/synthetic_scenarios.csv \
  --output-dir "${LAYA_DIR}" \
  --model-dir "${MODEL_DIR}" \
  --epochs 10 \
  --validation-size 0.2 \
  --test-size 0.2

"${PYTHON_BIN}" -m diagnostic_model.export_stopping_visuals \
  "${LAYA_DIR}/laya_sequential_metrics.json" \
  "${LAYA_DIR}/laya_sequential_decisions.jsonl" \
  --output-dir "${LAYA_DIR}/visualizations" \
  --approach Laya

"${PYTHON_BIN}" -m diagnostic_model.confidence_narrowing \
  "${LAYA_DIR}/laya_sequential_decisions.jsonl" \
  --output-dir "${LAYA_DIR}/visualizations" \
  --approach Laya

"${PYTHON_BIN}" -m diagnostic_model.compare_sequential \
  --ebm "${EBM_DIR}/ebm_sequential_metrics.json" \
  --laya "${LAYA_DIR}/laya_sequential_metrics.json" \
  --output-dir "${COMPARISON_DIR}"

printf 'Laya v2 results: %s\n' "${RESULTS_DIR}"
printf 'Laya v2 model:   %s\n' "${MODEL_DIR}"
printf 'Finished: %s\n' "$(date --iso-8601=seconds)"
