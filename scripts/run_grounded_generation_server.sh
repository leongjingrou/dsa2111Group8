#!/usr/bin/env bash
set -euo pipefail

DDX_ROOT="${DDX_ROOT:-/home/ubuntu/dsa2111-persistent/datasets/ddxplus-english}"
RESULTS_DIR="${RESULTS_DIR:-/home/ubuntu/dsa2111-persistent/results/grounded-v4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVIDENCES="${EVIDENCES:-${DDX_ROOT}/release_evidences.json}"

DDX_ROOT="${DDX_ROOT}" scripts/prepare_ddxplus_english.sh
mkdir -p "${RESULTS_DIR}"
"${PYTHON_BIN}" -m diagnostic_model.fetch_regional_seeds \
  "${RESULTS_DIR}/sea-style-seeds.jsonl"

for split in train validate test; do
  case "${split}" in
    train) sample_size=5000; variants=2 ;;
    validate) sample_size=1000; variants=1 ;;
    test) sample_size=1000; variants=1 ;;
  esac
  "${PYTHON_BIN}" -m diagnostic_model.grounding convert-ddxplus \
    "${DDX_ROOT}/release_${split}_patients.csv" \
    "${EVIDENCES}" "${RESULTS_DIR}/ddxplus-${split}-seeds.jsonl" \
    --split "${split}" --sample-size "${sample_size}" --seed 2111
  "${PYTHON_BIN}" -m diagnostic_model.grounded_generation \
    "${RESULTS_DIR}/ddxplus-${split}-seeds.jsonl" \
    "${RESULTS_DIR}/sea-style-seeds.jsonl" \
    "${RESULTS_DIR}/grounded-${split}.jsonl" \
    --variants-per-case "${variants}" --seed 2111
done

sha256sum "${RESULTS_DIR}"/*.jsonl >"${RESULTS_DIR}/sha256sums.txt"
