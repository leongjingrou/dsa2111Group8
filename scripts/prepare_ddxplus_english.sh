#!/usr/bin/env bash
set -euo pipefail

DDX_ROOT="${DDX_ROOT:-/home/ubuntu/dsa2111-persistent/datasets/ddxplus-english}"
ARTICLE_URL="https://figshare.com/ndownloader/articles/22687585/versions/2"
ARCHIVE="${DDX_ROOT}/ddxplus-english-v2.zip"

mkdir -p "${DDX_ROOT}"
if [[ ! -f "${DDX_ROOT}/release_train_patients" ]] || \
   [[ ! -f "${DDX_ROOT}/release_validate_patients" ]] || \
   [[ ! -f "${DDX_ROOT}/release_test_patients" ]]; then
  if [[ ! -f "${ARCHIVE}" ]]; then
    curl --fail --location --retry 3 "${ARTICLE_URL}" --output "${ARCHIVE}"
  fi
  unzip -n "${ARCHIVE}" -d "${DDX_ROOT}"
  for split in train validate test; do
    unzip -n "${DDX_ROOT}/release_${split}_patients.zip" -d "${DDX_ROOT}"
  done
fi

for required in release_evidences.json release_conditions.json \
                release_train_patients release_validate_patients \
                release_test_patients; do
  test -s "${DDX_ROOT}/${required}"
done

sha256sum "${DDX_ROOT}"/release_evidences.json \
  "${DDX_ROOT}"/release_conditions.json \
  "${DDX_ROOT}"/release_train_patients \
  "${DDX_ROOT}"/release_validate_patients \
  "${DDX_ROOT}"/release_test_patients \
  >"${DDX_ROOT}/extracted-sha256sums.txt"
