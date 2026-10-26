#!/usr/bin/env bash
set -euo pipefail

DDX_ROOT="${DDX_ROOT:-/home/ubuntu/dsa2111-persistent/datasets/ddxplus-english}"

mkdir -p "${DDX_ROOT}"

download_and_verify() {
  local name="$1"
  local file_id="$2"
  local expected_md5="$3"
  local destination="${DDX_ROOT}/${name}"
  if [[ -f "${destination}" ]] && \
     echo "${expected_md5}  ${destination}" | md5sum --check --status; then
    return
  fi
  local temporary="${destination}.download"
  curl --fail --location --retry 3 \
    "https://ndownloader.figshare.com/files/${file_id}" --output "${temporary}"
  echo "${expected_md5}  ${temporary}" | md5sum --check --status
  mv "${temporary}" "${destination}"
}

# Fixed manifest from English Figshare v2 (doi:10.6084/m9.figshare.22687585.v2).
download_and_verify release_evidences.json 40278013 0660962abe0bbf99e1fb77188c253597
download_and_verify release_conditions.json 62561569 a992eadcff3099dd72e1820fbd2115aa
download_and_verify release_train_patients.zip 40278019 f75a78420ccb061951daa4db2d62139b
download_and_verify release_validate_patients.zip 40278022 e4db88e9082f18410de6fd0cb7e74d18
download_and_verify release_test_patients.zip 40278016 6fd53d331cf19ed4439bf4f46b92ee94

for split in train validate test; do
  if [[ ! -s "${DDX_ROOT}/release_${split}_patients" ]]; then
    unzip -n "${DDX_ROOT}/release_${split}_patients.zip" -d "${DDX_ROOT}"
  fi
done

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
