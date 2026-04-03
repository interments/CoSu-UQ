#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.env"

RUN_SETTINGS_FILE="${1:-}"
if [[ -z "${RUN_SETTINGS_FILE}" ]]; then
  echo "Usage: $0 <run_settings_file>"
  echo "Each non-empty line should be one run_setting. Lines starting with # are ignored."
  exit 1
fi

if [[ ! -f "${RUN_SETTINGS_FILE}" ]]; then
  echo "File not found: ${RUN_SETTINGS_FILE}"
  exit 2
fi

while IFS= read -r line; do
  rs="${line## }"
  rs="${rs%% }"
  [[ -z "${rs}" ]] && continue
  [[ "${rs}" =~ ^# ]] && continue

  echo "[batch] run_setting=${rs}"
  RUN_SETTING="${rs}" "${SCRIPT_DIR}/run_pipeline.sh"
done < "${RUN_SETTINGS_FILE}"

echo "[done] batch finished"
