#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.env"

# Stage toggles
DO_STAGE0="${DO_STAGE0:-false}"
DO_STAGE1="${DO_STAGE1:-true}"
DO_STAGE2="${DO_STAGE2:-true}"
DO_STAGE3="${DO_STAGE3:-true}"
DO_STAGE4="${DO_STAGE4:-true}"
DO_STAGE5J="${DO_STAGE5J:-true}"
DO_STAGE5L="${DO_STAGE5L:-true}"
DO_STAGE6S="${DO_STAGE6S:-true}"
DO_STAGE6T="${DO_STAGE6T:-true}"
DO_STAGE6B="${DO_STAGE6B:-true}"
DO_STAGE6C="${DO_STAGE6C:-true}"
DO_STAGE7="${DO_STAGE7:-true}"

run_if_enabled() {
  local enabled="$1"
  local stage="$2"
  shopt -s nocasematch
  if [[ "${enabled}" == "1" || "${enabled}" == "true" || "${enabled}" == "yes" ]]; then
    echo "[run] stage=${stage}"
    "${SCRIPT_DIR}/run_stage.sh" "${stage}"
  else
    echo "[skip] stage=${stage}"
  fi
  shopt -u nocasematch
}

run_if_enabled "${DO_STAGE0}" 0
run_if_enabled "${DO_STAGE1}" 1
run_if_enabled "${DO_STAGE2}" 2
run_if_enabled "${DO_STAGE3}" 3
run_if_enabled "${DO_STAGE4}" 4
run_if_enabled "${DO_STAGE5J}" 5j
run_if_enabled "${DO_STAGE5L}" 5l
run_if_enabled "${DO_STAGE6S}" 6s
run_if_enabled "${DO_STAGE6T}" 6t
run_if_enabled "${DO_STAGE6B}" 6b
run_if_enabled "${DO_STAGE6C}" 6c
run_if_enabled "${DO_STAGE7}" 7

echo "[done] full pipeline script finished"
