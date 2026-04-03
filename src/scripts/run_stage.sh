#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.env"

STAGE="${1:-}"
if [[ -z "${STAGE}" ]]; then
  echo "Usage: $0 <stage>"
  echo "  stages: 0 1 2 3 4 5j 5l 6s 6t 6b 6c 7"
  exit 1
fi

export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

str_to_bool() {
  local v="${1:-false}"
  shopt -s nocasematch
  if [[ "${v}" == "1" || "${v}" == "true" || "${v}" == "yes" ]]; then
    echo "true"
  else
    echo "false"
  fi
  shopt -u nocasematch
}

derive_run_setting() {
  local dataset_name
  dataset_name="$(basename "${DATA_FILE}")"
  dataset_name="${dataset_name%.jsonl}"
  dataset_name="${dataset_name%.json}"

  if [[ "$(str_to_bool "${USE_API}")" == "true" ]]; then
    echo "${MODEL_NAME}_API_${dataset_name}_fraction_${FRACTION}_max_length_${MAX_LENGTH}_num_generations_${NUM_GENERATIONS}_temperature_${TEMPERATURE}_top_p_${TOP_P}_seed_${SEED}"
  else
    echo "${MODEL_NAME}_${dataset_name}_fraction_${FRACTION}_max_length_${MAX_LENGTH}_num_generations_${NUM_GENERATIONS}_temperature_${TEMPERATURE}_top_k_${TOP_K}_top_p_${TOP_P}_decode_method_${DECODE_METHOD}_seed_${SEED}"
  fi
}

if [[ -z "${RUN_SETTING}" ]]; then
  RUN_SETTING="$(derive_run_setting)"
fi

echo "[info] stage=${STAGE} run_setting=${RUN_SETTING}"

case "${STAGE}" in
  0)
    # shellcheck disable=SC2086
    "${PYTHON_BIN}" -m pipeline.stage0_build_datasets \
      --datasets ${DATASETS} \
      --num_samples "${NUM_SAMPLES}" \
      --output_dir "${DATASETS_DIR}" \
      --wiki2_hf_repo "${WIKI2_HF_REPO}" \
      --wiki2_hf_split "${WIKI2_HF_SPLIT}"
    ;;
  1)
    cmd=("${PYTHON_BIN}" -m pipeline.stage1_generate_cot
      --model_dir "${MODEL_DIR}"
      --model_name "${MODEL_NAME}"
      --data_file "${DATA_FILE}"
      --output_dir "${RESULTS_DIR}"
      --fraction "${FRACTION}"
      --max_length "${MAX_LENGTH}"
      --num_generations_per_prompt "${NUM_GENERATIONS}"
      --temperature "${TEMPERATURE}"
      --top_k "${TOP_K}"
      --top_p "${TOP_P}"
      --seed "${SEED}"
      --device "${DEVICE}"
      --decode_method "${DECODE_METHOD}")

    if [[ "$(str_to_bool "${USE_API}")" == "true" ]]; then
      cmd+=(--use_api --api_key "${API_KEY}" --api_base "${API_BASE}")
      if [[ -n "${API_MODEL_NAME}" ]]; then
        cmd+=(--api_model_name "${API_MODEL_NAME}")
      fi
    fi
    "${cmd[@]}"
    ;;
  2)
    "${PYTHON_BIN}" -m pipeline.stage2_semantic_cluster \
      --device "${DEVICE}" \
      --seed "${SEED}" \
      --run_setting "${RUN_SETTING}" \
      --nli_model "${NLI_MODEL}"
    ;;
  3)
    cmd=("${PYTHON_BIN}" -m pipeline.stage3_compute_support
      --device "${DEVICE}"
      --model_name "${NLI_MODEL}"
      --run_setting "${RUN_SETTING}"
      --luq_method "${LUQ_METHOD}"
      --split_method "${SPLIT_METHOD}")

    if [[ "$(str_to_bool "${SAVE_MATRIX}")" == "true" ]]; then
      cmd+=(--save_matrix)
    fi

    "${cmd[@]}"
    ;;
  4)
    "${PYTHON_BIN}" -m pipeline.stage4_extract_confidence \
      --device "${DEVICE}" \
      --run_setting "${RUN_SETTING}" \
      --model_dir "${TOKENIZER_MODEL_DIR}" \
      --split_method "${SPLIT_METHOD}"
    ;;
  5j)
    cmd=("${PYTHON_BIN}" -m eval.judge_responses
      --run_setting "${RUN_SETTING}"
      --max_retries "${MAX_RETRIES}"
      --retry_interval "${RETRY_INTERVAL}")

    if [[ -n "${JUDGERS}" ]]; then
      # shellcheck disable=SC2206
      local_judgers=( ${JUDGERS} )
      cmd+=(--judgers "${local_judgers[@]}")
    fi

    "${cmd[@]}"
    ;;
  5l)
    cmd=("${PYTHON_BIN}" -m eval.build_auroc_labels
      --results_dir "${RESULTS_DIR}"
      --run_setting "${RUN_SETTING}")

    if [[ "$(str_to_bool "${OVERWRITE_AUROC_LABELS}")" == "true" ]]; then
      cmd+=(--overwrite)
    fi

    "${cmd[@]}"
    ;;
  6s)
    "${PYTHON_BIN}" -m baselines.cache_sentence_similarity \
      --measurement_model "${SENT_SIM_MODEL}" \
      --run_setting "${RUN_SETTING}" \
      --use_cleaned "True"
    ;;
  6t)
    "${PYTHON_BIN}" -m baselines.cache_token_importance \
      --measurement_model "${TOKEN_IMPT_MODEL}" \
      --tokenizer_model "${TOKENIZER_MODEL_DIR}" \
      --run_setting "${RUN_SETTING}"
    ;;
  6b)
    "${PYTHON_BIN}" -m baselines.build_baseline_scores \
      --output_dir "${RESULTS_DIR}" \
      --run_setting "${RUN_SETTING}" \
      --senten_sim_meas_model "${SENT_SIM_MODEL}" \
      --token_impt_meas_model "${TOKEN_IMPT_MODEL}" \
      --num_generation "${NUM_GENERATION_FOR_BASELINE}" \
      --temperature "${SAR_TEMPERATURE}" \
      --luq_split_method "${SPLIT_METHOD}" \
      --luq_json_prefix "${LUQ_JSON_PREFIX}"
    ;;
  6c)
    kw_cmd=("${PYTHON_BIN}" -m pipeline.Keywords_extraction_and_scoring
      --device "${DEVICE}"
      --model_dir "${TOKENIZER_MODEL_DIR}"
      --prompt_type "${COTUQ_PROMPT_TYPE}"
      --run_setting "${RUN_SETTING}")
    if [[ "$(str_to_bool "${COTUQ_USE_GREEDY}")" == "true" ]]; then
      kw_cmd+=(--use_greedy)
    fi
    "${kw_cmd[@]}"

    agg_cmd=("${PYTHON_BIN}" -m pipeline.Aggregated_probs
      --device "${DEVICE}"
      --model_dir "${TOKENIZER_MODEL_DIR}"
      --measure_model "${TOKEN_IMPT_MODEL}"
      --aggregated_method "${COTUQ_AGG_METHOD}"
      --prompt_type "${COTUQ_PROMPT_TYPE}"
      --run_setting "${RUN_SETTING}")
    if [[ "$(str_to_bool "${COTUQ_USE_GREEDY}")" == "true" ]]; then
      agg_cmd+=(--use_greedy)
    fi
    "${agg_cmd[@]}"
    ;;
  7)
    cmd=("${PYTHON_BIN}" -m eval.final_compare
      --results-root "${RESULTS_DIR}"
      --max-workers "${MAX_WORKERS}"
      --output-csv "${OUTPUT_CSV}")

    if [[ -n "${RUN_SETTINGS_JSON}" ]]; then
      cmd+=(--run-settings-json "${RUN_SETTINGS_JSON}")
    fi

    "${cmd[@]}"
    ;;
  *)
    echo "Unknown stage: ${STAGE}"
    exit 2
    ;;
esac
