#!/usr/bin/env bash
# 03_run_benchmark.sh — Config-tagged SWE-bench runner + evaluator (mini-swe-agent).
#
# WHAT IT DOES: sources the SAME sub-phase config as 02_serve.sh (CONFIG=... ),
# points mini-swe-agent at the local /v1 endpoint (litellm hosted_vllm provider),
# runs SWE-bench to produce predictions/trajectories, evaluates them with the
# swebench harness (needs Docker), then records a compact resolve-rate summary
# TAGGED WITH THE CONFIG NAME under records/results/. Tagging by config is what
# lets compare_results.py build the per-lever delta table (PLAN §3.7).
#
# For the 2e reasoning-effort lever, ENABLE_THINKING in the config is injected as
# chat_template_kwargs {enable_thinking: false} via a mini-swe-agent model config,
# WITHOUT changing how the server is launched.
#
# WHERE TO RUN: on the g6e.2xlarge box, AFTER 02_serve.sh is up ("Application
# startup complete"). Docker must be available for evaluation:
#   CONFIG=configs/2e_reasoning_low.env bash 03_run_benchmark.sh
#
# ITERATION NOTE:
#   (a) Use SUBSET=lite + small INSTANCE_LIMIT to wire a lever, THEN SUBSET=verified.
#   (b) Confirm CLI flags against the installed versions before a big run:
#         mini-extra swebench --help
#         python -m swebench.harness.run_evaluation --help

set -euo pipefail

# ==== CONFIG (edit me) ====
CONFIG="${CONFIG:-${1:-configs/2a_fp8_baseline.env}}"
API_BASE="${API_BASE:-http://127.0.0.1:8000/v1}"
BENCH_VENV_DIR="${HOME}/bench-venv"
# ==== END CONFIG ====

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "${CONFIG}" && -f "${SCRIPT_DIR}/${CONFIG}" ]]; then
  CONFIG="${SCRIPT_DIR}/${CONFIG}"
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config file not found: ${CONFIG}"
  exit 1
fi
# shellcheck source=/dev/null
source "${CONFIG}"
: "${CONFIG_NAME:?config is missing CONFIG_NAME}"

# Benchmark-side levers (with the same defaults as the baseline config).
SUBSET="${SUBSET:-lite}"
SPLIT="${SPLIT:-test}"
INSTANCE_LIMIT="${INSTANCE_LIMIT:-5}"
WORKERS="${WORKERS:-4}"
MAX_MODEL_TOKENS="${MAX_MODEL_TOKENS:-64000}"
LITELLM_MODEL="hosted_vllm/${SERVED_MODEL_NAME}"

RESULTS_DIR="${SCRIPT_DIR}/records/results"

# ---- inline recording (tee stdout/stderr to a timestamped log under records/) ----
TS="$(date -u +"%Y%m%dT%H%M%SZ")"
RECORD_DIR="${SCRIPT_DIR}/records"
mkdir -p "${RECORD_DIR}"
RAW_RECORD_FILE="${RECORD_DIR}/${TS}_run_benchmark_${CONFIG_NAME}.log"
exec > >(tee -a "${RAW_RECORD_FILE}") 2>&1
echo "record_name=run_benchmark"
echo "config_name=${CONFIG_NAME}"
echo "started_utc=${TS}"
echo "host=$(hostname)"
echo "command=$0 $*"
echo "--- output ---"
_finish_recording() {
  local status=$?
  trap - EXIT
  echo "--- end ---"
  echo "finished_utc=$(date -u +"%Y%m%dT%H%M%SZ")"
  echo "exit_status=${status}"
  echo "raw_record=${RAW_RECORD_FILE}"
  exit "${status}"
}
trap _finish_recording EXIT
# ---- end inline recording ----

echo "=== Preflight: endpoint check (${API_BASE}) ==="
if ! curl -fsS "${API_BASE}/models" >/dev/null; then
  echo "ERROR: endpoint not reachable at ${API_BASE}/models."
  echo "Start it first with: CONFIG=${CONFIG} bash 02_serve.sh"
  exit 1
fi
echo "Endpoint OK."

echo ""
echo "=== Preflight: Docker (needed for swebench evaluation) ==="
if ! command -v docker &>/dev/null; then
  echo "WARNING: docker not found. Prediction may run, but swebench evaluation needs Docker."
else
  echo "docker found: $(docker --version 2>/dev/null || true)"
fi

echo ""
echo "=== Benchmark venv at ${BENCH_VENV_DIR} ==="
if [[ ! -d "${BENCH_VENV_DIR}" ]]; then
  python3 -m venv "${BENCH_VENV_DIR}"
else
  echo "Reusing existing venv."
fi
# shellcheck disable=SC1091
source "${BENCH_VENV_DIR}/bin/activate"
pip install -U pip
pip install "mini-swe-agent" "swebench"

# Map subset -> HF dataset for the swebench evaluation harness.
case "${SUBSET}" in
  lite)     DATASET="princeton-nlp/SWE-bench_Lite" ;;
  verified) DATASET="princeton-nlp/SWE-bench_Verified" ;;
  *) echo "ERROR: Unsupported SUBSET='${SUBSET}' (expected 'lite' or 'verified')."; exit 1 ;;
esac

echo ""
echo "=== litellm registry for the hosted_vllm model ==="
REGISTRY_PATH="${SCRIPT_DIR}/registry.json"
cat > "${REGISTRY_PATH}" <<EOF
{
  "${SERVED_MODEL_NAME}": {
    "max_tokens": ${MAX_MODEL_TOKENS},
    "max_input_tokens": ${MAX_MODEL_TOKENS},
    "max_output_tokens": ${MAX_MODEL_TOKENS},
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
    "litellm_provider": "hosted_vllm",
    "mode": "chat",
    "api_base": "${API_BASE}"
  }
}
EOF
export LITELLM_MODEL_REGISTRY_PATH="${REGISTRY_PATH}"
export OPENAI_API_KEY="dummy"
export HOSTED_VLLM_API_BASE="${API_BASE}"
export OPENAI_BASE_URL="${API_BASE}"
echo "Wrote ${REGISTRY_PATH}."

OUT_DIR="${RESULTS_DIR}/run_${CONFIG_NAME}_${TS}"
mkdir -p "${OUT_DIR}"

echo ""
echo "=== Generate predictions with mini-swe-agent ==="
echo "config=${CONFIG_NAME} model=${LITELLM_MODEL} subset=${SUBSET} split=${SPLIT} workers=${WORKERS}"

# NOTE: confirm these flags with `mini-extra swebench --help` for the installed version.
MSWEA_ARGS=(
  --model "${LITELLM_MODEL}"
  --subset "${SUBSET}"
  --split "${SPLIT}"
  --output "${OUT_DIR}"
  --workers "${WORKERS}"
)
if [[ -n "${INSTANCE_LIMIT}" && "${INSTANCE_LIMIT}" =~ ^[0-9]+$ && "${INSTANCE_LIMIT}" -gt 0 ]]; then
  echo "Limiting to first ${INSTANCE_LIMIT} instances (smoke slice)."
  MSWEA_ARGS+=(--slice ":${INSTANCE_LIMIT}")
fi

# --- 2e reasoning-effort lever: inject chat_template_kwargs {enable_thinking: false} ---
# Passed through a mini-swe-agent model config so ONLY the request changes, not the server.
# NOTE: confirm the exact key mini-swe-agent/litellm expects for chat_template_kwargs.
if [[ -n "${ENABLE_THINKING:-}" ]]; then
  MODEL_CONFIG_PATH="${OUT_DIR}/model_config.yaml"
  cat > "${MODEL_CONFIG_PATH}" <<EOF
model:
  model_name: "${LITELLM_MODEL}"
  model_kwargs:
    chat_template_kwargs:
      enable_thinking: ${ENABLE_THINKING}
EOF
  echo "Reasoning lever: enable_thinking=${ENABLE_THINKING} (wrote ${MODEL_CONFIG_PATH})."
  MSWEA_ARGS+=(--config "${MODEL_CONFIG_PATH}")
fi

mini-extra swebench "${MSWEA_ARGS[@]}"

echo ""
echo "=== Locate predictions file ==="
PREDICTIONS_PATH="${OUT_DIR}/preds.json"
if [[ ! -f "${PREDICTIONS_PATH}" ]]; then
  PREDICTIONS_PATH="$(find "${OUT_DIR}" -maxdepth 2 -name '*.json' -type f -print0 \
    | xargs -0 ls -t 2>/dev/null | head -n1 || true)"
fi
if [[ -z "${PREDICTIONS_PATH:-}" || ! -f "${PREDICTIONS_PATH}" ]]; then
  echo "ERROR: Could not find a predictions JSON under ${OUT_DIR}."
  exit 1
fi
echo "Predictions: ${PREDICTIONS_PATH}"

RUN_ID="phase2_${CONFIG_NAME}_${TS}"
echo ""
echo "=== Evaluate with swebench harness (run_id=${RUN_ID}) ==="
echo "(Builds/uses Docker images per repo; can take a while.)"
# NOTE: confirm flags with `python -m swebench.harness.run_evaluation --help`.
python -m swebench.harness.run_evaluation \
  --dataset_name "${DATASET}" \
  --predictions_path "${PREDICTIONS_PATH}" \
  --run_id "${RUN_ID}" \
  --max_workers "${WORKERS}"

echo ""
echo "=== Locate + parse evaluation report ==="
REPORT_PATH="$(find "${SCRIPT_DIR}" "${OUT_DIR}" "${HOME}" -maxdepth 3 \
  -name "*${RUN_ID}*.json" -type f -print0 2>/dev/null \
  | xargs -0 ls -t 2>/dev/null | head -n1 || true)"

SUMMARY_PATH="${RESULTS_DIR}/swebench_${CONFIG_NAME}_${TS}.json"
mkdir -p "${RESULTS_DIR}"

if [[ -z "${REPORT_PATH:-}" || ! -f "${REPORT_PATH}" ]]; then
  echo "WARNING: could not auto-locate the swebench report JSON for ${RUN_ID}."
  python - "$SUMMARY_PATH" <<PY
import json, sys
summary = {
    "timestamp_utc": "${TS}",
    "config_name": "${CONFIG_NAME}",
    "subphase": "${SUBPHASE:-}",
    "model": "${SERVED_MODEL_NAME}",
    "litellm_model": "${LITELLM_MODEL}",
    "subset": "${SUBSET}",
    "split": "${SPLIT}",
    "dataset": "${DATASET}",
    "instances_attempted": None,
    "resolved": None,
    "resolve_rate_pct": None,
    "config": {
        "engine": "${ENGINE:-vllm}",
        "model_id": "${MODEL_ID}",
        "maxlen": ${MAXLEN},
        "kv_cache_dtype": "${KV_CACHE_DTYPE:-}",
        "enable_prefix_caching": "${ENABLE_PREFIX_CACHING:-}",
        "enable_chunked_prefill": "${ENABLE_CHUNKED_PREFILL:-}",
        "max_num_seqs": "${MAX_NUM_SEQS:-}",
        "spec_config": "${SPEC_CONFIG:-}",
        "enable_thinking": "${ENABLE_THINKING:-}",
        "workers": ${WORKERS},
        "instance_limit": "${INSTANCE_LIMIT}",
        "max_model_tokens": ${MAX_MODEL_TOKENS},
    },
    "note": "swebench report JSON not auto-located; edit resolved/total manually.",
}
with open(sys.argv[1], "w") as f:
    json.dump(summary, f, indent=2)
print("Wrote partial summary:", sys.argv[1])
PY
else
  echo "Report: ${REPORT_PATH}"
  python - "$REPORT_PATH" "$SUMMARY_PATH" <<PY
import json, sys

report_path, summary_path = sys.argv[1], sys.argv[2]
with open(report_path) as f:
    report = json.load(f)

def find_int(keys):
    for k in keys:
        v = report.get(k)
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, list):
            return len(v)
    return None

resolved = find_int(["resolved_instances", "resolved"])
total = find_int(["total_instances", "submitted_instances", "total"])
rate = round(100.0 * resolved / total, 2) if resolved is not None and total else None

summary = {
    "timestamp_utc": "${TS}",
    "config_name": "${CONFIG_NAME}",
    "subphase": "${SUBPHASE:-}",
    "model": "${SERVED_MODEL_NAME}",
    "litellm_model": "${LITELLM_MODEL}",
    "subset": "${SUBSET}",
    "split": "${SPLIT}",
    "dataset": "${DATASET}",
    "instances_attempted": total,
    "resolved": resolved,
    "resolve_rate_pct": rate,
    "config": {
        "engine": "${ENGINE:-vllm}",
        "model_id": "${MODEL_ID}",
        "maxlen": ${MAXLEN},
        "kv_cache_dtype": "${KV_CACHE_DTYPE:-}",
        "enable_prefix_caching": "${ENABLE_PREFIX_CACHING:-}",
        "enable_chunked_prefill": "${ENABLE_CHUNKED_PREFILL:-}",
        "max_num_seqs": "${MAX_NUM_SEQS:-}",
        "spec_config": "${SPEC_CONFIG:-}",
        "enable_thinking": "${ENABLE_THINKING:-}",
        "workers": ${WORKERS},
        "instance_limit": "${INSTANCE_LIMIT}",
        "max_model_tokens": ${MAX_MODEL_TOKENS},
    },
    "source_report": report_path,
}
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"resolved={resolved} total={total} resolve_rate={rate}%")
print("Wrote summary:", summary_path)
PY
fi

echo ""
echo "=== Done (${CONFIG_NAME}) ==="
echo "Summary: ${SUMMARY_PATH}"
if [[ -f "${SUMMARY_PATH}" ]]; then
  RATE="$(python -c "import json;print(json.load(open('${SUMMARY_PATH}')).get('resolve_rate_pct'))" 2>/dev/null || true)"
  echo "SWE-bench ${SUBSET}/${SPLIT} resolve rate for ${CONFIG_NAME}: ${RATE}%"
fi
