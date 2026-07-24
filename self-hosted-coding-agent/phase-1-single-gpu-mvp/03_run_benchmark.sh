#!/usr/bin/env bash
# 03_run_benchmark.sh — Self-contained SWE-bench runner + evaluator (mini-swe-agent).
#
# WHAT IT DOES: points mini-swe-agent at the local vLLM /v1 endpoint (via litellm's
# hosted_vllm provider), runs SWE-bench to produce predictions/trajectories,
# evaluates them with the swebench harness (needs Docker), then records a compact
# resolve-rate summary under records/results/.
#
# WHERE TO RUN: on the g6e.2xlarge box, AFTER 02_serve_vllm.sh is up and serving
# on PORT (wait for "Application startup complete"). Docker must be available for
# the evaluation step. Copy it up with scp (see 01_infra_setup.sh "Next steps"):
#   bash 03_run_benchmark.sh
#
# ITERATION NOTE:
#   (a) First run should use SUBSET=lite + a small INSTANCE_LIMIT to validate the
#       end-to-end wiring, THEN scale to SUBSET=verified for the headline number.
#   (b) Exact mini-swe-agent / swebench CLI flag names change across versions.
#       Confirm them against the installed versions before a big run:
#         mini-extra swebench --help
#         python -m swebench.harness.run_evaluation --help

set -euo pipefail

# ==== CONFIG (edit me) ====
API_BASE="http://127.0.0.1:8000/v1"
SERVED_MODEL_NAME="qwen3.6-35b-a3b-fp8"
LITELLM_MODEL="hosted_vllm/qwen3.6-35b-a3b-fp8"

SUBSET="lite"           # `lite` for fast iteration, `verified` for the headline number
SPLIT="test"
INSTANCE_LIMIT="5"      # small smoke slice first; set empty or 0 to run the full subset
WORKERS="4"
# IMPORTANT: keep in sync with the vLLM server's --max-model-len in 02_serve_vllm.sh (65536).
# The earlier ContextWindowExceededError was caused by advertising 64000 here while the server
# only allowed 32768 — the agent thought it had headroom it didn't and never trimmed, so the
# server rejected the request. We now advertise the real context and reserve room for output.
MAX_MODEL_TOKENS="65536"
MAX_OUTPUT_TOKENS="8192"   # generation reserve; input is capped at MAX_MODEL_TOKENS - this
MAX_INPUT_TOKENS="$(( MAX_MODEL_TOKENS - MAX_OUTPUT_TOKENS ))"

BENCH_VENV_DIR="${HOME}/bench-venv"
# ==== END CONFIG ====

# Base the records dir on THIS script's own directory (it may live anywhere on the box).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/records/results"

# ---- inline recording (tee stdout/stderr to a timestamped log under records/) ----
TS="$(date -u +"%Y%m%dT%H%M%SZ")"
RECORD_DIR="${SCRIPT_DIR}/records"
mkdir -p "${RECORD_DIR}"
RAW_RECORD_FILE="${RECORD_DIR}/${TS}_run_benchmark.log"
exec > >(tee -a "${RAW_RECORD_FILE}") 2>&1
echo "record_name=run_benchmark"
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

echo "=== Preflight: endpoint check ==="
if ! curl -fsS "${API_BASE}/models" >/dev/null; then
  echo "ERROR: vLLM endpoint not reachable at ${API_BASE}/models."
  echo "Start it first with 02_serve_vllm.sh and wait for 'Application startup complete'."
  exit 1
fi
echo "Endpoint OK: ${API_BASE}/models"

echo ""
echo "=== Preflight: Docker (needed for swebench evaluation) ==="
if ! command -v docker &>/dev/null; then
  echo "WARNING: docker not found. The prediction step may run, but the swebench"
  echo "evaluation harness needs Docker. Install/enable Docker before evaluating."
else
  echo "docker found: $(docker --version 2>/dev/null || true)"
fi

echo ""
echo "=== Benchmark venv at ${BENCH_VENV_DIR} ==="
if [[ ! -d "${BENCH_VENV_DIR}" ]]; then
  echo "Creating venv ..."
  python3 -m venv "${BENCH_VENV_DIR}"
else
  echo "Reusing existing venv."
fi
# shellcheck disable=SC1091
source "${BENCH_VENV_DIR}/bin/activate"

pip install -U pip
pip install "mini-swe-agent" "swebench"

# Map subset -> HF dataset name for the swebench evaluation harness.
case "${SUBSET}" in
  lite)     DATASET="princeton-nlp/SWE-bench_Lite" ;;
  verified) DATASET="princeton-nlp/SWE-bench_Verified" ;;
  *)
    echo "ERROR: Unsupported SUBSET='${SUBSET}' (expected 'lite' or 'verified')."
    exit 1
    ;;
esac

echo ""
echo "=== litellm registry for the hosted_vllm model ==="
# Tell litellm about the local zero-cost hosted_vllm model. Keyed by served name.
REGISTRY_PATH="${SCRIPT_DIR}/registry.json"
cat > "${REGISTRY_PATH}" <<EOF
{
  "${SERVED_MODEL_NAME}": {
    "max_tokens": ${MAX_MODEL_TOKENS},
    "max_input_tokens": ${MAX_INPUT_TOKENS},
    "max_output_tokens": ${MAX_OUTPUT_TOKENS},
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
    "litellm_provider": "hosted_vllm",
    "mode": "chat",
    "api_base": "${API_BASE}"
  }
}
EOF
export LITELLM_MODEL_REGISTRY_PATH="${REGISTRY_PATH}"
# vLLM ignores the key, but litellm/openai clients often require a non-empty value.
export OPENAI_API_KEY="dummy"
export HOSTED_VLLM_API_BASE="${API_BASE}"
export OPENAI_BASE_URL="${API_BASE}"
# Self-hosted model is free -> per-token cost is 0.0 in registry.json. mini-swe-agent's
# default cost tracking treats a 0.0 cost as "unregistered" and aborts every task with
# "RuntimeError: Cost must be > 0.0". Tell it to ignore cost-tracking errors.
export MSWEA_COST_TRACKING="ignore_errors"
echo "Wrote ${REGISTRY_PATH} (LITELLM_MODEL_REGISTRY_PATH set; MSWEA_COST_TRACKING=ignore_errors)."

OUT_DIR="${RESULTS_DIR}/run_${TS}"
mkdir -p "${OUT_DIR}"

echo ""
echo "=== Generate predictions with mini-swe-agent ==="
echo "model=${LITELLM_MODEL} subset=${SUBSET} split=${SPLIT} workers=${WORKERS}"

# NOTE: confirm these flags with `mini-extra swebench --help` for the installed version.
MSWEA_ARGS=(
  --model "${LITELLM_MODEL}"
  --subset "${SUBSET}"
  --split "${SPLIT}"
  --output "${OUT_DIR}"
  --workers "${WORKERS}"
)
# If INSTANCE_LIMIT is a positive number, only run a small slice.
# The exact flag may differ across versions (e.g. --slice / --limit); confirm with --help.
if [[ -n "${INSTANCE_LIMIT}" && "${INSTANCE_LIMIT}" =~ ^[0-9]+$ && "${INSTANCE_LIMIT}" -gt 0 ]]; then
  echo "Limiting to first ${INSTANCE_LIMIT} instances (smoke slice)."
  MSWEA_ARGS+=(--slice ":${INSTANCE_LIMIT}")
fi

mini-extra swebench "${MSWEA_ARGS[@]}"

echo ""
echo "=== Locate predictions file ==="
# mini-swe-agent writes a predictions JSON into the output dir (commonly preds.json).
PREDICTIONS_PATH="${OUT_DIR}/preds.json"
if [[ ! -f "${PREDICTIONS_PATH}" ]]; then
  # Fall back to the newest *.json under the output dir.
  PREDICTIONS_PATH="$(find "${OUT_DIR}" -maxdepth 2 -name '*.json' -type f -print0 \
    | xargs -0 ls -t 2>/dev/null | head -n1 || true)"
fi
if [[ -z "${PREDICTIONS_PATH:-}" || ! -f "${PREDICTIONS_PATH}" ]]; then
  echo "ERROR: Could not find a predictions JSON under ${OUT_DIR}."
  echo "Inspect the output dir and set PREDICTIONS_PATH manually."
  exit 1
fi
echo "Predictions: ${PREDICTIONS_PATH}"
PRED_COUNT="$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "${PREDICTIONS_PATH}" 2>/dev/null || echo 0)"
echo "Predictions submitted: ${PRED_COUNT}"

RUN_ID="phase1_${TS}"
echo ""
echo "=== Evaluate with swebench harness (run_id=${RUN_ID}) ==="
echo "(This builds/uses Docker images per repo and can take a while.)"
# NOTE: confirm flags with `python -m swebench.harness.run_evaluation --help`.
python -m swebench.harness.run_evaluation \
  --dataset_name "${DATASET}" \
  --predictions_path "${PREDICTIONS_PATH}" \
  --run_id "${RUN_ID}" \
  --max_workers "${WORKERS}"

echo ""
echo "=== Locate + parse evaluation report ==="
# swebench writes a report JSON (name typically includes the run_id). Grab the newest.
REPORT_PATH="$(find "${SCRIPT_DIR}" "${OUT_DIR}" "${HOME}" -maxdepth 3 \
  -name "*${RUN_ID}*.json" -type f -print0 2>/dev/null \
  | xargs -0 ls -t 2>/dev/null | head -n1 || true)"

SUMMARY_PATH="${RESULTS_DIR}/swebench_${TS}.json"
mkdir -p "${RESULTS_DIR}"

if [[ -z "${REPORT_PATH:-}" || ! -f "${REPORT_PATH}" ]]; then
  echo "WARNING: could not auto-locate the swebench report JSON for ${RUN_ID}."
  echo "Recording a partial summary; fill in resolved/total from the harness output."
  python - "$SUMMARY_PATH" <<PY
import json, sys
summary = {
    "timestamp_utc": "${TS}",
    "model": "${SERVED_MODEL_NAME}",
    "litellm_model": "${LITELLM_MODEL}",
    "subset": "${SUBSET}",
    "split": "${SPLIT}",
    "dataset": "${DATASET}",
    "instances_attempted": ${PRED_COUNT},
    "resolved": None,
    "resolve_rate_pct": None,
    "config": {
        "api_base": "${API_BASE}",
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

pred_count = ${PRED_COUNT}

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
rate = round(100.0 * resolved / pred_count, 2) if resolved is not None and pred_count else None

summary = {
    "timestamp_utc": "${TS}",
    "model": "${SERVED_MODEL_NAME}",
    "litellm_model": "${LITELLM_MODEL}",
    "subset": "${SUBSET}",
    "split": "${SPLIT}",
    "dataset": "${DATASET}",
    "instances_attempted": pred_count,
    "dataset_total": total,
    "resolved": resolved,
    "resolve_rate_pct": rate,
    "config": {
        "api_base": "${API_BASE}",
        "workers": ${WORKERS},
        "instance_limit": "${INSTANCE_LIMIT}",
        "max_model_tokens": ${MAX_MODEL_TOKENS},
    },
    "source_report": report_path,
}
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"resolved={resolved} attempted={pred_count} dataset_total={total} resolve_rate={rate}%")
print("Wrote summary:", summary_path)
PY
fi

echo ""
echo "=== Done ==="
echo "Summary: ${SUMMARY_PATH}"
if [[ -f "${SUMMARY_PATH}" ]]; then
  RATE="$(python -c "import json;print(json.load(open('${SUMMARY_PATH}')).get('resolve_rate_pct'))" 2>/dev/null || true)"
  echo "SWE-bench ${SUBSET}/${SPLIT} resolve rate: ${RATE}%"
fi
