#!/usr/bin/env bash
# 02_serve.sh — Config-driven, one-lever-at-a-time serve for the Phase 2 sweep.
#
# WHAT IT DOES (all in one script): sources ONE sub-phase config from configs/
# (via the CONFIG env var or first arg), sets up a Python venv, downloads the
# model, then launches the serving engine (vLLM by default, SGLang for 2f) with
# the OpenAI-compatible /v1 API and reasoning ON. Every lever (precision, KV
# dtype, prefix/chunked prefill, batching, CUDA graphs, speculative decoding,
# engine) is read from the config file, so the SAME script serves every 2a-2f
# variant — only the config changes. The serve command BLOCKS; run it in a
# dedicated terminal / tmux and wait for "Application startup complete".
#
# WHERE TO RUN: on the g6e.2xlarge box (Deep Learning AMI with a working
# `nvidia-smi`). Copy up 02_serve.sh, 03_run_benchmark.sh and the configs/ dir:
#   CONFIG=configs/2b_prefix_chunked.env bash 02_serve.sh
#
# The Phase 1 baseline lever set corresponds to configs/2a_fp8_baseline.env.

set -euo pipefail

# ==== CONFIG (edit me) ====
# Which sub-phase config to serve. Override with CONFIG=... or pass as $1.
CONFIG="${CONFIG:-${1:-configs/2a_fp8_baseline.env}}"

PORT="${PORT:-8000}"
VENV_DIR="${HOME}/serve-venv"
HF_HOME="${HF_HOME:-${HOME}/hf}"
VLLM_SPEC="vllm>=0.24"     # >=0.24 required for --reasoning-parser + qwen3_xml tool parser
SGLANG_SPEC="sglang[all]"  # only installed when a config sets ENGINE=sglang (2f)
# ==== END CONFIG ====

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve CONFIG relative to this script if it is not an absolute/existing path.
if [[ ! -f "${CONFIG}" && -f "${SCRIPT_DIR}/${CONFIG}" ]]; then
  CONFIG="${SCRIPT_DIR}/${CONFIG}"
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config file not found: ${CONFIG}"
  echo "Pass one of configs/*.env via CONFIG=... or as the first argument."
  exit 1
fi
# shellcheck source=/dev/null
source "${CONFIG}"
: "${CONFIG_NAME:?config is missing CONFIG_NAME}"
ENGINE="${ENGINE:-vllm}"

# ---- inline recording (tee stdout/stderr to a timestamped log under records/) ----
TS="$(date -u +"%Y%m%dT%H%M%SZ")"
RECORD_DIR="${SCRIPT_DIR}/records"
mkdir -p "${RECORD_DIR}"
RAW_RECORD_FILE="${RECORD_DIR}/${TS}_serve_${CONFIG_NAME}.log"
exec > >(tee -a "${RAW_RECORD_FILE}") 2>&1
echo "record_name=serve"
echo "config_name=${CONFIG_NAME}"
echo "config_file=${CONFIG}"
echo "engine=${ENGINE}"
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

echo "=== GPU check ==="
if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. Use the Deep Learning AMI or install NVIDIA drivers."
  exit 1
fi
nvidia-smi

echo ""
echo "=== Python venv prerequisite ==="
if ! dpkg -s python3-venv &>/dev/null; then
  echo "python3-venv is missing; installing it with apt ..."
  sudo apt-get update
  sudo apt-get install -y python3-venv
else
  echo "python3-venv is already installed."
fi

echo ""
echo "=== Python venv at ${VENV_DIR} ==="
if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
else
  echo "Reusing existing venv."
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install -U pip wheel

echo ""
echo "=== Installing serving engine for ENGINE=${ENGINE} ==="
case "${ENGINE}" in
  vllm)
    pip install "${VLLM_SPEC}"
    python -c "import vllm; print('vllm', vllm.__version__)"
    ;;
  sglang)
    pip install "${SGLANG_SPEC}"
    python -c "import sglang; print('sglang', getattr(sglang, '__version__', 'unknown'))"
    ;;
  *)
    echo "ERROR: unsupported ENGINE='${ENGINE}' (expected 'vllm' or 'sglang')."
    exit 1
    ;;
esac
pip install -U huggingface_hub

echo ""
echo "=== Torch CUDA sanity check ==="
python - <<'PY'
import sys
import torch

count = torch.cuda.device_count()
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"Visible CUDA devices: {count}")
if count == 0:
    print("ERROR: PyTorch cannot see a CUDA device.", file=sys.stderr)
    raise SystemExit(1)
for index in range(count):
    print(f"  GPU {index}: {torch.cuda.get_device_name(index)}")
PY

echo ""
echo "=== Hugging Face download: ${MODEL_ID} ==="
export HF_HOME
mkdir -p "${HF_HOME}"
echo "HF_HOME=${HF_HOME}"
echo "If rate-limited or gated, run: hf auth login"
hf download "${MODEL_ID}"

echo ""
echo "=== Serving config summary (lever = ${CONFIG_NAME}) ==="
echo "engine=${ENGINE} model=${MODEL_ID} served=${SERVED_MODEL_NAME}"
echo "MAXLEN=${MAXLEN} GPU_MEM_UTIL=${GPU_MEM_UTIL} MAX_NUM_SEQS=${MAX_NUM_SEQS} PORT=${PORT}"
echo "KV_CACHE_DTYPE='${KV_CACHE_DTYPE:-}' PREFIX_CACHING=${ENABLE_PREFIX_CACHING} CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL}"
echo "ENFORCE_EAGER=${ENFORCE_EAGER} COMPILATION_LEVEL='${COMPILATION_LEVEL:-}' SPEC_CONFIG='${SPEC_CONFIG:-}'"
echo "This command BLOCKS — wait for 'Application startup complete'. Endpoint: http://127.0.0.1:${PORT}/v1"
echo ""

# ============================================================================
# Build + launch the engine command from the sourced levers.
# NOTE: flag names drift across vLLM/SGLang releases — confirm with `vllm serve --help`
#       or `python -m sglang.launch_server --help` for the installed version.
# ============================================================================
if [[ "${ENGINE}" == "vllm" ]]; then
  ARGS=(
    "${MODEL_ID}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --tensor-parallel-size 1
    --max-model-len "${MAXLEN}"
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --port "${PORT}"
    # Reasoning + tool parsing (correctness prerequisite, evaluation.md §3.3).
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser qwen3_xml
  )

  if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
    ARGS+=(--enable-prefix-caching)
  else
    ARGS+=(--no-enable-prefix-caching)
  fi
  [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]] && ARGS+=(--enable-chunked-prefill)
  [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]] && ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
  [[ -n "${KV_CACHE_DTYPE:-}" ]] && ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
  [[ "${ENFORCE_EAGER}" == "1" ]] && ARGS+=(--enforce-eager)
  [[ -n "${COMPILATION_LEVEL:-}" ]] && ARGS+=("-O${COMPILATION_LEVEL}")
  [[ -n "${SPEC_CONFIG:-}" ]] && ARGS+=(--speculative-config "${SPEC_CONFIG}")
  # shellcheck disable=SC2206
  [[ -n "${EXTRA_SERVE_ARGS:-}" ]] && ARGS+=(${EXTRA_SERVE_ARGS})

  echo "+ vllm serve ${ARGS[*]}"
  exec vllm serve "${ARGS[@]}"

elif [[ "${ENGINE}" == "sglang" ]]; then
  # SGLang: RadixAttention (prefix cache) is on by default. Speculative decoding /
  # KV-dtype flags differ from vLLM; keep the engine A/B on the baseline lever set.
  ARGS=(
    --model-path "${MODEL_ID}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --tp-size 1
    --context-length "${MAXLEN}"
    --mem-fraction-static "${GPU_MEM_UTIL}"
    --host 0.0.0.0
    --port "${PORT}"
    # Reasoning + tool parsing (SGLang parser names; confirm for the installed version).
    --reasoning-parser qwen3
    --tool-call-parser qwen3
  )
  [[ "${ENABLE_PREFIX_CACHING}" != "1" ]] && ARGS+=(--disable-radix-cache)
  [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]] && ARGS+=(--chunked-prefill-size "${MAX_NUM_BATCHED_TOKENS}")
  # shellcheck disable=SC2206
  [[ -n "${EXTRA_SERVE_ARGS:-}" ]] && ARGS+=(${EXTRA_SERVE_ARGS})

  echo "+ python -m sglang.launch_server ${ARGS[*]}"
  exec python -m sglang.launch_server "${ARGS[@]}"
fi
