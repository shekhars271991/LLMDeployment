#!/usr/bin/env bash
# 02_serve_vllm.sh — Self-contained: set up venv + download model + serve vLLM.
#
# WHAT IT DOES (all in one script): creates a Python venv, installs vLLM >= 0.24
# (needed for the reasoning + tool-streaming parsers), downloads the FP8 model
# from Hugging Face, then launches the vLLM OpenAI-compatible server with
# reasoning ON. The `vllm serve` command BLOCKS — run this in a dedicated
# terminal or tmux window and wait for "Application startup complete".
#
# WHERE TO RUN: on the g6e.2xlarge box (Deep Learning AMI with a working
# `nvidia-smi`). Copy it up with scp (see 01_infra_setup.sh "Next steps"):
#   bash 02_serve_vllm.sh

set -euo pipefail

# ==== CONFIG (edit me) ====
MODEL_ID="Qwen/Qwen3.6-35B-A3B-FP8"
SERVED_MODEL_NAME="qwen3.6-35b-a3b-fp8"

# FP8 weights are ~35GB; on a 48GB L40S that leaves ~10-13GB for KV (see reference.md).
# We raise MAXLEN to 65536 (from 32768) to stop agentic runs overflowing the window
# (reasoning traces + big file observations inflate KV 2-10x). Two things make this fit:
#   - KV_CACHE_DTYPE=fp8 roughly halves KV memory (~2x effective context).
#   - MAX_NUM_SEQS lowered to 8 so a single long sequence's KV actually allocates.
# The model is hybrid-attention (Gated DeltaNet linear layers) so KV grows sub-linearly.
# If vLLM logs a KV/OOM error at startup, lower MAXLEN (49152) or MAX_NUM_SEQS first.
MAXLEN=65536
GPU_MEM_UTIL=0.92
MAX_NUM_SEQS=8
KV_CACHE_DTYPE=fp8
PORT=8000

VENV_DIR="${HOME}/vllm-venv"
HF_HOME="${HOME}/hf"
VLLM_SPEC="vllm>=0.24"   # >=0.24 required for --reasoning-parser + qwen3_xml tool parser
# ==== END CONFIG ====

# Base the records dir on THIS script's own directory (it may live anywhere on the box).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- inline recording (tee stdout/stderr to a timestamped log under records/) ----
TS="$(date -u +"%Y%m%dT%H%M%SZ")"
RECORD_DIR="${SCRIPT_DIR}/records"
mkdir -p "${RECORD_DIR}"
RAW_RECORD_FILE="${RECORD_DIR}/${TS}_serve_vllm.log"
exec > >(tee -a "${RAW_RECORD_FILE}") 2>&1
echo "record_name=serve_vllm"
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
  echo "Creating venv ..."
  python3 -m venv "${VENV_DIR}"
else
  echo "Reusing existing venv."
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo ""
echo "=== Installing vLLM (${VLLM_SPEC}) + huggingface_hub ==="
pip install -U pip wheel
pip install "${VLLM_SPEC}"
pip install -U huggingface_hub
python -c "import vllm; print('vllm', vllm.__version__)"

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
echo "=== vLLM serve (reasoning ON) ==="
echo "Model: ${MODEL_ID} (served as ${SERVED_MODEL_NAME})"
echo "MAXLEN=${MAXLEN} GPU_MEM_UTIL=${GPU_MEM_UTIL} MAX_NUM_SEQS=${MAX_NUM_SEQS} KV_CACHE_DTYPE=${KV_CACHE_DTYPE} PORT=${PORT}"
echo "Reasoning is ON (--reasoning-parser qwen3). Endpoint will be at http://127.0.0.1:${PORT}/v1"
echo "This command BLOCKS — wait for 'Application startup complete'."
echo ""

exec vllm serve "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size 1 \
  --max-model-len "${MAXLEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --port "${PORT}"
