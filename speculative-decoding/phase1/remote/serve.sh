#!/usr/bin/env bash
# serve.sh — Start vLLM OpenAI-compatible server (baseline, no extra optimizations).
# Prerequisite: setup_venv.sh, download_weights.sh (optional but recommended).
# Run on remote box: bash serve.sh
# Wait for: "Application startup complete" / Uvicorn on port ${PORT}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/record.sh
source "${SCRIPT_DIR}/../common/record.sh"
start_recording "serve" "${SCRIPT_DIR}/records"

# shellcheck source=config.env
source "${SCRIPT_DIR}/config.env"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export HF_HOME

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
  unset CUDA_VISIBLE_DEVICES
  echo "CUDA_VISIBLE_DEVICES is unset (all GPUs visible)"
fi

if [[ "${NCCL_P2P_DISABLE:-}" == "1" ]]; then
  export NCCL_P2P_DISABLE=1
  echo "NCCL_P2P_DISABLE=1 (TP=2 PCIe workaround)"
else
  unset NCCL_P2P_DISABLE
fi

EXTRA_ARGS=()
if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-custom-all-reduce)
  echo "Using --disable-custom-all-reduce"
fi

echo "=== vLLM serve (baseline) ==="
echo "Model: ${MODEL_ID}"
echo "TP=${TP} MAXLEN=${MAXLEN} GPU_MEM_UTIL=${GPU_MEM_UTIL} PORT=${PORT}"
echo "Prefix caching: disabled (uncached baseline)"
echo ""

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

exec vllm serve "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAXLEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --no-enable-prefix-caching \
  --port "${PORT}" \
  "${EXTRA_ARGS[@]}"
