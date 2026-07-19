#!/usr/bin/env bash
# Shared preamble for Phase 2 vLLM serve launchers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../common/record.sh
source "${SCRIPT_DIR}/../../common/record.sh"

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

preflight_cuda() {
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
}
