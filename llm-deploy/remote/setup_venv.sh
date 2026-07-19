#!/usr/bin/env bash
# setup_venv.sh — Create fresh venv and install vLLM (>=0.8.5 for Qwen3 AWQ).
# Prerequisite: SSH on GPU box; nvidia-smi works (Deep Learning AMI).
# Run on remote box: bash setup_venv.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/record.sh
source "${SCRIPT_DIR}/../common/record.sh"
start_recording "setup_venv" "${SCRIPT_DIR}/records"

# shellcheck source=config.env
source "${SCRIPT_DIR}/config.env"

echo "=== GPU check ==="
if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. Use Deep Learning AMI or install NVIDIA drivers."
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
echo "=== Creating venv at ${VENV_DIR} ==="
python3 -m venv --clear "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

pip install -U pip wheel
pip install "vllm>=0.8.5"

echo ""
echo "=== vLLM version ==="
python -c "import vllm; print('vllm', vllm.__version__)"

echo ""
echo "Done. Activate before other scripts:"
echo "  source ${VENV_DIR}/bin/activate"
