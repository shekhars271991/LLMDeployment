#!/usr/bin/env bash
# download_weights.sh — Pre-download Qwen3-32B-AWQ from Hugging Face into HF_HOME.
# Prerequisite: setup_venv.sh (venv active or script activates it).
# Run on remote box: bash download_weights.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/record.sh
source "${SCRIPT_DIR}/../common/record.sh"
start_recording "download_weights" "${SCRIPT_DIR}/records"

# shellcheck source=config.env
source "${SCRIPT_DIR}/config.env"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export HF_HOME
mkdir -p "${HF_HOME}"

pip install -q -U "huggingface_hub"

if ! command -v hf &>/dev/null; then
  echo "ERROR: Hugging Face 'hf' CLI was not installed in ${VENV_DIR}."
  exit 1
fi

echo "=== Hugging Face download: ${MODEL_ID} ==="
echo "HF_HOME=${HF_HOME}"
echo "If rate-limited, run: hf auth login"
echo ""

hf download "${MODEL_ID}"

echo ""
echo "Download complete. Weights cached under ${HF_HOME}"
