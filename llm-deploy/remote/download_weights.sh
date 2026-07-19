#!/usr/bin/env bash
# download_weights.sh — Pre-download Qwen3-32B-AWQ from Hugging Face into HF_HOME.
# Prerequisite: setup_venv.sh (venv active or script activates it).
# Run on remote box: bash download_weights.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.env
source "${SCRIPT_DIR}/config.env"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export HF_HOME
mkdir -p "${HF_HOME}"

pip install -q "huggingface_hub[cli]"

echo "=== Hugging Face download: ${MODEL_ID} ==="
echo "HF_HOME=${HF_HOME}"
echo "If rate-limited, run: huggingface-cli login"
echo ""

huggingface-cli download "${MODEL_ID}"

echo ""
echo "Download complete. Weights cached under ${HF_HOME}"
