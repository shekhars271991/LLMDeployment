#!/usr/bin/env bash
# download_draft_weights.sh — Pre-download draft model for speculative decoding.
# Prerequisite: venv from Phase 1 setup_venv.sh
# Run on remote box: ./download_draft_weights.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../common/record.sh
source "${SCRIPT_DIR}/../../common/record.sh"
start_recording "download_draft_weights" "${SCRIPT_DIR}/records"

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

echo "=== Hugging Face download (draft model): ${DRAFT_MODEL_ID} ==="
echo "HF_HOME=${HF_HOME}"
echo "If rate-limited, run: hf auth login"
echo ""

hf download "${DRAFT_MODEL_ID}"

echo ""
echo "Draft model cached under ${HF_HOME}"
