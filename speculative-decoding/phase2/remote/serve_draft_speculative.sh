#!/usr/bin/env bash
# serve_draft_speculative.sh — vLLM with Qwen3-0.6B draft-model speculation.
# Prerequisite: download_draft_weights.sh
# Run on remote box: ./serve_draft_speculative.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=serve_common.sh
source "${SCRIPT_DIR}/serve_common.sh"
start_recording "serve_draft" "${SCRIPT_DIR}/records"

: "${DRAFT_MODEL_ID:?Set DRAFT_MODEL_ID in config.env}"
: "${DRAFT_NUM_SPEC_TOKENS:?Set DRAFT_NUM_SPEC_TOKENS in config.env}"

SPEC_JSON=$(
  python - <<PY
import json
print(json.dumps({
    "method": "draft_model",
    "model": "${DRAFT_MODEL_ID}",
    "num_speculative_tokens": int("${DRAFT_NUM_SPEC_TOKENS}"),
    "draft_tensor_parallel_size": int("${DRAFT_TP:-1}"),
}))
PY
)

echo "=== vLLM serve (draft-model speculative) ==="
echo "Target model: ${MODEL_ID}"
echo "Draft model: ${DRAFT_MODEL_ID}"
echo "TP=${TP} MAXLEN=${MAXLEN} GPU_MEM_UTIL=${GPU_MEM_UTIL} PORT=${PORT}"
echo "Prefix caching: disabled"
echo "Speculative config: ${SPEC_JSON}"
echo ""

preflight_cuda
echo ""

exec vllm serve "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAXLEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --no-enable-prefix-caching \
  --port "${PORT}" \
  --speculative-config "${SPEC_JSON}" \
  "${EXTRA_ARGS[@]}"
