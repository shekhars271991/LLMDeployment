#!/usr/bin/env bash
# serve.sh — Phase 2 control server (no speculative decoding).
# Prerequisite: venv + target weights from Phase 1 or download_weights on EC2.
# Run on remote box: ./serve.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=serve_common.sh
source "${SCRIPT_DIR}/serve_common.sh"
start_recording "serve_control" "${SCRIPT_DIR}/records"

echo "=== vLLM serve (Phase 2 control) ==="
echo "Model: ${MODEL_ID}"
echo "TP=${TP} MAXLEN=${MAXLEN} GPU_MEM_UTIL=${GPU_MEM_UTIL} PORT=${PORT}"
echo "Prefix caching: disabled"
echo "Speculative decoding: disabled"
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
  "${EXTRA_ARGS[@]}"
