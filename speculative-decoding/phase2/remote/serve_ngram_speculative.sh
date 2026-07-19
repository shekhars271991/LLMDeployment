#!/usr/bin/env bash
# serve_ngram_speculative.sh — vLLM with n-gram speculative decoding.
# Set NGRAM_NUM_SPEC_TOKENS in config.env (2, 4, or 8 for sweeps).
# Run on remote box: ./serve_ngram_speculative.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=serve_common.sh
source "${SCRIPT_DIR}/serve_common.sh"
start_recording "serve_ngram" "${SCRIPT_DIR}/records"

: "${NGRAM_NUM_SPEC_TOKENS:?Set NGRAM_NUM_SPEC_TOKENS in config.env (e.g. 2, 4, or 8)}"
: "${NGRAM_PROMPT_LOOKUP_MIN:=2}"
: "${NGRAM_PROMPT_LOOKUP_MAX:=5}"

SPEC_JSON=$(
  python - <<PY
import json
print(json.dumps({
    "method": "ngram",
    "num_speculative_tokens": int("${NGRAM_NUM_SPEC_TOKENS}"),
    "prompt_lookup_min": int("${NGRAM_PROMPT_LOOKUP_MIN}"),
    "prompt_lookup_max": int("${NGRAM_PROMPT_LOOKUP_MAX}"),
}))
PY
)

echo "=== vLLM serve (n-gram speculative) ==="
echo "Model: ${MODEL_ID}"
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
