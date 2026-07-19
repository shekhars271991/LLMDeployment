#!/usr/bin/env bash
# smoke_test.sh — Hit /v1/models and one chat completion.
# Prerequisite: serve.sh running in another terminal.
# Run on remote box: bash smoke_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/record.sh
source "${SCRIPT_DIR}/../common/record.sh"
start_recording "smoke_test" "${SCRIPT_DIR}/records"

# shellcheck source=config.env
source "${SCRIPT_DIR}/config.env"

BASE="http://127.0.0.1:${PORT}"

echo "=== GET ${BASE}/v1/models ==="
curl -sf "${BASE}/v1/models" | python3 -m json.tool

echo ""
echo "=== POST ${BASE}/v1/chat/completions ==="
curl -sf "${BASE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${SERVED_MODEL_NAME}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Say hi in one short sentence.\"}],
    \"max_tokens\": 32,
    \"chat_template_kwargs\": {\"enable_thinking\": false}
  }" | python3 -m json.tool

echo ""
echo "Smoke test OK."
