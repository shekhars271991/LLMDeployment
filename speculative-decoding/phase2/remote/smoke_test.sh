#!/usr/bin/env bash
# smoke_test.sh — Quick API check for Phase 2 server.
# Run on remote box: ./smoke_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../common/record.sh
source "${SCRIPT_DIR}/../../common/record.sh"
start_recording "smoke_test" "${SCRIPT_DIR}/records"

# shellcheck source=config.env
source "${SCRIPT_DIR}/config.env"

BASE="http://127.0.0.1:${PORT}"

echo "=== Phase 2 smoke test ==="
echo "GET ${BASE}/v1/models"
curl -fsS "${BASE}/v1/models" | python3 -m json.tool | head -20
echo ""

echo "POST ${BASE}/v1/chat/completions (greedy, short)"
curl -fsS "${BASE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${SERVED_MODEL_NAME}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly: ok\"}],
    \"max_tokens\": 8,
    \"temperature\": 0,
    \"chat_template_kwargs\": {\"enable_thinking\": false}
  }" | python3 -m json.tool | head -30

echo ""
echo "GET ${BASE}/metrics (speculative counters)"
curl -fsS "${BASE}/metrics" | python3 -c '
import sys
for line in sys.stdin:
    if "spec_decode" in line and not line.startswith("#"):
        print(line.rstrip())
' | head -10 || true

echo ""
echo "Smoke test complete."
