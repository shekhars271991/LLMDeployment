#!/usr/bin/env bash
# ssh_tunnel.sh — SSH into the GPU box with port-forward 8000→8000 for vLLM.
# Prerequisite: launch_instance.sh (or set INSTANCE_PUBLIC_IP in aws/instance.env).
# Run: bash aws/ssh_tunnel.sh
# Keeps session open; vLLM on remote :8000 reachable at http://localhost:8000 on your Mac.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/record.sh
source "${SCRIPT_DIR}/../common/record.sh"
start_recording "ssh_tunnel" "${SCRIPT_DIR}/../records/aws"

# shellcheck source=config.env
source "${SCRIPT_DIR}/config.env"

INSTANCE_ENV="${SCRIPT_DIR}/instance.env"
if [[ -f "${INSTANCE_ENV}" ]]; then
  # shellcheck source=instance.env
  source "${INSTANCE_ENV}"
fi

if [[ -z "${INSTANCE_PUBLIC_IP:-}" ]] || [[ "${INSTANCE_PUBLIC_IP}" == "None" ]]; then
  echo "ERROR: INSTANCE_PUBLIC_IP not set. Run launch_instance.sh or edit aws/instance.env"
  exit 1
fi

if [[ ! -f "${SSH_KEY_PATH}" ]]; then
  echo "ERROR: SSH key not found at ${SSH_KEY_PATH}"
  echo "Set SSH_KEY_PATH in aws/config.env"
  exit 1
fi

chmod 400 "${SSH_KEY_PATH}" 2>/dev/null || true

echo "Connecting to ${SSH_USER}@${INSTANCE_PUBLIC_IP} (tunnel localhost:8000 → remote:8000)"
echo "Press Ctrl+D or 'exit' to close."
echo ""

exec ssh -i "${SSH_KEY_PATH}" \
  -o StrictHostKeyChecking=accept-new \
  -L 8000:localhost:8000 \
  "${SSH_USER}@${INSTANCE_PUBLIC_IP}"
