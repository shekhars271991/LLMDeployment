#!/usr/bin/env bash
# load_env.sh — Source Phase 2 remote config into the current shell.
# Usage (on EC2):
#   cd ~/phase2/remote && source load_env.sh

_REMOTE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

if [[ ! -f "${_REMOTE_DIR}/config.env" ]]; then
  echo "ERROR: ${_REMOTE_DIR}/config.env not found." >&2
  echo "Create it: cp config.env.example config.env" >&2
  return 1 2>/dev/null || exit 1
fi

set -a
# shellcheck source=config.env
source "${_REMOTE_DIR}/config.env"
set +a

unset _REMOTE_DIR
