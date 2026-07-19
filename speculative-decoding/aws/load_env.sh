#!/usr/bin/env bash
# load_env.sh — Source AWS config + runtime instance state into the current shell.
# Usage (from speculative-decoding/):
#   source aws/load_env.sh

_AWS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

set -a
# shellcheck source=config.env
source "${_AWS_DIR}/config.env"
if [[ -f "${_AWS_DIR}/instance.env" ]]; then
  # shellcheck source=instance.env
  source "${_AWS_DIR}/instance.env"
fi
set +a

unset _AWS_DIR
