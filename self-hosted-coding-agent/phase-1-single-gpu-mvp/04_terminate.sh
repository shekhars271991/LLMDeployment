#!/usr/bin/env bash
# 04_terminate.sh — Terminate the EC2 instance created by 01_infra_setup.sh.
#
# WHAT IT DOES: terminates the single GPU EC2 instance recorded in instance.env,
# which STOPS the compute billing for this Phase 1 run. The EBS root volume is
# also deleted (DeleteOnTermination=true, per the launcher), so its storage cost
# stops too. On success it removes instance.env.
#
# WHERE TO RUN: on your LOCAL Mac, with the `aws` CLI installed and configured
# (aws configure / SSO). It does NOT run on the GPU box.
#
# PREREQUISITE: instance.env must exist next to this script (written by
# 01_infra_setup.sh). Without it there is nothing to terminate.
#
#   bash 04_terminate.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- inline recording (tee stdout/stderr to a timestamped log under records/) ----
TS="$(date -u +"%Y%m%dT%H%M%SZ")"
RECORD_DIR="${SCRIPT_DIR}/records"
mkdir -p "${RECORD_DIR}"
RAW_RECORD_FILE="${RECORD_DIR}/${TS}_terminate.log"
exec > >(tee -a "${RAW_RECORD_FILE}") 2>&1
echo "record_name=terminate"
echo "started_utc=${TS}"
echo "host=$(hostname)"
echo "command=$0 $*"
echo "--- output ---"
_finish_recording() {
  local status=$?
  trap - EXIT
  echo "--- end ---"
  echo "finished_utc=$(date -u +"%Y%m%dT%H%M%SZ")"
  echo "exit_status=${status}"
  echo "raw_record=${RAW_RECORD_FILE}"
  exit "${status}"
}
trap _finish_recording EXIT
# ---- end inline recording ----

INSTANCE_ENV="${SCRIPT_DIR}/instance.env"

if [[ ! -f "${INSTANCE_ENV}" ]]; then
  echo "ERROR: ${INSTANCE_ENV} not found. Nothing to terminate."
  echo "(Was the instance already terminated, or has 01_infra_setup.sh not run yet?)"
  exit 1
fi

# shellcheck source=instance.env
source "${INSTANCE_ENV}"

if [[ -z "${INSTANCE_ID:-}" ]]; then
  echo "ERROR: INSTANCE_ID empty or unset in ${INSTANCE_ENV}. Nothing to terminate."
  exit 1
fi

if ! command -v aws &>/dev/null; then
  echo "ERROR: aws CLI not found. Install and run 'aws configure' first."
  exit 1
fi

# Fall back to us-east-1 if the region was somehow not recorded.
if [[ -z "${AWS_REGION:-}" ]]; then
  echo "NOTE: AWS_REGION empty in ${INSTANCE_ENV}; falling back to us-east-1."
  AWS_REGION="us-east-1"
fi

echo "=== About to terminate ==="
echo "INSTANCE_ID:        ${INSTANCE_ID}"
echo "INSTANCE_PUBLIC_IP: ${INSTANCE_PUBLIC_IP:-unknown}"
echo "AWS_REGION:         ${AWS_REGION}"
echo ""
echo "This will TERMINATE the instance above and STOP its compute billing."
read -r -p "Type 'yes' to confirm: " CONFIRM
if [[ "${CONFIRM}" != "yes" ]]; then
  echo "Aborted. Nothing was terminated."
  exit 0
fi

aws ec2 terminate-instances --region "${AWS_REGION}" --instance-ids "${INSTANCE_ID}"
echo ""
echo "Terminate requested for ${INSTANCE_ID}."
echo "The EBS root volume is deleted because DeleteOnTermination=true (per 01_infra_setup.sh)."

rm -f "${INSTANCE_ENV}"
echo "Removed ${INSTANCE_ENV}."
