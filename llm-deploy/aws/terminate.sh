#!/usr/bin/env bash
# terminate.sh — Terminate the EC2 instance created by launch_instance.sh.
# Prerequisite: aws/instance.env with INSTANCE_ID.
# Run: bash aws/terminate.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/record.sh
source "${SCRIPT_DIR}/../common/record.sh"
start_recording "terminate" "${SCRIPT_DIR}/../records/aws"

# shellcheck source=config.env
source "${SCRIPT_DIR}/config.env"

INSTANCE_ENV="${SCRIPT_DIR}/instance.env"
if [[ ! -f "${INSTANCE_ENV}" ]]; then
  echo "ERROR: ${INSTANCE_ENV} not found. Nothing to terminate."
  exit 1
fi

# shellcheck source=instance.env
source "${INSTANCE_ENV}"

if [[ -z "${INSTANCE_ID:-}" ]]; then
  echo "ERROR: INSTANCE_ID empty in ${INSTANCE_ENV}"
  exit 1
fi

echo "This will TERMINATE instance ${INSTANCE_ID} (${INSTANCE_PUBLIC_IP:-unknown IP})."
read -r -p "Type 'yes' to confirm: " CONFIRM
if [[ "${CONFIRM}" != "yes" ]]; then
  echo "Aborted."
  exit 0
fi

aws ec2 terminate-instances --region "${AWS_REGION}" --instance-ids "${INSTANCE_ID}"
echo "Terminate requested. EBS root volume deleted if DeleteOnTermination=true."
rm -f "${INSTANCE_ENV}"
