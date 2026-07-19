#!/usr/bin/env bash
# check_quota.sh — Print G/VT instance vCPU quota vs what INSTANCE_TYPE needs.
# Prerequisite: aws CLI configured (aws configure); edit aws/config.env first.
# Run: bash aws/check_quota.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/record.sh
source "${SCRIPT_DIR}/../common/record.sh"
start_recording "check_quota" "${SCRIPT_DIR}/../records/aws"

# shellcheck source=config.env
source "${SCRIPT_DIR}/config.env"

if ! command -v aws &>/dev/null; then
  echo "ERROR: aws CLI not found. Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
  exit 1
fi

# vCPUs required per instance type (case = portable on macOS bash 3.2)
lookup_vcpus() {
  case "$1" in
    g6e.xlarge)  echo 4 ;;
    g6e.12xlarge) echo 48 ;;
    g5.12xlarge) echo 48 ;;
    g5.xlarge)   echo 4 ;;
    g6.xlarge)   echo 4 ;;
    *)           echo unknown ;;
  esac
}

REQUIRED="$(lookup_vcpus "${INSTANCE_TYPE}")"
if [[ "${REQUIRED}" == "unknown" ]]; then
  echo "WARN: vCPU count not in script lookup table for ${INSTANCE_TYPE}."
  echo "      Check AWS instance specs and compare to quota below."
  REQUIRED="?"
fi

echo "=== G/VT vCPU quota check (${AWS_REGION}) ==="
echo "Target instance type: ${INSTANCE_TYPE} (needs ~${REQUIRED} vCPUs)"
echo ""

QUOTA_CODE="L-DB2E81BA"
QUOTA_NAME="Running On-Demand G and VT instances"

QUOTA_JSON="$(aws service-quotas get-service-quota \
  --region "${AWS_REGION}" \
  --service-code ec2 \
  --quota-code "${QUOTA_CODE}" \
  2>/dev/null || true)"

if [[ -z "${QUOTA_JSON}" ]]; then
  echo "Could not fetch quota via API (permissions or quota code changed)."
  echo "Check manually: AWS Console → Service Quotas → EC2 → '${QUOTA_NAME}'"
  exit 0
fi

LIMIT="$(echo "${QUOTA_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin)['Quota']['Value'])")"
echo "Current on-demand G/VT vCPU limit: ${LIMIT}"

if [[ "${REQUIRED}" != "?" ]] && python3 -c "exit(0 if float('${LIMIT}') >= float('${REQUIRED}') else 1)"; then
  echo "OK: quota appears sufficient for one ${INSTANCE_TYPE}."
else
  if [[ "${REQUIRED}" != "?" ]]; then
    echo "NOT OK: need at least ${REQUIRED} vCPUs. Request increase in Service Quotas."
  fi
fi

echo ""
echo "Spot quota (if using spot): 'All G and VT Spot Instance Requests' — request separately if needed."
