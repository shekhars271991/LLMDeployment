#!/usr/bin/env bash
# 01_infra_setup.sh — Provide ONE EC2 GPU instance for Phase 2 (single-node sweeps).
#
# WHAT IT DOES: Phase 2 runs on the SAME single L40S as Phase 1 — the sweep just
# re-serves the box with one changed lever at a time, so ideally you REUSE the
# Phase 1 instance instead of paying for a second one. This script therefore:
#   (a) REUSE PATH (default): if an instance.env already exists next to this
#       script (copy Phase 1's here), it validates that the instance is alive and
#       reuses it. Nothing new is launched.
#   (b) LAUNCH PATH: if there is no instance.env, it launches a fresh
#       g6e.2xlarge (1x NVIDIA L40S 48 GB) with a phase2 name tag, waits for
#       status checks, and writes instance.env next to this script.
#
# WHERE TO RUN: on your LOCAL Mac, with the `aws` CLI installed and configured
# (aws configure / SSO). It does NOT run on the GPU box.
#
# To reuse the Phase 1 box:
#   cp ../phase-1-single-gpu-mvp/instance.env ./instance.env
#   bash 01_infra_setup.sh
# To launch a dedicated Phase 2 box:
#   rm -f ./instance.env && bash 01_infra_setup.sh

set -euo pipefail

# ==== CONFIG (edit me) ====
AWS_REGION="us-east-1"
export AWS_DEFAULT_REGION="${AWS_REGION}"

INSTANCE_TYPE="g6e.2xlarge"                 # 1x NVIDIA L40S 48 GB, 8 vCPU, single node

# Region-specific Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04). Same as Phase 1.
AMI_ID="ami-0f14e146be5a0b944"   # us-east-1 Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04)

KEY_NAME="skrgpuuseast"                       # existing EC2 key-pair name (same as Phase 1)
SSH_KEY_PATH="${HOME}/.ssh/${KEY_NAME}.pem"  # local path to the matching private key

# Security group (same as Phase 1). NOTE: permissive temporary group (all TCP open).
# Fine for a short-lived run; tighten to SSH-from-your-IP + an SSH tunnel for real use.
SG_ID="sg-09c9aa5873bbd5f48"

SUBNET_ID=""                                 # leave empty to let EC2 pick a default subnet
ROOT_VOLUME_GB=350                           # room for FP8 + AWQ weights + SWE-bench docker images
INSTANCE_NAME_TAG="coding-agent-phase2-qwen36-35b"
SSH_USER="ubuntu"
# ==== END CONFIG ====

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- inline recording (tee stdout/stderr to a timestamped log under records/) ----
TS="$(date -u +"%Y%m%dT%H%M%SZ")"
RECORD_DIR="${SCRIPT_DIR}/records"
mkdir -p "${RECORD_DIR}"
RAW_RECORD_FILE="${RECORD_DIR}/${TS}_infra_setup.log"
exec > >(tee -a "${RAW_RECORD_FILE}") 2>&1
echo "record_name=infra_setup"
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

if ! command -v aws &>/dev/null; then
  echo "ERROR: aws CLI not found. Install and run 'aws configure' first."
  exit 1
fi

# ---- (a) REUSE PATH: an instance.env already exists ----
if [[ -f "${INSTANCE_ENV}" ]]; then
  echo "=== Found existing instance.env — reuse path ==="
  # shellcheck source=instance.env
  source "${INSTANCE_ENV}"
  : "${INSTANCE_ID:?instance.env present but INSTANCE_ID is empty}"
  REUSE_REGION="${AWS_REGION:-us-east-1}"

  STATE="$(aws ec2 describe-instances \
    --region "${REUSE_REGION}" \
    --instance-ids "${INSTANCE_ID}" \
    --query 'Reservations[0].Instances[0].State.Name' \
    --output text 2>/dev/null || true)"

  if [[ "${STATE}" == "running" ]]; then
    # Refresh the public IP (it can change across stop/start).
    PUBLIC_IP="$(aws ec2 describe-instances \
      --region "${REUSE_REGION}" \
      --instance-ids "${INSTANCE_ID}" \
      --query 'Reservations[0].Instances[0].PublicIpAddress' \
      --output text)"
    cat > "${INSTANCE_ENV}" <<EOF
# Generated/refreshed by phase-2 01_infra_setup.sh — do NOT commit (contains a public IP).
export INSTANCE_ID="${INSTANCE_ID}"
export INSTANCE_PUBLIC_IP="${PUBLIC_IP}"
export AWS_REGION="${REUSE_REGION}"
export INSTANCE_TYPE="${INSTANCE_TYPE}"
export SSH_USER="${SSH_USER}"
export SSH_KEY_PATH="${SSH_KEY_PATH}"
EOF
    echo "Reusing running instance ${INSTANCE_ID} at ${PUBLIC_IP}."
    echo "Saved refreshed IP to: ${INSTANCE_ENV}"
    echo "Next: bash run_sweep.sh   (or copy 02_serve.sh + 03_run_benchmark.sh up and run a single config)."
    exit 0
  fi

  echo "WARNING: instance ${INSTANCE_ID} is in state '${STATE:-unknown}', not 'running'."
  echo "It cannot be reused. Remove instance.env to launch a fresh Phase 2 box:"
  echo "  rm -f ${INSTANCE_ENV} && bash 01_infra_setup.sh"
  exit 1
fi

# ---- (b) LAUNCH PATH: no instance.env, launch a dedicated Phase 2 box ----
echo "=== No instance.env — launch path ==="
echo "=== Preflight checks ==="
for var in AMI_ID KEY_NAME SG_ID; do
  if [[ "${!var}" == *"REPLACE_ME"* ]]; then
    echo "ERROR: Set ${var} in the CONFIG block (still contains REPLACE_ME)."
    exit 1
  fi
done

echo ""
echo "=== On-Demand G and VT vCPU quota (informational) ==="
echo "(g6e.2xlarge needs 8 vCPUs of 'Running On-Demand G and VT instances' quota)"
aws service-quotas get-service-quota \
  --region "${AWS_REGION}" \
  --service-code ec2 \
  --quota-code L-DB2E81BA \
  --query 'Quota.Value' --output text 2>/dev/null \
  | awk '{print "Current G/VT vCPU quota:", $0}' || true

echo ""
echo "=== Launching ${INSTANCE_TYPE} in ${AWS_REGION} ==="

RUN_ARGS=(
  --region "${AWS_REGION}"
  --image-id "${AMI_ID}"
  --instance-type "${INSTANCE_TYPE}"
  --key-name "${KEY_NAME}"
  --security-group-ids "${SG_ID}"
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${ROOT_VOLUME_GB},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]"
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME_TAG}}]"
  --count 1
)

if [[ -n "${SUBNET_ID}" ]]; then
  RUN_ARGS+=(--subnet-id "${SUBNET_ID}")
fi

INSTANCE_ID="$(aws ec2 run-instances "${RUN_ARGS[@]}" --query 'Instances[0].InstanceId' --output text)"
echo "InstanceId: ${INSTANCE_ID}"

echo "Waiting for instance to be running ..."
aws ec2 wait instance-running --region "${AWS_REGION}" --instance-ids "${INSTANCE_ID}"

echo "Waiting for EC2 system and instance status checks (this can take a few minutes) ..."
aws ec2 wait instance-status-ok --region "${AWS_REGION}" --instance-ids "${INSTANCE_ID}"

PUBLIC_IP="$(aws ec2 describe-instances \
  --region "${AWS_REGION}" \
  --instance-ids "${INSTANCE_ID}" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)"

cat > "${INSTANCE_ENV}" <<EOF
# Generated by phase-2 01_infra_setup.sh — do NOT commit (contains a public IP).
export INSTANCE_ID="${INSTANCE_ID}"
export INSTANCE_PUBLIC_IP="${PUBLIC_IP}"
export AWS_REGION="${AWS_REGION}"
export INSTANCE_TYPE="${INSTANCE_TYPE}"
export SSH_USER="${SSH_USER}"
export SSH_KEY_PATH="${SSH_KEY_PATH}"
EOF

echo ""
echo "=== Launch complete ==="
echo "INSTANCE_ID=${INSTANCE_ID}"
echo "PUBLIC_IP=${PUBLIC_IP}"
echo "Saved to: ${INSTANCE_ENV}"
echo ""
echo "=== Next steps ==="
echo "# Run the full one-lever-at-a-time sweep from your Mac:"
echo "bash run_sweep.sh"
echo ""
echo "# Or drive a single config by hand:"
echo "scp -i \"${SSH_KEY_PATH}\" 02_serve.sh 03_run_benchmark.sh -r configs ${SSH_USER}@${PUBLIC_IP}:~/"
echo "ssh -i \"${SSH_KEY_PATH}\" ${SSH_USER}@${PUBLIC_IP}"
echo "#   then, on the box:"
echo "CONFIG=configs/2b_prefix_chunked.env bash 02_serve.sh   # wait for 'Application startup complete'"
echo "CONFIG=configs/2b_prefix_chunked.env bash 03_run_benchmark.sh   # in a second terminal"
echo ""
echo "# Reach the endpoint from your Mac (port 8000 is not open publicly), tunnel:"
echo "ssh -i \"${SSH_KEY_PATH}\" -L 8000:localhost:8000 ${SSH_USER}@${PUBLIC_IP}"
