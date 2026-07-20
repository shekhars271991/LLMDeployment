#!/usr/bin/env bash
# run_all.sh — One-shot Phase 1 orchestrator (run on your LOCAL Mac).
#
# Calls the phase-1 steps in order:
#   1) 01_infra_setup.sh   -> launch the g6e.2xlarge, write instance.env
#   2) copy 02/03 to the box, start vLLM (02_serve_vllm.sh) detached, wait until
#      the OpenAI-compatible endpoint is live
#   3) 03_run_benchmark.sh on the box -> SWE-bench + record
#   4) download records/results back to this machine
#   5) (optional) terminate the instance
#
# PREREQS: aws CLI configured; the private key at SSH_KEY_PATH; AMI/KEY/SG already
# filled in 01_infra_setup.sh (they are, reused from the speculative-decoding project).
#
# WHERE TO RUN: local Mac.  Run:  bash run_all.sh
#
# NOTE: this uses whatever config is baked into 02_serve_vllm.sh and
# 03_run_benchmark.sh (model, MAXLEN, SUBSET=lite, INSTANCE_LIMIT=5, etc.).
# Edit those scripts' CONFIG blocks first if you want a full SWE-bench Verified run.

set -euo pipefail

# ==== CONFIG (edit me) ====
PORT=8000                    # must match 02_serve_vllm.sh PORT
POLL_TIMEOUT_SECS=3600       # max wait for model download + vLLM startup (~35GB weights)
SSH_WAIT_SECS=300            # max wait for sshd to start accepting connections
AUTO_TERMINATE="false"       # "true" to terminate the instance after results download
# ==== END CONFIG ====

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- inline recording (tee stdout/stderr to a timestamped log under records/) ----
TS="$(date -u +"%Y%m%dT%H%M%SZ")"
RECORD_DIR="${SCRIPT_DIR}/records"
mkdir -p "${RECORD_DIR}"
RAW_RECORD_FILE="${RECORD_DIR}/${TS}_run_all.log"
exec > >(tee -a "${RAW_RECORD_FILE}") 2>&1
echo "record_name=run_all"
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

echo "=================================================="
echo "STEP 1/5 - Infra setup (launch instance)"
echo "=================================================="
bash "${SCRIPT_DIR}/01_infra_setup.sh"

INSTANCE_ENV="${SCRIPT_DIR}/instance.env"
if [[ ! -f "${INSTANCE_ENV}" ]]; then
  echo "ERROR: ${INSTANCE_ENV} was not written by 01_infra_setup.sh"
  exit 1
fi
# shellcheck source=instance.env
source "${INSTANCE_ENV}"
: "${INSTANCE_PUBLIC_IP:?missing INSTANCE_PUBLIC_IP}"
: "${SSH_USER:?missing SSH_USER}"
: "${SSH_KEY_PATH:?missing SSH_KEY_PATH}"
IP="${INSTANCE_PUBLIC_IP}"
REMOTE_USER="${SSH_USER}"

SSH_OPTS=(-i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15)

echo ""
echo "=================================================="
echo "STEP 2/5 - Copy scripts + start vLLM on ${IP}"
echo "=================================================="

echo "Waiting for SSH on ${IP} (up to ${SSH_WAIT_SECS}s) ..."
ssh_deadline=$(( $(date +%s) + SSH_WAIT_SECS ))
until ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${IP}" "echo ok" >/dev/null 2>&1; do
  if (( $(date +%s) > ssh_deadline )); then
    echo "ERROR: SSH not ready after ${SSH_WAIT_SECS}s."
    exit 1
  fi
  sleep 5
done
echo "SSH ready."

echo "Copying 02_serve_vllm.sh and 03_run_benchmark.sh to the box ..."
scp "${SSH_OPTS[@]}" \
  "${SCRIPT_DIR}/02_serve_vllm.sh" \
  "${SCRIPT_DIR}/03_run_benchmark.sh" \
  "${REMOTE_USER}@${IP}:~/"

echo "Starting vLLM (detached via nohup) on the box ..."
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${IP}" \
  "nohup bash ~/02_serve_vllm.sh > ~/serve_nohup.out 2>&1 & echo 'serve started, pid' \$!"

echo "Waiting for the vLLM endpoint http://127.0.0.1:${PORT}/v1/models (up to ${POLL_TIMEOUT_SECS}s) ..."
echo "(first run installs vLLM + downloads ~35GB of weights, so this can take a while)"
serve_deadline=$(( $(date +%s) + POLL_TIMEOUT_SECS ))
until ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${IP}" "curl -fsS http://127.0.0.1:${PORT}/v1/models >/dev/null 2>&1"; do
  if (( $(date +%s) > serve_deadline )); then
    echo "ERROR: vLLM did not become ready within ${POLL_TIMEOUT_SECS}s. Last serve log lines:"
    ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${IP}" "tail -n 40 ~/serve_nohup.out" || true
    exit 1
  fi
  if ! ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${IP}" "pgrep -f 'vllm serve' >/dev/null 2>&1 || pgrep -f '02_serve_vllm.sh' >/dev/null 2>&1"; then
    echo "ERROR: the vLLM serve process is no longer running (startup failed). Last log lines:"
    ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${IP}" "tail -n 40 ~/serve_nohup.out" || true
    exit 1
  fi
  sleep 15
done
echo "vLLM endpoint is live."

echo ""
echo "=================================================="
echo "STEP 3/5 - Run SWE-bench benchmark on ${IP}"
echo "=================================================="
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${IP}" "bash ~/03_run_benchmark.sh"

echo ""
echo "=================================================="
echo "STEP 4/5 - Download results to this machine"
echo "=================================================="
LOCAL_DL="${SCRIPT_DIR}/downloaded-results/${TS}"
mkdir -p "${LOCAL_DL}"
scp "${SSH_OPTS[@]}" -r "${REMOTE_USER}@${IP}:~/records" "${LOCAL_DL}/" || echo "WARN: could not copy ~/records"
scp "${SSH_OPTS[@]}" "${REMOTE_USER}@${IP}:~/serve_nohup.out" "${LOCAL_DL}/" || true
echo "Results downloaded to: ${LOCAL_DL}"

SUMMARY="$(find "${LOCAL_DL}" -name 'swebench_*.json' -type f 2>/dev/null | sort | tail -n1 || true)"
if [[ -n "${SUMMARY}" ]]; then
  echo "SWE-bench summary: ${SUMMARY}"
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('resolve_rate_pct=', d.get('resolve_rate_pct'), ' resolved=', d.get('resolved'), '/', d.get('instances_attempted'))" "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
fi

echo ""
echo "=================================================="
echo "STEP 5/5 - Teardown"
echo "=================================================="
if [[ "${AUTO_TERMINATE}" == "true" ]]; then
  echo "AUTO_TERMINATE=true -> terminating instance ${INSTANCE_ID:-?} ..."
  echo yes | bash "${SCRIPT_DIR}/04_terminate.sh"
else
  echo "Instance ${INSTANCE_ID:-?} (${IP}) is STILL RUNNING and billing."
  echo "Terminate it when you are done:"
  echo "  bash ${SCRIPT_DIR}/04_terminate.sh"
fi

echo ""
echo "=== run_all complete ==="
