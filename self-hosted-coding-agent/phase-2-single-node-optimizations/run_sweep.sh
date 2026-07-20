#!/usr/bin/env bash
# run_sweep.sh — One-shot Phase 2 sweep orchestrator (run on your LOCAL Mac).
#
# The Phase 2 analog of Phase 1's run_all.sh, but instead of one config it walks
# the sub-phase configs ONE LEVER AT A TIME on the SAME single GPU box. For each
# config it:
#   1) (re)starts the engine on the box with that config's levers (02_serve.sh),
#      detached, and waits for the OpenAI-compatible endpoint to go live,
#   2) runs SWE-bench for that config (03_run_benchmark.sh) and records a
#      config-tagged summary,
#   3) stops the engine so the next lever starts from a clean server.
# Then it downloads all records and prints the per-lever delta table
# (compare_results.py), which is the Phase 2 §3.7 deliverable.
#
# PREREQS: run 01_infra_setup.sh first (reuse the Phase 1 box or launch one);
# aws CLI + the private key configured.
#
# WHERE TO RUN: local Mac.  bash run_sweep.sh
#
# Edit CONFIGS below to choose which levers to sweep. Each config file bakes in
# its own SUBSET/INSTANCE_LIMIT (start with SUBSET=lite for a fast smoke sweep,
# then rerun the winners with SUBSET=verified for the headline deltas).

set -euo pipefail

# ==== CONFIG (edit me) ====
PORT=8000                    # must match 02_serve.sh PORT
POLL_TIMEOUT_SECS=3600       # max wait for model download + engine startup per config
SSH_WAIT_SECS=300            # max wait for sshd
AUTO_TERMINATE="false"       # "true" to terminate the box after the sweep

# The sweep list. Baseline FIRST so it becomes the reference for all deltas.
# Comment out lines to skip levers; reorder freely (baseline should stay first).
CONFIGS=(
  "configs/2a_fp8_baseline.env"
  "configs/2a_fp8_kv.env"
  "configs/2a_int4_awq.env"
  "configs/2b_prefix_chunked.env"
  "configs/2c_batching_cudagraph.env"
  "configs/2d_spec_mtp.env"
  "configs/2d_spec_ngram.env"
  "configs/2e_reasoning_low.env"
  "configs/2f_sglang.env"
)
# ==== END CONFIG ====

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- inline recording (tee stdout/stderr to a timestamped log under records/) ----
TS="$(date -u +"%Y%m%dT%H%M%SZ")"
RECORD_DIR="${SCRIPT_DIR}/records"
mkdir -p "${RECORD_DIR}"
RAW_RECORD_FILE="${RECORD_DIR}/${TS}_run_sweep.log"
exec > >(tee -a "${RAW_RECORD_FILE}") 2>&1
echo "record_name=run_sweep"
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
  echo "ERROR: ${INSTANCE_ENV} not found. Run 01_infra_setup.sh first"
  echo "(reuse the Phase 1 box: cp ../phase-1-single-gpu-mvp/instance.env ./instance.env && bash 01_infra_setup.sh)."
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

echo "=================================================="
echo "STEP 1 - Wait for SSH + copy scripts/configs to ${IP}"
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

echo "Copying 02_serve.sh, 03_run_benchmark.sh and configs/ to the box ..."
scp "${SSH_OPTS[@]}" \
  "${SCRIPT_DIR}/02_serve.sh" \
  "${SCRIPT_DIR}/03_run_benchmark.sh" \
  "${REMOTE_USER}@${IP}:~/"
scp "${SSH_OPTS[@]}" -r "${SCRIPT_DIR}/configs" "${REMOTE_USER}@${IP}:~/"

stop_engine() {
  echo "Stopping any running engine on the box ..."
  ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${IP}" \
    "pkill -f 'vllm serve' 2>/dev/null; pkill -f 'sglang.launch_server' 2>/dev/null; pkill -f '02_serve.sh' 2>/dev/null; sleep 5; true"
}

# Ensure we always leave the box without a stray server holding the GPU.
trap 'stop_engine || true; _finish_recording' EXIT

for CONFIG in "${CONFIGS[@]}"; do
  CONFIG_NAME="$(basename "${CONFIG}" .env)"
  echo ""
  echo "=================================================="
  echo "LEVER ${CONFIG_NAME} - serve + benchmark"
  echo "=================================================="

  stop_engine

  echo "Starting engine for ${CONFIG_NAME} (detached via nohup) ..."
  ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${IP}" \
    "CONFIG=${CONFIG} PORT=${PORT} nohup bash ~/02_serve.sh > ~/serve_${CONFIG_NAME}.out 2>&1 & echo 'serve started, pid' \$!"

  echo "Waiting for endpoint http://127.0.0.1:${PORT}/v1/models (up to ${POLL_TIMEOUT_SECS}s) ..."
  serve_deadline=$(( $(date +%s) + POLL_TIMEOUT_SECS ))
  until ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${IP}" "curl -fsS http://127.0.0.1:${PORT}/v1/models >/dev/null 2>&1"; do
    if (( $(date +%s) > serve_deadline )); then
      echo "ERROR: engine not ready within ${POLL_TIMEOUT_SECS}s for ${CONFIG_NAME}. Last log lines:"
      ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${IP}" "tail -n 40 ~/serve_${CONFIG_NAME}.out" || true
      echo "Skipping ${CONFIG_NAME} and moving on."
      continue 2
    fi
    if ! ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${IP}" "pgrep -f 'vllm serve' >/dev/null 2>&1 || pgrep -f 'sglang.launch_server' >/dev/null 2>&1 || pgrep -f '02_serve.sh' >/dev/null 2>&1"; then
      echo "ERROR: engine process for ${CONFIG_NAME} died during startup. Last log lines:"
      ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${IP}" "tail -n 40 ~/serve_${CONFIG_NAME}.out" || true
      echo "Skipping ${CONFIG_NAME} and moving on."
      continue 2
    fi
    sleep 15
  done
  echo "Endpoint live for ${CONFIG_NAME}."

  echo "Running SWE-bench for ${CONFIG_NAME} ..."
  ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${IP}" "CONFIG=${CONFIG} bash ~/03_run_benchmark.sh" \
    || echo "WARN: benchmark for ${CONFIG_NAME} exited non-zero; continuing sweep."
done

stop_engine
trap '_finish_recording' EXIT

echo ""
echo "=================================================="
echo "STEP 3 - Download all records to this machine"
echo "=================================================="
LOCAL_DL="${SCRIPT_DIR}/downloaded-results/${TS}"
mkdir -p "${LOCAL_DL}"
scp "${SSH_OPTS[@]}" -r "${REMOTE_USER}@${IP}:~/records" "${LOCAL_DL}/" || echo "WARN: could not copy ~/records"
scp "${SSH_OPTS[@]}" "${REMOTE_USER}@${IP}:~/serve_*.out" "${LOCAL_DL}/" 2>/dev/null || true
echo "Results downloaded to: ${LOCAL_DL}"

echo ""
echo "=================================================="
echo "STEP 4 - Per-lever delta table (Phase 2 §3.7 deliverable)"
echo "=================================================="
RESULTS_GLOB_DIR="${LOCAL_DL}/records/results"
if [[ -d "${RESULTS_GLOB_DIR}" ]]; then
  python3 "${SCRIPT_DIR}/compare_results.py" "${RESULTS_GLOB_DIR}" || echo "WARN: compare_results.py failed."
else
  echo "No downloaded results/ dir found at ${RESULTS_GLOB_DIR}; skipping comparison."
fi

echo ""
echo "=================================================="
echo "STEP 5 - Teardown"
echo "=================================================="
if [[ "${AUTO_TERMINATE}" == "true" ]]; then
  echo "AUTO_TERMINATE=true -> terminating instance ${INSTANCE_ID:-?} ..."
  echo yes | bash "${SCRIPT_DIR}/04_terminate.sh"
else
  echo "Instance ${INSTANCE_ID:-?} (${IP}) is STILL RUNNING and billing."
  echo "Terminate it when done:  bash ${SCRIPT_DIR}/04_terminate.sh"
fi

echo ""
echo "=== run_sweep complete ==="
