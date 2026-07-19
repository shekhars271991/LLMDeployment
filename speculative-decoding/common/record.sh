#!/usr/bin/env bash
# Shared raw-output recorder. Source it, then call:
#   start_recording "script_name" "/path/to/records/category"
# Console output remains visible and is also appended to a timestamped .log file.

start_recording() {
  local record_name="$1"
  local record_dir="$2"
  local timestamp

  timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
  mkdir -p "${record_dir}"

  export RAW_RECORD_FILE="${record_dir}/${timestamp}_${record_name}.log"

  exec > >(tee -a "${RAW_RECORD_FILE}") 2>&1

  printf '%s\n' \
    "record_name=${record_name}" \
    "started_utc=${timestamp}" \
    "host=$(hostname)" \
    "user=$(id -un)" \
    "working_directory=$(pwd)" \
    "command=$0 $*" \
    "--- output ---"

  _finish_recording() {
    local status=$?
    trap - EXIT
    printf '%s\n' \
      "--- end ---" \
      "finished_utc=$(date -u +"%Y%m%dT%H%M%SZ")" \
      "exit_status=${status}" \
      "raw_record=${RAW_RECORD_FILE}"
    exit "${status}"
  }

  trap _finish_recording EXIT
}
