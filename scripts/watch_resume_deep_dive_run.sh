#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 INITIAL_PID METADATA_PATH LOG_PATH COMMAND..." >&2
  exit 2
fi

initial_pid="$1"
metadata_path="$2"
log_path="$3"
shift 3

while kill -0 "$initial_pid" 2>/dev/null; do
  sleep 30
done

# The scorer writes its terminal wall-time metadata immediately before process
# exit.  On busy filesystems the watcher can observe the exited PID a moment
# before the atomic metadata replacement, so wait briefly for that state.
for _poll in $(seq 1 12); do
  status="$(jq -r '.status // "missing"' "$metadata_path")"
  if [[ "$status" != "initializing" && "$status" != "running" ]]; then
    break
  fi
  sleep 5
done

attempt=0
while true; do
  status="$(jq -r '.status // "missing"' "$metadata_path")"
  case "$status" in
    completed)
      exit 0
      ;;
    wall_time_exhausted)
      attempt=$((attempt + 1))
      if (( attempt > 20 )); then
        echo "resume attempt limit reached for $metadata_path" >> "$log_path"
        exit 3
      fi
      echo "resume_attempt=$attempt status=$status" >> "$log_path"
      if ! "$@" >> "$log_path" 2>&1; then
        echo "resume command failed; preserving terminal metadata for inspection" >> "$log_path"
        exit 4
      fi
      sleep 5
      ;;
    *)
      echo "not resuming terminal/non-wall status=$status metadata=$metadata_path" >> "$log_path"
      exit 0
      ;;
  esac
done
