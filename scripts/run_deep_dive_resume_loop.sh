#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 METADATA_PATH LOG_PATH -- COMMAND..." >&2
  exit 2
fi

metadata_path="$1"
log_path="$2"
shift 2
if [[ "$1" != "--" ]]; then
  echo "missing -- command separator" >&2
  exit 2
fi
shift
if [[ $# -eq 0 ]]; then
  echo "resume-loop command is required" >&2
  exit 2
fi

canonical_metadata_path="$(realpath -m "$metadata_path")"
lock_key="$(printf '%s' "$canonical_metadata_path" | sha256sum | cut -d' ' -f1)"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(realpath -m "$script_dir/..")"
lock_root="${MYREC_DEEP_DIVE_RESUME_LOCK_DIR:-$repository_root/tmp/deep_dive_resume_locks}"
mkdir -p "$lock_root"
lock_path="$lock_root/$lock_key.lock"
lock_fd=""
exec {lock_fd}> "$lock_path"
if ! flock -n "$lock_fd"; then
  echo "resume-loop writer lock is busy metadata=$canonical_metadata_path" >> "$log_path"
  exit 7
fi

attempt=0
resume=false
if [[ -f "$metadata_path" ]]; then
  status="$(jq -r '.status // "missing"' "$metadata_path")"
  case "$status" in
    completed)
      exit 0
      ;;
    wall_time_exhausted)
      resume=true
      ;;
    *)
      echo "refusing existing non-resumable status=$status metadata=$metadata_path" >> "$log_path"
      exit 3
      ;;
  esac
fi

while true; do
  attempt=$((attempt + 1))
  if (( attempt > 20 )); then
    echo "resume-loop attempt limit reached metadata=$metadata_path" >> "$log_path"
    exit 4
  fi
  echo "attempt=$attempt resume=$resume command=$*" >> "$log_path"
  if [[ "$resume" == true ]]; then
    if ! "$@" --resume >> "$log_path" 2>&1; then
      echo "resume-loop command failed metadata=$metadata_path" >> "$log_path"
      exit 5
    fi
  else
    if ! "$@" >> "$log_path" 2>&1; then
      echo "resume-loop command failed metadata=$metadata_path" >> "$log_path"
      exit 5
    fi
  fi
  status="$(jq -r '.status // "missing"' "$metadata_path")"
  case "$status" in
    completed)
      exit 0
      ;;
    wall_time_exhausted)
      resume=true
      sleep 5
      ;;
    *)
      echo "resume-loop terminal status=$status metadata=$metadata_path" >> "$log_path"
      exit 6
      ;;
  esac
done
