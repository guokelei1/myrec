#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 LOG_PATH METADATA... -- COMMAND..." >&2
  exit 2
fi

log_path="$1"
shift
metadata_paths=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
  metadata_paths+=("$1")
  shift
done
if [[ $# -eq 0 || "$1" != "--" ]]; then
  echo "missing -- command separator" >&2
  exit 2
fi
shift
if [[ ${#metadata_paths[@]} -eq 0 || $# -eq 0 ]]; then
  echo "metadata and command are required" >&2
  exit 2
fi

while true; do
  all_completed=true
  for metadata_path in "${metadata_paths[@]}"; do
    if [[ -f "$metadata_path" ]]; then
      status="$(jq -r '.status // "missing"' "$metadata_path")"
    else
      status="missing"
    fi
    case "$status" in
      completed)
        ;;
      missing|initializing|running|selection_finalized|wall_time_exhausted)
        all_completed=false
        ;;
      *)
        echo "upstream terminal status=$status metadata=$metadata_path" >> "$log_path"
        exit 3
        ;;
    esac
  done
  if [[ "$all_completed" == true ]]; then
    break
  fi
  sleep 30
done

echo "all upstream bundles completed; running: $*" >> "$log_path"
"$@" >> "$log_path" 2>&1
