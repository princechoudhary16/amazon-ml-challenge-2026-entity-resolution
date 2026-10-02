#!/usr/bin/env bash
# Run a long job in the background and kill it if the resident memory of its whole
# process tree exceeds a cap. On a 16 GB laptop this prevents the OS from killing other
# applications (editor, terminal) under memory pressure; the pipeline caches every stage
# on disk, so a killed run can simply be resumed.
#
# usage: scripts/run_with_memory_cap.sh <max_GB> <log_file> <command> [args...]
# e.g.   scripts/run_with_memory_cap.sh 9.5 run.log python -m ber.run --art artifacts --out output
set -uo pipefail

MAX_GB=${1:?usage: $0 <max_GB> <log_file> <command...>}
LOG=${2:?usage: $0 <max_GB> <log_file> <command...>}
shift 2

nohup "$@" > "$LOG" 2>&1 &
PID=$!
echo "started pid $PID (log: $LOG, memory cap: ${MAX_GB} GB)"

descendants() {  # all descendant pids of $1
  local kids
  kids=$(pgrep -P "$1" 2>/dev/null) || return 0
  for k in $kids; do echo "$k"; descendants "$k"; done
}

(
  while kill -0 "$PID" 2>/dev/null; do
    TREE="$PID $(descendants "$PID" | tr '\n' ' ')"
    # shellcheck disable=SC2086
    RSS_GB=$(ps -o rss= -p ${TREE// /,} 2>/dev/null | awk '{s += $1} END {printf "%.2f", s / 1048576}')
    if awk -v r="$RSS_GB" -v m="$MAX_GB" 'BEGIN {exit !(r > m)}'; then
      echo "MEMORY CAP: killed process tree at ${RSS_GB} GB (cap ${MAX_GB} GB)" >> "$LOG"
      # shellcheck disable=SC2086
      kill -9 $TREE 2>/dev/null
      break
    fi
    sleep 2
  done
) > /dev/null 2>&1 &
