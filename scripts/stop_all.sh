#!/usr/bin/env bash
# Stop every training / evaluation process this repo has running.
#
#   ./scripts/stop_all.sh              # stop them
#   ./scripts/stop_all.sh -n           # dry run: list, kill nothing
#   ./scripts/stop_all.sh -a           # also stop the dashboard and duel game
#   GRACE=30 ./scripts/stop_all.sh     # longer wait before SIGKILL
#
# ORCHESTRATORS DIE FIRST, AND THAT MATTERS.  train_all_options.sh loops over
# runs sequentially, so killing a worker while its parent is alive just makes
# the parent start the NEXT run -- you would be fighting the script.  So the
# launcher scripts get SIGTERM before any trainer does.
#
# Trainings are resumable, but only from the last checkpoint: src.train saves
# every `save_interval` updates (50 by default, ~64k steps at the paper batch),
# so anything since the last save is lost.  Resume with --resume, or just re-run
# train_parallel.sh, which resumes partial runs by itself.
#
# Viewers (dashboard, duel game) are left alone unless -a: they only read
# checkpoints, so they cost nothing and are usually what you want kept.
set -uo pipefail
cd "$(dirname "$0")/.."

DRY=0; VIEWERS=0; GRACE=${GRACE:-10}
for a in "$@"; do
  case $a in
    -n|--dry-run) DRY=1;;
    -a|--all)     VIEWERS=1;;
    -h|--help)    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown option '$a' (try -h)"; exit 1;;
  esac
done

ORCHESTRATORS='(train_parallel|train_all_options|run_all|chain_after)\.sh|scripts\.ab_throughput'
WORKERS='\-m +src\.(train|evaluate|viz\.report|viz\.replay3d|viz\.heatmap)'
VIEWER_RE='\-m +src\.viz\.(dashboard|game)'

# Never match this script, its shell, or the pgrep doing the matching -- a
# pattern broad enough to catch the workers also catches the process hunting
# for them, and a stop_all that kills itself stops halfway through.
pids_for () {
  pgrep -f -- "$1" 2>/dev/null | while read -r p; do
    if [ "$p" = "$$" ] || [ "$p" = "${PPID:-0}" ]; then continue; fi
    cmd=$(ps -o args= -p "$p" 2>/dev/null) || continue
    case $cmd in
      *stop_all*) continue;;
      # A `bash -c '... train_parallel.sh ...'` wrapper -- an editor terminal, a
      # CI step, an agent shell -- only MENTIONS the script in its command line.
      # Killing those takes down the caller's own shell, not a training job. A
      # real invocation looks like `/bin/bash ./scripts/train_parallel.sh`.
      *\ -c\ *) continue;;
    esac
    printf '%s\n' "$p"
  done
}

show () {                                  # show <pid> <kind>
  local cmd run step
  cmd=$(ps -o args= -p "$1" 2>/dev/null) || return
  run=$(printf '%s' "$cmd" | sed -n 's/.*--run  *\([^ ]*\).*/\1/p')
  step=""
  if [ -n "$run" ] && [ -f "runs/$run/metrics.jsonl" ]; then
    step=$(tail -c 20000 "runs/$run/metrics.jsonl" | sed -n '$s/.*"step": *\([0-9]*\).*/\1/p')
    [ -n "$step" ] && step=$(printf ' @ %.2fM steps' "$(echo "$step/1000000" | bc -l)")
  fi
  printf '  %-7s pid %-7s %s%s\n' "$2" "$1" "${run:-$(printf '%.60s' "$cmd")}" "$step"
}

mapfile -t ORCH < <(pids_for "$ORCHESTRATORS")
mapfile -t WORK < <(pids_for "$WORKERS")
VIEW=()
[ "$VIEWERS" = 1 ] && mapfile -t VIEW < <(pids_for "$VIEWER_RE")
ALL=("${ORCH[@]}" "${WORK[@]}" "${VIEW[@]}")

if [ "${#ALL[@]}" -eq 0 ]; then
  echo "nothing to stop -- no training or evaluation running"
  [ "$VIEWERS" = 0 ] && pids_for "$VIEWER_RE" | grep -q . && \
    echo "(a dashboard/game is up; -a stops those too)"
  exit 0
fi

echo "### ${#ALL[@]} process(es):"
for p in ${ORCH[@]+"${ORCH[@]}"}; do show "$p" launcher; done
for p in ${WORK[@]+"${WORK[@]}"}; do show "$p" worker;   done
for p in ${VIEW[@]+"${VIEW[@]}"}; do show "$p" viewer;   done

if [ "$DRY" = 1 ]; then echo "### dry run - nothing killed"; exit 0; fi

# launchers first, so none of them relaunches a worker we are about to kill
for p in ${ORCH[@]+"${ORCH[@]}"}; do kill -TERM "$p" 2>/dev/null; done
sleep 1
for p in ${WORK[@]+"${WORK[@]}"} ${VIEW[@]+"${VIEW[@]}"}; do kill -TERM "$p" 2>/dev/null; done

for _ in $(seq "$GRACE"); do
  alive=0
  for p in "${ALL[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
  [ "$alive" = 0 ] && break
  sleep 1
done

hard=0
for p in "${ALL[@]}"; do
  if kill -0 "$p" 2>/dev/null; then kill -KILL "$p" 2>/dev/null; hard=$((hard+1)); fi
done
[ "$hard" -gt 0 ] && echo "### $hard did not exit in ${GRACE}s - SIGKILLed"

sleep 1
left=0
for p in "${ALL[@]}"; do kill -0 "$p" 2>/dev/null && left=$((left+1)); done
if [ "$left" -eq 0 ]; then
  echo "### stopped. Resume with:  ./scripts/train_parallel.sh   (it resumes partial runs)"
else
  echo "### WARNING: $left process(es) still alive"; exit 1
fi
