#!/usr/bin/env bash
# Train several configurations AT THE SAME TIME, one process each.
#
#   ./scripts/train_parallel.sh                  # no args = the full 6-network matrix
#   ./scripts/train_parallel.sh RS CUT-1
#   ./scripts/train_parallel.sh RS:2 RS CUT-1 CUT-2 CUT-2:2
#   STEPS=30000000 MAX_PARALLEL=3 ./scripts/train_parallel.sh RS CUT-2
#
# A spec is  DATASET[:ORIENTATIONS]  -- "CUT-2" is 1 pose, "CUT-2:2" is 2 poses.
# Run names match the rest of the repo: bpp1_cut2 / bpp1_orient_cut2.
#
# WHY THIS HELPS, AND WHAT IT DOES NOT DO
#   Each training process uses exactly ONE core (measured: 101% CPU, n=531
#   samples across four runs) because VecPackingEnv.step is a serial Python
#   loop.  With 16 cores idle, running several trainings side by side uses
#   them.  It does NOT make any single run finish sooner -- the runs do not
#   share anything, they just stop taking turns.  For one run to go faster you
#   need SubprocVecEnv (TODO D.1), which is a different change.
#
#   Returns are sub-linear because every process contends for the one GPU --
#   NOT for GPU memory, which is never the limit (a run holds ~200-300 MB of
#   8 GB; at 6 runs the card was 11% full but 79% busy).  The model is tiny, so
#   every kernel is launch-latency bound and the driver time-slices between CUDA
#   contexts rather than overlapping them.  Measured, CUT-2, 400k steps each:
#
#     N   per-run    aggregate   vs N=1   marginal
#     1   4,101/s     4,101/s     1.00x
#     2   3,424/s     6,848/s     1.67x    +2,747
#     3   2,968/s     8,904/s     2.17x    +2,056
#     4   2,517/s    10,068/s     2.46x    +1,164
#     5   2,161/s    10,805/s     2.63x      +737
#     6   1,848/s    11,088/s     2.70x      +283
#
#   Aggregate is still rising at 6, so for "finish all of them" the answer is
#   ALWAYS launch them all at once -- that is why MAX_PARALLEL defaults to the
#   number of specs.  For 5 jobs x 30M steps: all-5-at-once 3.86 h, batches of
#   three 5.24 h.  The trade is that each individual run is slower (2,161 vs
#   4,101 steps/s at N=5), so if you need ONE result early, run that one alone
#   first.  Past 6 is untested here.
#
# Resumable, like train_all_options.sh: a finished run is skipped, a partial
# one resumes from its last checkpoint.  Logs land in runs/<name>/train.log.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

PY=${PY:-$(command -v python3 || command -v python)}
[ -x "$PY" ] || { echo "FATAL: no python3 found"; exit 1; }

STEPS=${STEPS:-30000000}
SEQ_POOL=${SEQ_POOL:-40000}        # the one knob measured free (-0.01 pp at matched steps)
MAX_PARALLEL=${MAX_PARALLEL:-0}    # 0 = all of them at once (measured optimal)
EXTRA=${EXTRA:-}                   # anything else to pass to src.train
RUN_SUFFIX=${RUN_SUFFIX:-}         # appended to every run name; use it to
                                   # experiment without touching a real run

case ${1:-} in -h|--help)
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0;; esac

# No specs given -> the whole matrix train_all_options.sh covers: 3 streams x
# 2 orientation settings = the 6 networks the duel game offers as opponents.
# CUT-2 first, as in train_all_options.sh, so the paper's headline config gets
# a slot first if MAX_PARALLEL is capped below the job count.
if [ $# -eq 0 ]; then
  set -- CUT-2 CUT-1 RS CUT-2:2 CUT-1:2 RS:2
  echo "### no specs given -- training the full matrix: {RS, CUT-1, CUT-2} x {1 pose, 2 poses}"
fi

run_name () {                      # run_name <dataset> <orientations>
  local ds slug
  ds=$(echo "$1" | tr 'A-Z' 'a-z' | tr -d '-')     # CUT-2 -> cut2
  case $ds in rs) slug=rs;; cut1) slug=cut1;; cut2) slug=cut2;;
    *) echo "FATAL: unknown dataset '$1' (want RS, CUT-1 or CUT-2)" >&2; exit 1;; esac
  [ "$2" = 2 ] && echo "bpp1_orient_$slug$RUN_SUFFIX" || echo "bpp1_$slug$RUN_SUFFIX"
}

steps_done () {
  [ -f "runs/$1/metrics.jsonl" ] || { echo 0; return; }
  "$PY" - "$1" <<'PYEOF'
import json,sys
try:
    with open(f"runs/{sys.argv[1]}/metrics.jsonl") as f:
        print(json.loads([l for l in f if l.strip()][-1])["step"])
except Exception:
    print(0)
PYEOF
}

# ---- parse every spec up front so a typo fails before anything launches ----
declare -a NAMES DSS ORIS
for spec in "$@"; do
  ds=${spec%%:*}
  ori=${spec#*:}; [ "$ori" = "$spec" ] && ori=1
  case $ds in RS|rs) ds=RS;; CUT-1|cut-1|CUT1|cut1) ds=CUT-1;;
              CUT-2|cut-2|CUT2|cut2) ds=CUT-2;;
              *) echo "FATAL: unknown dataset '$ds' in spec '$spec'"; exit 1;; esac
  case $ori in 1|2) ;; ori|orient) ori=2;;
              *) echo "FATAL: orientations must be 1 or 2, got '$ori'"; exit 1;; esac
  NAMES+=("$(run_name "$ds" "$ori")"); DSS+=("$ds"); ORIS+=("$ori")
done

[ "$MAX_PARALLEL" -eq 0 ] && MAX_PARALLEL=${#NAMES[@]}
echo "### ${#NAMES[@]} configuration(s), $STEPS steps each, up to $MAX_PARALLEL at a time"
[ "${#NAMES[@]}" -gt "$MAX_PARALLEL" ] && \
  echo "### (the rest queue; raise with MAX_PARALLEL=N)"
[ "$MAX_PARALLEL" -gt 6 ] && \
  echo "### note: concurrency was only measured to 6; past that is untested"

launch () {                        # launch <run> <dataset> <orientations>
  local run=$1 ds=$2 ori=$3 done_steps
  done_steps=$(steps_done "$run")
  if [ "$done_steps" -ge "$((STEPS * 995 / 1000))" ] 2>/dev/null; then
    echo "-- skip $run (already at $done_steps steps)"; return
  fi
  mkdir -p "runs/$run"
  local args=(--run "$run" --preset paper --dataset "$ds" --orientations "$ori"
              --total-steps "$STEPS" --seq-pool "$SEQ_POOL" $EXTRA)
  [ -f "runs/$run/latest.pt" ] && args+=(--resume) && \
    echo "== resume $run from $done_steps steps" || \
    echo "== start  $run  ($ds, $ori pose(s))"
  "$PY" -m src.train "${args[@]}" > "runs/$run/train.log" 2>&1 &
  echo "   pid $! -> runs/$run/train.log"
}

for i in "${!NAMES[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do wait -n; done
  launch "${NAMES[$i]}" "${DSS[$i]}" "${ORIS[$i]}"
done
wait

echo
echo "### all done"
for n in "${NAMES[@]}"; do
  printf '  %-22s %s steps\n' "$n" "$(steps_done "$n")"
done
echo "### evaluate with:  $PY -m src.evaluate --run <name> --episodes 500 --bppk 3 5"
