#!/usr/bin/env bash
# Train every option the duel game can offer:
#   3 box streams (RS / CUT-1 / CUT-2)  x  2 orientation settings (1 pose / 2 poses)
#   = 6 networks.
#
# BPP-1, BPP-3 and BPP-5 are NOT separate trainings.  BPP-k reuses the BPP-1
# network unchanged and only searches harder at test time (paper Sec. 3.3), so
# the algorithms axis costs training time zero -- it is an --bppk eval flag.
#
# Resumable: finished runs are skipped, partial ones are resumed.
# Re-running is cheap: finished trainings are skipped, a partial one is resumed
# from its last checkpoint (the LR schedule picks up from the restored step), and
# an evaluation whose eval.json is newer than best.pt is not repeated.
#   ./scripts/train_all_options.sh                 # everything
#   ORIENT_ONLY=1 ./scripts/train_all_options.sh   # only the 2-pose runs
#   NO_EVAL=1     ./scripts/train_all_options.sh   # train, evaluate later
#   FORCE_EVAL=1  ./scripts/train_all_options.sh   # re-run evals even if fresh
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

PY=${PY:-$(command -v python3 || command -v python)}
[ -x "$PY" ] || { echo "FATAL: no python3 found"; exit 1; }

STEPS=${STEPS:-100000000}
EPISODES=${EPISODES:-500}
BPPK_EPISODES=${BPPK_EPISODES:-100}

soft () { echo "== $*"; "$@" || echo "!! FAILED (continuing): $*"; }

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

train_one () {                       # train_one <run> <dataset> <orientations>
  local run=$1 ds=$2 orient=$3
  local done_steps; done_steps=$(steps_done "$run")
  if [ "$done_steps" -ge "$((STEPS * 995 / 1000))" ] 2>/dev/null; then
    echo "-- skip $run (already at $done_steps steps)"; return
  fi
  local args=(--run "$run" --preset paper --dataset "$ds"
              --orientations "$orient" --total-steps "$STEPS")
  if [ -f "runs/$run/latest.pt" ]; then
    echo "== resume $run from $done_steps steps"
    "$PY" -m src.train "${args[@]}" --resume || echo "!! train failed: $run"
  else
    echo "== train $run  ($ds, $orient orientation(s))"
    "$PY" -m src.train "${args[@]}" || echo "!! train failed: $run"
  fi
}

eval_one () {                        # eval_one <run> <dataset>
  [ -n "$NO_EVAL" ] && return
  [ -f "runs/$1/best.pt" ] || [ -f "runs/$1/latest.pt" ] || { echo "-- no ckpt for $1"; return; }
  # an eval newer than the checkpoint it scored is still valid - re-running it
  # would burn ~9 min to reproduce the same numbers.  FORCE_EVAL=1 overrides.
  if [ -z "$FORCE_EVAL" ] && [ -f "runs/$1/eval.json" ] && [ "runs/$1/eval.json" -nt "runs/$1/best.pt" ]; then
    echo "-- skip eval $1 (eval.json is newer than best.pt)"; return
  fi
  # one sweep covers all three algorithms: BPP-1 greedy, then MCTS at k=3 and k=5
  soft "$PY" -m src.evaluate --run "$1" --datasets "$2" \
       --episodes "$EPISODES" --bppk 3 5 --bppk-episodes "$BPPK_EPISODES" --record 3
  soft "$PY" -m src.viz.report --run "$1"
}

echo "### 6 networks: {RS, CUT-1, CUT-2} x {1 pose, 2 poses}, $STEPS steps each"

if [ -z "$ORIENT_ONLY" ]; then
  for DS in CUT-2 CUT-1 RS; do
    case $DS in CUT-2) R=bpp1_cut2;; CUT-1) R=bpp1_cut1;; RS) R=bpp1_rs;; esac
    train_one "$R" "$DS" 1
    eval_one  "$R" "$DS"
  done
fi

for DS in CUT-2 CUT-1 RS; do
  case $DS in CUT-2) R=bpp1_orient_cut2;; CUT-1) R=bpp1_orient_cut1;; RS) R=bpp1_orient_rs;; esac
  train_one "$R" "$DS" 2
  eval_one  "$R" "$DS"
done

echo "### done - every combination is now an opponent in:  $PY -m src.viz.game"
