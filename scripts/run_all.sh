#!/usr/bin/env bash
# Full reproduction pipeline, ordered so the most valuable results land first.
# Resumable: completed stages are skipped, partially-trained ones are resumed.
# Watch live:  $PY -m src.viz.dashboard --port 8080
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

# pick an interpreter that actually exists (this box has no bare `python`)
PY=${PY:-$(command -v python3 || command -v python)}
[ -x "$PY" ] || { echo "FATAL: no python3 found"; exit 1; }
echo "using interpreter: $PY"

MAIN_STEPS=${MAIN_STEPS:-30000000}
ABL_STEPS=${ABL_STEPS:-8000000}
name_of () { case "$1" in CUT-2) echo bpp1_cut2;; CUT-1) echo bpp1_cut1;; RS) echo bpp1_rs;; esac; }

# A stage that must not take the whole queue down with it (eval, plots, reports).
soft () { echo "== $*"; "$@" || echo "!! FAILED (continuing): $*"; }

# Train unless already finished. A run that exists but stopped short is RESUMED,
# not skipped -- checking only for latest.pt would silently drop 28M steps.
train_if_missing () {
  local run=$1; shift
  local target=$MAIN_STEPS
  for ((i=1;i<=$#;i++)); do
    [ "${!i}" = "--total-steps" ] && { local j=$((i+1)); target=${!j}; }
  done
  local done_steps=0
  if [ -f "runs/$run/metrics.jsonl" ]; then
    done_steps=$("$PY" - "$run" <<'PYEOF'
import json,sys
try:
    with open(f"runs/{sys.argv[1]}/metrics.jsonl") as f:
        last=[l for l in f if l.strip()][-1]
    print(json.loads(last)["step"])
except Exception:
    print(0)
PYEOF
)
  fi
  if [ "$done_steps" -ge "$((target * 995 / 1000))" ] 2>/dev/null; then
    echo "-- skip $run (done: $done_steps steps)"; return
  fi
  if [ -f "runs/$run/latest.pt" ]; then
    echo "== resume $run from $done_steps steps"
    "$PY" -m src.train --run "$run" "$@" --resume || echo "!! train failed: $run"
  else
    echo "== train $run"
    "$PY" -m src.train --run "$run" "$@" || echo "!! train failed: $run"
  fi
}

# 1. main model on CUT-2
train_if_missing bpp1_cut2 --preset paper --dataset CUT-2 --total-steps $MAIN_STEPS

# 2. evaluate + visualise it straight away
soft "$PY" -m src.evaluate --run bpp1_cut2 --datasets RS CUT-1 CUT-2 \
     --episodes 500 --bppk 2 3 5 --bppk-episodes 100 --record 3
soft "$PY" -m src.viz.replay3d --run bpp1_cut2 --all
soft "$PY" -m src.viz.heatmap  --run bpp1_cut2 --dataset CUT-2
soft "$PY" -m src.viz.report   --run bpp1_cut2

# 3. the other two benchmarks
for DS in CUT-1 RS; do
  R=$(name_of $DS)
  train_if_missing "$R" --preset paper --dataset $DS --total-steps $MAIN_STEPS
  soft "$PY" -m src.evaluate --run "$R" --datasets $DS --episodes 500 --bppk 2 3 5 \
       --bppk-episodes 100 --record 3
  soft "$PY" -m src.viz.replay3d --run "$R" --all
  soft "$PY" -m src.viz.report   --run "$R"
done

# 4. re-orientation extension (paper Table 4)
train_if_missing bpp1_orient_rs --preset orient --dataset RS --total-steps $MAIN_STEPS
soft "$PY" -m src.evaluate --run bpp1_orient_rs --datasets RS --episodes 500 --record 3
soft "$PY" -m src.viz.replay3d --run bpp1_orient_rs --all
soft "$PY" -m src.viz.heatmap  --run bpp1_orient_rs --dataset RS

# 5. MP / MC / FE ablation (paper Table 1)
train_if_missing abl_none  --preset paper --dataset CUT-2 --total-steps $ABL_STEPS --no-mp --no-mc --no-fe
train_if_missing abl_mp_fe --preset paper --dataset CUT-2 --total-steps $ABL_STEPS --no-mc
train_if_missing abl_mp_mc --preset paper --dataset CUT-2 --total-steps $ABL_STEPS --no-fe
train_if_missing abl_mc_fe --preset paper --dataset CUT-2 --total-steps $ABL_STEPS --no-mp

# 6. final report with everything in it
soft "$PY" -m src.viz.report --run bpp1_cut2 \
     --ablation abl_none abl_mp_fe abl_mp_mc abl_mc_fe bpp1_cut2
echo "REPORT: runs/bpp1_cut2/report.html"
