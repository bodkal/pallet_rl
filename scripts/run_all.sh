#!/usr/bin/env bash
# End-to-end: datasets -> policies -> attackers -> tables -> figures -> report.
# Every stage is resumable; re-running picks up where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."

NBS="${NBS:-10 20}"
ITERS="${ITERS:-4000}"
ATT_ITERS="${ATT_ITERS:-1500}"
PAR="${PAR:-4}"

echo "== test set"
[ -f data/discrete_test.npy ] || python3 -c \
  "import numpy as np; from ar2l.evaluate import make_dataset; \
   np.save('data/discrete_test.npy', make_dataset(3000, 150))"

echo "== packing policies"
python3 scripts/jobs.py policies --nbs $NBS --par $PAR --iters $ITERS

echo "== one attacker per policy"
python3 scripts/jobs.py attackers --nbs $NBS --par $PAR --iters $ATT_ITERS

echo "== attackers against the heuristics (Table 1)"
python3 scripts/attack_heur.py --nbs $NBS --par 3 --iters $ATT_ITERS

echo "== evaluation"
python3 scripts/eval_all.py table2 --nbs $NBS --n_inst 3000
python3 scripts/eval_all.py table1 --nbs $NBS --n_inst 3000

echo "== figures and report"
python3 scripts/plot_curves.py --nbs $NBS
python3 -m ar2l.viz.attack --runs $(for n in $NBS; do echo att_pct_nb$n; done | tr '\n' ' ')
python3 -m ar2l.viz.report --packing $(for n in $NBS; do echo pct_nb$n ex10_nb$n; done | tr '\n' ' ')
python3 scripts/summarize.py
python3 scripts/make_readme.py

echo "== done; see results/report.html"
