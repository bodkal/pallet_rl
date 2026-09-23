#!/usr/bin/env bash
# Runs once the policy grid (and its second pass) have drained: one attacker
# per policy, attackers against the heuristics, then the tables and figures.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "== waiting for the policy grid"
until grep -q "all done" runs/GRID2.log 2>/dev/null; do sleep 60; done

echo "== one attacker per policy  $(date +%H:%M)"
python3 scripts/jobs.py attackers --nbs 10 20 --par 6 --iters 800

echo "== attackers against the heuristics  $(date +%H:%M)"
python3 scripts/attack_heur.py --nbs 5 20 --par 5 --iters 800

echo "== evaluation  $(date +%H:%M)"
python3 scripts/eval_all.py table2 --nbs 10 20 --n_inst 3000
python3 scripts/eval_all.py table1 --nbs 5 20 --n_inst 3000

echo "== figures and report  $(date +%H:%M)"
python3 scripts/plot_curves.py --nbs 10 20
python3 -m ar2l.viz.attack --runs att_pct_nb10 att_pct_nb20 --n_inst 512
python3 -m ar2l.viz.report --packing pct_nb10 pct_nb10@run:att_pct_nb10 \
    ex10_nb10 ex10_nb10@run:att_ex10_nb10
python3 scripts/summarize.py
python3 scripts/make_readme.py
echo "== PIPELINE DONE  $(date +%H:%M)"
