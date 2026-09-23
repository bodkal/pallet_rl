#!/usr/bin/env bash
# The 800-iteration attackers cost PCT ~10 points where the paper's cost ~21
# (N_B=10) and ~35 (N_B=20), which compresses every comparison in Table 2.
# Train them out to 3000 iterations and re-evaluate.
set -uo pipefail
cd "$(dirname "$0")/.."

until grep -q "PIPELINE DONE" runs/PIPE.log 2>/dev/null; do sleep 60; done

echo "== keeping the weak-attacker results  $(date +%H:%M)"
cp results/table2.json results/table2_att800.json
cp results/table1.json results/table1_att800.json 2>/dev/null || true

echo "== attackers to 3000 iterations  $(date +%H:%M)"
python3 scripts/jobs.py attackers --nbs 10 20 --par 6 --iters 3000

echo "== re-evaluating  $(date +%H:%M)"
python3 scripts/eval_all.py table2 --nbs 10 20 --n_inst 3000

echo "== report  $(date +%H:%M)"
python3 -m ar2l.viz.attack --runs att_pct_nb10 att_pct_nb20 --n_inst 512
python3 -m ar2l.viz.report --packing pct_nb10 pct_nb10@run:att_pct_nb10 \
    ex10_nb10 ex10_nb10@run:att_ex10_nb10
python3 scripts/summarize.py
python3 scripts/make_readme.py
echo "== STRONG ATTACK DONE  $(date +%H:%M)"
