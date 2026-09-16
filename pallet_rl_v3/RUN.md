# Reproducing the run

Everything below assumes a CUDA GPU; pass `--device cpu` to fall back (slowly).

```bash
pip install -r requirements.txt
python3 -c "import numpy as np; from ar2l.evaluate import make_dataset; \
            np.save('data/discrete_test.npy', make_dataset(3000, 150))"
```

## 1. Train the packing policies

One policy per (method, N_B) cell of Table 2, four at a time:

```bash
python3 scripts/jobs.py policies --nbs 10 20 --par 4 --iters 4000
```

`--resume` is passed to every job, so re-running the same command picks each
run up from its last checkpoint.  Single runs can be launched directly:

```bash
python3 -m ar2l.train --name ex10_nb10 --algo exact --alpha 1.0 --nb 10 --iters 4000
```

Algorithms: `pct`, `cppo`, `rarl`, `rfmdp`, `exact`, `approx`, `attack`.

## 2. Train one attacker per policy

Table 2 evaluates every policy under *its own* permutation-based attacker, so
each frozen policy gets a dedicated one:

```bash
python3 scripts/jobs.py attackers --nbs 10 20 --par 4 --iters 1500
```

and, for Table 1, an attacker against each heuristic:

```bash
python3 scripts/attack_heur.py --nbs 5 10 15 20 --par 3 --iters 1500
```

## 3. Evaluate

```bash
python3 scripts/eval_all.py table2 --nbs 10 20 --n_inst 3000
python3 scripts/eval_all.py table1 --nbs 5 10 15 20 --n_inst 3000
python3 scripts/summarize.py                  # markdown tables + claim checks
python3 scripts/make_readme.py                # write them into README.md
```

## 4. Look at it

```bash
python3 -m ar2l.viz.dashboard --port 8095    # live training curves
python3 -m ar2l.viz.game      --port 8096    # play the packer yourself
python3 -m ar2l.viz.report --packing pct_nb10 pct_nb10@run:att_pct_nb10 \
                                     ex10_nb10 ex10_nb10@run:att_ex10_nb10
python3 -m ar2l.viz.replay3d --policy run:pct_nb10 --gif
python3 -m ar2l.viz.heatmap  --policy run:ex10_nb10 --attacker run:att_ex10_nb10 --gif
python3 -m ar2l.viz.attack   --runs att_pct_nb10 att_pct_nb20
```

Both servers bind `127.0.0.1` by default; pass `--host 0.0.0.0` to reach them
from another machine. The file viewers take `--root` (where `runs/`, `data/`
and `results/` live) and `--fps` for the GIFs.

## Checks

```bash
python3 -m pytest tests -q        # simulator vs. brute force, the duals, the loops
python3 scripts/calibrate.py      # the stability-rule identification table
```
