# Reproducing the run

Everything below assumes a CUDA GPU; pass `--device cpu` to fall back (slowly).

```bash
pip install -r requirements.txt
python3 -c "import numpy as np; from ar2l.evaluate import make_dataset; \
            np.save('data/pallet_2cm_test.npy', make_dataset(3000, 120))"
```

An instance set is `(n_inst, n_items, 4)`: three sides and a `type_id`, drawn
from the `types:` size classes in [`config.yaml`](config.yaml). A three-column
file from before box types still loads — every box in it is type 0 — so
`data/discrete_test.npy` and `data/pallet_2cm_untyped.npy` remain playable as
single-type instances. Pass `types=False` to `make_dataset` to draw one
yourself.

## 0a. Box types and the stacking rule

Every box carries a `type_id` and may only be stacked on boxes of its own type;
the floor takes any type. The type is also the SKU — `types:` in
[`config.yaml`](config.yaml) gives one item-size class per type, so the stream
draws a type and then its sides from that type's bounds.

The rule bites hardest on a FIFO conveyor: with `--n_pick 1` there is nothing
to choose, and a box whose type has nowhere to go ends the episode. A reach is
what lets a run group a type together —

```bash
--n_types 3 --nb 10 --n_pick 5 --algo select     # the permuter picks *for* the packer
--n_types 3 --nb 10 --n_pick 5 --algo exact      # and against it, which is AR2L
```

With `--n_pick > 1` the station's `b_pick` hides the reachable boxes that have
nowhere legal to go, so the permuter is pointed only at the ones it can hand
over, and the episode ends when the whole station is blocked.

To turn the whole thing off:

```bash
--n_types 1                  # with `types: null` in the config: the pre-type simulator
--type_constraint 0          # keeps type_id in the state, lets anything stack on anything
```

A policy trained with types must be *evaluated* with them: the node projections
are 16 columns wider, and `ar2l.evaluate` takes `--n_types` and
`--type_constraint` for exactly that reason. A checkpoint from before box types
still loads — `evaluate.load_nets` pads its narrow projections with zeros,
which is the identity on the features it was trained with, and leaves the type
embedding at its fresh initialisation.

## 0b. Your own pallets: an orders CSV

Real orders go in as a CSV, one row per box in arrival order
([`data/orders_example.csv`](data/orders_example.csv)):

```csv
pallet_id,seq,length_cm,width_cm,height_cm,type,qty
P001,1,24,14,12,0,8
P001,2,16,12,10,2,5
```

`pallet_id` and the three sides are required; `seq` orders boxes within a
pallet (file order otherwise), `type` is the stacking-rule type (default 0) and
`qty` repeats a row. Sides are rounded *up* to 2 cm cells (`--cell_cm`).

The pallet the orders are packed onto is `--pallet_cm L W H` (or
`eval.pallet_cm` in [`config.yaml`](config.yaml)), rounded *down* to cells;
without it they go onto `env.bin`, the 60 × 48 × 80 cm training pallet, which
real cartons outgrow. A box that fits the pallet in no orientation is an error
rather than a silently short pallet. A `pallet_id` may be spread through the
file — a cell building two pallets at once — and its boxes are gathered in
file order. Pallets of different lengths share one array, padded with zero
rows the env reads as "end of this pallet".

```bash
python3 -m ar2l.evaluate --data data/orders_all.csv --pallet_cm 120 80 180 --heuristic dbl --beta 0 \
        --per_pallet results/orders_dbl.csv
python3 -m ar2l.evaluate --data data/orders_all.csv --pallet_cm 120 80 180 \
        --ckpt runs/sel_types_k5/best.pt --attacker runs/sel_types_k5/best.pt \
        --permuter mixer --nb 10 --n_pick 5 --beta 100 --per_pallet results/orders_sel.csv
python3 -m ar2l.viz.replay3d --data data/orders_all.csv --pallet_cm 120 80 180 --seq 0 \
        --policy run:sel_types_k5 --attacker mix:sel_types_k5 --gif
python3 -m ar2l.orders data/orders_all.csv data/orders_all.npy --pallet_cm 120 80 180   # convert once, if wanted
```

`--box_scale N` (or `eval.box_scale`) divides every box side — x, y and z
alike — by `N` before the rounding up to cells: `0` (or `1`) keeps the file's
sizes, `2` halves them, `3` takes a third. The pallet is not scaled.

`--order_random R` (or `eval.order_random`) randomises each pallet's box order
before it is played: `0` is the order in the file, `1` a uniformly random
order, and in between each box drifts from its place by about `R / (1 - R)`
of the pallet — `0.1` moves a box ~5 places in a 50-box pallet. The draw is
fixed by `--order_seed`, and `replay3d`/`heatmap` take the same two flags, so
a replay shows exactly the order `evaluate` scored.

The game plays them too: `python3 -m ar2l.viz.game`, then under *Game
parameters* set **boxes from** to *an orders file*. Pick a pallet (or let the
seed deal one), and the pallet in cm, cell size, box scale, rounding and order
randomness work as they do here; the bin, item bounds and box count are then
the file's. It starts from `train.data` and the `eval:` values in the config.

`All` in the output is the share of pallets whose every box was placed; the
`--per_pallet` CSV has `pallet_id,beta,boxes,placed,util_pct` per pallet. For a
`--algo select` run the selector only acts at `--beta 100`; at `--beta 0` the
packer takes the boxes in CSV order.

## 0c. Training on your own pallets

`train.data` in [`config.yaml`](config.yaml) (or `--data`) picks where the
training episodes come from: empty/`null` is the random generator exactly as
before, a path — an orders `.csv` or an instance `.npy` — makes every episode
one of its pallets, drawn at random.

```bash
python3 -m ar2l.train --name sel_orders --data data/orders_all.csv \
        --pallet_cm 120 80 180 --cell_cm 4 --algo select --nb 10 --n_pick 5 \
        --iters 10000 --order_random 0.2
```

- `--holdout` (default `0.2`) keeps that share of the pallets out of training;
  the held-out evals that choose `best.pt` score them. The split is seeded and
  written to `runs/<name>/train_ids.txt` and `holdout_ids.txt`.
- `--order_random` randomises each drawn pallet's box order, for variety from
  few pallets. `--pallet_cm` also sets the bin trained on; `--cell_cm` and
  `--box_scale` read the CSV as `evaluate` does.
- On a 120 × 80 × 180 pallet a `select --n_pick 5` iteration takes ~22 s at
  2 cm cells and ~6.5 s at 4 cm; 4 cm is the grid scale the earlier runs used.
- The data file's SHA-1 is recorded in `args.json`, and `--resume` refuses to
  continue if the file has changed since.
- Few pallets overfit: with ~100 of them, trust the held-out numbers, not the
  training `util`.

## 0. The stability rule the numbers were measured under

The simulator now refuses any placement resting on less than **80%** of the
item's base (`--min_support`, `env.min_support` in
[`config.yaml`](config.yaml), default `0.80`). Every published number in
`README.md` and `results/`, and every checkpoint that produced them, predates
that floor, so reproducing them means turning it off:

```bash
--min_support 0        # the bare centre-of-mass rule the tables were run under
```

It is accepted by both `ar2l.train` and `ar2l.evaluate`, and a policy must be
evaluated under the rule it was trained under.

The checkpoints those numbers came from are no longer in `runs/`: they were
archived to `old_run/runs_nosupport/` when the floor became the default, so
`runs/` is empty until something is retrained under it. `scripts/eval_all.py`
and the viewers resolve `runs/<name>`, so scoring an archived policy means
pointing `ar2l.evaluate --ckpt/--attacker` straight at its file:

```bash
python3 -m ar2l.evaluate --ckpt old_run/runs_nosupport/ex10_nb10/best.pt \
    --attacker old_run/runs_nosupport/att_ex10_nb10/best.pt \
    --nb 10 --n_inst 3000 --min_support 0
```

## 1. The strongest policy — what to run

**`--algo exact --alpha 1.0`, exact AR2L at `α = 1`.** It is the best packer in
`README.md` once anything reorders the conveyor, and it gives up nothing
nominally: at `N_B = 10` it holds **65.1** utilisation under its own attacker
against PCT's 64.2, from a nominal 75.4 against PCT's 75.9, and it packs at
least as many items as PCT in 9 of 10 cells. `--alpha 0.5` is the variant to
pick if the conveyor is never adversarial — it is the highest nominal number in
the table at `N_B = 20` (75.7) — and RARL edges ExactAR2L(1.0) out at `N_B = 10`
under attack (65.7) by training on the attacker's orderings alone, which is the
trade the paper is arguing against. There is no "bigger" trained net in the
repo: all three networks are the paper's 65k-parameter transformer, and
`--width / --heads / --layers` are what would change that (see below).

Three commands: the packer, its own attacker, and the evaluation.

```bash
# 1. the packer — exact AR2L at alpha = 1, N_B = 10
python3 -m ar2l.train --name best_nb10 --algo exact --alpha 1.0 --nb 10 \
    --iters 20000 --eval_every 250 --save_every 250 --resume --min_support 0

# 2. a dedicated attacker against that frozen packer
python3 -m ar2l.train --name att_best_nb10 --algo attack --nb 10 \
    --init runs/best_nb10/best.pt --freeze_pack --ent_final 0.001 \
    --iters 4000 --eval_every 250 --save_every 250 --resume --min_support 0

# 3. score it on the 3000 held-out instances, nominal through fully attacked
python3 -m ar2l.evaluate --ckpt runs/best_nb10/best.pt \
    --attacker runs/att_best_nb10/best.pt --nb 10 --n_inst 3000 --min_support 0
```

`best.pt` is the checkpoint with the best held-out score — nominal utilisation
for a packer, and for `--algo attack` the *lowest* utilisation it drove the
frozen packer to. `--resume` picks the run up from `last.pt`, so the same
command can be re-issued after an interruption or to extend `--iters`.

**Which `--min_support` to train under.** The commands above pass
`--min_support 0`, which is the rule every number in `README.md` was measured
under and the only way to compare against them or against the archived runs.
For a policy meant to drive a real palletiser, **drop the flag from all three
commands** and take the 80% contact-area default instead — that is the rule the
simulator now runs by default, and it is the one to train under unless the
point of the run is the paper's table. The floor costs roughly 15 points of
utilisation, so its numbers are not comparable to anything in `README.md`;
what stays comparable is one method against another under the same floor.

**What the budget buys.** 4000 iterations is what `README.md` reports; the
paper runs ~28,000, and the gap in
[TODO.md](TODO.md#c1-what-is-left-the-attacker-and-only-the-attacker) is
attributed to the attacker and to that budget, not to network capacity — which
is why `--iters 20000` above, and why raising `--width`/`--layers` is a lever
nothing here has measured. Measured on an RTX 4070 Laptop, one run at a time,
`--n_env 64 --T 30`:

| run | ms/iteration | 4000 iters | 20000 iters |
|---|---|---|---|
| `--algo exact --nb 10` | 1120 | 1.2 h | 6.2 h |
| `--algo exact --nb 20` | 1180 | 1.3 h | 6.6 h |
| `--algo pct --nb 10` | 485 | 0.5 h | 2.7 h |
| `--algo attack --freeze_pack` | 470 | 0.5 h | 2.6 h |

Exact AR2L is ~2.3× PCT because it runs three PPO updates per iteration
(attacker, mixture model, packer). The 80% floor costs nothing measurable
(1120 vs 1122 ms/it). Four runs sharing this GPU take about 1.8× as long each.

**Bigger networks, if you want to try.** `--width 128 --heads 4 --layers 2`
applies to all three networks at once; the checkpoint records them and
`ar2l.evaluate` rebuilds the right shape, so nothing else has to be told. The
paper's 64/1/1 is what every number here was produced with.

## 2. Train the packing policies

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

## 3. Train one attacker per policy

Table 2 evaluates every policy under *its own* permutation-based attacker, so
each frozen policy gets a dedicated one:

```bash
python3 scripts/jobs.py attackers --nbs 10 20 --par 4 --iters 1500
```

and, for Table 1, an attacker against each heuristic:

```bash
python3 scripts/attack_heur.py --nbs 5 10 15 20 --par 3 --iters 1500
```

## 4. Evaluate

```bash
python3 scripts/eval_all.py table2 --nbs 10 20 --n_inst 3000
python3 scripts/eval_all.py table1 --nbs 5 10 15 20 --n_inst 3000
python3 scripts/summarize.py                  # markdown tables + claim checks
python3 scripts/make_readme.py                # write them into README.md
```

## 5. Look at it

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

---

# Every parameter

## Where the defaults live: `config.yaml`

Every default is in [`config.yaml`](config.yaml), in six sections — `env`,
`train`, `ppo`, `model`, `run`, `eval`. That file *is* the defaults:
[`ar2l/config.py`](ar2l/config.py) reads it at import, the argparsers of
`ar2l.train` and `ar2l.evaluate` take theirs from it, and the values that have
no flag of their own read it directly. Edit a value there and it changes
everywhere; pass a flag and the flag wins.

```
config.yaml  <  --config other.yaml / AR2L_CONFIG=other.yaml  <  --flag
```

An override file is a *patch*: it needs only the keys it changes, and the rest
keep their shipped values. An unknown key or section is an error rather than a
silent no-op.

```yaml
# mine.yaml — a 12^3 bin with the support floor off
env:
  bin: 12
  min_support: 0.0
model:
  width: 96
```

```bash
python3 -m ar2l.train --name mine --algo exact --config mine.yaml
AR2L_CONFIG=mine.yaml python3 -m ar2l.evaluate --ckpt runs/mine/best.pt
```

`runs/<name>/args.json` and the checkpoint record the values the run resolved
to, `--config` path included, so a finished run still says what it was trained
under — `min_support` among them, which is the one that must match at
evaluation time.

Five things used to be reachable only by editing the source and now have both
a config entry and a flag: `ppo.clip` (`--clip`), `ppo.vf_coef` (`--vf_coef`),
`ppo.max_grad` (`--max_grad`), `model.c_temp` (`--c_temp`, the pointer
temperature of Eq. 28) and `env.max_c` (`--max_c`, the initial packed-item
capacity). Their defaults are unchanged: 0.2, 0.5, 0.5, 10.0 and 80.

## Every flag, spelled out

Each command below sets **every** flag its entry point accepts, to its default
unless noted; all of them were run to check that they parse. They are
copy-pasteable as they stand — delete the lines you do not care about, since
nothing here has to be passed. What each flag means is in the tables below.

**A packing policy** (`exact`, `pct`, `cppo`, `rarl`, `rfmdp`, `approx`):

```bash
python3 -m ar2l.train \
  --name best_nb10 --algo exact --nb 10 \
  --config config.yaml \
  --alpha 1.0 --rho 0.1 --dist_coef 1.0 --cvar_q 0.5 \
  --iters 20000 --n_env 64 --T 30 \
  --bin 10 --n_items 150 --size_lo 1 --size_hi 5 --max_l 120 --max_c 80 \
  --rot 2 --ems 1 --stability com --min_support 0.80 \
  --lr 3e-4 --gamma 1.0 --lam 0.95 --epochs 4 --minibatches 4 \
  --ent_coef 0.01 --ent_final 0.001 --clip 0.2 --vf_coef 0.5 --max_grad 0.5 \
  --width 64 --heads 1 --layers 1 --c_temp 10.0 \
  --init runs/other/best.pt \
  --seed 0 --device cuda --compile 0 --resume \
  --eval_every 250 --eval_inst 256 --log_every 50 --save_every 250 --progress auto
```

Every value here is the `config.yaml` default except: `--iters` (default 8000), `--ent_final`
(off by default, and only the attacker's bonus is annealed), `--init` (off by
default — drop the line to start from scratch), and the four `--eval/--save/--log`
periods, which the grid runs at 250/256/50/250. `--alpha` is read by `exact` and
`approx`, `--rho` by `approx` and `rfmdp`, `--dist_coef` by `exact`, `--cvar_q`
by `cppo`; the rest are accepted and ignored. `--freeze_pack` and `--heur_pack`
are the two flags left out: they belong to an attacker run, and on a packer run
they would train nothing.

**An attacker** against that frozen policy — the same flag set, plus those two:

```bash
python3 -m ar2l.train \
  --name att_best_nb10 --algo attack --nb 10 \
  --init runs/best_nb10/best.pt --freeze_pack \
  --alpha 1.0 --rho 0.1 --dist_coef 1.0 --cvar_q 0.5 \
  --iters 4000 --n_env 64 --T 30 \
  --config config.yaml \
  --bin 10 --n_items 150 --size_lo 1 --size_hi 5 --max_l 120 --max_c 80 \
  --rot 2 --ems 1 --stability com --min_support 0.80 \
  --lr 3e-4 --gamma 1.0 --lam 0.95 --epochs 4 --minibatches 4 \
  --ent_coef 0.01 --ent_final 0.001 --clip 0.2 --vf_coef 0.5 --max_grad 0.5 \
  --width 64 --heads 1 --layers 1 --c_temp 10.0 \
  --seed 0 --device cuda --compile 0 --resume \
  --eval_every 250 --eval_inst 256 --log_every 50 --save_every 250 --progress auto
```

`--init` names the packer to attack and `--freeze_pack` holds it fixed and plays
it greedily. Swap both for `--heur_pack dbl` to attack a heuristic instead
(`dbl|bmf|lsah|onlinebph|hmm|macs`) — that is what Table 1 needs. The
environment flags must match the ones the packer was trained under.

**Evaluation** — every flag of the scorer:

```bash
python3 -m ar2l.evaluate \
  --ckpt runs/best_nb10/best.pt \
  --attacker runs/att_best_nb10/best.pt \
  --config config.yaml \
  --nb 10 --beta 0 25 50 75 100 \
  --data data/discrete_test.npy --n_inst 3000 --batch 256 \
  --rot 2 --min_support 0.80 --seed 0 \
  --out results/best_nb10.json --device cuda
```

`--heuristic dbl` replaces `--ckpt` to score a heuristic. Drop `--attacker` for
the nominal column alone; it is meaningless on a `pct`, `cppo` or `rfmdp`
checkpoint, which carry an untrained attacker head. `--rot` and `--min_support`
must match what the policy was trained under, `--seed` picks which instances
each `--beta` subset reorders.

**The grid, end to end** — every flag of each driver:

```bash
python3 scripts/jobs.py policies  --nbs 10 20 --methods pct cppo rarl rfmdp ex05 ex10 ap05 ap10 \
                                  --par 4 --iters 4000 --extra "--min_support 0"
python3 scripts/jobs.py attackers --nbs 10 20 --methods pct cppo rarl rfmdp ex05 ex10 ap05 ap10 \
                                  --par 4 --iters 1500 --extra "--min_support 0"
python3 scripts/attack_heur.py --nbs 5 10 15 20 \
                               --heuristics dbl bmf lsah onlinebph hmm macs \
                               --par 3 --iters 1500
python3 scripts/eval_all.py table2 --nbs 10 20 --betas 0 25 50 75 100 \
                                   --data data/discrete_test.npy --n_inst 3000 \
                                   --batch 375 --min_support 0 --device cuda
python3 scripts/eval_all.py table1 --nbs 5 10 15 20 --betas 0 100 \
                                   --data data/discrete_test.npy --n_inst 3000 \
                                   --batch 375 --min_support 0 --device cuda
python3 scripts/summarize.py --which table1 table2
python3 scripts/make_readme.py
python3 scripts/calibrate.py  --n 256 --rot 2 --out results/calibration.json
python3 scripts/ablate_env.py --n 512 --seed 11 --out results/ablate_env.json
python3 scripts/plot_curves.py --nbs 10 20 --methods pct cppo rarl rfmdp ex05 ex10 ap05 ap10 \
                               --out results/training_curves.png
```

`jobs.py --extra` is appended verbatim to every training command it launches,
which is the only way to give the whole grid a flag like `--min_support 0`.
`make_readme.py` takes none.

**The viewers**, each with its full flag set:

```bash
python3 -m ar2l.viz.dashboard --port 8095 --host 127.0.0.1
python3 -m ar2l.viz.game      --port 8096 --host 127.0.0.1 --device cuda \
                              --bin 30x25x40 --size_hi 15x10x10 --max_l 256 \
                              --nb 10 --n_pick 5
python3 -m ar2l.viz.report    --packing best_nb10 best_nb10@run:att_best_nb10 \
                              --curves best_nb10 --out results/report.html --device cuda
python3 -m ar2l.viz.replay3d  --policy run:best_nb10 --attacker run:att_best_nb10 \
                              --nb 10 --seq 0 --data data/discrete_test.npy --root . \
                              --gif --fps 2.4 --tag mine --device cuda
python3 -m ar2l.viz.heatmap   --policy run:best_nb10 --attacker run:att_best_nb10 \
                              --nb 10 --seq 0 --step 8 --data data/discrete_test.npy \
                              --root . --gif --fps 1.4 --device cuda
python3 -m ar2l.viz.attack    --runs att_best_nb10 att_best_nb20 \
                              --data data/discrete_test.npy --n_inst 512 --root . \
                              --out results/attacker_behaviour.png --device cuda
```

`--policy` and `--attacker` take `run:<name>` (a checkpoint under `runs/`),
`heur:<name>` or `random`; `--attacker mix:<name>` uses an exact-AR2L run's
mixture model instead of its attacker. `--step` renders one decision rather
than the whole episode, and `--nb` defaults to the run's own.

`game`'s own flags **seed** its setup form, which is otherwise refilled from
whichever run you choose as the opponent — its bin, item bounds, `nb`,
`n_pick`, `max_l`, `rot`, `ems`, stability rule and support floor, straight out
of `runs/<name>/args.json`. A flag you did pass outranks the run, as it
outranks `config.yaml` everywhere else; the rest come from the run, and every
field stays editable. The server
re-validates the whole set on each keystroke and refuses one it could not run,
so a combination the page accepts is one `BPPBatch` accepts; fields you move
away from the run's own value are marked, because the agent is then being asked
a question it was never trained on. `?opp=`, `?att=`, `?bin=`, `?nb=`, `?k=`,
`?items=`, `?cell=` and `?play=1` set the same things from the URL.

With `n_pick = k > 1` the conveyor is a pick station: all `k` reachable boxes
are shown with the placements each still has, and you click one (or press
<kbd>1</kbd>…<kbd>9</kbd>) to pack it. Turn the pick off in setup to be handed
the front box instead, as a FIFO conveyor would.

On the result screen, **Recalculate** replays the same boxes in the same order
with one model alone, under a model and parameters you pick — one row per
setting, each naming the model, what it moved, and by how much the fill
changed. Any policy or heuristic the setup form offers can be dropped onto the
instance, whether or not it was the opponent. `n_items`, `size_lo` and
`size_hi` are held at the game's values, because they are what deals the
sequence; change those from setup instead.

## `python3 -m ar2l.train`

Every default below is the `config.yaml` value; the section it lives in is
named in each table's heading.

**What to run, and for how long** (`run:`)

| flag | default | meaning |
|---|---|---|
| `--name` | *required* | run directory: `runs/<name>/` holds `args.json`, `log.jsonl`, `best.pt`, `last.pt` |
| `--algo` | `pct` | `pct`, `cppo`, `rarl`, `rfmdp`, `exact`, `approx`, `attack` |
| `--iters` | `8000` | PPO iterations. One iteration is `--n_env` × `--T` steps per network trained |
| `--resume` | off | continue from `runs/<name>/last.pt`, keeping the recorded best score |
| `--seed` | `0` | seeds the env and torch |
| `--device` | `cuda` | `cpu` works and is much slower; there is no silent fallback |
| `--compile` | `0` | `torch.compile`. Off by default: with dynamic node counts it miscompiles the pointer head for some `N_B` and surfaces as an illegal memory access |
| `--config` | — | a YAML file of defaults merged over `config.yaml`; the same as `AR2L_CONFIG=` |

**Algorithm knobs** (`train:`) — each is read by some algorithms and ignored by
the rest.

| flag | default | read by | meaning |
|---|---|---|---|
| `--alpha` | `1.0` | `exact`, `approx` | robustness weight α: how far the mixture dynamics may move from the nominal ordering toward the attacker's |
| `--rho` | `0.1` | `approx`, `rfmdp` | TV uncertainty radius in the Eq. 18 dual. `0` collapses the dual to the α-weighted mean of the nominal and attacked next-state values (a unit test) |
| `--dist_coef` | `1.0` | `exact` | weight on the mixture model's distance loss; the official code's coefficient is 1 |
| `--cvar_q` | `0.5` | `cppo` | CVaR level — the worst fraction of trajectories the policy update sees. The paper does not give it |
| `--init` | — | any | checkpoint to initialise the packer from |
| `--freeze_pack` | off | `attack` | hold the packer fixed and play it greedily; this is what makes the attacker train against the packer it will be tested against |
| `--heur_pack` | — | `attack` | attack a heuristic (`dbl`, `bmf`, `lsah`, `onlinebph`, `hmm`, `macs`) instead of a network |
| `--ent_final` | — | `attack` | anneal the attacker's entropy bonus linearly from `--ent_coef` to this value (the grid uses `0.001`) |

**The environment** (`env:`) — these define the problem, and a policy is only
comparable to another trained with the same ones.

| flag | default | meaning |
|---|---|---|
| `--nb` | `10` | observable items `N_B`: how far down the conveyor both the packer and the attacker can see |
| `--bin` | `10` | bin extent; an int for a cube or `WxLxH`, e.g. `60x50x80` |
| `--n_items` | `150` | items per episode |
| `--size_lo` / `--size_hi` | `1` / `5` | item side bounds; `--size_hi` also takes `WxLxH` for per-axis caps |
| `--rot` | `2` | orientations offered: `1` = none, `2` = also yawed 90°. Worth +6.7 points on the heuristics |
| `--ems` | `1` | `1` = candidates are the empty-maximal-space corners, `2` = corner cells of the height map (walls, taller boxes, and the edges of box tops), `3` = both, `0` = every loading position on the grid |
| `--max_l` | `120` | leaf-node cap, the length of the candidate list. Raise it on a large bin if the EMS corners hit it |
| `--max_c` | `80` | initial packed-item capacity — the packer's memory of the bin. Not a limit: it doubles when a bin holds more |
| `--stability` | `com` | `com` = centre of mass over the support (the identified rule), `cdrl` = the 60%-area/4-corner rule as the paper's citations write it |
| `--n_types` | `3` | box types. Must match the `types:` size classes in the config; `1` with `types: null` is the pre-type simulator |
| `--type_constraint` | `1` | `0` keeps `type_id` in the state but lets any box be stacked on any other |
| `--min_support` | `env.min_support`, `0.80` | fraction of the item's base that must rest on the layer below for a placement to be offered. `0` is the bare centre-of-mass rule every published number was measured under. Ignored by `--stability cdrl` |

**PPO** (`ppo:`, and `n_env`/`T` from `train:`)

| flag | default | meaning |
|---|---|---|
| `--n_env` | `64` | parallel bins |
| `--T` | `30` | rollout length per iteration |
| `--lr` | `3e-4` | Adam, no decay (the official code decays to a 1e-4 floor) |
| `--gamma` | `1.0` | undiscounted, as in the paper |
| `--lam` | `0.95` | GAE λ |
| `--epochs` | `4` | passes over each rollout |
| `--minibatches` | `4` | minibatches per pass (the official code runs 1 × 32) |
| `--ent_coef` | `0.01` | entropy bonus |
| `--clip` | `0.2` | PPO ratio clip (the official code uses 0.1) |
| `--vf_coef` | `0.5` | value loss weight (the official code uses 1.0) |
| `--max_grad` | `0.5` | gradient-norm clip |

**The networks** (`model:`)

| flag | default | meaning |
|---|---|---|
| `--width` / `--heads` / `--layers` | `64` / `1` / `1` | the transformer, shared by all three networks. The paper's size; 65k parameters each |
| `--c_temp` | `10.0` | pointer-head temperature `c` of Eq. 28 |
| `--type_embed` | `16` | width of the trainable box-type embedding, concatenated onto every node's physical features before its element-wise projection |

**Logging and checkpoints** (`run:`)

| flag | default | meaning |
|---|---|---|
| `--eval_every` | `200` | iterations between held-out evaluations, which is what selects `best.pt` |
| `--eval_inst` | `256` | instances per held-out evaluation |
| `--log_every` | `50` | iterations between log lines and `log.jsonl` records |
| `--save_every` | `200` | iterations between `last.pt` writes |
| `--progress` | `auto` | progress bar on stderr; `auto` = only on a terminal, so a redirected log stays clean |

## `python3 -m ar2l.evaluate`

Defaults from the `eval:` section, plus `env:` for `--rot` / `--min_support`.

| flag | default | meaning |
|---|---|---|
| `--ckpt` | — | packing policy checkpoint; the network shape is read from it |
| `--config` | — | a YAML file of defaults merged over `config.yaml` |
| `--heuristic` | — | score a heuristic instead: `dbl`, `bmf`, `lsah`, `onlinebph`, `hmm`, `macs` |
| `--attacker` | — | checkpoint holding the attacker to reorder with. Only `rarl`, `exact`, `approx` and `attack` runs carry a trained one; every other checkpoint holds an unused randomly-initialised head that would report a harmless attack |
| `--nb` | `10` | observable items; match the policy's |
| `--beta` | `0 25 50 75 100` | percentage of instances the attacker reorders |
| `--data` | `data/pallet_2cm_test.npy` | the held-out set, `(n_inst, n_items, 4)`; a three-column file from before box types loads as all type 0 |
| `--n_inst` | `3000` | instances scored |
| `--batch` | `256` | bins stepped at once |
| `--rot` | `2` | orientations; match the policy's |
| `--min_support` | `env.min_support`, `0.80` | contact-area floor; **match the policy's** |
| `--n_pick` | `1` | how many of the `N_B` are within reach; **match the policy's** |
| `--n_types` | `3` | box types the dataset carries; **match the policy's** |
| `--type_constraint` | `1` | `0` scores the policy with the stacking rule lifted |
| `--seed` | `0` | which instances the β subsets pick |
| `--out` | — | write the metrics to this JSON path |
| `--device` | `cuda` | |

The CLI evaluates on the 10³ default bin: `--bin`, `--stability` and `--max_l`
are not exposed here. For a non-cubic bin or the `cdrl` rule, call
`ar2l.evaluate.run(...)`, which takes `S`, `stability`, `max_l` and `ems`.

## `scripts/`

| script | flags | what it does |
|---|---|---|
| `jobs.py policies\|attackers` | `--nbs 10 20`, `--methods`, `--par 4`, `--iters 4000`, `--extra ""` | the training grid, `--par` jobs at a time. `--extra` is appended to every command, which is how a grid-wide `--min_support 0` gets in |
| `attack_heur.py` | `--nbs 5 10 15 20`, `--heuristics`, `--par 3`, `--iters 1500` | one attacker per heuristic, for Table 1 |
| `eval_all.py table1\|table2` | `--nbs`, `--betas 0 25 50 75 100`, `--data`, `--n_inst 3000`, `--batch 375`, `--min_support`, `--device` | scores the whole grid into `results/table{1,2}.json` |
| `summarize.py` | `--which table1 table2` | markdown tables and the paper's comparative claims |
| `make_readme.py` | — | writes those between the `RESULTS` markers in `README.md` |
| `calibrate.py` | `--n 256`, `--rot 2`, `--out` | the stability-rule identification table. It pins `min_support` per row, so it does not move when the env default does |
| `ablate_env.py` | `--n 512`, `--seed 11`, `--out` | the heuristics under each `rot` × `ems` combination — what the action space is worth |
| `plot_curves.py` | `--nbs`, `--methods`, `--out` | training curves as a PNG |

## `ar2l.viz`

| viewer | flags |
|---|---|
| `dashboard` | `--port 8095`, `--host 127.0.0.1` |
| `game` | `--port 8096`, `--host`, `--device`, `--bin 10`, `--size_hi 5`, `--max_l 120` — the last three must match the opponent's training configuration |
| `report` | `--packing <run>[@run:<attacker>] ...`, `--curves`, `--out results/report.html`, `--device` |
| `replay3d` | `--policy run:<name>\|heur:<name>\|random`, `--attacker run:<name>\|mix:<name>`, `--nb`, `--seq 0`, `--data`, `--root .`, `--gif`, `--fps 2.4`, `--tag`, `--device` |
| `heatmap` | the same, plus `--step` to render one decision; `--fps 1.4` |
| `attack` | `--runs att_pct_nb10 ...`, `--data`, `--n_inst 512`, `--root`, `--out`, `--device` |
