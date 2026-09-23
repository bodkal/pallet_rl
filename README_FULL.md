# pallet_rl — Complete Technical Reference

**Adjustable Robust Reinforcement Learning (AR2L) for online 3D bin packing and
palletising: code, architecture, representation, and the reasons behind every
design choice.**

This document is meant to stand on its own. It covers what the system does,
how each part is built, how the data looks at every stage, and, for each
significant decision, which alternatives existed and why this one was chosen.
It was written from the source code on the `add_type_to_box` branch
(September 2026).

---

## Table of contents

1. What this project is
2. The problem, in plain terms
3. System architecture at a glance
4. Repository map
5. Configuration system
6. The environment (simulator) — `ar2l/env.py`
7. The neural networks — `ar2l/model.py`
8. PPO and the robust value targets — `ar2l/ppo.py`
9. The training algorithms — `ar2l/train.py`
10. The heuristic baselines — `ar2l/heuristics.py`
11. Evaluation — `ar2l/evaluate.py`
12. Real pallet orders — `ar2l/orders.py`
13. Viewers and tools — `ar2l/viz/`
14. Scripts and the experiment pipeline — `scripts/`
15. Tests — what is guaranteed
16. Design decisions: a consolidated table
17. Deviations from the paper
18. Results and current status
19. Current defaults and known documentation drift
20. Glossary
21. Frequently asked questions

---

## 1. What this project is

The project began as a from-scratch reproduction of the NeurIPS 2023 paper
**"Adjustable Robust Reinforcement Learning for Online 3D Bin Packing"** by
Pan, Chen and Lin (arXiv 2310.04323; the PDF is `2310.04323v1.pdf` in the repo
root). The official implementation (github.com/panyxy/ar2l_bpp) came out after
this reproduction was written. It was later read and compared against this one
(see `TODO.md`, section A), but **no code was copied from it**.

On top of the reproduction, the project adds features that a real palletising
cell needs:

- **Non-cubic bins.** A real pallet is, for example, 60 × 48 × 80 cm on a 2 cm
  grid, so the bin is 30 × 24 × 40 cells.
- **A contact-area stability floor.** A box must rest on a real share of its
  base, not just balance.
- **Box types (SKUs) with a same-type stacking rule.** A box may only be placed
  on boxes of its own type; the floor accepts any type.
- **A pick station** (`n_pick`). The robot can reach the first *k* boxes on the
  conveyor and can see some more upstream.
- **A cooperative selector** (`--algo select`). It chooses which reachable box
  to pack next *for* the packer, not against it.
- **Real orders.** Pallets are read from a CSV of box dimensions in cm, used
  for both training and evaluation.
- **Interactive tools.** These include a browser game in which a human packs
  against the agent, a live training dashboard, 3D replays and decision
  heatmaps.

The Python package is `ar2l/`. It has five core modules (env, model, ppo,
train, evaluate), plus heuristics, orders, config and a `viz/` sub-package.

---

## 2. The problem, in plain terms

**Online 3D bin packing (3D-BPP).** Boxes arrive one at a time on a conveyor.
The packer sees the next `N_B` boxes (the *observable window*). It must place
the front box now, somewhere in the bin, and can never move it later. The goal
is to maximise **space utilisation**: the packed volume divided by the bin
volume.

**Why robustness matters.** A policy trained on random box orders does well on
average but can collapse on a bad ordering, for example many small boxes first
and then a large one that no longer fits. AR2L trains a policy that holds up
against the *worst* ordering an adversary can build, without giving up
average-case performance.

**The paper's three parts, all implemented here:**

1. **A permutation-based attacker.** This policy looks at the packed bin and
   the observable window, then moves one visible box to the front of the
   conveyor to make the instance as hard as possible for the packer.
2. **Exact AR2L (Algorithm 1).** A third network, the *mixture-dynamics model*
   `π_mix`, is trained to maximise the packer's return. It is also kept close
   both to the nominal ordering and to the attacker's ordering. A weight `α`
   sets the balance between average-case and worst-case. The packer is then
   trained on the orderings `π_mix` produces.
3. **Approximate AR2L (Algorithm 2).** This has the same goal but no third
   network. The critic's target is replaced with the dual of an *adjustable
   robust Bellman operator* (the paper's Eq. 18), computed from the next-state
   values under both the nominal and the attacked conveyor.

**The palletising extension.** With box types and the same-type stacking rule,
the *order* boxes arrive in decides much of what can be packed. If the robot
can reach several boxes (`n_pick > 1`), choosing *which* box to take becomes a
policy of its own. The same permuter network can learn that choice
cooperatively (`select`) or adversarially (`rarl` / `exact` / `approx`).

---

## 3. System architecture at a glance

```
                        config.yaml  (every default, 6 sections)
                              │
                     ar2l/config.py  → CFG dict (+ --config / AR2L_CONFIG patch)
                              │
         ┌────────────────────┼─────────────────────────────┐
         │                    │                             │
   ar2l/orders.py        ar2l/env.py                   ar2l/model.py
   CSV (cm) → cells      BPPBatch: n_env bins          Encoder + Pointer
   (n, max_boxes, 4)     stepped in lockstep            PackNet  π_pack(l | C,B,L), V(C,B)
         │               state (C_t, B_t, L_t)          PermNet  π_perm / π_mix(b_i | C,B), V(C,B)
         │               height map, type map                     │
         │               EMS candidates, stability                │
         │               stacking rule, conveyor                  │
         └────────► pool ──┘   │                                  │
                               │ obs dicts (numpy)                │
                               ▼                                  ▼
                         ar2l/train.py  ─────────────►  ar2l/ppo.py
                         Runner.collect (rollouts)       PPO clipped update
                         8 algorithms (pct … select)     GAE (+ robust next-value)
                         held-out evals → best.pt        sup_tv_dual (Eq. 18), inf_tv_dual
                               │
                   runs/<name>/{args.json, log.jsonl, best.pt, last.pt}
                               │
         ┌─────────────────────┼──────────────────────────┐
   ar2l/evaluate.py       ar2l/heuristics.py          ar2l/viz/*
   β-mixture datasets     DBL BMF LSAH OnlineBPH      dashboard, game, report,
   Uti / Std / Num / All  HMM MACS (same cand. list)  replay3d, heatmap, attack
                               │
                    scripts/* (job queue, tables, figures, README results)
```

**Data flow during one training iteration** (exact AR2L shown):

1. `Runner.collect(T=30, permuter=attacker)`: at each of 30 steps, for each of
   the 64 bins, the attacker picks a window slot to move to the front, and
   then the packer picks a placement. The environment steps, and rewards and
   done flags are recorded.
2. The attacker is updated with PPO on the **negated** packer reward.
3. `Runner.collect(T=30, permuter=mixer)`: a new rollout, this time under the
   mixture model.
4. The mixer is updated with PPO on the packer's reward, plus a distance loss
   that keeps it near "no permutation" and near the attacker.
5. The packer is updated with PPO on that same mixer rollout.
6. Every `eval_every` iterations, a greedy held-out evaluation runs. If it
   improves, `best.pt` is saved.

---

## 4. Repository map

| Path | What it contains |
|---|---|
| `config.yaml` | Every default, in six sections: `env`, `train`, `ppo`, `model`, `run`, `eval`. |
| `ar2l/config.py` | Loads `config.yaml` into `CFG`, merges override files as patches, and rejects unknown keys. |
| `ar2l/env.py` | `BPPBatch`, the batched simulator: height map, type map, window-max sweeps, EMS enumeration, stability, stacking rule, conveyor, observation building. |
| `ar2l/model.py` | `Encoder`, `Pointer` (Eq. 28), `Critic`, `PackNet`, `PermNet`, `sample`. |
| `ar2l/ppo.py` | `PPO` (clipped surrogate), `gae`, `sup_tv_dual` (Eq. 18), `inf_tv_dual` (RfMDP). |
| `ar2l/train.py` | `Runner` (rollouts), `move_to_front`, the eight training loops, checkpointing, data hold-out, CLI. |
| `ar2l/heuristics.py` | Six hand-designed packers that score the same candidate list the network sees. |
| `ar2l/evaluate.py` | `run` (play instances), `metrics`, `nominal_score`, `attack_score`, `load_nets` (with pre-type checkpoint migration), CLI. |
| `ar2l/orders.py` | Orders-CSV loader, cm-to-cell rounding, `box_scale`, `pallet_cm`, `randomize_order`. |
| `ar2l/viz/agents.py` | Shared episode driver and policy/attacker loaders for the viewers. |
| `ar2l/viz/game.py` + `game.html` | Browser game: a human packs the same instance as an agent. |
| `ar2l/viz/dashboard.py` + `.html` | Live training curves from `runs/*/log.jsonl`. |
| `ar2l/viz/report.py` | A single self-contained HTML report (inline SVG and base64 PNG). |
| `ar2l/viz/replay3d.py` | Step-by-step 3D PNGs or GIF of a bin being built. |
| `ar2l/viz/heatmap.py` | Per-decision panels: height map, landing height, feasibility, `π_pack`, `π_perm`. |
| `ar2l/viz/attack.py` | What the attacker learned: size distribution of the boxes it promotes, how far it reaches, the damage it does. |
| `scripts/` | Job queue (`jobs.py`), evaluation driver (`eval_all.py`), `calibrate.py`, `ablate_env.py`, `attack_heur.py`, plots, summary, README filler, shell pipelines. |
| `tests/` | Simulator vs brute force, box types vs voxel grid, duals vs closed forms, game server, orders, training on data. |
| `data/` | Test sets (`.npy`) and real orders (`.csv`). |
| `results/` | JSON tables, calibration, ablation, training-curve PNG. |
| `runs/` | Training runs: `args.json`, `log.jsonl`, `best.pt`, `last.pt`, and for data runs `train_ids.txt` / `holdout_ids.txt`. |
| `old_run/` | Archived pre-rotation results and figures. Their checkpoints cannot be re-run. |
| `README.md`, `RUN.md`, `TODO.md` | Results summary, how to run, status and open items. |

---

## 5. Configuration system

### 5.1 How it works

`config.yaml` holds **every** default. `ar2l/config.py` reads it at import time
into a module-level dict `CFG`. The argparsers of `ar2l.train` and
`ar2l.evaluate` take their defaults from `CFG` *when the parser is built*.
Values that have no command-line flag, such as the pointer temperature and the
packed-item capacity, are read from `CFG` directly.

The precedence order is:

```
config.yaml   <   --config other.yaml  /  AR2L_CONFIG=other.yaml   <   --flag on the command line
```

- An override file is a **patch**. It lists only the keys it changes, and
  everything else keeps the shipped value (`_merge(BASE, override)`).
- An unknown section or key is an **error** (`ValueError`), not silently
  ignored.
- `--config` must be applied *before* the parser is built, because the parser's
  defaults come from it. `main()` therefore runs a small pre-parser that reads
  only `--config`, calls `load_config`, and then builds the real parser.
- `load()` changes `CFG` in place (`CFG.clear(); CFG.update(...)`), so any code
  that already imported `CFG` sees the new values.

### 5.2 Why this design

- **One source of truth.** Before this, defaults were spread across argparsers
  and constructors, and five of them (PPO clip, value coefficient, gradient
  clip, pointer temperature, `max_c`) could only be changed by editing source.
- **Patches, not full copies.** An experiment file lists only what it changes,
  so it states exactly how that experiment differs.
- **Loud failures.** A misspelled key that silently did nothing would produce a
  run that looks correct but tests something else.
- **Every run records what it resolved to.** `runs/<name>/args.json` and every
  checkpoint store the fully resolved argument set, for example
  `min_support: 0.3` rather than `null`. `main()` resolves `min_support`
  explicitly for exactly this reason.

---

## 6. The environment (simulator) — `ar2l/env.py`

### 6.1 World model: a height map of dropped boxes

The bin is an integer grid of `Lx × Ly × Lz` cells (`S=` accepts an int for a
cube or an `(Lx, Ly, Lz)` triple). Boxes are axis-aligned, integer-sized, and
**dropped from above**. A box lands at height `z = max(hmap)` over its
footprint and cannot slide in from the side.

Because every box is dropped, the reachable free space is exactly
`{(x, y, z) : z ≥ hmap[x, y]}`. The full state of the bin's free space is
therefore just the **height map** `hmap[x, y]` (int16). Voids under overhangs
cannot be reached, so they never need to be tracked.

*Why a height map and not a voxel grid:* it is `O(Lx·Ly)` instead of
`O(Lx·Ly·Lz)`. Every feasibility question becomes a windowed maximum over a 2D
array. And it matches how a top-down robot actually places boxes.

Alongside the height map, the environment keeps:

- `tmap[x, y]` (int8): the **type of the topmost box** in each column, or
  `TYPE_FLOOR = -1` for bare floor.
- `packed[n, max_c, 6]` (float32): `C_t`, the packed boxes as normalised
  `(x, y, z, sx, sy, sz)`.
- `ptype[n, max_c]` (int8): the type of each packed box, stored beside `packed`
  rather than inside it, so every caller that unpacks a 6-tuple still works.
- `seq[n, n_items, 4]` (int16): each bin's box sequence as
  `(sx, sy, sz, type_id)`.
- `head[n]`: the index of the current front box; `length[n]`: the real number
  of boxes in each instance (so padding is supported).
- `volume`, `n_packed`, `done`.

### 6.2 Batching: all bins in lockstep

`BPPBatch(n_env, ...)` steps `n_env` independent bins (64 by default) **as
array operations over the whole batch**. There is no Python loop over bins in
the hot path. The one exception is the pruned EMS enumeration for large bins,
explained in 6.6.

*Why:* PPO needs many transitions per iteration (64 × 30 = 1920 per network),
and the networks run on the GPU in batch. A per-bin Python loop would make the
simulator the bottleneck. On a 10³ bin, 64 bins step in about 1.7 ms.

### 6.3 The state representation: the PCT triple (C_t, B_t, L_t)

This follows the paper, which follows PCT (Packing Configuration Trees, Zhao
et al. 2022). The observation is a dict of arrays:

| Key | Shape | Meaning |
|---|---|---|
| `c` | (n, N_C, 6) | Packed boxes `(x, y, z, sx, sy, sz) / scale`. N_C is trimmed to the batch's current maximum. |
| `c_mask` | (n, N_C) bool | Which packed-box slots are real. |
| `c_type` | (n, N_C) int64 | Type of each packed box, as an embedding row index. |
| `b` | (n, N_B, 3) | The observable window `(sx, sy, sz) / scale`. |
| `b_mask` | (n, N_B) bool | Which window slots hold a real box (the sequence may have ended). |
| `b_type` | (n, N_B) int64 | Type of each window box. |
| `b_pick` | (n, N_B) bool | Which window boxes the permuter may choose: within reach, and (if `pick_feasible`) placeable. |
| `l` | (n, N_L, 6) | Candidate placements `(x, y, z, sx', sy', sz') / scale`, where `sx', sy'` are the *rotated* footprint. |
| `l_mask` | (n, N_L) bool | Which candidate slots are real. |
| `l_type` | (n, N_L) int64 | The type of whatever each candidate would **rest on** (the floor has its own index). |

`obs_cb()` returns only the `C` and `B` parts, for the attacker and the
critics. It skips building `L`, which is the expensive part.

**Normalisation.** Every coordinate and size is divided by a *single* scale,
`max(Lx, Ly, Lz)`, not by each axis separately. *Why:* per-axis normalisation
would stretch a lopsided bin into a unit cube and distort box shapes. The
packer is reasoning about geometric fit, and a 12 × 7 box must look
elongated to the network.

**Type indices.** The environment stores `TYPE_FLOOR = -1`. `_emb_type` maps
`-1` to row `n_types`, so the embedding table has `n_types + 1` rows: one per
box type, plus one for the floor. *Why the floor gets its own row:* the floor
accepts every type while a box accepts only its own, so the packer has to tell
the two apart.

### 6.4 The action space: (orientation, x, y) candidates

A placement is a triple `(orientation r, x, y)`:

- With `rot = 2` (the default), the front box is offered both as `(w, l, h)`
  and as `(l, w, h)`, a 90° yaw. Height never rotates. For a square footprint
  the second orientation is a duplicate and is masked out.
- For each orientation, the environment computes a feasibility grid
  `feas[n, Lx, Ly]` and a landing-height grid `z[n, Lx, Ly]`.
- With `ems = 1` (the default), feasibility is further limited to **the four
  bottom corners of every empty maximal space (EMS)** large enough for the
  box. These are the "leaf nodes" of PCT and AR2L.

**Building the candidate list L** (`obs()`):

1. Flatten `feas` over `(orientation, x, y)`.
2. Sort with `lexsort` on `(index, z, ~feasible)`: feasible candidates first,
   then **lowest landing height first**, with index as the tie-break.
3. Keep the first `N_L = min(max_l, #candidates, max feasible in batch)`.
4. Record `_lxy` (x, y, r), `_lz`, `_ldim` (rotated dims) and `_lunder` (type
   underneath), so `step(action)` can decode the index the network chose.

*Why EMS corners and not every grid position:* measured with the heuristics
(`scripts/ablate_env.py`), EMS corners cost nothing once rotation is enabled
(61.7% either way) and cut the candidate list from about 72 to 37. That makes
training faster and gives the pointer fewer, better options. EMS corners are
also the paper's action space.

*Why keep the deepest candidates when truncating:* the cap `max_l` rarely
binds (about 32 candidates at the 90th percentile on the pallet config against
`max_l = 256`). When it does bind, the lowest placements are the most useful,
and the ordering is deterministic.

*Why rotation:* it is the paper's discrete setting, and it is worth **+6.7
points** of utilisation on the heuristics (measured).

### 6.5 Feasibility: running window maxima

**The central trick.** A box of footprint `sx × sy` placed at `(x, y)` lands at
`z = max(hmap[x : x+sx, y : y+sy])`. Computing this for every position and
every footprint size at once is a **2D sliding-window maximum**, and it is
separable: first take a window max along y, then along x.

`_win_max(m, axis, L, kmax)` computes the window max for **every width
1..kmax** incrementally:

```
out[0]   = m
out[k]   = max(out[k-1][..., : L-k], m[..., k:])        # width k+1 = width k plus one more cell
```

so `out[w-1][x]` is the max over `[x, x+w)`. The result is indexed by
`[width-1, env, x, y]`, so each bin reads the window matching its own box size
with fancy indexing (`my[cy - 1, ar]`). Bins with different box sizes are
handled in one batched call.

**Contact area in the same pass** (`_win_maxcount`). Stability rules need the
number of footprint cells that actually touch the box, meaning the cells whose
column height equals `z`. The sweep carries the pair **(max, count of cells
attaining the max)**. This pair is associative under "append one more cell":
the new cell beats the running max (count resets to the new cell's count), ties
it (counts add), or loses (count unchanged). So the y-sweep's `(max, count)`
feeds straight into the x-sweep, and the counts of disjoint columns add up
correctly.

*Why:* an earlier version encoded counts as a base-32 histogram (`32^h` per
column). It was cheaper by one sweep but only representable while heights
stayed under 12 and footprints under 25 cells, which ruled out the area rules
on exactly the realistic pallet sizes. The `(max, count)` form has no limit on
bin height or box size.

**Caching.** Sweeps that depend only on the bin (`_sweeps`: y-max, y-count,
x-max) are cached until the height map changes. The type sweep is cached per
incoming-type vector. `_invalidate(hmap=False)` is used after a permutation:
the front box changed but the bin did not, so the bin-level sweeps are reused.

**What `_feas_one(dims, types)` combines for one orientation:**

```
feas = stable  &  fits in x  &  fits in y  &  z + sz ≤ Lz
       & (EMS corner, if ems)
       & (type-legal, if type_constraint and n_types > 1)
```

### 6.6 Empty maximal spaces (EMS): exact, stateless enumeration

**Definition used.** Because boxes are dropped, an EMS is a footprint
`[x0, x0+wx) × [y0, y0+wy)` whose floor `h` (the window max of the height map
over it) **cannot be lowered by widening the footprint**, extended up to the
lid. A footprint is maximal when widening it by one cell on each of its four
sides would either raise the floor or hit a wall.

**Two exact implementations** return the same set:

- `_ems_exhaustive()` scores all `(Lx(Lx+1)/2) · (Ly(Ly+1)/2)` footprints with
  batched window maxima and keeps those where all four one-cell extensions
  (`left`, `right`, `back`, `front`) are strictly higher than the floor. Walls
  use the sentinel `Lz + 1`. This is `O(S⁴)` per bin: fine at S = 10 (55 × 55),
  but 6.2 million rectangles at S = 70.
- `_ems_pruned()` uses a necessary condition. The left edge `x0` of a maximal
  space must sit where the height map **steps down going right**
  (`hmap[x0-1, y] > hmap[x0, y]` for some y in the interval). Otherwise the
  footprint could widen for free. The same holds for the other three sides.
  Only `O(#boxes)` rows and columns can bound a space, which gives about 2,900
  candidates at S = 70 instead of 6.2 million. It loops over bins in Python,
  but each iteration is vectorised.

**Dispatch:** the pruned version is used when `Lx · Ly ≥ EMS_PRUNE_S² = 625`.
This crossover was measured: the pruned version is 0.2× the speed of the
exhaustive one at S = 10, 1.0× at S = 24, 3.5× faster at S = 40 and 18× faster
at S = 70 (4,693 ms to 255 ms per step with 64 bins). Area is used rather than
side length because the footprint count drives the cost.

**Candidates from EMS** (`_ems_corners`): for every space that is wide enough,
deep enough and tall enough (`floor + sz ≤ Lz`), mark its four bottom corners
`(x0 or x0 + wx - sx) × (y0 or y0 + wy - sy)` as valid positions.

*Why enumerate instead of maintaining the list incrementally (the Lai & Chan
difference process used by the official code):* the enumeration reads only the
height map, so it is **stateless**. It cannot carry stale spaces across an
episode boundary, imposes no ordering artefacts, and can be tested against
brute force (`test_ems_list_is_exact`, `test_pruned_ems_matches_the_exhaustive_scan`,
`test_pruned_ems_survives_episode_boundaries`). Forcing the pruned path at
S = 10 was measured to leave every observation bit-identical over 2,000 steps
× 64 bins for all six heuristics, and to reproduce six trained checkpoints'
held-out scores with a maximum difference of 0.

### 6.7 Stability rules

Two rules can be selected with `stability=`:

**`"com"` (default): centre of mass over the support, plus a contact-area
floor.**

- *Centre-of-mass span test* (inner function `span` in `_feas_one`). Along
  each axis, the support must cover both the near half and the far half of the
  box, meaning some contact cell lies at or before the centre and some at or
  after it. This is the standard grid test for "the projection of the centre
  of mass falls inside the convex hull of the contact cells". It is computed
  from sub-window maxima: a sub-window touches the contact layer exactly when
  its max equals `z`.
- *Contact-area floor* (`min_support`). The fraction of the base resting on
  the contact layer (`count / (sx · sy)`, from the `(max, count)` sweep) must
  be at least `min_support`. `SUPPORT_EPS = 1e-6` absorbs float rounding at
  exact ratios such as 4/5 vs 0.80.
- When `min_support = 0` and the stability rule is `com`, the count sweep is
  skipped entirely (`_needs_count`), because nothing reads it.

**`"cdrl"`: the rule as the paper's citations write it** (Zhao et al.). Stable
if (support > 60% and all 4 bottom corners supported), or (> 80% and 3
corners), or (> 95%). It is kept for comparison and ignores `min_support`.

**Why `com`: the rule was identified, not read off the paper.** AR2L says
stability follows Zhao et al.'s constraints. Implemented literally (`cdrl`),
*every* method caps out around 36% utilisation, about 25 points below every
number the paper reports, heuristics included. So that cannot be what was run.
The paper's six heuristics are fully specified algorithms, so they pin the
simulator down. `scripts/calibrate.py` scores them under each candidate rule:

| Stability rule | Mean absolute gap to the paper's heuristic numbers |
|---|---|
| 60% + 4 corners (as written) | 24.2 |
| 4 corners only | 25.8 |
| area ≥ 60% | 9.6 |
| area ≥ 40% | 6.3 |
| area ≥ 20% | 6.8 |
| no stability check | 7.4 |
| **centre of mass over support** | **6.7** |

A bare 40% threshold fits marginally better, but it is a number chosen to fit
the data. The centre-of-mass rule is physics, and it also reproduces HMM's
item count (22.4 against the paper's 22.6). The official code, read later,
confirmed that it uses a centre-of-mass test (`space.py: check_box`).

**Why a contact-area floor on top.** The centre-of-mass rule accepts a box
balanced on a third of its base. That is a valid state in the simulator but a
fallen box on a real pallet. The floor makes placements physically credible.
It costs utilisation, and the right value depends on the box mix:

- On the 10³ paper setup, an 80% floor cost about 15 points on every
  heuristic.
- On the current pallet config, 0.70 left a median of only 4 legal placements
  and capped DBL at 18%. **0.30 costs about 2 points** against the bare
  centre-of-mass rule and leaves the action space intact, so the current
  default is **0.30**.

**Known limitation.** The official code's stability test is stricter in other
ways: the convex hull is shrunk by 10%, the centre of mass of the *whole stack
above* is used, and the check recurses down the pile. This simulator checks
only the new box against its immediate support, so it is more permissive and
cannot represent toppling (open item in `TODO.md`).

### 6.8 Box types and the same-type stacking rule

**The type is the SKU.** `types:` in the config lists one size class per type:
`{p: share, lo: [x, y, z], hi: [x, y, z]}`. `sample_items` draws a type
first (`rng.choice` with the normalised shares), then that type's sides from
its own per-axis bounds, in a single vectorised `rng.integers` call with
per-item bounds. The environment recomputes `size_lo` / `size_hi` as the
envelope of the classes, because the sweeps are sized from it.

**The rule.** A box may rest only on boxes of its own type; the floor takes
any type. What "rest on" means is the whole content of the rule: **only the
cells the box actually touches are checked.** An overhang bridging a foreign
type is legal; a contact patch on a foreign type is not.

**Implementation (`_type_sweep`, `_type_ok`).** For an incoming type `T`,
build a *blocked-height map*:

```
hbad[x, y] = hmap[x, y]   if the column is topped by a box of a type other than T
           = -1           if the column is bare floor, or topped by a box of type T
```

A footprint is type-legal **exactly when `max(hbad)` over it is below the
landing height `z`**. The reasoning: every cell the box touches has height
`z`. If any of those cells belongs to a foreign type, `hbad` there equals `z`,
so the max is not below `z`. Cells lower than `z` are voids the box bridges,
and their `hbad` is below `z` whatever their type. This reuses the same
window-max machinery: one extra y-sweep (cached per type vector) and one
x-sweep. `tests/test_types.py` checks it against an explicit voxel grid.

**What each candidate reports as "the type underneath" (`_under_sweep`,
`_type_under`).** A second coded sweep packs height and type into one integer:

```
code = h · (n_types + 1) + (t + 1)      (bare column → -1)
```

This is monotone in `h`, because `t + 1 < n_types + 1`. So the window max still
gives the landing height in its quotient and, among the cells at that height,
the largest type in its remainder. Under the stacking rule, every touched cell
has the same type, so the remainder *is* the type underneath. With one type,
`_type_under` skips the sweep: it is 0 above the floor and `TYPE_FLOOR` on it.

**Switches:**

- `n_types: 1` with `types: null` gives exactly the pre-type simulator.
- `type_constraint: 0` keeps `type_id` in the state but lets anything stack on
  anything.
- `type_blocked()` is a diagnostic: it reports bins whose front box is blocked
  *only* by the type rule (it has geometric room but none on its own type).
  With DBL on a FIFO conveyor, 240 of 256 episodes ended this way, not because
  the pallet was full.

**Why this matters and what it motivates.** On a FIFO conveyor (reach 1) with
three interleaved types, the stream decides most of the packing. The first box
of each type grabs floor space, and then the episode ends as soon as no box
can go anywhere. Measured with DBL on the 30 × 25 × 40 pallet, `min_support
0.30`:

| Setting | Utilisation | Boxes |
|---|---|---|
| One type, no rule (pre-type simulator) | 61.1% | 40.1 |
| Three types, rule off | 64.5% | 53.6 |
| Three types, rule on, FIFO | 29.4% | 24.3 |

A naive selector ("take the reachable box with the most open placements")
does even *worse* (17.2%), because "most placements open" means "the empty
floor". Choosing which box to hand over is a real policy, which is what the
`select` algorithm and the AR2L permuter learn.

### 6.9 The conveyor: window, reach, preview, permutation

- **`window()`** returns the next `nb` boxes from `head`, with a validity mask
  (the sequence may be shorter than the window near its end).
- **`n_pick`** splits the window into a **pickable prefix** (the pick station,
  the first `n_pick` slots) and a **preview tail** (seen but not reachable).
  For example, `nb = 11, n_pick = 5` means the robot reaches 5 boxes and sees 6
  more upstream. `n_pick = 1` is the paper's FIFO conveyor; `n_pick = nb`
  (the default when `null`) is free choice over the whole window.
  - The tail is masked out of the *pointer* only (`b_pick`). The *encoder*
    still attends to it (`b_mask`). So preview boxes inform the choice without
    being choosable.
  - *Why masks rather than a different architecture:* it makes "utilisation
    vs how many boxes the station can reach" a one-parameter sweep with the
    same network.
- **`permute(idx)`** moves window slot `idx` to the front and keeps the others
  in order, a *move-to-front* permutation (the paper's attacker semantics).
  Moving slot `i < n_pick` permutes only the prefix among itself, and
  `head += 1` in `step` then slides exactly one preview box into reach, so the
  split stays consistent automatically.
- **`pick_feasible`** (the station filter). When a type rule can block one box
  while its neighbour is fine, `b_pick` also removes reachable boxes with *no
  legal placement* (`_placeable`, one feasibility sweep per reachable slot).
  The permuter is then pointed only at boxes it can actually hand over, and the
  episode ends only when the **whole station** is blocked. It defaults to on
  only when it can matter (`type_constraint and n_types > 1 and n_pick > 1`),
  because it roughly doubles the step cost at `n_pick = 5`.

### 6.10 Stepping, reward and termination

`step(action)`:

1. Decode `action` (an index into L) to `x, y`, rotated dims and `z`.
2. Raise the footprint in `hmap` to `z + sz`, and set `tmap` over the footprint
   to the box's type (it is now the top of those columns).
3. Append the box to `packed` / `ptype`. If `n_packed` reaches capacity,
   **double the arrays** (`max_c` is an initial capacity, not a limit).
4. `reward = box volume / bin volume`, only for bins that were alive.
5. `head += 1`, invalidate the caches.
6. `done` is set if the box could not be placed, the sequence ended, or the new
   front box has no feasible placement.

*Why a dense per-step reward:* the paper uses one terminal reward equal to
final utilisation with γ = 1. The sum of per-step volume fractions is exactly
that same number, so the return is identical, but the variance of the
advantage estimates is much lower.

*Why grow `max_c` instead of clamping:* an earlier version clamped the write
slot, so box 81 silently overwrote box 80. That was a wrong *state*, not just a
wrong count, and callers indexing `packed[n_packed - 1]` crashed. Large pallets
of small boxes hold hundreds of boxes.

`reset_done()` re-rolls only finished bins during rollout collection, so all
64 bins stay busy.

### 6.11 Instance sources: generator or pool

- **Generator** (default when `train.data` is empty): `sample_items` from the
  type classes. The untyped cube case deliberately keeps the original scalar
  `rng.integers` call, because passing array bounds consumes the RNG stream in a
  different order and would silently re-roll every stored dataset.
- **Pool** (`pool=`): a fixed `(n, max_boxes, 4)` table of real pallets. Each
  episode draws one pallet at random with replacement, optionally reordered by
  `pool_order_random`. Trailing all-zero rows are padding, and `length[n]`
  marks each pallet's real end. `_as_seq` validates that padding only appears
  at the end, that no box has a zero side, and that the types are in range. A
  three-column array (a pre-type dataset) loads as all type 0.

---

## 7. The neural networks — `ar2l/model.py`

### 7.1 Overview

There are three networks with the same backbone, following paper Appendix A.4:

| Network | Class | Reads | Pointer over | Trained with |
|---|---|---|---|---|
| Packer `π_pack` | `PackNet` | C, B, L | candidates L (`l_mask`) | PPO on the packer's return |
| Attacker `π_perm` | `PermNet` | C, B | window B (`b_pick`) | PPO on the **negated** packer return |
| Mixture model `π_mix` / selector | `PermNet` | C, B | window B (`b_pick`) | PPO on the packer return (+ distance loss for `exact`) |

Each network also has its own **critic** `V(C, B)`. The default size is width
64, 1 head, 1 attention layer: about 65–70k parameters per network, the
paper's size.

### 7.2 Encoder

```
for each node group k in (C, B[, L]):
    h_k = Linear_k( concat(features_k, TypeEmbedding(type_k)) )      # element-wise projection
    if k is B: h_k += sinusoidal positional encoding                  # conveyor order matters
    h_k *= mask_k                                                      # zero out padding
x = concat(h_C, h_B[, h_L])                                            # one token sequence
repeat n_layer times:
    x = LayerNorm(x + MultiHeadAttention(x, x, x, key_padding_mask))
    x = LayerNorm(x + FF(x))              # FF = Linear → ReLU → Linear
x[pad] = 0
xbar = mean of x over non-padding tokens                               # global context
return x, xbar
```

Design points:

- **Independent projections per node group.** Packed boxes have 6 features,
  window boxes 3, candidates 6, and they mean different things, so each group
  gets its own linear layer into the shared 64-dimensional space. This is the
  paper's design. The official code instead uses one uniform 9-dimensional
  node vector with a 2-layer embedder (a noted, minor difference).
- **Type embedding concatenated before the projection.** A trainable
  `nn.Embedding(n_types + 1, type_embed = 16)` per encoder. Concatenating it
  onto the physical features makes each projection 16 columns wider (C: 6 + 16
  = 22, B: 3 + 16 = 19, L: 6 + 16 = 22), and *nothing downstream changes*.
  *Why:* it is the smallest change to the architecture, and it makes old
  checkpoints loadable (see 11.4).
- **Positional encoding on B only.** The window is a *sequence* (which box is
  in front matters), while packed boxes and candidates are *sets*.
- **All-padding guard.** A bin with nothing packed and nothing feasible would
  be all padding, and attention over it would produce NaN. The first key is
  always unmasked (`safe[:, 0] = False`), and the output is re-masked
  afterwards.

### 7.3 Pointer head (paper Eq. 28)

```
logits_i = c_temp · tanh( (W_q · xbar)ᵀ (W_k · x_i) / √d )  +  (0 if mask_i else -1e9)
π(·) = softmax(logits)
```

`c_temp = 10` bounds the logits to [-10, 10]. This keeps the policy from
becoming infinitely sharp early in training while still allowing near-greedy
choices. Masked nodes get -1e9. The pointer outputs one score per *existing
node*, so the action space can vary in size from state to state, which is
exactly what a candidate list needs.

### 7.4 Critic V(C, B)

The critic has its own encoder over C and B only, followed by a
`Linear → ReLU → Linear(1)` head on `xbar`. It **does not read L**. *Why:*
approximate AR2L (Algorithm 2) must evaluate the *same* state under both the
nominal conveyor and an attacked (permuted) conveyor. A permuted state has a
different front box and therefore a different candidate list. Building that
list would mean stepping a copy of the environment. With `V(C, B)`, the
permuted state is just the observation with the B arrays reordered
(`move_to_front`). It is also cheaper.

The critic takes the observation *dict* rather than loose tensors. A robust
target built by hand from rollout data that silently lacked one key would be a
value computed on the wrong state rather than an error.

### 7.5 PermNet masks

The encoder receives `b_mask` (everything visible). The pointer receives
`b_pick` (only what can be taken). If a hand-built dict lacks `b_pick` (the
approx bootstrap), it falls back to `b_mask`.

### 7.6 Sampling

`sample(logits, greedy)` replaces any non-finite logit with -1e9 before the
softmax. *Why:* a NaN logit makes `Categorical` return an out-of-range index,
which only surfaces much later as an illegal CUDA memory access in a gather
call. It returns the action, its log-probability and the entropy. Greedy mode
(argmax) is used at evaluation.

---

## 8. PPO and the robust value targets — `ar2l/ppo.py`

### 8.1 GAE with an optional robust next-value

```
for t = T-1 … 0:
    v_next = next_val[t] if given else V(s_{t+1})
    δ_t    = r_t + γ · v_next · (1 - done_t) - V(s_t)
    A_t    = δ_t + γ · λ · (1 - done_t) · A_{t+1}
return A, A + V                     # advantages, value targets
```

Defaults: γ = 1 (undiscounted, as in the paper), λ = 0.95.

**Design choice: the robust value goes inside the TD residual.** A robust
Bellman operator only defines the value of the *next* state. Substituting the
dual value for `V(s_{t+1})` inside δ keeps multi-step credit assignment. The
alternative, using the dual as a one-step target, cost RfMDP and ApproxAR2L
several points: `TODO.md` records RfMDP going from 61.7% to 66.2% (+4.5) when
the dual was moved into the residual, and `README.md` puts the cost at about 7
points. The loss had nothing to do with robustness: those two would have been
the only methods doing one-step credit assignment over a 30-step episode.

### 8.2 The PPO update

- The optimiser is Adam (lr 3e-4, eps 1e-5), with no learning-rate decay.
- There are 4 epochs × 4 minibatches over each rollout of 64 × 30 = 1,920
  transitions.
- Advantages are normalised per update.
- Loss = clipped surrogate (clip 0.2) + 0.5 · MSE value loss − 0.01 · entropy,
  plus an optional `aux(obs, idx, logp_all)` term.
- An optional per-sample `weight` (used by CVaR-PPO) multiplies the surrogate.
- Gradient-norm clip 0.5.
- Statistics are accumulated on the device and synchronised once at the end,
  to avoid a GPU sync per minibatch.

### 8.3 `sup_tv_dual`: Eq. 18 for approximate AR2L

**Setting.** The uncertainty set mixes the nominal next-state distribution and
the attacked one, with weight α, within a total-variation radius ρ. The
supremum of E[V] over that set has a dual form in two Lagrange multipliers μ₁
and μ₂. Given `v_o` (nominal next-state values), `v_w` (attacked next-state
values) and `v_all` (the state values standing in for the λ term):

```
λ(μ₁, μ₂) = max( max_s ReLU(V(s) - μ₁),  max_s ReLU(V(s) - μ₂) )
objective(μ₁, μ₂) = mean ReLU(v_o - μ₁) + α · mean ReLU(v_w - μ₂) + μ₁ + α·μ₂ + ρ(1+α)·λ
(μ₁*, μ₂*) = argmin over a 96 × 96 grid spanning [min V - 0.05, max V + 0.05]
target_i = [ ReLU(v_o,i - μ₁*) + α·ReLU(v_w,i - μ₂*) + μ₁* + α·μ₂* + ρ(1+α)·λ* ] / (1 + α)
target clamped to [0, 1]
```

Design decisions:

- **μ₁ and μ₂ are shared across the rollout batch**, not per-state functions
  as in the paper. With one next-state sample per state, per-state
  multipliers are not identifiable. Sharing them across the batch is Panaganti
  et al.'s practical choice, and it avoids training two more networks.
- **Grid search instead of gradient descent.** The problem is 2D and convex. A
  96 × 96 grid is exact enough, cheap on the GPU, and has no step size to tune.
- **Clamping to [0, 1].** The operator is optimistic by up to `ρ · range(V)`
  per application, and the paper's Theorem 2 guarantees a contraction only for
  γ < 1. This MDP is undiscounted. Without an anchor, the critic was measured
  running away to V ≈ 3,400, where the true return is a utilisation in
  [0, 1]. Projecting onto [0, 1], which contains the true fixed point, is
  non-expansive and restores a fixed point. This lifted ApproxAR2L from 60.5%
  to 64.7%.
- **Sanity property:** with ρ = 0 the dual collapses to the α-weighted mean
  `(v_o + α·v_w) / (1 + α)`. This is a unit test.

### 8.4 `inf_tv_dual`: the RfMDP baseline

The pessimistic mirror: the infimum of E[V] over a TV ball of radius ρ around
the nominal distribution.

```
ν* = argmax over a 192-point grid of   ν - mean ReLU(ν - v_o) - ρ · max ReLU(ν - V)
target_i = ν* - ReLU(ν* - v_o,i) - ρ · max ReLU(ν* - V),   clamped to [0, 1]
```

---

## 9. The training algorithms — `ar2l/train.py`

### 9.1 Rollout collection: `Runner.collect(T, permuter, greedy, keep_perm)`

For each of T steps:

1. `env.reset_done()` restarts finished bins.
2. If a permuter is given: build the `(C, B)` observation, sample a slot,
   optionally record the permuter's observation, action, log-probability and
   value, then `env.permute(idx)`. **The permuter acts before every packing
   step.**
3. The packer (a network, or a heuristic when attacking one) chooses a
   candidate from the full observation. The environment steps.
4. Rewards, done flags, values and log-probabilities are recorded. Finished
   episodes update a rolling tracker of utilisation and box count.
5. **CVaR labelling:** every step is tagged with the return of the episode it
   belongs to. For episodes still running at the end of the rollout, the label
   is the reward banked so far plus the critic's estimate. An earlier bug
   relabelled steps of a *previous* episode in the same window; it is fixed.

Final values `V(s_T)` are bootstrapped for both the packer and the permuter.
An earlier bug bootstrapped the permuter critics from zero.

`greedy_pack=True` (set by `--freeze_pack`) makes the frozen packer act
greedily during attacker training.

### 9.2 `move_to_front(o, idx)`

This is a *tensor-side* version of `env.permute`, used by the approx bootstrap.
All window-indexed arrays (`b`, `b_mask`, `b_type`, `b_pick`) are permuted
**together**, so a permuted state stays consistent and the sizes cannot drift
away from their types. It is tested against the environment's own permutation.

### 9.3 The eight algorithms

**`pct`: PCT baseline.** Nominal rollout, then a plain GAE and PPO update of
the packer.

**`cppo`: CVaR-PPO.** Nominal rollout. Each sample gets weight
`1{episode return ≤ q-quantile} / q`, so the policy update only sees the worst
`q` fraction of trajectories (`cvar_q = 0.5`; the paper does not state it).

**`rarl`: robust adversarial RL.** One rollout under the attacker. The
attacker is updated on `−reward`, and the packer is updated on the *same*
attacked rollout. The packer never sees the nominal ordering.

**`rfmdp`: robust-feasible MDP.** Nominal rollout. `v_o = V(next states)`, the
pessimistic target is `inf_tv_dual(v_o, ρ, V)`, and it is used as `next_val`
in GAE.

**`exact`: exact AR2L (Algorithm 1).** Three PPO updates per iteration:

1. Attacker rollout; attacker update on `−reward`.
2. Mixer rollout (the mixer permutes the conveyor); mixer update on `+reward`
   with the auxiliary distance loss of Eq. 6:
   ```
   L_mix = −η(π_pack, π_mix)  +  dist_coef · [ −log π_mix(slot 0)  +  α · KL(π_mix ‖ π_perm) ]
   ```
   - `−log π_mix(0)` is the cross-entropy to "no permutation" (slot 0 stays in
     front). It stands in for the paper's `D_KL(π_mix ‖ 1{x = b_{t+1,1}})`,
     which is infinite for any non-point-mass `π_mix`. The official code uses
     the same surrogate.
   - The KL to the attacker is computed with the attacker's logits under
     `no_grad`, so only the mixer moves.
3. Packer update on the same mixer rollout.

   α sets how far the training dynamics may move toward the adversary: α = 0
   means stay nominal, and larger α means more robust.

**`approx`: approximate AR2L (Algorithm 2).** A nominal rollout only. For each
next state `s'`:

- `v_o = V(s')` on the nominal conveyor.
- `v_w = V(move_to_front(s', attacker sample))` on the attacked conveyor.
- The target is `sup_tv_dual(v_o, v_w, α, ρ, V(s))`, used as `next_val` in GAE.

  The attacker is not trained by an `approx` run's own nominal stage. It is
  trained in stage 1, which runs for `rarl`, `exact`, `approx` and `attack`.

**`attack`: train an attacker against a frozen packer.** It uses `--init
<packer ckpt> --freeze_pack` (or `--heur_pack dbl` etc. for a heuristic).
Three choices make it stronger, all aimed at training it under the same
conditions it is tested in:

1. The frozen packer plays **greedily**, as it will at test time.
2. `best.pt` is chosen by **held-out attacked utilisation, with both sides
   greedy** (`attack_score`), and *lower is better*.
3. The entropy bonus is **annealed** to `--ent_final` (0.001 in the grid), so a
   policy deployed greedily actually sharpens.

**`select`: cooperative selector (a project extension).** The mirror of
`rarl`: one rollout under the mixer network, which is updated on `+reward`
with **no distance loss**, and the packer is updated on that same rollout. The
permuter learns to hand over the box that packs best. No attacker is trained,
which is the saving over `exact --dist_coef 0`. With `--n_pick k` it chooses
among the `k` boxes the cell can reach. This is the algorithm for real
palletising with box types.

### 9.4 Held-out evaluation and checkpoint selection

Every `eval_every` iterations (and at the last iteration):

| Algorithm | Held-out metric | `best.pt` criterion |
|---|---|---|
| `attack` | `attack_score(packer, attacker)` → `att_util` | lowest `att_util` |
| `select` | `attack_score(packer, mixer)` → `sel_util` (the packer *with* its selector) | highest |
| all others | `nominal_score(packer)` → `nom_util` | highest |

*Why `select` is scored with its selector:* scoring the packer alone on the
plain conveyor would measure it without the component it was trained with,
and would pick `best.pt` on a number the run is not optimising.

Evaluation instances are either a fixed generated slice (seeded) or, when
training on data, the **held-out pallets**.

### 9.5 Checkpoints, resume and data safety

- `last.pt` is written every `save_every` iterations. `best.pt` is written on
  improvement. Both contain all three networks, all three Adam states, the
  full `args`, the iteration and the best score.
- `--resume` restores networks, **optimiser moments** (dropping them causes a
  visible transient on every restart) and **the best score** (otherwise the
  first save after a resume always "wins" and overwrites a better `best.pt`).
- **Data fingerprint:** the SHA-1 of the `--data` file is stored in `args`.
  `--resume` refuses to continue if the file changed. This was added after
  runs were accidentally glued together from two different datasets.
- **Hold-out split:** `--holdout 0.2` sets aside 20% of pallets (seeded by
  `--seed`). The split is written to `train_ids.txt` / `holdout_ids.txt`. With
  `--holdout 0`, evaluation uses the training pallets.

### 9.6 Logging

`log.jsonl` gets one JSON record every `log_every` iterations: iteration,
rolling training utilisation and box count, wall time, the held-out metric,
and the PPO statistics per network (`pack_pg`, `pack_vf`, `pack_ent`,
`pack_kl`, `mix_aux`, and so on). A tqdm progress bar writes to **stderr**, so
stdout stays a clean, pipeable log. `torch.compile` is off by default, because
with dynamic node counts it miscompiled the pointer head for some `N_B`
(showing up as illegal memory access).

---

## 10. The heuristic baselines — `ar2l/heuristics.py`

All six heuristics score **the same candidate list L the network sees**
(`env.obs()["l_mask"]`, `env._lxy`, `_lz`, `_ldim`), so they can never drift
from the simulator. Each is a primary score plus a deepest-bottom-left
tie-break, and the action is the argmax.

| Name | Idea | Score used here |
|---|---|---|
| `dbl` | Deepest-bottom-left | `−(z·10⁴ + y·10² + x)` |
| `hmm` | Height-map minimisation | Minimise the void sealed under the box: `sx·sy·z − Σhmap(footprint)`, from a 2D prefix sum |
| `lsah` | Least surface area of the pile | Minimise the surface area of the bounding box of all packed boxes plus this one |
| `bmf` | Best-match-first | Maximise how many of the box's sides match the free extents (x, y and height to the lid), then minimise wasted free volume |
| `onlinebph` | Deepest-bottom-left with fill | Lowest z first, then least wasted free volume |
| `macs` | Maximise accessible convex space | For the top 24 DBL candidates, simulate the placement and keep the one that leaves the largest empty cuboid |

Scores are combined with large multipliers (10⁷, 10¹³) so that the primary
criterion dominates and DBL only breaks ties. BMF, OnlineBPH and MACS are
reconstructions from citations (the paper does not state them), so the spread
among them is not evidence about the simulator.

*Why heuristics matter here:* they never train, so they isolate the
**simulator and action space** from the learning. That is how the stability
rule was identified (section 6.7) and how rotation and EMS were valued
(section 6.4).

---

## 11. Evaluation — `ar2l/evaluate.py`

### 11.1 `run(seqs, policy, nb, attacker, attacked, …)`

Plays every instance once, in batches of `eval.batch` (256) bins, greedy by
default. `policy` is a network or a heuristic name. For instances flagged in
`attacked`, the attacker (or selector) permutes the conveyor before every
packing step. Unflagged instances use slot 0, which leaves the order unchanged.
The environment is built with `types=False` because the instances carry their
own types, and `size_hi` is taken from the data. It returns per-instance
utilisation and box counts.

### 11.2 The β-mixture datasets (paper Table 2)

`β` is the **percentage of test instances reordered by the policy's own
trained attacker**. A seeded permutation picks which instances are attacked
for each β in {0, 25, 50, 75, 100}. β = 0 is nominal; β = 100 is fully
adversarial.

### 11.3 Metrics

- **Uti.**: mean space utilisation (%).
- **Std.**: standard deviation of utilisation across instances.
- **Num.**: mean number of packed boxes.
- **All** (real orders): the share of pallets whose *every* box was placed.
- `--per_pallet` writes `pallet_id, beta, boxes, placed, util_pct` rows.

### 11.4 Loading checkpoints and pre-type migration (`load_nets`, `_fit_state`)

The network shape (width, heads, layers, `c_temp`, `n_types`, `type_embed`) is
read from the checkpoint's `args`. A checkpoint trained **before box types**
has narrower input projections. `_fit_state` pads the missing columns with
**zeros**, which is the identity on the features the network was trained with,
so the loaded packer computes exactly what it used to. The type embedding is
left at its fresh initialisation. Any *other* mismatch (a missing key other
than the embedding, or stray keys) is an error, because silently keeping fresh
weights would report a partly untrained policy as trained.

### 11.5 CLI

`python -m ar2l.evaluate --ckpt … [--attacker … --permuter attacker|mixer]
--nb … --n_pick … --beta … --data <npy|csv> [--pallet_cm L W H --cell_cm
--box_scale --box_round] [--order_random R --order_seed] --n_inst --batch
--rot --min_support --n_types --type_constraint --out --per_pallet`.

**The environment flags must match what the policy was trained under**: `nb`,
`n_pick`, `rot`, `min_support`, `n_types`.

---

## 12. Real pallet orders — `ar2l/orders.py`

### 12.1 CSV format

One row per box, in arrival order:

```
pallet_id,seq,length_cm,width_cm,height_cm,type,qty
P001,1,24,14,12,0,8
```

- Required columns: `pallet_id`, `length_cm`, `width_cm`, `height_cm`.
- Optional columns: `seq` (order within the pallet; file order if absent),
  `type` (default 0) and `qty` (repeat the row).
- Height is the vertical side. Length and width lie along x and y, and the
  environment yaws the box itself when `rot = 2`.
- A `pallet_id` may be spread through the file (a cell building two pallets
  from one conveyor); its boxes are gathered in file order.

### 12.2 From centimetres to cells

1. Divide every side by `box_scale` (0 or 1 means unchanged; 2 halves every
   side; 4 takes a quarter). The pallet is **not** scaled. Values between 0 and
   1 are rejected, because 0.5 would *grow* boxes and reads too easily as
   "halve".
2. Divide by `cell_cm` and round with `box_round`:
   - `up`: a box is never modelled smaller than it is.
   - `down`: never bigger (a packing may then be slightly too tight in
     reality).
   - `nearest`: an exact half goes down (1.5 → 1, 1.51 → 2).
   - Every side is at least 1 cell. `EPS = 1e-6` stops float noise such as
     24/2 = 12.000001 from moving a side by a cell.
3. The pallet `pallet_cm` is rounded **down** to cells, so it is never modelled
   bigger than it is. With no `pallet_cm`, boxes go onto `env.bin`.
4. A box that fits in no allowed orientation is an **error** with a message
   that points at `--pallet_cm`, not a silently shortened pallet.

The output is `(n_pallets, max_boxes, 4)` int16, zero-padded at the end, plus
the pallet IDs.

### 12.3 `randomize_order(seqs, amount, seed)`

This is a *noisy sort*. Box `i` of an `n`-box pallet gets the key
`(1 − amount)·i/n + amount·U(0, 1)`, and the boxes are re-sorted by key.
`amount = 0` keeps the data order and `amount = 1` gives a uniform random
permutation. In between, a box drifts by about `amount / (1 − amount)` of the
pallet: local swaps, the way a real conveyor jitters, rather than a few boxes
thrown to the far end. Padding stays last. It is used both for training
variety (`train.order_random`) and for evaluation robustness
(`eval.order_random`).

### 12.4 The data in `data/`

| File | Content |
|---|---|
| `orders_all.orig.csv` | 4,375 real boxes in 89 pallets (1–97 boxes each); sides 7–67 cm; types 0 (1,002 boxes), 1 (1,057), 2 (2,316). |
| `orders_all.csv` | The same boxes regrouped into **59 pallets of 74–75 boxes**. |
| `orders_all.merge_map.csv` | Which source pallets and box counts make up each merged pallet. |
| `orders_example.csv` | A small example showing every column, including `seq` and `qty`. |
| `pallet_2cm_test.npy` | 3,000 × 120 × 4: generated test set for the pallet config (typed). The default evaluation set. |
| `pallet_2cm_untyped.npy` | 3,000 × 120 × 3: the same geometry, no types. |
| `discrete_test.npy` | 3,000 × 150 × 3: the paper's 10³ setup (sides 1–5), untyped. |

---

## 13. Viewers and tools — `ar2l/viz/`

All viewers play through `BPPBatch`, so what they draw comes from exactly the
simulator the agent was trained in. They share a style (dark/light palette,
card layout, HTML in a sibling `.html` file), a `QuietServer` that ignores the
aborted response a browser refresh causes, and the flags `--port`, `--host`
(default 127.0.0.1), `--root` and `--fps`.

Policy IDs (`agents.py`): `run:<name>[:best|last]`, `heur:<dbl|…>`, `random`.
Attacker IDs: `run:<name>` (the checkpoint's attacker), `mix:<name>` (its
mixer or selector), or none.

- **`game`** (port 8096). A human packs a bin by clicking placements, then the
  same instance is handed to an opponent (a trained run or a heuristic) and
  the two are scored side by side.
  - The setup form is filled from **the opponent's own `args.json`** (bin, box
    bounds, `nb`, `n_pick`, `max_l`, `rot`, `ems`, stability, support floor),
    so the duel happens on the geometry the agent knows. A command-line flag
    still takes precedence over the run's value.
  - `POST /api/check` re-validates the whole parameter set on each keystroke,
    and `POST /api/new` refuses sets the simulator could not run. So any
    accepted set is one `BPPBatch` accepts (pinned by tests). Fields moved
    away from the run's values are flagged as "off-distribution".
  - With `n_pick = k > 1`, the conveyor is drawn as a pick station showing how
    many placements each reachable box still has. A pick is applied to the
    order shown on screen, so changing your mind is one permutation, not two
    composed ones.
  - An attacker can be chosen to reorder the conveyor for both players. A
    `select` run replayed without an attacker gets its own selector back.
  - `POST /api/recalc` replays the **same boxes in the same order** with any
    model under edited parameters, one row per setting. `n_items`, `size_lo`
    and `size_hi` are held at the session's values, because changing them
    would deal a different sequence.
  - Boxes can come from the generator or from an orders CSV (one pallet per
    game).
  - The top view scales to fit, with a cell-size slider saved in
    `localStorage`.
- **`dashboard`** (port 8095). Polls `runs/*/log.jsonl`. It shows held-out
  utilisation, training utilisation and the losses, with the paper's number
  drawn as a dashed target line.
- **`report`**. One self-contained HTML page with Tables 1 and 2 (the paper's
  numbers beside this project's), training curves as SVG and rendered bins as
  base64 PNG.
- **`replay3d`**. The bin built box by box, as PNG frames or a GIF. Each box
  keeps a colour from a golden-angle hue walk.
- **`heatmap`**. The debugging view for one decision: height map, landing
  height per position, which positions survive the EMS and stability filters,
  `π_pack` over them, and `π_perm` over the conveyor.
- **`attack`**. What the attacker learned: the sizes of the boxes it promotes
  against the nominal distribution, how deep in the window it reaches, and
  the damage per `N_B`. This is the measurement behind the paper's claim that
  the attacker prefers small boxes as `N_B` grows, which reproduced here.

---

## 14. Scripts and the experiment pipeline — `scripts/`

| Script | Role |
|---|---|
| `jobs.py policies\|attackers` | A job queue for the training grid: `--par` jobs at a time. `policies` trains one packer per (method, N_B) cell. `attackers` trains a dedicated attacker per frozen packer (greedy packer, annealed entropy, best by attacked utilisation). `--extra` is appended to every command. |
| `attack_heur.py` | One attacker per heuristic, for Table 1. |
| `eval_all.py table1\|table2` | Scores the whole grid into `results/table{1,2}.json`. |
| `summarize.py` | Markdown tables with the paper's numbers and gaps, plus the paper's comparative claims checked cell by cell. |
| `make_readme.py` | Rewrites the text between the `RESULTS` markers in `README.md`. Idempotent. |
| `calibrate.py` | The stability-rule identification table (section 6.7). |
| `ablate_env.py` | Heuristics under rot ∈ {1, 2} × ems ∈ {0, 1}: what the action space is worth. |
| `plot_curves.py` | Training curves (paper Fig. 3a/b). |
| `run_all.sh` | End to end: test set → policies → attackers → heuristic attackers → tables → figures → report → README. Every stage is resumable. |
| `after_grid.sh`, `strong_attack.sh` | Follow-up pipelines, including longer 3,000-iteration attackers. |

Method names in the grid: `pct`, `cppo`, `rarl`, `rfmdp`, `ex05` / `ex10`
(exact, α = 0.5 / 1.0), `ap05` / `ap10` (approx, α = 0.5 / 1.0).

---

## 15. Tests — what is guaranteed

Run with `python3 -m pytest tests -q`.

- **`test_env.py`**: feasibility and landing height **cell by cell against
  brute force** (0 mismatches over 192,000 positions); the EMS list is exact;
  the pruned EMS equals the exhaustive one and does not leak across episodes;
  rotation offers both footprints; no overlaps; every box rests on support;
  volume bookkeeping; move-to-front; the `n_pick` reach/preview split and
  refill; heuristics place legally; six deliberately lopsided bins (7×5×9,
  5×9×6, 11×4×7, 6×6×13, 9×12×5, 4×13×11). These catch x/y swaps and
  height/side confusions that a cube would hide. They found two real bugs: a
  wrong EMS wall sentinel, and an RNG-order change that re-rolled datasets.
- **`test_types.py`**: the type stream honours the classes and shares; the
  stacking rule against a **voxel-grid reference**; overhangs over foreign
  types are legal; the floor takes all types; `l_type` is correct; one type
  equals the untyped simulator; `b_pick` hides blocked boxes; the episode ends
  when the whole station is blocked; networks consume types and get gradients
  through the embedding; pre-type checkpoints migrate; no episode ever
  produces a mixed-type contact.
- **`test_algos.py`**: at ρ = 0 the sup-dual equals the mixture mean; the
  duals move in the right direction and stay in [0, 1]; GAE at γ = 1 sums the
  rewards; `move_to_front` matches the environment; ragged rollouts are
  re-padded; networks only choose unmasked nodes.
- **`test_orders.py`**: CSV parsing, rounding modes, `box_scale`, `pallet_cm`,
  error messages, `randomize_order` properties; padding is invisible (a padded
  pallet plays exactly as it would alone); short pallets end at their last box.
- **`test_train_data.py`**: without `--data` the generator is untouched; pool
  episodes come from the pool; the hold-out split is disjoint, seeded and
  written; training on a CSV works end to end; resume refuses a changed file.
- **`test_game.py`**: everything the game form accepts, the simulator runs;
  picks permute exactly once; recalculation replays the same boxes; orders
  games work.

---

## 16. Design decisions: a consolidated table

| # | Decision | Alternative(s) | Why this one |
|---|---|---|---|
| 1 | Height map + type map as the bin state | Voxel grid | Boxes are dropped, so free space is exactly above the height map. O(Lx·Ly) memory, and every query becomes a window max. |
| 2 | Batched NumPy over all bins | Per-bin Python loop; GPU simulator | PPO needs thousands of transitions per iteration. NumPy vectorisation is fast enough (1.7 ms per step for 64 bins) and easy to test against brute force. |
| 3 | Separable running window maxima for landing heights | Per-position brute force; 2D max filter per size | Computes every footprint width at once, reused across box sizes and bins via indexing. |
| 4 | (max, count) pair in the same sweep for contact area | Base-32 histogram (old); a separate count pass | Exact at any bin height or box size. The old histogram overflowed beyond height 12 or 25-cell footprints. |
| 5 | Candidates = EMS bottom corners, both orientations | Every grid position | Same utilisation with rotation, half the candidates, and it is the paper's action space. |
| 6 | Exact stateless EMS enumeration (exhaustive or pruned) | Incremental Lai & Chan difference process | Exact, has no stale state across episodes, is order-free and testable. Pruning gives 18× on large bins with an identical set. |
| 7 | Centre-of-mass stability | The 60%/4-corner rule as literally written | Identified from the paper's heuristics (gap 6.7 vs 24.2), physically grounded, later confirmed in the official code. |
| 8 | Contact-area floor (`min_support`, now 0.30) | Centre of mass alone; 0.80 | Real pallets need boxes to sit on the box below. 0.30 costs about 2 points on the pallet config; 0.70–0.80 destroyed the action space for these box sizes. |
| 9 | One normalisation scale `max(Lx, Ly, Lz)` | Per-axis normalisation | Keeps box shapes undistorted for the network. |
| 10 | Dense per-step volume reward | The paper's sparse terminal utilisation | Identical return at γ = 1, much lower variance. |
| 11 | Type = SKU with its own size class | Types as labels independent of size | Matches real warehouses, where a SKU has fixed dimensions. |
| 12 | Type rule via the `hbad` window max | Per-cell loops; voxel checks | Reuses the existing sweep machinery; exact "only what it touches" semantics; allows overhangs over foreign types. |
| 13 | Coded sweep `h·(K) + t + 1` for the type underneath | A separate lookup after placement | Landing height and underlying type come from one window max. |
| 14 | Type embedding concatenated before the projection, floor as its own row | Separate type tokens; one-hot features | Minimal architecture change, backward-compatible checkpoints (zero padding = identity), and the floor is distinguishable. |
| 15 | `n_pick` via masks (`b_mask` vs `b_pick`) | Separate reach and preview encoders | One dial from FIFO to free choice, same network; the preview informs without being choosable. |
| 16 | `pick_feasible` only when it can matter | Always on | It costs a feasibility sweep per reachable box; it is only needed with a type rule, more than one type and more than one reachable box. |
| 17 | Critic `V(C, B)` without L, separate encoder | Critic over (C, B, L); shared encoder | Algorithm 2 values permuted states without rebuilding L; cheaper; the value and policy heads do not compete for the same features. |
| 18 | Pointer head with tanh and temperature 10 (Eq. 28) | Plain dot-product softmax | Bounded logits keep the policy stable; it handles variable-size action sets. |
| 19 | Cross-entropy `−log π_mix(0)` surrogate | The paper's `D_KL` to an indicator (infinite) | A finite standard surrogate; the official code does the same. |
| 20 | Dual multipliers shared across the batch, grid-searched | Per-state multiplier networks | One next-state sample per state makes per-state multipliers unidentifiable; the 2D convex grid is cheap and exact enough. |
| 21 | Clamp robust targets to [0, 1] | No anchor | At γ = 1 the operator is not a contraction; the critic diverged to about 3,400. The projection is non-expansive and restores a fixed point. |
| 22 | Robust value inside the GAE residual | One-step robust target | Keeps multi-step credit assignment; the one-step version cost about 7 points for reasons unrelated to robustness. |
| 23 | Attacker trained against a greedy frozen packer, best by held-out attacked utilisation, entropy annealed | Sampling packer; rolling training average | Train under test conditions; the attacker was the weakest link. |
| 24 | `select` = rarl with the sign flipped and no attacker | `exact --dist_coef 0` | Same goal, one fewer network and PPO update per iteration. |
| 25 | `max_c` grows by doubling | Clamp the write slot | Clamping silently corrupted the state. |
| 26 | Single config file with patch overrides and hard errors on unknown keys | Defaults spread across argparsers | One source of truth; typos fail loudly; runs record resolved values. |
| 27 | SHA-1 of the data file checked on resume | Trust the user | Prevents two experiments being silently glued together. |
| 28 | Noisy-sort order randomisation | Uniform shuffle with probability p | Produces local jitter like a real conveyor, with one continuous knob. |
| 29 | Pallet rounded down, box rounding configurable, oversize boxes are an error | Silent clipping | Never model the pallet bigger than it is; never silently shorten a pallet. |
| 30 | Keep the RNG call order for untyped data | Refactor freely | Preserves every stored dataset and published number bit for bit. |
| 31 | `torch.compile` off by default | On | It miscompiled the pointer head with dynamic shapes. |
| 32 | Progress bar on stderr | On stdout | stdout stays a clean, pipeable log. |

---

## 17. Deviations from the paper

- **Stability rule:** identified from the paper's heuristics (centre of mass),
  not the rule the citations write down. This is the one substantive
  difference, and it was later confirmed by the official code.
- **EMS:** enumerated exactly rather than maintained incrementally.
- **Reward:** a dense per-box volume fraction instead of a sparse terminal
  reward (same return).
- **Eq. 6 distance term:** the cross-entropy surrogate for an infinite KL.
- **Eq. 18 multipliers:** shared per batch rather than per-state networks.
- **Robust value placement:** inside the GAE residual.
- **CPPO level:** q = 0.5 (not stated in the paper).
- **BMF, OnlineBPH, MACS:** score functions reconstructed from citations.
- **Differences from the official code, not adopted (yet):** PPO 1 epoch × 32
  minibatches, clip 0.1, value coefficient 1.0, LR decay; block alternation
  (10 iterations each of attacker, mixer, packer); reverse KL direction;
  infeasible candidates shown with a flag; a recursive stack-level stability
  check.
- **Not implemented:** the continuous setting and its Intersection-Points
  candidates, the Mixed-Item dataset (Table 3), the CDRL baseline (Table 1),
  and the ρ sweep (Fig. 3c/d).
- **Training budget:** 4,000 iterations against the paper's roughly 28,000.
- **Extensions not in the paper:** box types and the stacking rule, the pick
  station, the cooperative selector, the contact-area floor, non-cubic bins,
  real orders, and the interactive game.

---

## 18. Results and current status

### 18.1 Reproduction (10³ bin, sides 1–5, 150 boxes, `min_support = 0`, 3,000 held-out instances)

Space utilisation (%) at β = 0 (nominal) and β = 100 (fully attacked by the
policy's own attacker):

| Method | N_B = 10 ours | N_B = 10 paper | N_B = 20 ours | N_B = 20 paper |
|---|---|---|---|---|
| PCT | 75.9 / 64.2 | 76.4 / 55.7 | 75.5 / 63.4 | 77.0 / 41.9 |
| CPPO | 75.5 / 64.0 | 75.6 / 57.4 | 74.6 / 56.3 | 74.1 / 45.8 |
| RARL | 75.9 / 65.7 | 74.3 / 63.3 | 74.7 / 62.3 | 72.0 / 58.7 |
| RfMDP | 72.6 / 58.4 | 74.4 / 55.9 | 74.1 / 58.2 | 73.8 / 54.4 |
| ExactAR2L(0.5) | 75.9 / 64.6 | 77.6 / 59.7 | 75.7 / 63.8 | 76.8 / 54.4 |
| ExactAR2L(1.0) | 75.4 / 65.1 | 76.0 / 63.8 | 75.5 / 62.6 | 76.1 / 58.5 |
| ApproxAR2L(0.5) | 69.1 / 57.6 | 76.2 / 56.1 | 71.2 / 50.8 | 75.0 / 53.1 |
| ApproxAR2L(1.0) | 70.1 / 57.8 | 73.6 / 57.1 | 70.5 / 53.3 | 73.4 / 57.6 |

Across all 80 cells of Table 2, the mean signed gap is +1.7 points and the mean
absolute gap is 3.5. Nominal utilisation is within a point of the paper for
five of the eight methods.

**Interpretation.**

- The remaining gap is almost entirely in the **attacked columns, and in one
  direction: the attacker is too weak.** It costs PCT 11.6 points at N_B = 10
  where the paper's costs 20.7, and 12.1 at N_B = 20 where the paper's costs
  35.1.
- Part of that is real: rotation doubles the placements, so a hostile ordering
  has less effect. On DBL, the attack damage is −23.2 with one orientation
  and −18.1 with two, so about 5 points are absorbed by rotation.
- The remaining suspects are block alternation, the PPO settings, and the
  training budget.
- Because the attacked numbers bunch together, the paper's comparative claims
  score near chance. The claim that utilisation falls monotonically in β holds
  in 16/16 cases; ExactAR2L(1.0) packing at least as many boxes as PCT holds in
  9/10.
- **ApproxAR2L** is 4–7 points short nominally. It is the only method with no
  reference implementation, and the suspect is Eq. 18 at γ = 1.

### 18.2 What the action space is worth (heuristics, 512 instances)

| Heuristic | rot1 + EMS | **rot2 + EMS (default)** | rot1, all positions | rot2, all positions | Paper |
|---|---|---|---|---|---|
| DBL | 56.1 | **62.3** | 57.8 | 63.1 | 63.6 |
| BMF | 50.4 | **57.3** | 50.5 | 56.9 | 62.0 |
| LSAH | 52.7 | **58.0** | 53.3 | 57.7 | 60.9 |
| OnlineBPH | 55.0 | **62.8** | 56.6 | 63.5 | 64.1 |
| HMM | 56.3 | **62.6** | 56.0 | 62.0 | 56.1 |
| MACS | 59.5 | **67.4** | 61.3 | 67.2 | 53.0 |

### 18.3 Palletising runs in `runs/` (last logged held-out value)

These are from each run's `log.jsonl`. Each was measured under the config at
the time of that run, so treat them as indicative, not as a controlled table.

| Run | Setup | Held-out utilisation |
|---|---|---|
| `pct_2cm_baseline` | PCT, untyped, 30×25×40 pallet, FIFO, 8,000 iterations | 71.7% (nominal) |
| `sel_2cm_k5` | select, untyped, reach 5 of 10, 8,000 iterations | 72.6% (with selector) |
| `pct_types` | PCT, 3 types, FIFO, 6,000 iterations | 61.7% (nominal) |
| `sel_types_k5` | select, 3 types, reach 5 of 10, 10,000 iterations | **75.5%** (with selector) |
| `sel_orders` | select on real orders (`orders_all.csv`, box_scale 4), reach 5, stopped at iteration 700 of 5,000 | 60.8% (early) |

The headline: with box types, a FIFO packer loses about 10 points, and a
learned selector with a reach of 5 recovers them and more.

### 18.4 Open items (from `TODO.md`)

- A recursive convex-hull stack-stability check.
- Re-running the stability calibration on top of rotation.
- The official PPO settings; block alternation for ExactAR2L.
- Showing infeasible candidates with a flag; a reverse-KL ablation.
- Table 1 rows (PCT with 1 observable box; N_B = 10 and 15).
- A per-step ρ budget, or γ < 1 for the robust value only (Eq. 18 at γ = 1).
- Continuous setting, Mixed-Item dataset, ρ sweep, CDRL baseline.

---

## 19. Current defaults and known documentation drift

**`config.yaml` is authoritative.** The current shipped values:

| Key | Value | Meaning |
|---|---|---|
| `env.bin` | [30, 24, 40] | A 60 × 48 × 80 cm pallet on a 2 cm grid. |
| `env.n_items` | 120 | Boxes per generated episode (episodes end by infeasibility around 40). |
| `env.size_lo` / `size_hi` | 4 / [15, 10, 10] | Box side envelope (8 cm up to 30 × 20 × 20 cm). |
| `env.max_l` / `max_c` | 256 / 128 | Candidate cap / initial packed capacity. |
| `env.rot`, `ems`, `stability` | 2, 1, com | |
| `env.min_support` | 0.30 | |
| `env.n_types`, `type_constraint` | 3, 1 | |
| `env.types` | type 0 and type 1: 12×7×6–7; type 2: 2–12 × 2–12 × 2–10 | Types 0 and 1 share their geometry and differ only by label, which isolates the effect of the stacking rule. |
| `train.algo`, `nb`, `n_pick` | pct, 10, 1 | |
| `train.iters`, `n_env`, `T` | 8000, 64, 30 | |
| `train.alpha`, `rho`, `dist_coef`, `cvar_q` | 1.0, 0.1, 1.0, 0.5 | |
| `train.data` | data/orders_all.csv | Training uses the real orders by default. |
| `train.holdout`, `order_random` | 0.2, 0.0 | |
| `ppo.*` | lr 3e-4, γ 1.0, λ 0.95, 4 epochs, 4 minibatches, entropy 0.01, clip 0.2, value coefficient 0.5, gradient clip 0.5 | |
| `model.*` | width 64, 1 head, 1 layer, `c_temp` 10, `type_embed` 16 | |
| `run.*` | seed 0, cuda, compile 0, `eval_every` 250, `eval_inst` 512, `log_every` 50, `save_every` 250 | |
| `eval.data` | data/pallet_2cm_test.npy | |
| `eval.pallet_cm`, `cell_cm`, `box_scale`, `box_round` | [30, 25, 40], 1.0, 4, nearest | Real boxes quartered onto a 30 × 25 × 40-cell pallet. This is equivalent to a 120 × 100 × 160 cm pallet at 4 cm cells. |

**Where other documents are out of date:**

- `RUN.md` and parts of `README.md` describe older defaults in places: a 10³
  bin, 150 boxes, `min_support` 0.80, `max_c` 80, `max_l` 120. The
  reproduction tables (section 18.1 here, and the results block of
  `README.md`) were measured at `min_support = 0` on the 10³ bin.
  To reproduce them, pass `--bin 10 --size_lo 1 --size_hi 5 --n_items 150
  --min_support 0 --n_types 1` with `types: null`.
- When training on an orders CSV, `pallet_cm` (here [30, 25, 40]) **replaces**
  `env.bin` ([30, 24, 40]) as the bin trained on.
- The checkpoints behind the published tables were archived. The pre-rotation
  ones under `old_run/` cannot be re-run, because the old candidate generator
  (`_ems_axis`) no longer exists.

---

## 20. Glossary

- **3D-BPP**: three-dimensional bin packing problem. *Online* means boxes
  arrive one at a time and placements are final.
- **AR2L**: Adjustable Robust Reinforcement Learning, the paper's method.
- **C_t / B_t / L_t**: packed boxes, observable window, candidate placements
  (the PCT state).
- **N_B (`nb`)**: the number of observable boxes on the conveyor.
- **n_pick (reach)**: how many of the observable boxes the robot can take.
  The rest are preview.
- **EMS**: empty maximal space, a maximal empty box in the bin. Its bottom
  corners are the candidate placements.
- **PCT**: Packing Configuration Tree (Zhao et al. 2022), the baseline
  representation and policy.
- **Attacker / π_perm**: a policy that reorders the conveyor to hurt the
  packer.
- **Mixture model / π_mix**: exact AR2L's policy that generates training
  orderings between nominal and adversarial.
- **Selector**: `π_mix` trained cooperatively (`select`) to help the packer.
- **α (alpha)**: the robustness weight, i.e. how much the adversarial ordering
  counts.
- **ρ (rho)**: the total-variation radius of the uncertainty set.
- **β (beta)**: the percentage of test instances that are attacked.
- **TV dual**: the Lagrangian dual form of an optimisation over a
  total-variation ball of distributions.
- **GAE**: Generalised Advantage Estimation.
- **PPO**: Proximal Policy Optimisation.
- **CVaR**: Conditional Value at Risk, the mean of the worst q-fraction of
  outcomes.
- **RARL**: Robust Adversarial RL (Pinto et al. 2017).
- **RfMDP**: robust MDP with a pessimistic TV-ball value (Ho, Panaganti et
  al.).
- **Height map / type map**: per-column top height and top box type.
- **`TYPE_FLOOR`**: the sentinel −1 for bare floor. It is embedding row
  `n_types`.
- **`min_support`**: the minimum fraction of a box's base that must rest on
  the layer below.
- **Uti. / Std. / Num. / All**: mean utilisation, its standard deviation, mean
  packed boxes, share of pallets fully packed.

---

## 21. Frequently asked questions

**Q: Why does the packer only see candidate positions, not the whole height
map?**
Following PCT, the state is a set of nodes (packed boxes, window boxes,
candidate placements) processed by attention, and the action is a pointer to
one candidate node. This makes the action space exactly the set of legal,
sensible placements. The network never has to learn what is illegal, and it
works on any bin size without changing the architecture. The heatmap viewer
shows the height map for people; the network gets the equivalent information
through C_t.

**Q: How does the network handle a different number of candidates at each
step?**
Padding plus masks. `obs()` trims N_L to the batch maximum, pads the rest with
`l_mask = False`, and the pointer gives masked slots −1e9. Across a rollout,
`flat_obs` re-pads observations of different lengths to a common size.

**Q: What exactly does the attacker change?**
At every step, before the packer acts, it moves one box within reach
(`b_pick`) to the front of the conveyor. The other boxes keep their relative
order (move-to-front). It never changes which boxes exist, only their order.

**Q: Why train a separate attacker for every policy at evaluation time?**
The paper evaluates each policy under *its own* worst case. An attacker
trained against a different packer would measure transfer, not robustness.

**Q: What is the difference between `exact`, `rarl` and `select`?**
All three put a permuter in front of the packer. In `rarl`, the packer trains
purely against the adversary. In `exact`, the packer trains against a mixture
policy that tries to *help* the packer while being kept close to both "no
reordering" and the adversary (α sets the balance). In `select`, the permuter
simply helps the packer with no pull toward the adversary; it is a learned
box-picking policy for a pick station.

**Q: Why does `approx` not need the third network?**
It never generates mixed orderings. It changes only the critic's target: for
each next state it evaluates both the nominal next value and the attacked next
value (by permuting the window in tensor form), and combines them with the
Eq. 18 dual. The adversarial influence enters through the value function
alone.

**Q: Why is the reward given at every step if the paper uses a terminal
reward?**
With γ = 1, the sum of per-box volume fractions equals the final utilisation
exactly, so the objective is unchanged, and the variance is much lower.

**Q: Why can a box overhang a foreign type but not touch it?**
The rule concerns what a box *rests on*. Cells under the box that are lower
than its landing height are voids it bridges, not supports. The `hbad`
window-max test encodes exactly that.

**Q: What happens when no box in the station can be placed?**
With `n_pick = 1`, the episode ends when the front box is blocked. With
`n_pick > 1` and `pick_feasible`, blocked boxes are removed from `b_pick` and
the episode ends only when every reachable box is blocked.

**Q: Can I train on my own pallets?**
Yes. Write a CSV in the orders format, set `train.data` (or `--data`), set
`--pallet_cm` to the real pallet and `--cell_cm` / `--box_scale` for the grid
resolution, and use `--algo select --nb 10 --n_pick 5` for a pick station.
Held-out pallets choose `best.pt`. With few pallets, trust the held-out
numbers, not the training utilisation.

**Q: How do I reproduce the paper's table?**
Use the 10³ settings (`--bin 10 --size_lo 1 --size_hi 5 --n_items 150
--min_support 0`, one type), then run `scripts/run_all.sh`, or the steps in
`RUN.md`: `jobs.py policies`, `jobs.py attackers`, `eval_all.py table2`,
`summarize.py`.

**Q: How do I check that the simulator is correct?**
Run `python3 -m pytest tests -q`. The geometry is checked against brute force
and a voxel grid, and the duals against their closed forms. `python3
scripts/calibrate.py` regenerates the stability identification.
