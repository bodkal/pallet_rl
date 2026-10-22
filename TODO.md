# TODO

Reproduction of **"Adjustable Robust Reinforcement Learning for Online 3D Bin
Packing"** — Pan, Chen & Lin, NeurIPS 2023
([arXiv:2310.04323](https://arxiv.org/abs/2310.04323)), and its extension to
real orders, box types and the UR20 cell.

*Last updated: 2026-10-21* — only open items are listed; finished work is in
the git history (the full list as of 2026-09-27 is `git show eb38e8c:TODO.md`).

The official implementation is at <https://github.com/panyxy/ar2l_bpp>. **No
code was taken from it** — anything adopted below is to be written from
scratch.

---

## A. Remaining differences against the official implementation

Rotation (2 orientations) and the exact EMS list are adopted and no longer
listed. What is left:

| # | Official | Ours | Expected impact |
|---|---|---|---|
| 3 | **Stability**: convex hull of the contact rectangles, **shrunk 10%** toward its centroid; centre of mass of the **whole stack above**; checked **recursively** down the pile | per-axis span test, new box only, plus the `min_support` area floor | Ours is more permissive and cannot represent toppling |
| 4 | Infeasible candidates are **shown** to the network with an `isFeasi = 0` flag | only feasible candidates exposed, rest masked | Small |
| 5 | PPO: **1 epoch x 32 minibatches**, clip **0.1**, vf coef **1.0**, entropy **0.01**, linear LR decay to a 1e-4 floor | 4 x 4, clip 0.2, vf 0.5, entropy 0.01, no decay | Moderate. 32 small tightly-clipped steps vs our 16 large ones |
| 6 | **Block alternation**: 10 PPO iterations of the attacker, then 10 of the mixture model, then 10 of the packer | one of each, every iteration | Moderate for ExactAR2L — their mixture model converges against a *fixed* attacker before the packer moves |
| 7 | `KLDivLoss(mix, wor, log_target=True)` = **KL(pi_perm ‖ pi_mix)** — the reverse of the paper's text | `KL(pi_mix ‖ pi_perm)`, as written | Small |
| 9 | Clipped value loss (PPO2 style) | plain MSE | Small |
| 10 | Node vector is a uniform 9-dim `[x,y,z,w,l,h,density,isFeasi,isEmbed]` with a 2-layer LeakyReLU embedder per type | 6/3/6-dim type-specific nodes, single linear embedder | Small |

Item 8 (sparse terminal reward) is deliberately **not** adopted: our dense
per-item reward has the same return at `gamma = 1` and far lower variance.

**The official code implements only exact AR2L.** There is no ApproxAR2L, no
RfMDP, no CPPO and no attacker against the heuristics, so those — and
[B](#b-open-question-eq-18-at-gamma--1) — have no reference to check against.

---

## B. Open question: Eq. 18 at `gamma = 1`

The adjustable robust Bellman operator is optimistic by up to `rho * range(V)`
per application, and Theorem 2 only makes it a contraction for `gamma < 1` —
but this MDP is undiscounted. The ApproxAR2L critic ran away to `V ~ 3411`;
projecting the target onto `[0, 1]` restores a fixed point (60.5 -> 64.7), but
ApproxAR2L still trails and the projection saturates for early states when
`rho = 0.1` over a 30-step episode. Worth trying: a per-step `rho` budget, or
`gamma < 1` for the robust value only.

---

## C. Where the paper numbers stand (2026-09-15)

Space utilisation, 3000 held-out instances of 150 items, `--min_support 0`.
Policies and attackers both trained to 4000 iterations. `results/table2.json`.

| | `N_B=10` ours `b=0`/`b=100` | paper | `N_B=20` ours `b=0`/`b=100` | paper |
|---|---|---|---|---|
| PCT | 75.9 / 64.2 | 76.4 / 55.7 | 75.5 / 63.4 | 77.0 / 41.9 |
| CPPO | 75.5 / 64.0 | 75.6 / 57.4 | 74.6 / 56.3 | 74.1 / 45.8 |
| RARL | 75.9 / 65.7 | 74.3 / 63.3 | 74.7 / 62.3 | 72.0 / 58.7 |
| RfMDP | 72.6 / 58.4 | 74.4 / 55.9 | 74.1 / 58.2 | 73.8 / 54.4 |
| ExactAR2L(0.5) | 75.9 / 64.6 | 77.6 / 59.7 | 75.7 / 63.8 | 76.8 / 54.4 |
| ExactAR2L(1.0) | 75.4 / 65.1 | 76.0 / 63.8 | 75.5 / 62.6 | 76.1 / 58.5 |
| ApproxAR2L(0.5) | 69.1 / 57.6 | 76.2 / 56.1 | 71.2 / 50.8 | 75.0 / 53.1 |
| ApproxAR2L(1.0) | 70.1 / 57.8 | 73.6 / 57.1 | 70.5 / 53.3 | 73.4 / 57.6 |

Over all 80 cells: mean gap +1.7 points, mean |gap| 3.5.

### C.1 What is left: the attacker, and only the attacker

Our attacker costs PCT **11.7** points at `N_B = 10` where the paper's costs
20.7, and **12.1** at `N_B = 20` where the paper's costs **35.1**. With the
attacked columns that high, every method bunches together and the paper's
comparative claims cannot separate (`scripts/summarize.py`). Rotation absorbs
about 5 points of attacker damage (measured on DBL, `runs/attrot{1,2}_dbl_nb10`),
which is roughly half the shortfall at `N_B = 10` and almost none of it at
`N_B = 20`. Remaining suspects: block alternation (A.6), the PPO settings
(A.5), and a budget of 4000 iterations against the paper's ~28,000.

### C.2 ApproxAR2L is the one method still clearly short

4–7 points low nominally, with no reference implementation. This is
[B](#b-open-question-eq-18-at-gamma--1), not the action space.

---

## D. Open

### D.1 The rewrite

- [ ] **Stability: recursive convex-hull centre of mass** (A.3). Contact
      polygon = convex hull of the overlap rectangles, shrunk 10%; centre of
      mass of the stack above; recurse into supporting boxes. Keep
      `stability="com"` and `"cdrl"` selectable for comparison
- [ ] Re-run `scripts/calibrate.py` — the stability sweep has to be redone on
      top of rotation, since the two interact (`results/calibration.json` is
      from 2026-09-23)
- [ ] **PPO settings** (A.5): 1 epoch x 32 minibatches, clip 0.1, vf coef 1.0,
      linear LR decay to a 1e-4 floor, clipped value loss (A.9)
- [ ] **Block alternation** for ExactAR2L (A.6): `--block 10`, cycling
      attacker -> mixture -> packer
- [ ] Table 1 on the rotation action space: `results/table1.json` does not
      exist yet — only the pre-rotation `results/table1_norot.json`

### D.2 Cheap, independent of the rewrite

- [ ] Show infeasible candidates to the network with a feasibility flag (A.4)
- [ ] Try the reverse KL in Eq. 6 (A.7) as an ablation, one run
- [ ] `pct_nb1` + its attackers, to fill the missing PCT row of Table 1
- [ ] Table 1 at `N_B = 10, 15` (currently only 5 and 20)

### D.3 Not started

- [ ] Continuous setting and its Intersection-Points candidate generator
      (paper Appendix A.1.1/A.1.2)
- [ ] Mixed-Item dataset, paper Table 3
- [ ] `rho` sweep, paper Figure 3(c,d): `rho` in {0.1, 0.2, 0.3, 0.4}
- [ ] CDRL baseline for Table 1 (reproduced separately in `../pallet_rl_v1`)
- [ ] Revisit Eq. 18 at `gamma = 1` — see [B](#b-open-question-eq-18-at-gamma--1)

### D.4 Robot-arm filter (`env.arm_collision`)

The batched check (`ArmPackChecker.collides_batch`) costs ~25 ms extra per env
step (64 bins, ~1500 candidates); on the orders pallet a `select` iteration is
~13 s. Verdicts are cached per `(bin, item size, x, y)` and kept across steps
unless the new box raised a column they rest on (`_keep_arm`).

- [ ] **`--arm_collision` flag for `train` / `evaluate`**, so `args.json`
      records whether a run trained with the filter. None of the runs in
      `runs/` records it today
- [ ] Skip the check when the other rules leave a single candidate (it can
      only end the episode, not change the choice)
- [ ] Numba (not installed yet) or a small C++ extension for the column test —
      the pair-mask build and `nonzero` are most of the time
- [ ] Multiprocessing across bins over a process pool (16 cores). Expect 6-10x
      at best after pickling/IPC; awkward next to the threaded game server and
      CUDA in the training process

### D.5 Tests that depend on `config.yaml`

35 failures on 2026-10-12, the same set on HEAD `8172bcc`. All but the
experiment's come from a test inheriting what `config.yaml` now ships:

- [ ] The brute-force feasibility tests in `tests/test_env.py`
      (`test_feasibility_matches_brute_force`, `..._non_cubic_...`,
      `test_contact_area_...`) inherit `env.ems` and fail with `ems: 3`
      (11 failures); pin `ems` in `plain()` or parametrise over it
- [ ] The fallback tests in `tests/test_mixed.py` (14) inherit `soft_mix: 1`,
      which turns a box's own fallback into the station's last resort; pin
      `soft_mix=False` in `mixed()` -- with `soft_mix: 0` all 96 pass
- [ ] `tests/test_experiment.py` (6): `experiment.start` now refuses a run
      with no name; give the tests one
- [ ] Boxes read from an orders file come out wider and longer than the
      tests expect, heights unchanged -- the config's `eval.box_pad_m`
      padding (`test_experiment` 1, `test_game` 2); pin it in the tests
- [ ] `tests/test_types.py::test_no_box_is_ever_stacked_on_another_type`
      pins `type_rule: touch` but inherits the rest (`ems`, arm, stack cap,
      `min_support`); no box is ever stacked, so it tests nothing

### D.6 Stack cap (`env.stack_cap`)

A small brown box limits how high a later box may start over its footprint
(its top plus an allowance set by its shorter side: 12 cm -> 8 cm,
20 cm -> 64 cm, 32 cm and up -> none). Measured 2026-10-12 on 128 pallets of
`data/train_data/all_data.csv` with `runs/w_128_h_1_l_1_soft_mix_all_data`:
65.2% utilisation, 48.9 boxes a pallet, and 126 of the 128 end with boxes
left. The candidate points are not what ends them -- the outline step over
the whole grid (`outline_area: all`) would have rescued none -- the rules are.

- [ ] **Stack-cap last resort**: when the station is stuck -- after soft mix
      and the outline step -- allow one placement with the cap loosened,
      then restore it. Two forms measured: lifted (+0.4 boxes a pallet,
      65.4%; 41 rescues in 128 pallets) and every limit +20 cm (+0.3, 65.3%;
      29 rescues). As a config switch, off by default (`stack_cap_resort`,
      `stack_cap_resort_cm`), with a field in the game. Per bin through
      `cmap`: save it, loosen it in the stuck bins, `step`, then
      `cmap = min(saved, cmap)` so a brown box placed in that step keeps its
      own limit. **Waiting on operations**: it puts more on a small brown
      box than the cell's C++ `stable_grid` allows, and the real cell would
      refuse the placement unless that rule changes too

### D.7 Picker lookahead (search over the station)

Idea: before the picker (`PermNet`) chooses, search a few picks ahead and
keep it off boxes that lead to a stuck station. Each branch picks a station
box, places it at the **packer's top-1 position** (no branching over
placements), and checks whether the station is then stuck -- the same check
`step` already ends with under `pick_feasible`, which is this search at depth
0. The station refills from the preview after every pick, so each level has
`n_pick` = 5 choices: 5 branches at depth 1, 5 + 25 = 30 at depth 2,
155 at depth 3. Only as deep as the visible conveyor (`nb` = 10); past it
the boxes are unknown. A box is bad only if **every** branch under it gets
stuck -- "any branch" would block nearly every box, since one bad placement
is always possible; the fraction stuck is the soft version.

Three ways to use the result:

| | Search in training | Search at deploy | Picker learns |
|---|---|---|---|
| Mask | yes | yes | can't pick a dead box |
| Penalty | yes | no | to avoid dead picks, from the reward |
| Input feature (4th number per box: dead-end risk) | yes | yes, always -- an input missing at deploy breaks the net | to weigh the risk itself |

A fourth, free: a **hindsight penalty** on the last 1-3 picks of an episode
that ended stuck -- no search, but no idea which pick would have saved it.

Measured 2026-10-21 on `all_data.csv` (`mixed_column`, arm on), with another
training run sharing the machine. One branch = deepcopy + permute + `obs` +
packer forward + `step`:

| | One branch | Normal move | Depth 1 | Depth 2 | Depth 3 |
|---|---|---|---|---|---|
| 1 pallet, 4 cm grid (31x27x40) | ~18 ms | ~30 ms | 0.09 s | 0.54 s | 2.8 s |
| 1 pallet, 2 cm grid (62x54x80) | ~32 ms | ~45 ms | 0.16 s | 0.96 s | 5.0 s |
| 1 pallet, 1 cm grid (124x108x160) | ~69 ms | ~85 ms | 0.35 s | 2.1 s | 10.7 s |
| 64 bins, 4 cm (training, per move) | ~0.37 s | ~0.4 s | 1.9 s | 11 s | 57 s |

A 4x finer grid costs ~4x, not 16x: `ems: 3` keeps the candidates to
corners. At 1 cm the p90 `step` is ~2x its mean. In training, depth 1 adds
~55 s to a ~35 s iteration (T = 30) and depth 2 ~5.5 min -- 20,000 iterations
at depth 2 is months -- so the search is cheap at deploy and dear in training.

- [ ] **Measure first**: on the held-out pallets, how many episodes end with
      the station stuck vs the pallet full, and how many of the stuck ones a
      different pick 1-2 moves earlier would have saved. That is the most
      any form of this can gain
- [ ] **Deploy-time mask, no retraining**: in `evaluate.py`, drop station
      boxes whose depth-2 search is stuck on every branch before the trained
      picker chooses; compare boxes packed with and without. Safe because a
      mask changes the choices, not what the net sees. Also bounds what a
      search-trained picker could gain
- [ ] An env snapshot/restore cheaper than `deepcopy` (copy the maps and the
      sequence, rebuild or keep the caches) -- ~20 ms of the 69 ms branch at
      1 cm
- [ ] Training with it, if the mask pays: depth 1, all 5 branches batched as
      one 320-bin env, and only when the station is at risk (<= 2 boxes with
      a place, or a brown in reach). Penalty first; input feature if the
      picker does not pick it up; hindsight penalty as the free baseline

#### D.7.1 Review of the idea (2026-10-21)

It is a good idea, and the codebase already has the first level of it.
`pick_feasible` (`ar2l/env.py:866`) removes a station box from the picker's
mask when it has no legal place right now; this idea is that check, looking
`n` steps further ahead. It targets the real failure mode: an episode ends
when the whole station is blocked (`ar2l/env.py:2239`), and the stuck-station
experiments say the main cause is the type rule (brown-on-white), not the
pallet being full. Picking in the wrong order can create those dead ends, so
a lookahead could see them coming.

Four things to change or check before building it:

1. **"All branches die", not "one branch dies".** At each later step the
   packer has many placements and the picker several boxes, so some branch
   almost always dies -- a bad placement is always possible. If one dead
   branch blocked a box, nearly every box would be blocked. Box *i* is bad
   only if **every** continuation dies within `n` steps. The softer version is
   the fraction of branches that die, as a penalty or an extra input.
2. **Prune the tree.** Unpruned, each level is up to 5 boxes x ~32 feasible
   placements (p90), ~160 branches a level and ~25,000 at depth 2, each
   needing a feasibility sweep with the arm check -- the expensive part of
   this env. So:
   - expand only the packer's top 1-3 placements, not all of them -- which
     also tests whether *our* packer gets stuck, the question that matters;
   - stay within what the cell can see: the station plus the preview. Past
     that the boxes are unknown;
   - depth 2 probably catches most cases: "pick A, and then the other four
     are all blocked" is the typical stuck pattern.
3. **Mask or penalty** -- they differ at deploy:
   - mask: a hard constraint, but the search runs on the real cell at every
     pick, adding latency;
   - penalty in training only: the picker learns to avoid dead ends and no
     search runs at deploy. Keep it small, or make it an auxiliary loss, so it
     does not swamp the main objective of packing more boxes;
   - input feature: give the search result to `PermNet` as one more number
     per box ("dead-end risk") and let the network learn how far to trust it.
4. **Measure the problem first.** Count how many episodes end with the
   station stuck vs the pallet actually full, then replay the stuck ones and
   ask whether a different pick 1-2 moves earlier would have avoided the dead
   end. That is the most this idea can gain. The experiments so far give only
   partial bounds (cap-at-stuck +0.4 boxes, arm-off +2.2).

Implementation cost: the env has no clone or snapshot method and keeps a lot
of cached state (`_pos_cache`, `_slot_geo`, ...). The lookahead needs a safe
way to copy one env, try steps on the copy and throw it away, so that comes
first.

#### D.7.2 Top-1 packer position, and the input-feature option

With only the packer's recommended position, the tree branches only on which
box is picked. The station refills from the preview after every pick, so
every level has 5 choices: 5 + 25 = 30 branches at depth 2, 155 at depth 3
(an earlier estimate of 20 and 60 assumed the station shrinks; it does not)
-- small enough to run at every pick on one pallet (D.7 table).

The input-feature option gives each conveyor box a 4th number. Today
`PermNet` sees each box as 3 numbers (its size) plus its type
(`ar2l/model.py:225`, encoder `[6, 3]`); the 4th is its dead-end risk from
the search:

- 0: picking it leaves a way to keep packing for the next `n` steps;
- 1: every branch after picking it gets stuck;
- in between: the fraction of branches that get stuck.

The picker then decides how much to trust that number instead of being
forced by a mask. It has to be computed **always** -- in training, in
evaluation and on the real cell: whatever the network sees as input in
training must be there at deploy. Computed in training only, the network
learns to rely on it, and at deploy the number is missing (or 0) and it
behaves badly.

| Option | Search in training | Search at deploy | How the picker learns |
|---|---|---|---|
| 1. Mask | yes | yes | can't pick a dead box at all |
| 2. Penalty | yes | no | from punishment, to avoid dead picks |
| 3. Extra input | yes | yes | sees the risk and decides itself |

For training, start with the penalty: it changes only the reward, not the
network or the deploy code, so it is the fastest to test. If the picker does
not learn to avoid dead ends well enough, switch to the input; the two also
combine -- the input says where the danger is, the penalty makes the picker
care.
