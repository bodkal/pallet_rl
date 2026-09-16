# TODO / Status

Reproduction of **"Adjustable Robust Reinforcement Learning for Online 3D Bin
Packing"** — Pan, Chen & Lin, NeurIPS 2023
([arXiv:2310.04323](https://arxiv.org/abs/2310.04323)).

*Last updated: 2026-09-14*

- **`[x]`** — implemented **and** verified by a test or a measurement
- **`[ ]`** — open

> Implemented is not reproduced. The two are tracked separately: everything in
> [DONE](#done) is code that works; the [results](#c-where-the-numbers-stand)
> say how close it gets to the paper.

The official implementation was published at
<https://github.com/panyxy/ar2l_bpp> *after* this reproduction was written.
Section [A](#a-differences-against-the-official-implementation) is a read of
that repository against ours. **No code was taken from it** — anything adopted
below is to be written from scratch.

---

## A. Differences against the official implementation

### A.1 Confirmed correct (no action)

Their code settles several things the paper leaves ambiguous, and we had them
right:

- **Stability is a centre-of-mass test**, not the 60%-area/4-corner rule the
  paper's citations describe. `space.py: check_box` → `calculated_impact` tests
  whether the centre of mass lies inside the contact polygon. Our
  identification of this from the heuristic baselines (`scripts/calibrate.py`)
  was correct.
- **`normal_loss = (-log pi_mix(index 0)).mean()`** — the same finite
  cross-entropy surrogate we chose for the infinite `D_KL(pi_mix || 1{x=b})`
  term of Eq. 6.
- **`distance_loss = normal + alpha * worst`, coefficient 1** — matches our
  `--dist_coef 1.0`.
- Identical: 125 item types (1..5)^3, 150 items/episode, 80 packed-item nodes,
  120 candidate slots, 64 processes x 30 steps, `gamma = 1`, GAE `lambda = 0.95`,
  `lr = 3e-4`, coordinates normalised by 1/10, 1 attention layer, embedding 64,
  and the attacker's move-to-front semantics.

### A.2 Differences

| # | Official | Ours | Expected impact |
|---|---|---|---|
| ~~1~~ | **Rotation: 2 orientations.** `orientation = 2 if setting != 2 else 6` — every candidate is emitted for `(w,l,h)` *and* `(l,w,h)` | **adopted** — `--rot 2`, the default | **Confirmed large: +6.7 points** on the heuristics. See [A.4](#a4-adopted-rotation-and-a-true-ems-list) |
| ~~2~~ | **True 3-D EMS list** — Lai & Chan difference process plus inscribed-EMS elimination; 4 bottom corners per EMS per orientation | **adopted** — exact maximal-footprint enumeration, `env._ems_list` | **Free once rotation is in**, and cuts the candidate list from 72 to 37. See [A.4](#a4-adopted-rotation-and-a-true-ems-list) |
| 3 | **Stability**: convex hull of the contact rectangles, **shrunk 10%** toward its centroid; centre of mass of the **whole stack above**; checked **recursively** down the pile | per-axis span test, new box only, no margin | Ours is more permissive and cannot represent toppling |
| 4 | Infeasible candidates are **shown** to the network with an `isFeasi = 0` flag | only feasible candidates exposed, rest masked | Small |
| 5 | PPO: **1 epoch x 32 minibatches**, clip **0.1**, vf coef **1.0**, entropy **0.01**, linear LR decay to a 1e-4 floor | 4 x 4, clip 0.2, vf 0.5, entropy 0.002, no decay | Moderate. 32 small tightly-clipped steps vs our 16 large ones |
| 6 | **Block alternation**: 10 PPO iterations of the attacker, then 10 of the mixture model, then 10 of the packer | one of each, every iteration | Moderate for ExactAR2L — their mixture model converges against a *fixed* attacker before the packer moves |
| 7 | `KLDivLoss(mix, wor, log_target=True)` = **KL(pi_perm ‖ pi_mix)** — the reverse of the paper's text | `KL(pi_mix ‖ pi_perm)`, as written | Small |
| 8 | Sparse terminal reward `10 x utilisation` | dense per-item volume fraction | **Keep ours.** Same return at `gamma = 1`, far lower variance |
| 9 | Clipped value loss (PPO2 style) | plain MSE | Small |
| 10 | Node vector is a uniform 9-dim `[x,y,z,w,l,h,density,isFeasi,isEmbed]` with a 2-layer LeakyReLU embedder per type | 6/3/6-dim type-specific nodes, single linear embedder | Small |

### A.3 Not in their repository at all

**The official code implements only exact AR2L.** There is no ApproxAR2L (no
`rho`, no `mu1`/`mu2`, no dual), no RfMDP, no CPPO, and no attacker against the
heuristics. Our Eq. 18 implementation, RfMDP, CPPO and all of Table 1 have no
reference to check against — including the value-divergence documented in
[B.2](#b2-open-question-eq-18-at-gamma--1), which stays ours to solve.

### A.4 Adopted: rotation and a true EMS list

Items 1 and 2 of the table above are now implemented, from scratch.

**Rotation** (`--rot 2`, the default). A placement is an `(orientation, x, y)`
triple: the leading item is offered both as `(w, l, h)` and as `(l, w, h)`, with
the height never rotating. A square base yields only one distinct placement, so
the duplicate is masked. The leaf node carries the *rotated* size, so the
network sees the footprint it would actually get.

**Empty maximal spaces.** Because every item is dropped, the reachable free
volume is exactly `{(x, y, z) : z >= hmap[x, y]}`, so an empty maximal space is
a footprint `[x0, x0+wx) x [y0, y0+wy)` whose floor — the window max of the
height map — cannot be lowered by widening it, stacked to the lid. At `S = 10`
there are only 55 x 55 candidate footprints per bin, so `env._ems_list` scores
all of them with running window maxima and keeps the maximal ones. That gives
the *exact* EMS list, with none of the bookkeeping or ordering artefacts of an
incremental difference process, and it is checked against brute force in
`tests/test_env.py::test_ems_list_is_exact`. Candidates are then the four bottom
corners of every space wide and tall enough to take the item.

Measured without any learning — the heuristics never train, so they isolate the
action space from the policy (`scripts/ablate_env.py`, 512 instances):

| heuristic | rot1 ems | **rot2 ems** | rot1 all | rot2 all | paper |
|---|---|---|---|---|---|
| dbl | 56.1 | **62.3** | 57.8 | 63.1 | 63.6 |
| bmf | 50.4 | **57.3** | 50.5 | 56.9 | 62.0 |
| lsah | 52.7 | **58.0** | 53.3 | 57.7 | 60.9 |
| onlinebph | 55.0 | **62.8** | 56.6 | 63.5 | 64.1 |
| hmm | 56.3 | **62.6** | 56.0 | 62.0 | 56.1 |
| macs | 59.5 | **67.4** | 61.3 | 67.2 | 53.0 |
| **mean** | 55.0 | **61.7** | 55.9 | 61.7 | 59.9 |
| **mean \|gap\|** | 7.2 | **5.2** | 6.8 | 4.9 | — |

Rotation is worth **+6.7 points** with EMS corners (+5.8 with every position);
restricting to EMS corners is **free** once rotation is in and costs 0.9 without
it, while cutting the candidate list from 72 to 37 — which is why iterations got
*faster*, not slower, despite the richer action space. Mean absolute gap to the
paper: 7.2 -> 5.2.

DBL, LSAH and ONLINEBPH now land within 3 points of the paper. HMM, the one that
matched exactly before, overshoots by 6.5, and MACS by 14.4 — so the earlier
reading that "our environment is uniformly harder" was wrong in both directions:
what the heuristics measure is dominated by the action space, not by the
physics, and the three worst-fitting rows are exactly the three score functions
reconstructed from citations.

### A.5 Adopted: a stronger attacker (from [the diagnosis below](#c-where-the-numbers-stand))

Attacker strength is the binding constraint on every attacked column — ours cost
PCT 12.8 points at `N_B = 10` where the paper's costs 20.7. Three changes, all
aimed at the gap between how the attacker is trained and how it is tested:

1. **The frozen packer now plays greedily during attacker training.** It is
   attacked greedily at evaluation, so fitting an attacker against a *sampling*
   packer was fitting the wrong opponent (`Runner(greedy_pack=...)`).
2. **`best.pt` is chosen by held-out attacked utilisation**, both sides greedy
   (`evaluate.attack_score`), instead of by the rolling training average.
3. **The entropy bonus is annealed** to `--ent_final 0.001`, so the policy that
   is deployed greedily actually sharpens.

---

## B. Recommendation

### B.1 Do the full rewrite (items 1, 2, 3, 5, 6)

Not the cheap half. Items 1 and 2 push utilisation **up**; item 3 pushes it
**down**. Doing only 1 and 5 risks overshooting the paper rather than
converging on it, with no way to attribute the result.

The evidence that this is not a tuning problem: **at `N_B = 20` our ExactAR2L
is *less* robust than plain PCT** (48.7 vs 55.3 at `beta = 100`), the reverse of
the paper. The two mechanisms AR2L leans on at large `N_B` are exactly the two
we are missing — a mixture model given time to converge against a fixed
attacker (item 6), and a rotation-bearing action space that changes what a hard
permutation even is (item 1).

Cost: ~3 h implementation, ~7 h compute (16 policies + 16 attackers + a 3000-
instance evaluation). Existing results stay on disk; nothing measured is lost.

### B.2 Open question: Eq. 18 at `gamma = 1`

Independent of the rewrite. The adjustable robust Bellman operator is optimistic
by up to `rho * range(V)` per application, and Theorem 2 only makes it a
contraction for `gamma < 1` — but this MDP is undiscounted. Measured: the
ApproxAR2L critic ran away to `V ~ 3411` where the return is a utilisation in
`[0, 1]`. We anchor by projecting the target onto `[0, 1]`, which contains the
true fixed point, so the projection is non-expansive. That restores a fixed
point and lifted ApproxAR2L from 60.5 to 64.7 — but it still trails, and the
projection saturates for early states when `rho = 0.1` over a 30-step episode.
Worth revisiting: a per-step `rho` budget, or `gamma < 1` for the robust value
only.

---

## C. Where the numbers stand (2026-09-15, after A.4 and A.5)

Space utilisation, 3000 held-out instances of 150 items. Policies and attackers
both trained to 4000 iterations. `results/table2.json`; the pre-rotation results
are kept at `results/table2_norot.json`.

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

Over all 80 cells: **mean gap +1.7 points, mean |gap| 3.5** (was −5.5 / 6.5
before rotation). Nominal utilisation is now within a point of the paper for
five of the eight methods, and `N_B = 20` ExactAR2L is no longer anti-robust —
62.6 against PCT's 63.4, where it used to be 48.7 against 55.3.

### C.1 What is left: the attacker, and only the attacker

Every remaining discrepancy sits in the same place. Our attacker costs PCT
**11.7** points at `N_B = 10` where the paper's costs 20.7, and **12.1** at
`N_B = 20` where the paper's costs **35.1**. The paper's attacker gets much
stronger as `N_B` grows; ours barely moves. With the attacked columns that high,
every method bunches together and the paper's comparative claims cannot separate:

| claim in the paper | holds here | before |
|---|---|---|
| utilisation falls monotonically in `beta` | 16/16 | 16/16 |
| ExactAR2L(1.0) packs at least as many items as PCT | 9/10 | 8/10 |
| ExactAR2L(1.0) has a smaller Std. than PCT | 5/10 | 4/10 |
| ExactAR2L(1.0) beats PCT once a quarter of the set is attacked | 3/8 | 4/8 |
| ExactAR2L(1.0) Std. lies between RARL's and PCT's | 2/10 | 0/10 |
| RARL gives up nominal utilisation relative to PCT | 1/2 | 2/2 |
| ApproxAR2L(0.5) packs more items than RfMDP | 1/10 | 7/10 |
| ApproxAR2L(0.5) beats RfMDP on utilisation | 0/10 | 2/10 |

Part of the lost damage is real and expected: rotation gives the packer twice
the placements, so a hostile ordering has less bite. Measured directly on DBL,
which never trains, so the action space is the only variable (1500-iteration
attacker, `runs/attrot{1,2}_dbl_nb10`):

| | nominal | attacked | damage |
|---|---|---|---|
| DBL, 1 orientation | 56.1 | 32.9 | **−23.2** |
| DBL, 2 orientations | 62.3 | 44.2 | **−18.1** |

So rotation absorbs about **5 points** of attacker damage. That accounts for
roughly half the shortfall at `N_B = 10` and almost none of it at `N_B = 20`,
where the paper's attacker is three times as damaging as ours. The remaining
suspects are the two unimplemented items of [A.2](#a2-differences): the block
alternation (item 6) and the PPO settings (item 5), plus a training budget of
4000 iterations against the paper's ~28,000.

### C.2 ApproxAR2L is the one method still clearly short

4–7 points low nominally, and the only method with no reference implementation —
the official repository implements exact AR2L alone. This is
[B.2](#b2-open-question-eq-18-at-gamma--1), not the action space.

---

## D. Open

### D.1 The rewrite (blocked on a go/no-go)

- [x] **Rotation, 2 orientations.** Candidates emitted for `(w,l,h)` and
      `(l,w,h)`; `L` node carries the rotated size. `env._positions`,
      `env.step`, the `_lxy`/`_lz`/`_ldim` bookkeeping, `heuristics.scores`
      and the three viewers that draw candidates. Worth +6.7 points
- [x] **3-D EMS list.** `env._ems_list` enumerates every maximal footprint
      exactly; candidates = the 4 bottom corners of each space per
      orientation. Replaced `env._ems_axis`. Costs nothing alongside
      rotation, and −49% candidates
- [ ] **Stability: recursive convex-hull centre of mass.** Contact polygon =
      convex hull of the overlap rectangles, shrunk 10%; centre of mass of the
      stack above; recurse into supporting boxes. Keep `stability="com"` and
      `"cdrl"` selectable for comparison
- [x] `scripts/ablate_env.py` — the heuristic baselines under each combination
      of `rot` and `ems`, which is the only external check on the action space
- [ ] Re-run `scripts/calibrate.py` — the stability sweep now has to be redone
      on top of rotation, since the two interact
- [ ] **PPO settings**: 1 epoch x 32 minibatches, clip 0.1, vf coef 1.0,
      entropy 0.01, linear LR decay to a 1e-4 floor, clipped value loss
- [ ] **Block alternation** for ExactAR2L: `--block 10`, cycling
      attacker -> mixture -> packer
- [ ] Retrain the 16 policies and 16 attackers, re-evaluate, regenerate the
      report — **done for Table 2**, Table 1 running. The pre-rotation results
      are kept at `results/table2_norot.json` and `results/table1_norot.json`;
      everything else from before the rewrite is archived under `old_run/`,
      including the checkpoints, which can no longer be evaluated because
      `env._ems_axis` is gone

### D.2 Cheap, independent of the rewrite

- [ ] Show infeasible candidates to the network with a feasibility flag
      (item 4) — currently masked out entirely
- [ ] Try the reverse KL in Eq. 6 (item 7) as an ablation, one run
- [ ] `pct_nb1` + its attackers, to fill the missing PCT row of Table 1
- [ ] Table 1 at `N_B = 10, 15` (currently only 5 and 20)

### D.3 Not started

- [ ] Continuous setting and its Intersection-Points candidate generator
      (paper Appendix A.1.1/A.1.2)
- [ ] Mixed-Item dataset, paper Table 3
- [ ] `rho` sweep, paper Figure 3(c,d): `rho` in {0.1, 0.2, 0.3, 0.4}
- [ ] CDRL baseline for Table 1 (reproduced separately in `../pallet_rl_v1`)
- [ ] Revisit Eq. 18 at `gamma = 1` — see [B.2](#b2-open-question-eq-18-at-gamma--1)

---

## DONE

### Implementation

- [x] Batched PCT-state simulator: height map, EMS-style candidates, conveyor
      permutation. 64 bins step in 1.7 ms
- [x] Feasibility and landing height verified **cell-by-cell against brute
      force**, both stability modes, 0 mismatches over 192,000 positions
- [x] No-overlap and volume-bookkeeping invariants tested
- [x] Three transformers — packer, attacker, mixture model — with the pointer
      head of Eq. 28
- [x] PPO, GAE, and the TV duals: Eq. 18 for ApproxAR2L, its pessimistic mirror
      for RfMDP. `rho = 0` provably collapses Eq. 18 to the alpha-weighted
      mixture mean — a test
- [x] Seven training loops: `pct`, `cppo`, `rarl`, `rfmdp`, `exact` (Alg. 1),
      `approx` (Alg. 2), `attack`
- [x] Six heuristics: DBL, BMF, LSAH, OnlineBPH, HMM, MACS
- [x] Held-out evaluation, mixture datasets, `Uti./Std./Num.`
- [x] Five viewers: dashboard, game, report, replay3d, heatmap, attack
- [x] Job queue, resumable training, end-to-end pipeline scripts
- [x] 14 tests passing, no lint errors

### Bugs found and fixed

- [x] **CVaR episode labelling** — steps of a *previous* episode inside the same
      rollout window could be relabelled with the current episode's return
- [x] **Attacker / mixture critics bootstrapped from zero** at the rollout
      boundary instead of from their own value
- [x] **RfMDP and ApproxAR2L were handicapped by a one-step target.** A robust
      Bellman operator only specifies the *next-state* value, so the dual
      belongs inside the GAE residual. Worth ~4.5 points (RfMDP 61.7 -> 66.2)
- [x] **Robust critics diverged** — see [B.2](#b2-open-question-eq-18-at-gamma--1).
      Measured `V ~ 3411`; fixed by projecting onto `[0, 1]`; worth ~4 points
- [x] `torch.compile` miscompiles the pointer head for some `N_B` and surfaces
      as an illegal memory access — disabled by default
- [x] Isometric camera in the game viewer was inverted (drawing back faces)
- [x] Unquoted SVG attributes in the dashboard drew stray diagonals
- [x] Attacker-behaviour baseline was biased — items the attacker passes over
      linger in the window and were counted repeatedly

### Measurements

- [x] Stability rule identified from the paper's own heuristic baselines
      (`scripts/calibrate.py`, `results/calibration.json`) — later confirmed
      against the official code, see [A.1](#a1-confirmed-correct-no-action)
- [x] 16 policies x 2 `N_B` trained to 3000 iterations
- [x] 16 dedicated attackers, at 800 and at 3000 iterations; both result sets
      kept
- [x] Table 1 at `N_B = 5, 20`; Table 2 at `N_B = 10, 20`, `beta` in
      {0, 25, 50, 75, 100}
- [x] The paper's comparative claims scored cell-by-cell
      (`scripts/summarize.py`)
- [x] Attacker behaviour: prefers small items, earliest — the appendix claim,
      reproduced
