# TODO

Reproduction of **"Adjustable Robust Reinforcement Learning for Online 3D Bin
Packing"** — Pan, Chen & Lin, NeurIPS 2023
([arXiv:2310.04323](https://arxiv.org/abs/2310.04323)), and its extension to
real orders, box types and the UR20 cell.

*Last updated: 2026-09-28* — only open items are listed; finished work is in
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
~13 s. Verdicts are cached per `(bin, item size, x, y)` but the cache is
cleared on every height-map change.

- [ ] **`--arm_collision` flag for `train` / `evaluate`**, so `args.json`
      records whether a run trained with the filter. None of the runs in
      `runs/` records it today
- [ ] Reuse verdicts across steps: a placement only changes the height map
      under its footprint, so a candidate whose capsules' AABBs miss that
      patch keeps its verdict
- [ ] Skip the check when the other rules leave a single candidate (it can
      only end the episode, not change the choice)
- [ ] Numba (not installed yet) or a small C++ extension for the column test —
      the pair-mask build and `nonzero` are most of the time
- [ ] Multiprocessing across bins over a process pool (16 cores). Expect 6-10x
      at best after pickling/IPC; awkward next to the threaded game server and
      CUDA in the training process

### D.5 Tests that depend on `config.yaml`

- [ ] The brute-force feasibility tests in `tests/test_env.py`
      (`test_feasibility_matches_brute_force`, `..._non_cubic_...`,
      `test_contact_area_...`) inherit `env.ems` and fail with `ems: 3`
      (11 failures, 2026-09-28); pin `ems` in `plain()` or parametrise over it
