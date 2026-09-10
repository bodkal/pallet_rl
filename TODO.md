# TODO / Status

Reproduction of **"Online 3D Bin Packing with Constrained Deep Reinforcement Learning"**
— Zhao, She, Zhu, Yang & Xu, AAAI 2021 ([arXiv:2006.14978](https://arxiv.org/abs/2006.14978)).

*Last updated: 2026-09-10*

**Everything finished lives at the bottom, under [DONE](#done).** The sections
above it are only what is still open.

- **`[x]`** — code implemented **and** verified by a test
- **`[ ]`** — open: queued, running, or unanswered

> A completed implementation is not a reproduced result — the two are tracked
> separately inside DONE.

---

## A. In progress — running now

`STEPS=100000000 ./scripts/train_all_options.sh`. Every run was extended from its
30M milestone to 100M because **all six were still climbing at 30M** (+0.006 to
+0.015 utilisation per 5M steps) — 30M was a budget, not convergence.

| run | stream | orient | step | util | items | status |
|---|---|---|---|---|---|---|
| `bpp1_cut2` | CUT-2 | 1 | 100,000,000 / 100,000,000 (100%) | 0.6784 | 17.96 | **done** |
| `bpp1_cut1` | CUT-1 | 1 | 76,642,560 / 100,000,000 (77%) | 0.7213 | 18.43 | **running** |
| `bpp1_rs` | RS | 1 | 30,000,640 / 30,000,000 (100%) | 0.4930 | 11.93 | **done** |
| `bpp1_orient_cut2` | CUT-2 | 2 | 30,000,640 / 30,000,000 (100%) | 0.6166 | 16.35 | **done** |
| `bpp1_orient_cut1` | CUT-1 | 2 | 30,000,640 / 30,000,000 (100%) | 0.6279 | 15.95 | **done** |
| `bpp1_orient_rs` | RS | 2 | 30,000,640 / 30,000,000 (100%) | 0.5850 | 14.24 | **done** |

- [x] The three 20³ runs are scored — see "20³ reproduction, all three streams"
      under [DONE](#done). They had hit 100M but no `eval.json` had ever been
      written, so the eval guard had nothing to compare against.
- [ ] Finish the 100M extension for all six 10³ runs, then re-evaluate (the eval
      guard re-runs automatically once `best.pt` is newer than `eval.json`)
- [ ] MP/MC/FE ablation, 4 runs x 8M steps — not started
      (paper Table 1: 7.8% / 27.9% / 63.7% / 63.0% / 66.9%)
- [ ] Re-run `scripts/probe_buffer.py` against the 100M nets

> **Retired 2026-09-09:** `kl_lr`, `kl_tkl`, `kl_both` (lost; `kl_tkl`/`kl_both`
> measured the `ppo.py` starvation bug rather than a trust region) and
> `bpp1_cut2_20_deep` (stalled, superseded by the A.2 network grid) moved to
> `runs/old_test/`, with every number preserved in
> [runs/old_test/README.md](runs/old_test/README.md).
>
> **Retired 2026-09-09:** the six 15³ runs moved to `runs/old_test/` — 15 is
> neither the paper's 10³ headline size nor the 20³ of its Figure 14 resolution
> study, so no number from them is checkable. Three had completed 50M steps and
> three were aborted stubs under 0.05M; all their numbers are preserved in
> [runs/old_test/README.md](runs/old_test/README.md). Nothing referenced them.

> Extending `STEPS` resumes rather than restarts, but **resets the LR schedule to
> the new horizon** — measured jump 1.5e-05 -> 2.0e-04 at the hand-over. Expect a
> brief dip before the curve climbs past its old level.

### A.2 Lever-sweep follow-up — launched 2026-09-09

Five runs, 10M steps each, every one matching `kl_base` exactly (20³, CUT-2,
1 pose, `terminate`, seed 0, lr 3e-4, `num_envs` 32, `seq_pool` 40000,
`w_mask` 0.5) except the lever named. `kl_base` is the control for all five,
and is also the 2/64/256 cell of the network grid.

| run | lever | answers |
|---|---|---|
| `true_eps0` | `--use-true-mask --mask-eps 0` | the *clean* ceiling — with an exact mask, eps is the only remaining source of illegal moves, and 3.0% of episodes still died on one |
| `tkl_fixed` | `--target-kl 0.2` | the trust region asked honestly, on the fixed `ppo.py` |
| `net_depth` | 4 layers / 64 ch / 256 hidden | depth alone |
| `net_width` | 2 layers / 64 ch / 1024 hidden | width alone |
| `net_both` | 4 layers / 128 ch / 1024 hidden | the current default |

- [x] All five finished 2026-09-09 and are evaluated — results under
      "Network grid + follow-up levers" in [DONE](#done). The deep default
      earns its cost (+7.03 pp); depth is the axis that matters; the trust
      region works at 0.2; the ceiling is 46.77%.
- [ ] **Next runs, in priority order**, all following from the above:
    1. ~~Deep net at `--lr 1e-4`~~ — **done, and it LOST**: 35.36% vs 3e-4's
       36.94%. Keep 3e-4.
    2. ~~`4/128/256`, the missing grid cell~~ — **done**: 35.55%, completing an
       additive decomposition (see "Network grid completed" in DONE).
    3. ~~`--w-mask 5.0` stacked on the deep net~~ — **done, and it is the best
       result of the project so far**: `deep_stack` 43.03%.
    - [ ] Now: **100M `deep_stack` at lr 3e-4** (~26 h), plus `deep_wmask`
          (`w_mask 5.0` only) to finish attributing `deep_stack`'s gain between
          its two levers.

---

## B. Open questions raised by the reproduction

- [ ] Why is our spare-cuboid boundary rule **20 pp better** than the paper's reported boundary rule? Either their variant is weaker than the supplemental describes, or our maximal-cuboid enumeration is more thorough.
- [ ] **Does PPO actually match ACKTR here?** Cannot be answered until [C.1](#c-next-up--ordered-by-expected-value) lands. If PPO falls short of 66.9% on CUT-2, ACKTR becomes the prime suspect rather than a nice-to-have.
- [ ] **Why does re-orientation *hurt* on CUT-1?** Measured at 30M: RS +8.8 pts
      (0.4930 -> 0.5807, reproducing paper Table 4), CUT-2 +0.05 (a wash),
      CUT-1 **-4.3** (0.6712 -> 0.6279). A larger action space cannot lower the
      *optimum* — pose 0 stays available — so this must be an optimisation cost,
      not a representational one. Evidence gathered:
    - Rotation is equally *available* on all three (73% of states gain legal LPs,
      ~40% of placements actually rotated, 23% of boxes are square) — so it is
      not that CUT cannot use rotation.
    - 2-orient leads early on **every** stream (+0.03 CUT-1, +0.03 CUT-2,
      +0.08 RS at 2M) then is overtaken on CUT-1 at ~5M, and the deficit widens.
    - Episode length is the tell: RS 11.93 -> 14.24 items (**+19%**, rotation
      unlocks a higher ceiling), CUT-1 17.33 -> 15.95 (**-8%**, it just packs
      worse). CUT-1/CUT-2 have a perfect packing at the as-cut orientations, so
      rotation opens no new ceiling there — all cost, no benefit.
    - Confounded by budget: nothing had converged at 30M. The 100M runs in
      [A](#a-in-progress--running-now) should settle whether CUT-1 closes the gap.
    - Single seed each — a repeat would rule out seed noise.
- [ ] **Why is a net trained on the true mask *worse* than the control when
      deployed on its own predictor?** 23.57% against 29.91% held out, with
      79.6% of episodes dying illegal against 48.6%. A common-state probe puts
      most of it on the policy rather than the predictor: on `kl_base`'s own
      states `mask_true`'s false-feasible rate is 6.9% vs 4.1% — worse, but
      nowhere near enough to explain 6.3 pp. The policy never had to hedge
      against a predictor mistake, so it does not. Worth testing whether
      *annealing* true → predicted over training keeps the +7.4 pp of learning
      under a clean mask without the deployment collapse.
- [x] **Does the deep default actually help? YES — +7.03 pp, the largest
      faithful lever in the whole sweep.** Settled by A.2; see "Network grid"
      under [DONE](#done). `f60cc2b` was right to make it the default.
- [ ] **The deep net is far more LR-sensitive than the paper-sized one, and
      3e-4 is too high for it.** This is what briefly made the deep net look
      like a regression: `bpp1_cut2_20` vs `bpp1_cut2_20_deep` at matched steps
      put the deep net 1.6 pp BEHIND and widening — but both sit on a 100M LR
      schedule, so both were pinned near lr 2.8e-4 for the whole comparison.
      Same architecture on a 10M schedule that anneals properly is *ahead*, and
      the gap grows as the LR falls:

  | step | 100M horizon (lr) | 10M horizon (lr) | delta |
  |---|---|---|---|
  | 5.8M | 0.2497 (2.8e-4) | 0.2895 (1.2e-4) | +3.98 pp |
  | 7.0M | 0.2504 (2.8e-4) | 0.3074 (8.6e-5) | +5.70 pp |
  | 9.0M | 0.2592 (2.7e-4) | 0.3451 (2.6e-5) | **+8.59 pp** |

      The same test on the paper-sized net moves only +0.63 → +1.98 pp, so this
      is specific to the 4.5M-param net, not a general property of annealing. At
      lr 3e-4 the two architectures are indistinguishable (0.2497 vs 0.2500 at
      5.8M) — the deep net's entire advantage appears only once the LR drops.
    - ~~Consequence: the 100M reproduction runs are LR-crippled.~~ **Withdrawn**
      — see the correction above. `net_both` reaching 36.94% at 10M against
      `bpp1_cut2_20`'s 39.49% at 100M is a real sample-efficiency win for the
      deep net, but it is not evidence the 100M run was mis-tuned: that run
      annealed fully and its number is fair.
- [ ] `frac_seq_end` is 0 — no episode ever exhausts its sequence. Expected (the paper packs 17.5 of ~26 items), but worth confirming it stays that way at convergence rather than being a silent bug.

---

## C. Next up — ordered by expected value

### 1. ACKTR / K-FAC optimiser ← *the paper's actual optimiser*

- Kronecker-Factored Approximate Curvature for `Conv2d` + `Linear`
- Running estimates of `A` (input covariance) and `G` (grad-output covariance), eigendecomposition refreshed every ~100 steps (Wu et al. 2017)
- Trust-region step: scale the update so `KL(π_old ‖ π_new) ≈ δ` (0.001)
- Add `--algo acktr` alongside `--algo ppo`; compare curves on CUT-2
- The paper reports ACKTR > SAC; we should also settle ACKTR vs our PPO

### 2. Multi-bin packing (paper Table 2, Algorithm 2)

- N parallel BPP-1 instances sharing one network
- Bin selection score: drop in critic value `V(s_b, n)` vs the last value estimate; default score for an empty bin `s_def = −0.2`; pack into the argmax
- Reproduce: 1/4/9/16/25 bins → 67.4% … 77.8% space utilisation

### 3. Height parameterisation ablation (paper Figure 5)

- Height vector (HV): `L·W`-dim flattened column stack, MLP encoder
- Item sequence vector (ISV): 6 params `(x,y,z,l,w,h)` per packed item
- Expect HM to beat HV by ~16.0% and ISV by ~19.1% on CUT-1

### 4. Reward ablation: step-wise vs termination reward (paper §F, Table 5)

- Termination reward = final space utilisation only
- Also the "extra information" variant that disambiguates Figure 15 (a) vs (b)

### 5. Resolution study (paper Figure 14 left)

- Train/test at 20³ and 30³; needs a deeper CNN + more steps
- Check the reported drop: Size-10 0.669 / Size-20 0.654 / Size-30 0.634

### 6. Generalisation matrix (paper Tables 7 & 8)

- Train on X, test on Y for X, Y ∈ {RS, CUT-1, CUT-2}
- Unseen item dimensions at test time

### 7. Corner & boundary rule ablation (paper Table 6)

- Restrict the action space to corner points / boundary-adjacent LPs
- Expect this to **hurt**: 66.9% → 60.9%, confirming the learned policy beats the hand-coded regularity prior

### 8. Comparison baselines not implemented

- **BPH** (Ha et al. 2017) — online heuristic with lookahead re-ordering (Fig. 8)
- **LBP** (Martello, Pisinger & Vigo 2000) — offline branch-and-bound heuristic
- Brute-force `k!` permutation search, to reproduce the Figure 7(a,b) time/quality trade-off against our MCTS

---

## D. Engineering / performance

### D.1 Training throughput — where the hardware actually goes ← *measured, CPU/GPU almost idle*

Prompted by watching training run with 16 cores and an RTX 4070 barely used
(2.3 of 16 cores busy, GPU SM 9-43%, memory bandwidth 1-8%). Profiled a full PPO
iteration (`num_envs=32`, `epochs=4 x minibatches=8`) end to end rather than
guessing from a single forward pass, which gave a wrong first answer (see below):

| phase | share | bound by |
|---|---|---|
| `collect` — the env loop (`VecPackingEnv.step`) + policy forward | **66.2%** | 1 CPU core / GPU latency |
| `update` — 32 grad steps (PPO epochs x minibatches) | **33.8%** | GPU |

Re-measured 2026-09-04 at `num_envs=32` on an idle RTX 4070 Laptop + 16 cores
(3,374 steps/s, 6.3 h per 100M steps). Broken down over the whole iteration:

| component | share of iteration | Amdahl ceiling if free |
|---|---|---|
| `env.step` — all of it | 44.1% | 1.79x |
| — of which `bin3d.feasibility_mask` | 23.7% | 1.31x |
| — of which `items.gen_sequence` (amortised) | 5.7% | 1.06x |
| — of which `bin3d.is_feasible` | 3.4% | 1.04x |
| policy forward inside `collect` | 22.1% | 1.28x |
| `update` | 33.8% | 1.51x |
| *all of `collect`* | *66.2%* | *2.96x* |

Inside `collect`, `bin3d.feasibility_mask`/`pose_mask` alone is **54%** of
`env.step`, and `items.gen_sequence` (the CUT cutting-stock recursion, re-run on
every episode reset) is another **14%** — both currently plain numpy in a
Python `for` loop over `num_envs`, i.e. one core.

**~~Raising `num_envs` buys nothing~~ — CORRECTED 2026-09-04. It is worth 1.57x.**
The original measurement (flat at ~10,200 steps/s from 32 to 512) was of the
**env loop alone**, and that part stands: `env.step` scales perfectly linearly
with `num_envs` (3 -> 6 -> 12 -> 24 -> 48 ms), i.e. zero parallelism, because
the simulator is numpy on the CPU, not a GPU kernel — the IsaacGym "1000 robots
on one GPU" pattern does not transfer. But the *conclusion drawn from it was
wrong end-to-end*, because 56% of an iteration is GPU work that amortises over
a bigger batch. Measured end-to-end with a trained net (`bpp1_cut2/best.pt`,
so episode lengths are realistic) and a sequence pool:

| `num_envs` | steps/s | vs 32 | h per 100M |
|---|---|---|---|
| 32 | 4,613 | 1.00x | 6.0 |
| 128 | 7,257 | **1.57x** | 3.8 |
| 256 | 7,731 | 1.68x | 3.6 |
| 512 | 7,878 | 1.71x | 3.5 |

The source is the per-step GPU round-trip: one policy forward costs 0.70 ms at
batch 32 and 1.17 ms at batch 512 — 16x the work for 1.7x the time. Shipped as
the `fast` preset. **Caveat:** batch goes 1,280 -> 5,120, i.e. 4x fewer gradient
steps per sample, so this is a wall-clock win that still needs an A/B on
utilisation-vs-*step* before it is used for a reproduction number.

*Ruled out:* running the collect-phase forward on CPU to dodge the round-trip.
The GPU wins at every batch size (0.70 ms vs 3.46 ms at batch 32).

**Running several trainings concurrently helps, but less than a naive
env-only benchmark suggests** — measured on real end-to-end trainings
(env + GPU fwd/bwd), not just the env loop:

| concurrent runs | per-run steps/s | aggregate | scaling |
|---|---|---|---|
| 1 | 3,364 | 3,364 | 1.00x |
| 2 | 2,929 | 5,859 | **1.74x** |
| 3 | 2,447 | 7,340 | **2.18x** |

This is *not* a way to make one run faster — each process still uses one core;
concurrency only helps when there are several runs queued (e.g. re-seeding an
experiment, or the streams x orientations matrix), by using idle cores instead
of leaving them idle. A pure env-loop benchmark (no GPU) showed near-linear
CPU scaling to 4 processes (1.0x / 1.99x / 3.95x / 5.78x at 8) — the gap
between that and the 1.74-2.18x above is the GPU's 39.5% share, which every
concurrent process contends for. Amdahl bounds follow directly from the
60.5/39.5 split: fixing *only* the env loop caps total speedup at **2.53x**;
fixing *only* the GPU caps it at **1.65x** — so no single optimisation below
gets close to using the hardware fully; they need to land together.

- [ ] **`SubprocVecEnv`** (currently a synchronous loop over N envs in one
      process) — real CPU parallelism for `collect`, capped at ~2.53x alone
      per the Amdahl bound above; needs (with GPU-batched mask below) to clear
      that ceiling
- [ ] **GPU-batched feasibility mask** — prototyped and validated in
      `scripts/probe_buffer.py`-style throwaway script (not committed):
      `F.unfold` + max reproduces `bin3d.pose_mask`'s sliding-window reduction;
      grouped by the <=16 distinct `(l,w)` footprints per batch (item dims are
      `{2..5}`), output was **bit-identical** to the numpy path on 64 random
      bins. Timed numpy-loop vs torch-GPU:
      | batch | numpy | torch GPU | speedup |
      |---|---|---|---|
      | 32 (current `num_envs`) | 1.83 ms | 4.70 ms | **0.4x — slower** |
      | 128 | 6.58 ms | 5.31 ms | 1.2x |
      | 512 | 26.36 ms | 6.55 ms | 4.0x |
      | 2048 | 99.41 ms | 11.75 ms | 8.5x |
      Only pays off above ~128 envs (kernel-launch overhead dominates below
      that) — so this and `SubprocVecEnv` are a package: batching the mask is
      what makes raising `num_envs` worth anything once the env loop is no
      longer serial.
- [x] Pre-generate/cache a pool of CUT-1/CUT-2 sequences instead of running
      `gen_sequence`'s cutting-stock recursion on every episode reset — shipped
      as `--seq-pool N` (off by default; `fast` preset uses 40,000).
      > **Worth less than it looks.** Benchmarked with an *untrained* net it
      > shows 1.35-1.57x, but only because a random policy dies after 1.8 items
      > so `gen_sequence` runs almost every step. With a trained net (episode
      > length 12) it is **1.05x**, matching the 5.7% amortised cost. Take it —
      > it is ~5 lines — but the throughput is in `num_envs`, not here.
- [ ] **MCTS speed** — batch network evaluations across simulations (currently one forward pass per node visit). Target the paper's 3.6 s per decision at k=20. Add a transposition table keyed on `(height-map bytes, item, remaining set)`.
- [ ] Mixed precision / `torch.compile` for the CNN — revisit after the above:
      the GPU is 39.5% of a PPO iteration, not the ~2% a single forward pass
      would suggest, so this is worth more than first assumed
- [x] Larger minibatches / fewer PPO epochs — attacks the `update` 33.8% share
      directly; exposed as `--epochs` / `--minibatches`. Measured at
      `num_envs=32`: 4x8 -> 4x4 = 1.10x, 4x2 = 1.17x, 2x8 = 1.19x, 2x4 = 1.25x,
      1x4 = 1.32x; stacked with `fast` (128 envs, 2x4) the whole config reaches
      **1.98x** (3.2 h per 100M). Trades gradient steps for wall-clock — still
      needs the utilisation-vs-step A/B, so defaults stay at the paper's 4x8.

### D.2 Other

- [ ] Deterministic replay — store RNG state in checkpoints so a run is bit-exact resumable, not just weight-resumable
- [ ] Real test suite — the `bin3d`/`items` checks currently live in throwaway scripts and should be pytest cases guarding future refactors

---

## E. Research extensions beyond the paper

### E.1 Buffer / selection — let the agent choose *which* of the k boxes to place ← *measured, largest gain found so far*

The paper's own future work ("buffer zone", hold `b < k` items aside). In the
paper's BPP-k the agent **sees** k boxes but must always place the first one;
the permutation search only imagines reorderings. Letting it actually *pick*
turns out to be worth far more than searching with the same k.

**Measured** with a throwaway probe — [scripts/probe_buffer.py](scripts/probe_buffer.py),
the trained BPP-1 nets reused unchanged, 100 held-out sequences per cell, buffer `k=5`:

| policy | CUT-2 | RS |
|---|---|---|
| `forced` — must take the front box (= BPP-1) | 62.1% | 51.4% |
| `skip` — take the first one that *fits* | 71.0% **(+8.9)** | 62.3% **(+10.9)** |
| `choose` — take the best-scoring one | **78.0% (+15.9)** | **73.9% (+22.5)** |
| *BPP-5 MCTS at the same k, for comparison* | *70.8%* | *61.4%* |

- **Selection beats search at equal k** by +7.2 (CUT-2) / +12.6 (RS) points, and
  runs ~20× faster: 0.10 s/episode vs ~2.3 s for BPP-5 MCTS.
- Roughly **half the gain is just not terminating** when the front box does not
  fit — forced order dies there, a buffer only dies when *none* of the k fit.
  The rest is the choice itself.
- These are **lower bounds**: the nets were trained under forced order and used
  off-policy. A policy trained to select should do better still.
- ⚠ This is a **different, strictly easier problem** — BPP *with a buffer*, not
  online BPP. Never report it as a reproduction number or compare it to the
  paper's tables.
- ⚠ It costs hardware: `k` buffer slots means a physical accumulation / sorting
  station upstream of the pick.

To do properly:
- Buffer in the env: observation = height map + `b` item triples; action =
  `(which box, orientation, LP)`, so the action space is `b · O · L · W` with one
  feasibility mask per buffered box
- Train under the same constrained scheme (MP / MC / FE) rather than reusing a
  forced-order net; compare against forced order at matched `b`
- Ablate `b = 1…5`, and characterise the **clog** failure mode — a box nothing
  can ever place occupies a slot for the rest of the episode
- Relevant to the real palletiser: even `b = 3` is worth more than any amount of
  extra training or search

### E.2 Others

- [ ] **Irregular / non-cuboid items** (the paper's harder open problem)
- [ ] 6-DoF orientation (currently only the 2 horizontal yaw poses the paper uses)
- [ ] Real mass / centre-of-mass stability instead of the support-area proxy; couple to a rigid-body sim for verification
- [ ] **Sim-to-real** — feed real UR20 pallet dimensions + gripper reachability as extra action constraints folded into the feasibility mask
- [ ] Transformer / PCT-style attention over the lookahead set instead of MCTS — follow-up work: *"Learning Efficient Online 3D Bin Packing on Packing Configuration Trees"*, ICLR 2022, same authors

---

# DONE

## Completed experiments — results

**Trained networks** — 3 streams x 2 orientation settings, the full matrix the
duel game offers. All six reached 30M steps and were evaluated; the 100M
extension is in [A](#a-in-progress--running-now).

| stream | 1 orientation | 2 orientations | delta |
|---|---|---|---|
| CUT-1 | 0.6712 | 0.6279 | **-4.33** |
| CUT-2 | 0.6161 | 0.6166 | +0.05 |
| RS | 0.4930 | 0.5807 | **+8.77** |

RS reproduces paper Table 4 (62.1% w/ vs 50.5% w/o re-orientation); the CUT
results are an open question, see [B](#b-open-questions-raised-by-the-reproduction).

**20³ reproduction, all three streams** — evaluated 2026-09-09; the 100M runs
had reached their budget but had never been scored (no `eval.json` existed).
BPP-1, 500 held-out episodes each, `terminate`, paper-sized net, scored on the
stream it trained on. Paper column is Figure 14's Size-20 resolution study.

| run | stream | ours | paper Size-20 | gap |
|---|---|---|---|---|
| `bpp1_cut1_20` | CUT-1 | 39.58% | 63.4% | **−23.8** |
| `bpp1_cut2_20` | CUT-2 | 39.49% | 65.4% | **−25.9** |
| `bpp1_rs_20` | RS | 31.65% | 58.1% | **−26.5** |

> The gap is **24–26 pp and near-identical on all three streams**, which is
> itself informative: a uniform shortfall points at the optimiser / setup rather
> than anything stream-specific, and it matches the ~24 pp that the lever
> sweep's mask decomposition below could not account for. Two independent routes
> to the same residual is the strongest argument yet for
> [C.1](#c-next-up--ordered-by-expected-value) (ACKTR) being the real gap.
>
> Note 10³ CUT-2 reproduced cleanly (67.1% vs 66.9%). The paper loses only
> 1.5 pp going 10³ → 20³; we lose 26. Whatever we are missing scales with the
> action space.

**Network grid completed + the best config so far, 20³ CUT-2** — 3 runs x 10M
steps, 2026-09-10, each differing from `net_both` by exactly one lever. BPP-1 on
500 held-out CUT-2 episodes.

| run | config | params | eval util | items | ends illegal | vs control |
|---|---|---|---|---|---|---|
| **`deep_stack`** | 4/128/1024 + `w_mask 5.0` + `target_kl 0.2` | 4,547,877 | **43.03%** | 16.71 | 12.8% | **+13.12** |
| `net_both` | 4/128/1024 | 4,547,877 | 36.94% | 14.39 | 30.6% | +7.03 |
| `deep_ch` | 4/128/256 | 1,473,573 | 35.55% | 13.76 | 38.8% | +5.64 |
| `deep_lr1e4` | 4/128/1024, lr 1e-4 | 4,547,877 | 35.36% | 13.54 | 42.2% | +5.44 |
| `net_depth` | 4/64/256 | 1,138,981 | 33.04% | 12.60 | 43.4% | +3.12 |
| `kl_base` | *control, 2/64/256* | 1,065,125 | 29.91% | 11.46 | 48.6% | — |

**`deep_stack` at 43.03% beats the 100M reproduction run (39.49%) on one tenth
the steps.** It is the best result the project has produced, and it is faithful:
every lever only re-weights or constrains the existing objective. Its mechanism
is the mask — false-feasible rate 0.24% against `net_both`'s 0.96% (4x better),
which drops episodes ending illegally from 30.6% to 12.8%.

> `target_kl 0.2` turns out to be self-releasing rather than a permanent brake:
> it froze the actor on ~78% of minibatches early and only **0.5%** over the
> final 1M steps. It acts as a warm-up constraint that relaxes as the policy
> converges. Still confounded with `w_mask 5.0` though — the 4x mask improvement
> suggests `w_mask` does most of the work, but that is inference. `deep_wmask`
> would settle it.

**The network grid is now complete, and the three axes are almost perfectly
additive:**

| step | eval | increment |
|---|---|---|
| 2/64/256 baseline | 29.91% | — |
| + depth, 2 → 4 layers | 33.04% | **+3.12** |
| + channels, 64 → 128 | 35.55% | **+2.52** |
| + trunk width, 256 → 1024 | 36.94% | **+1.39** |

They sum to +7.03, exactly `net_both`'s total. Ranking is **depth > channels >
trunk width**, i.e. the cheapest axis is the most valuable — depth costs 7% more
parameters, trunk width costs 289% for less than half the gain. `deep_ch`
(4/128/256) captures 96% of `net_both`'s benefit at **one third the parameters**,
and is the config to prefer whenever memory or training cost binds.

> Inference cost is flat across all of these — 0.70-0.74 ms per placement at
> batch 1, against the control's 0.67 ms. The GPU is latency-bound at batch 1,
> so the 4x-larger network is essentially free to *serve*; it only costs
> training wall-clock. That matters for the UR20 cell.

**Projection.** `deep_stack`'s pre-anneal slope is 2.76 pp/doubling (lower than
`net_both`'s 4.76 — it starts higher, so there is less headroom). From 43.03% at
10M, 100M lands at **~55-58%** against the paper's 65.4%. The remaining 7-10 pp
sits squarely where the paper's own Table 10 puts the optimiser (ACKTR 66.9% vs
A2C 53.0%), which is now the third independent line of evidence pointing at
[C.1](#c-next-up--ordered-by-expected-value).

**Network grid + follow-up levers, 20³ CUT-2** — 5 runs x 10M steps,
2026-09-09 (section A.2), each matching the `kl_base` control except one lever.
BPP-1 on 500 held-out CUT-2 episodes.

| run | lever | params | eval util | items | ends illegal | vs control |
|---|---|---|---|---|---|---|
| `net_both` | 4 layers / 128 ch / 1024 | 4,547,877 | **36.94%** | 14.39 | 30.6% | **+7.03** |
| `net_depth` | 4 layers / 64 ch / 256 | 1,138,981 | 33.04% | 12.60 | 43.4% | +3.12 |
| `tkl_fixed` | `target_kl 0.2` | 1,065,125 | 31.06% | 11.90 | 53.4% | +1.15 |
| `net_width` | 2 layers / 64 ch / 1024 | 4,139,429 | 30.44% | 11.64 | 54.8% | +0.52 |
| `kl_base` | *control, 2/64/256* | 1,065,125 | 29.91% | 11.46 | 48.6% | — |
| `true_eps0` | true mask + `eps 0`, own predictor | 1,065,125 | 22.59% | 7.94 | 86.0% | −7.33 |
| *`true_eps0` scored with the true mask* | *(clean ceiling)* | | *46.77%* | *18.45* | *0.0%* | *+16.85* |

**Depth is the axis that matters; width alone is nearly worthless.** +3.12 pp
for 7% more parameters (4 layers gives a 9×9 receptive field against 5×5, i.e.
20% of a 20×20 height map instead of 6%), against +0.52 pp for 3.9x the
parameters. The full 4/128/1024 net adds another +3.9 pp over depth alone, but
it varies conv channels *and* trunk width together, so that increment is not
attributable to either — a 4/128/256 cell would separate them.

> Per-GPU-hour is NOT settled by this table. The logged fps (2034 control /
> 1498 depth / 1089 both) came from runs sharing a GPU with 2–3 others, and the
> waves had different contention, so they are not comparable. What *is* clean:
> `net_both` reaches 36.94% at 10M where the paper-sized net needs 100M to
> reach 39.49%, so the deep net buys ~10x the sample efficiency for <2x the
> per-step cost. See the LR-sensitivity item in
> [B](#b-open-questions-raised-by-the-reproduction) — that ratio only holds
> with an annealed LR.

**The trust region works once asked honestly: +1.15 pp at `target_kl 0.2`**,
freezing the actor on just 2.5% of minibatches. Against `target_kl 0.02`'s
22.87% (99–100% of updates aborted, ~10% of gradient steps taken) this is the
whole difference between a bug and a trust region. It also strengthens
[C.1](#c-next-up--ordered-by-expected-value): if a crude first-order KL bound is
worth +1.15 pp, a Fisher-preconditioned one should be worth more.

**The ceiling closed: `eps=0` on the true mask drove illegal moves to exactly
zero** — `invalid_rate` 0.0000 and `frac_invalid_end` 0.0000 for the entire run,
not merely small — and reached 46.77%. The full ladder:

| configuration | eval util |
|---|---|
| control, own predictor | 29.91% |
| control, handed the true mask at test time | 34.13% |
| trained under true mask, `eps=1e-3` (`mask_true`) | 41.54% |
| trained under true mask, `eps=0` (`true_eps0`) | **46.77%** |
| paper Size-20 | 65.4% |

So perfect, leak-free feasibility is worth **+16.9 pp of the 35.5 pp gap (48%)**,
and **~18.6 pp (52%) survives it**. Removing the `eps` leak alone added +5.2 pp
on top of `mask_true` — far more than its 3.0% illegal-ending rate suggested.
`true_eps0` also reproduces the `mask_true` trap more sharply: 22.59% on its own
predictor, −7.3 pp below control, 86% of episodes dying illegal. Diagnostic only.

**Lever sweep, 20³ CUT-2** — 7 runs x 10M steps, 2026-09-08, paper-sized net
(2/64/256), `terminate`, seed 0, one lever each off the `kl_base` control.
Utilisation is BPP-1 on 500 held-out CUT-2 episodes (`seed=999`); "illegal" is
the share of episodes ending on an illegal placement rather than a full bin.

| run | lever | eval util | items | ends illegal | vs control |
|---|---|---|---|---|---|
| `mask_w5` | `w_mask 5.0` | **31.98%** | 12.31 | 42.4% | **+2.07** |
| `mask_w2` | `w_mask 2.0` | 31.03% | 11.87 | 42.0% | +1.12 |
| `kl_base` | *control* | 29.91% | 11.46 | 48.6% | — |
| `kl_lr` | `lr 1e-4` | 26.26% | 9.99 | 56.2% | −3.65 |
| `kl_tkl` | `target_kl 0.02` | 22.87% | 8.60 | 63.2% | −7.04 |
| `kl_both` | both | 19.56% | 7.50 | 71.6% | −10.35 |
| `mask_true` | true mask to policy | 23.57% | 8.65 | 79.6% | −6.34 |
| *`mask_true` scored with the true mask* | *(ceiling)* | *41.54%* | *16.36* | *3.0%* | *+11.63* |
| *`kl_base` scored with the true mask* | *(ceiling)* | *34.13%* | *13.46* | *13.4%* | *+4.22* |

Three results, in order of how much they change the plan:

1. **The illegal-move rate is worth about a third of the gap, not all of it.**
   Handing the trained control a perfect mask at test time buys +4.2 pp;
   training under one as well buys +7.4 pp more. All mask error together is
   ~+11.6 pp of the 35 pp gap from the control to the paper's Size-20 65.4%.
   The remaining ~24 pp is packing quality — which points at the optimiser
   ([C.1](#c-next-up--ordered-by-expected-value)) and the network, not at the
   constrained-DRL scheme. This retires "fix the illegal moves first" as the
   headline plan.
2. **`w_mask 5.0` is the best faithful lever found: +2.1 pp**, and it only
   re-weights the existing loss. But note it HALVED the training false-feasible
   rate (1.98% → 1.01%) to move `invalid_rate` only 4.51% → 4.00%: per-cell
   mask accuracy has stopped being the binding constraint. Take it, stop tuning.
3. **`target_kl 0.02` measured nothing** — see the note in
   [src/config.py](src/config.py). It tripped on 99–100% of updates after a
   mean of ~3 of 32 minibatches, and the trip aborted the critic and mask head
   too, so those runs took ~10% of the intended gradient steps. Both faults are
   fixed; the honest question is re-asked at 0.2 in [A.2](#a2-lever-sweep-follow-up--launched-2026-09-09).

> ⚠ **Never quote a `use_true_mask_for_policy` run's TRAINING utilisation.** It
> is measured with the ground-truth mask in the loop and read 36.7% against a
> real 23.6%. Single seed (0) throughout, so the ±1–2 pp rows are inside
> plausible seed noise; the ±6–11 pp ones are not. All seven annealed their LR
> to the 5% floor at their 10M horizon, so none is a converged number.

**BPP-k costs no extra training** — the same BPP-1 network, more search at test
time (paper Sec. 3.3). Note the rows are at **different budgets**: CUT-2 has been
re-evaluated after its 100M extension, the other two are still 30M-milestone
numbers.

| run | stream | eval @ step | BPP-1 | BPP-3 | BPP-5 | paper BPP-1 |
|---|---|---|---|---|---|---|
| `bpp1_cut2` | CUT-2 | 96.3M | **67.1%** | 74.4% | 76.4% | 66.9% ✅ |
| `bpp1_cut1` | CUT-1 | 29.4M | 67.7% | 77.3% | 81.3% | 73.4% |
| `bpp1_rs` | RS | 29.6M | 51.1% | 58.9% | 61.3% | 50.5% ✅ |

> **CUT-2 is the first cell of the reproduction to land**: 67.1% against the
> paper's 66.9%, after the 100M extension took its training utilisation from
> 0.6161 to 0.6784. RS already matched at 30M (51.1% vs 50.5%). CUT-1 is 5.7 pts
> short at 30M and is the run to watch as it extends.

Monotone in k on all three streams, matching Fig. 7(b).

BPP-k end reasons contain **zero** `invalid_action` across every evaluation — the order-dependence blocking
guarantees the returned action is legal on the real bin.

**Human-vs-agent duel** — [src/viz/game.py](src/viz/game.py), logged to
`runs/duel_log.jsonl`. Same seeded stream for both sides, matched lookahead
(including the box BPP-k only peeks at), rotation gated on the checkpoint's
`orientations`, opponent list filtered to the agents trained on the chosen stream.

---

## Implemented and verified

### Environment / benchmarks

- [x] Height-map bin, `L=W=H=10`, action `a = x + L*y` — [src/bin3d.py](src/bin3d.py)
- [x] 64 pre-defined item types `{2,3,4,5}³`, all dims ≤ `L/2` — [src/items.py](src/items.py)
- [x] Physical stability criterion (paper §3.1): `>60%` area + 4 corners, **or** `>80%` + 3 corners, **or** `>95%` area
- [x] Feasibility mask `M`, vectorised via sliding-window max over the height map
  > **Verified:** matches a brute-force per-action feasibility test cell-for-cell, including the doubled 2-orientation action space.
- [x] RS / CUT-1 / CUT-2 sequence generators (cutting stock, Gilmore & Gomory)
  > **Verified:** replaying a CUT-2 sequence at its ground-truth positions gives 100% space utilisation in **300/300** trials — the benchmark is genuinely perfectly packable, so measured utilisation is meaningful.
- [x] Order-dependence blocking (raise height map to `H` over a later item's footprint) — used by BPP-k
- [x] Episode termination: sequence end / no feasible LP / illegal action chosen
- [x] Synchronous vector env — measured **~7,900 env FPS** (paper quotes 2,000)

### Model / constrained DRL

- [x] State CNN → `4·L·W = 400` features, matching the paper's Figure 9 bottleneck
- [x] Actor + critic heads, and an **independent** MLP mask predictor off the CNN
- [x] **MP** — mask prediction, MSE loss against the ground-truth mask
- [x] **MC** — projection: infeasible action probs multiplied by `ε = 1e-3` (applied in log space for numerical stability), not zeroed
- [x] **FE** — entropy bonus computed only over LPs with `M(x,y) = 1`
- [x] `E_inf` — summed probability mass on genuinely infeasible LPs
- [x] Composite loss (paper Eq. 1) with the paper's weights:
  `L = 1.0·L_actor + 0.5·L_critic + 0.5·L_mask + 0.01·E_inf − 0.01·E_entropy`
- [x] Step-wise reward `r = 10·l·w·h / (L·W·H)`, `γ = 1`
- [x] PPO trainer (clipped surrogate, GAE, value clipping, linear LR decay)
  > The paper uses ACKTR. See [C.1](#c-next-up--ordered-by-expected-value).
- [x] Checkpointing, `--resume`, best/latest, ablation switches `--no-mp` / `--no-mc` / `--no-fe`
- [x] `--mask-eps`, including a hard `0` projection
  > `log(0)` cannot be used for it: the feasibility-restricted entropy forms
  > `p·log(p)·M` and `0 · -inf` is NaN, which poisons the loss. A large finite
  > penalty underflows to `p = 0` in the softmax and keeps an all-infeasible row
  > uniform instead of NaN. **Verified:** `--use-true-mask --mask-eps 0` trains
  > with `invalid_rate` exactly 0.0000.
- [x] `target_kl` freezes the **actor only**, not the whole update
  > The original `break` also stopped the critic and the mask head — supervised
  > objectives with no trust region to respect — which is what made
  > `target_kl=0.02` catastrophic rather than merely conservative. `kl_stop` is
  > now the fraction of minibatches frozen (it used to be an unreadable `1/nb`).
  > **Verified:** at 0.02 the fixed code freezes 89% of minibatches at unchanged
  > steps/s, i.e. all 32 minibatches still run.

> **Fixed 2026-09-09: `src.evaluate --use-true-mask` had never done anything.**
> [0105920](src/evaluate.py) wired the flag into `cfg`, but `NetPolicy.__call__`
> hardcoded the predicted mask and never read
> `cfg.use_true_mask_for_policy` — so `yes` and `no` returned byte-identical
> numbers, and the warning about inheriting the flag was itself misleading. This
> is what hid the ceiling result: `mask_true` looked like a flat 23.57% either
> way, when its actual ceiling is 41.54%.

> **Settled during bring-up.** The mask predictor plateaued at ~92% accuracy in a short run, which killed 92% of episodes via illegal actions. A standalone supervised probe showed the cause was **undertraining, not architecture**: the paper's exact MLP + MSE head reaches **99.3% accuracy / 0.9% false-feasible** given enough gradient steps, matching the paper's "99.5% legit" claim. A conv head and a BCE loss were both tried and were **not** better. The architecture therefore stays exactly as the paper specifies, and `mask_fpr` (false-feasible rate) is now a logged metric.

### BPP-k

- [x] Monte-Carlo permutation tree search, paper Algorithm 1 — [src/mcts.py](src/mcts.py)
- [x] **Max**-return backup (paper supplemental B), not mean
- [x] UCB1 with returns normalised by the running min/max of the search
- [x] Tree policy hallucinates placements; default policy rolls out the remainder in arrival order and adds `V(s, Last)` at the leaf
- [x] Order dependence enforced inside the tree
- [x] `mcts_last_item` flag — `observed` (faithful to Algorithm 1, peeks one item past the lookahead window for the leaf value) vs `mean` (strictly online)

### Extensions / baselines

- [x] Item re-orientation: 2 horizontal poses, doubled action space, one mask per pose (paper Table 4) — `Config.orientations`
- [x] Boundary-rule heuristic via maximal spare cuboids (supplemental D)
  > **Caveat:** ours scores **60.7%** on CUT-2 vs the paper's reported **40.8%** — our baseline is *stronger* than theirs, so the comparison in the report runs against us. Worth investigating why they differ (see [B](#b-open-questions-raised-by-the-reproduction)).
- [x] Deepest-bottom-left baseline (55.3% CUT-2) and random-feasible (29.4% CUT-2)
- [x] Ablation harness for the MP/MC/FE grid (paper Table 1)

### Visualisation — the closed loop

- [x] Live training dashboard — stdlib HTTP + self-contained SVG, no deps; 12 charts, multi-run overlay, light/dark — [src/viz/dashboard.py](src/viz/dashboard.py)
  > **Verified:** rendered headless against live training data.
- [x] 3D packing replay → per-step GIF/MP4, final PNG, contact sheet — [src/viz/replay3d.py](src/viz/replay3d.py)
- [x] Live 2D view: height map │ true mask │ predicted mask (disagreements marked ×) │ projected action probs; `--live` steps with arrow keys — [src/viz/heatmap.py](src/viz/heatmap.py)
- [x] Self-contained HTML report: our numbers vs the paper's Tables 1/3/4, BPP-k curve, utilisation histograms, training curves, embedded replays — [src/viz/report.py](src/viz/report.py)
- [x] Resumable end-to-end pipeline — [scripts/run_all.sh](scripts/run_all.sh)
- [x] [README.md](README.md) (method summary) and [RUN.md](RUN.md) (commands)

---

