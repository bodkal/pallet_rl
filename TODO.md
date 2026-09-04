# TODO / Status

Reproduction of **"Online 3D Bin Packing with Constrained Deep Reinforcement Learning"**
— Zhao, She, Zhu, Yang & Xu, AAAI 2021 ([arXiv:2006.14978](https://arxiv.org/abs/2006.14978)).

*Last updated: 2026-08-31*

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

- [ ] Finish the 100M extension for all six, then re-evaluate (the eval guard
      re-runs automatically once `best.pt` is newer than `eval.json`)
- [ ] MP/MC/FE ablation, 4 runs x 8M steps — not started
      (paper Table 1: 7.8% / 27.9% / 63.7% / 63.0% / 66.9%)
- [ ] Re-run `scripts/probe_buffer.py` against the 100M nets

> Extending `STEPS` resumes rather than restarts, but **resets the LR schedule to
> the new horizon** — measured jump 1.5e-05 -> 2.0e-04 at the hand-over. Expect a
> brief dip before the curve climbs past its old level.

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
| `collect` — the env loop (`VecPackingEnv.step`) + policy forward | **60.5%** | 1 CPU core |
| `update` — 32 grad steps (PPO epochs x minibatches) | **39.5%** | GPU |

Inside `collect`, `bin3d.feasibility_mask`/`pose_mask` alone is **54%** of
`env.step`, and `items.gen_sequence` (the CUT cutting-stock recursion, re-run on
every episode reset) is another **14%** — both currently plain numpy in a
Python `for` loop over `num_envs`, i.e. one core.

**Raising `num_envs` buys nothing** — measured flat at ~10,200 steps/s from 32
envs up to 512; `env.step` scales perfectly *linearly* with `num_envs` (3 -> 6
-> 12 -> 24 -> 48 ms), i.e. zero parallelism, while the network handles 16x the
batch for 4x the time. The "1000 robots on one GPU" pattern (IsaacGym-style)
does not transfer here because the *simulator* is numpy on the CPU, not a GPU
kernel — unlike those environments, which put the sim itself on the GPU.

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
- [ ] Pre-generate/cache a pool of CUT-1/CUT-2 sequences instead of running
      `gen_sequence`'s cutting-stock recursion on every episode reset (14% of
      `env.step`) — cheap win, independent of the two above
- [ ] **MCTS speed** — batch network evaluations across simulations (currently one forward pass per node visit). Target the paper's 3.6 s per decision at k=20. Add a transposition table keyed on `(height-map bytes, item, remaining set)`.
- [ ] Mixed precision / `torch.compile` for the CNN — revisit after the above:
      the GPU is 39.5% of a PPO iteration, not the ~2% a single forward pass
      would suggest, so this is worth more than first assumed
- [ ] Larger minibatches / fewer PPO epochs — attacks the `update` 39.5% share
      directly; untested

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

