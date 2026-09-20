# Adjustable Robust RL for Online 3D Bin Packing
# https://github.com/panyxy/ar2l_bpp
A from-scratch reproduction of **"Adjustable Robust Reinforcement Learning for
Online 3D Bin Packing"** — Pan, Chen & Lin, NeurIPS 2023
([arXiv:2310.04323](https://arxiv.org/abs/2310.04323); the paper is
[`2310.04323v1.pdf`](2310.04323v1.pdf) in this directory).

Boxes arrive on a conveyor. You can see the next `N_B` of them, you must place
the front one now, and it can never be moved again. A packing policy trained on
random orderings does well *on average* — and then a bad ordering arrives and it
falls over. AR2L asks for both: a policy that survives the worst ordering an
adversary can construct, without giving up the average case.

The paper's three moving parts, all implemented here:

1. a **permutation-based attacker** — a policy that reorders the visible
   conveyor to make the instance as hard as possible for a given packer;
2. **exact AR2L** — a third network, the *mixture-dynamics model*, trained to
   maximise the packer's return while staying close to both the nominal
   ordering and the attacker's, with a weight `α` that dials the trade-off;
3. **approximate AR2L** — the same objective without the third network, by
   putting the dual of an adjustable robust Bellman operator into the critic.

<!-- RESULTS:start -->

### Headline

Space utilisation on 3000 held-out instances of 150 items; &beta; is the share of them reordered by that policy&rsquo;s own permutation-based attacker. *drop* is what the attacker costs.

**N_B = 10**

| method | &beta;=0 | &beta;=100 | drop | items at &beta;=100 | Std. at &beta;=100 |
|---|---|---|---|---|---|
| PCT | 75.9 | 64.2 | &minus;11.6 | 27.3 | 8.0 |
| CPPO | 75.5 | 64.0 | &minus;11.5 | 29.4 | 7.2 |
| RARL | 75.9 | 65.7 | &minus;10.2 | 26.2 | 7.7 |
| RfMDP | 72.6 | 58.4 | &minus;14.1 | 24.0 | 8.8 |
| ExactAR2L(0.5) | 75.9 | 64.6 | &minus;11.3 | 27.8 | 6.9 |
| ExactAR2L(1.0) | 75.4 | 65.1 | &minus;10.4 | 29.6 | 6.9 |
| ApproxAR2L(0.5) | 69.1 | 57.6 | &minus;11.5 | 24.0 | 10.4 |
| ApproxAR2L(1.0) | 70.1 | 57.8 | &minus;12.4 | 25.7 | 7.9 |

**N_B = 20**

| method | &beta;=0 | &beta;=100 | drop | items at &beta;=100 | Std. at &beta;=100 |
|---|---|---|---|---|---|
| PCT | 75.5 | 63.4 | &minus;12.1 | 30.4 | 7.6 |
| CPPO | 74.6 | 56.3 | &minus;18.3 | 27.8 | 13.3 |
| RARL | 74.7 | 62.3 | &minus;12.4 | 28.0 | 9.4 |
| RfMDP | 74.1 | 58.2 | &minus;15.9 | 25.3 | 10.4 |
| ExactAR2L(0.5) | 75.7 | 63.8 | &minus;11.9 | 28.7 | 7.3 |
| ExactAR2L(1.0) | 75.5 | 62.6 | &minus;12.9 | 33.0 | 10.5 |
| ApproxAR2L(0.5) | 71.2 | 50.8 | &minus;20.4 | 25.2 | 11.4 |
| ApproxAR2L(1.0) | 70.5 | 53.3 | &minus;17.2 | 27.1 | 7.9 |


### table2

**N_B = 10**

| method | β=0 ours | β=0 paper | β=25 ours | β=25 paper | β=50 ours | β=50 paper | β=75 ours | β=75 paper | β=100 ours | β=100 paper |
|---|---|---|---|---|---|---|---|---|---|---|
| PCT | 75.9 | 76.4 | 73.0 | 70.6 | 70.1 | 65.1 | 67.1 | 61.4 | 64.2 | 55.7 |
| CPPO | 75.5 | 75.6 | 72.6 | 70.7 | 69.8 | 66.2 | 66.9 | 62.3 | 64.0 | 57.4 |
| RARL | 75.9 | 74.3 | 73.3 | 71.1 | 70.8 | 69.2 | 68.2 | 65.6 | 65.7 | 63.3 |
| RfMDP | 72.6 | 74.4 | 69.1 | 70.5 | 65.6 | 65.7 | 62.0 | 60.8 | 58.4 | 55.9 |
| ExactAR2L(0.5) | 75.9 | 77.6 | 73.0 | 73.1 | 70.2 | 68.0 | 67.4 | 64.0 | 64.6 | 59.7 |
| ExactAR2L(1.0) | 75.4 | 76.0 | 72.8 | 72.4 | 70.2 | 70.3 | 67.7 | 66.7 | 65.1 | 63.8 |
| ApproxAR2L(0.5) | 69.1 | 76.2 | 66.3 | 72.1 | 63.4 | 66.9 | 60.5 | 62.1 | 57.6 | 56.1 |
| ApproxAR2L(1.0) | 70.1 | 73.6 | 67.0 | 69.3 | 63.9 | 66.1 | 61.0 | 61.9 | 57.8 | 57.1 |

**N_B = 20**

| method | β=0 ours | β=0 paper | β=25 ours | β=25 paper | β=50 ours | β=50 paper | β=75 ours | β=75 paper | β=100 ours | β=100 paper |
|---|---|---|---|---|---|---|---|---|---|---|
| PCT | 75.5 | 77.0 | 72.5 | 68.4 | 69.6 | 59.7 | 66.4 | 50.9 | 63.4 | 41.9 |
| CPPO | 74.6 | 74.1 | 70.1 | 66.7 | 65.6 | 59.1 | 61.0 | 53.2 | 56.3 | 45.8 |
| RARL | 74.7 | 72.0 | 71.5 | 68.6 | 68.3 | 64.6 | 65.2 | 61.7 | 62.3 | 58.7 |
| RfMDP | 74.1 | 73.8 | 70.1 | 69.4 | 66.2 | 64.7 | 62.2 | 59.4 | 58.2 | 54.4 |
| ExactAR2L(0.5) | 75.7 | 76.8 | 72.8 | 70.0 | 69.7 | 64.7 | 66.8 | 60.0 | 63.8 | 54.4 |
| ExactAR2L(1.0) | 75.5 | 76.1 | 72.2 | 70.9 | 69.0 | 66.7 | 65.8 | 62.8 | 62.6 | 58.5 |
| ApproxAR2L(0.5) | 71.2 | 75.0 | 66.1 | 70.1 | 61.0 | 63.9 | 56.0 | 58.8 | 50.8 | 53.1 |
| ApproxAR2L(1.0) | 70.5 | 73.4 | 66.3 | 68.2 | 61.8 | 65.6 | 57.5 | 61.9 | 53.3 | 57.6 |
80 cells: mean gap to the paper +1.7 points, mean |gap| 3.5, worst -7.1 / +21.5


### the paper's comparative claims, on our numbers

| claim in the paper | holds here |
|---|---|
| ExactAR2L(1.0) packs at least as many items as PCT | 9/10 |
| ExactAR2L(1.0) has a smaller Std. than PCT | 5/10 |
| ExactAR2L(1.0) Std. lies between RARL's and PCT's | 2/10 |
| RARL gives up nominal utilisation relative to PCT (beta=0) | 1/2 |
| ExactAR2L(1.0) beats PCT once a quarter of the set is attacked | 3/8 |
| ApproxAR2L(0.5) packs more items than RfMDP | 1/10 |
| ApproxAR2L(0.5) beats RfMDP on utilisation | 0/10 |
| the attacker hurts: utilisation falls monotonically in beta | 16/16 |

<!-- RESULTS:end -->

Table 1 is being regenerated against the current action space; the pre-rotation
version is kept at `results/table1_norot.json`. Everything superseded by the
rewrite is archived under [`old_run/`](old_run/README.md), which says what was
kept and why.

### How to read this

Across all 80 cells of Table 2 the mean absolute gap to the paper is **3.5
points** (mean signed **+1.7**). Nominal utilisation lands within a point of the
paper for five of the eight methods — PCT gets 75.9 against the paper's 76.4 at
`N_B = 10`.

**Everything that is still off is in the attacked columns, and in one
direction: our attacker is too weak.** It costs PCT 11.6 points at `N_B = 10`
where the paper's costs 20.7, and 12.1 at `N_B = 20` where the paper's costs
35.1 — the paper's attacker gets much more damaging as `N_B` grows, ours barely
moves. With every method's `β=100` column that high, they bunch together, which
is why the paper's comparative claims above score near chance even though the
absolute numbers are close.

Part of that is real rather than a defect: rotation doubles the placements
available, so a hostile ordering has less bite. Measured on DBL, which never
trains, so the action space is the only thing varying — 56.1 → 32.9 (−23.2) with
one orientation, 62.3 → 44.2 (−18.1) with two. **Rotation absorbs about 5 points
of attacker damage.** That explains roughly half the shortfall at `N_B = 10` and
almost none of it at `N_B = 20`. The rest is tracked in
[TODO.md](TODO.md#c1-what-is-left-the-attacker-and-only-the-attacker).

ApproxAR2L is the one method still clearly short nominally (4–7 points). It is
also the only one with no reference implementation — the official repository
implements exact AR2L alone — and the suspect is the Eq. 18 operator at `γ = 1`,
not the action space.

---

## What is implemented

| file | contents |
|---|---|
| `config.yaml` | every default, in six sections (`env`, `train`, `ppo`, `model`, `run`, `eval`); `ar2l/config.py` reads it, the argparsers take their defaults from it, and `--config other.yaml` patches it |
| `ar2l/env.py` | batched online 3D-BPP simulator with the PCT state `(C_t, B_t, L_t)`: height map, exact empty-maximal-space enumeration, two orientations, stability, conveyor permutation |
| `ar2l/model.py` | the three transformers — packer, attacker, mixture model — and the pointer head of Eq. 28 |
| `ar2l/ppo.py` | PPO, plus the TV-dual value targets: Eq. 18 for ApproxAR2L and its pessimistic mirror for RfMDP |
| `ar2l/train.py` | the seven training loops (Algorithms 1 and 2 and the four baselines) |
| `ar2l/heuristics.py` | DBL, BMF, LSAH, OnlineBPH, HMM, MACS — the Table 1 baselines |
| `ar2l/evaluate.py` | held-out evaluation, mixture datasets, the `Uti./Std./Num.` metrics |
| `ar2l/viz/` | the five viewers (below) |
| `scripts/` | the job queue, the evaluation driver, `calibrate.py` (the stability-rule identification), the figures |
| `tests/` | the simulator against brute force, and the duals against their closed forms |

### The environment

The state is the PCT triple: the packed items `C_t` (position and size each),
the observable conveyor `B_t` (sizes only), and the feasible positions `L_t`
generated for whichever item is at the front. An action picks one `l ∈ L_t`.

`n_pick` splits `B_t` into what the cell can reach and what it can merely see.
With `--nb 11 --n_pick 5` the permuter is shown eleven items and may move any
of the first five to the front; the remaining six are upstream on the conveyor,
attended to by the encoder but masked out of the pointer, so they inform the
choice without being choosable. Taking a box slides exactly one preview item
into reach. `n_pick = 1` is the FIFO conveyor the paper assumes and `n_pick =
N_B` is free choice, so it is a single dial from one to the other — which makes
"utilisation against how many boxes the station can reach" a sweep rather than
an architecture change. The default, `null`, is free choice.

Everything is stepped as array operations over the whole batch of bins. Landing
heights come from running window maxima of the height map, and the support
count rides along in the same sweep: each window carries the pair **(max, how
many cells attain it)**, which is associative over "append one more cell" just
as the max alone is, so widening a window updates both together. A cell
carries the item exactly when its column reaches the footprint maximum, so
that count *is* the contact area — at any bin height and any item side. 64
bins step in 1.7 ms.

(It used to be a base-32 histogram, `32^h` per column, read at the landing
height. That is cheaper by one sweep but only representable while heights stay
below 12 and footprints below 25 cells, which made the area rules unavailable
on exactly the bins worth running.)

Candidate positions are the corners of the empty maximal spaces, which on a
grid are exactly the offsets where the item ends up flush against a packed face
or a wall — `ar2l/env.py: _ems_axis`.

### The three networks

All three are the paper's small transformer: independent element-wise
projections per node type into 64 dimensions, one attention block with one
head, and a pointer head

    π(·) = softmax(c · tanh(x̄ᵀx / √d))

over the candidate nodes — the feasible positions `L_t` for the packer, the
observable items `B_t` for the attacker and the mixture model. 65k parameters
each. Critics read `(C_t, B_t)` only, which is what Algorithm 2 needs in order
to value the same state under both the nominal and the permuted conveyor.

### The two AR2L algorithms

**Exact AR2L** (Algorithm 1) runs three stages per iteration: the attacker is
improved against the current packer; the mixture model `π_mix` is improved with
the loss of Eq. 6,

    L_mix = −η(π_pack, π_mix) + ( D_KL(π_mix ‖ 1{x=b_t+1,1}) + α·D_KL(π_mix ‖ π_perm) )

and the packer is improved on the instances `π_mix` permuted.

**Approximate AR2L** (Algorithm 2) collects a nominal rollout, re-permutes each
stored conveyor with the attacker, and builds the critic target from Eq. 18 —
the dual of the sup of `E[V]` over the mixture uncertainty set. The dual is a
two-dimensional convex problem in `μ₁, μ₂`; with one next-state sample per
state it is solved over the rollout batch, which is what Panaganti et al. do in
practice. `ρ = 0` provably collapses it to the `α`-weighted mean of the nominal
and attacked next-state values — that is a unit test.

---

## The one thing that had to be identified rather than read off

AR2L says item stability is "checked based on constraints used in (Zhao et al.,
2022a,b)", which is the conservative rule those papers write down: an item is
stable if more than 60% of its base rests on the layer below **and** all four
bottom corners are supported (or 80%/3 corners, or 95%).

Implemented verbatim, with this paper's items (sides 1–5, so very jagged piles),
**every** method caps out around 36% utilisation — roughly 25 points below every
single number AR2L reports, heuristics included. So that cannot be the rule that
was run.

The paper does report six fully-specified heuristics, and those pin the
simulator down. Sweeping candidate rules against them:

| stability rule | DBL | BMF | LSAH | ONLINEBPH | HMM | MACS | mean abs. gap |
|---|---|---|---|---|---|---|---|
| **paper** | 63.6 | 62.0 | 60.9 | 64.1 | 56.1 | 53.0 | — |
| 60% + 4 corners (as written) | 35.2 | 37.1 | 35.5 | 34.4 | 34.9 | 37.2 | 24.2 |
| 4 corners only | 33.6 | 35.9 | 34.1 | 32.6 | 33.6 | 35.1 | 25.8 |
| support area >= 60% | 50.9 | 49.8 | 49.1 | 51.1 | 49.0 | 53.9 | 9.6 |
| support area >= 40% | 58.5 | 50.9 | 54.3 | 57.3 | 56.0 | 60.9 | 6.3 |
| support area >= 20% | 61.1 | 47.0 | 54.8 | 61.1 | 59.2 | 63.8 | 6.8 |
| no stability check | 61.3 | 43.8 | 54.6 | 61.5 | 59.9 | 64.1 | 7.4 |
| **centre of mass over support** | 57.5 | 51.0 | 52.2 | 57.0 | 56.0 | 60.5 | 6.7 |

Two rules fit about equally well, and both are ~18 points better than the rule
as written. We use the last one: the physically standard criterion — the
projection of the item's centre of mass must fall inside the convex hull of its
contact cells, tested on a grid by requiring the support to span the centre
along each axis (`stability="com"`, the default). A bare 40% area threshold
scores a hair closer on utilisation (6.3 vs 6.7) but is a number chosen to fit;
the centre-of-mass rule is physics, and it is the one that reproduces HMM's
item count as well as its utilisation (22.4 vs the paper's 22.6).
`stability="cdrl"` gives the rule as literally written, for comparison.
Regenerate the table with `python3 scripts/calibrate.py`.

Both are checked cell-by-cell against a brute-force reference in
`tests/test_env.py` — 0 mismatches over 192,000 positions.

### The default is stricter than any of them: an 80% contact-area floor

The table above answers "what did AR2L run?". It is not the answer to "what
will a palletiser actually carry". The centre-of-mass rule is happy with a box
balanced on a third of its base, which is a real stack in the simulator and a
fallen one on a pallet. So the simulator's default adds a **contact-area
floor**: a placement is offered only if at least 80% of the item's base rests
on the layer it lands on (`env.min_support` in `config.yaml`, `min_support=`
on `BPPBatch`, `--min_support` on `train`/`evaluate`).

The floor is the binding half of the rule — at 80% the centre-of-mass span
cannot fail, since that needs roughly half the base hanging free — but both
are evaluated, so lowering `min_support` degrades gracefully back to the bare
centre-of-mass rule at `0.0`.

It is not free. Scored the way the table above is scored (`--n 64`, so noisier
than the rows above, which is why it is quoted against a same-seed rerun of
its own baseline rather than against them):

| rule | DBL | BMF | LSAH | ONLINEBPH | HMM | MACS |
|---|---|---|---|---|---|---|
| centre of mass over support | 63.1 | 56.4 | 59.7 | 62.2 | 62.2 | 67.3 |
| **centre of mass + 80% area** (default) | 42.1 | 47.4 | 46.5 | 42.9 | 41.8 | 47.7 |

Roughly 15 points of utilisation, because sides 1–5 make very jagged piles and
most of the ledges in them are now unusable. **Every number elsewhere in this
README predates the floor and was measured at `min_support=0.0`**; to compare
against the paper, or against any run in `runs/`, pass `--min_support 0`.
`scripts/calibrate.py` pins the floor per row for exactly that reason, so the
identification table does not move when the default does.

> **The table above was measured before rotation was implemented**, so it is a
> sweep of the stability rule at a fixed, one-orientation action space. The
> ranking it produced still stands — the rule as literally written is ~18 points
> off, and the centre-of-mass rule was later confirmed against the official code
> — but the absolute numbers in it are superseded by the ones below.

## What the action space is worth

The heuristics never train, so running them under each combination of

* `rot` — 1 orientation, or also the item yawed 90°
* `ems` — candidates are the bottom corners of the empty maximal spaces, or
  every loading position on the grid

separates what the *action space* is worth from what the *policy* is worth
(`python3 scripts/ablate_env.py`, 512 instances):

| heuristic | rot1 ems | **rot2 ems** (default) | rot1 all | rot2 all | paper |
|---|---|---|---|---|---|
| DBL | 56.1 | **62.3** | 57.8 | 63.1 | 63.6 |
| BMF | 50.4 | **57.3** | 50.5 | 56.9 | 62.0 |
| LSAH | 52.7 | **58.0** | 53.3 | 57.7 | 60.9 |
| ONLINEBPH | 55.0 | **62.8** | 56.6 | 63.5 | 64.1 |
| HMM | 56.3 | **62.6** | 56.0 | 62.0 | 56.1 |
| MACS | 59.5 | **67.4** | 61.3 | 67.2 | 53.0 |
| **mean** | 55.0 | **61.7** | 55.9 | 61.7 | 59.9 |
| **mean \|gap\|** | 7.2 | **5.2** | 6.8 | 4.9 | — |

**Rotation is worth +6.7 points** with EMS corners, +5.8 with every position.
Restricting to EMS corners is free once rotation is in (61.7 either way) and
costs 0.9 without it — while cutting the candidate list from 72 to 37, which is
why training iterations got *faster* despite the richer action space. The mean
absolute gap to the paper falls from 7.2 to 5.2.

An earlier version of this README read the heuristic gap as "our environment is
uniformly about 5 points harder". That was wrong in both directions. DBL, LSAH
and ONLINEBPH now land within 3 points of the paper; HMM, which used to match
exactly, overshoots by 6.5, and MACS by 14.4. What the heuristic column measures
is dominated by the action space, not by the physics — and BMF, ONLINEBPH and
MACS are reconstructions from citations rather than from the paper, so the
spread across them is not evidence about the simulator. The comparison that
carries the claim is each method against **our own** PCT baseline.

---

## The five viewers

```bash
python3 -m ar2l.viz.dashboard --port 8095     # live training curves
python3 -m ar2l.viz.game      --port 8096     # play the packer yourself
python3 -m ar2l.viz.report --packing pct_nb10 # one self-contained HTML page
python3 -m ar2l.viz.replay3d --policy run:pct_nb10 --gif
python3 -m ar2l.viz.heatmap  --policy run:pct_nb10 --attacker run:att_pct_nb10 --gif
python3 -m ar2l.viz.attack   --runs att_pct_nb10 att_pct_nb20
```

They share the look and the conventions of the viewers in `../pallet_rl_v1`
and `../pallet_rl_v2`: the same dark/light palette and card layout, the page
markup in a sibling `.html` file, a `QuietServer` that ignores the aborted
response a browser refresh causes, and `--port` / `--host` / `--root` / `--fps`
meaning the same things.

**`game`** — a human-vs-agent duel with the paper's twist. You pack a bin by
clicking loading positions; the legal cells the page draws are
`env.obs()["l_mask"]`, the very candidate list the policy scores, so both
players face one action space. Then pick an **attacker** and the same learned
adversary reorders the conveyor for both of you — which is the fastest way to
feel what AR2L is about. The same instance is handed to the opponent
(any trained run, or a heuristic) and the two bins are scored side by side.

The instance is built from **the opponent's own `args.json`**: choose a run and
the setup form fills itself with the bin, the item bounds, `nb`, `n_pick`,
`max_l`, `rot`, `ems` and the stability rule that run was trained under, so the
duel is fought on the geometry the agent actually knows. A flag you passed on
the command line still outranks the run — otherwise the preset would silently
undo it the moment an opponent was chosen, which is every time. Every field
stays editable. `POST /api/check` re-validates the whole set on each keystroke and
`POST /api/new` refuses one the simulator could not run — a reach larger than
the window, an item side wider than its axis, a floor of more cells than the
top view can redraw — so no accepted combination can raise out of
`BPPBatch.__init__`. Anything moved away from the run's own value is marked on
the field and named in a warning: the agent will still play, it was just never
asked this. `tests/test_game.py` pins the pair — every set the form refuses,
and that every set it accepts runs.

With `n_pick = k > 1` the conveyor is a **pick station** rather than a queue:
all `k` reachable boxes are drawn with the number of placements each one still
has, and you pack whichever you like — the job a `select`-trained permuter does
for the agent. The rest of the window is drawn dashed: seen, reasoned about,
not reachable. A pick is applied to the order the strip is showing rather than
to whatever the last pick left behind, so changing your mind is one permutation
of the window and not two composed ones, and the strip keeps a stable order to
click. The bin is finished when *every* reachable box is stuck, not when the
front one is. `k = 1` collapses to the FIFO conveyor of the paper, and the
checkbox that withholds the pick puts you back on the front box with the rest
of the reach visible.

Because the human now chooses, the agent has to choose too or the duel is not
one: replayed with no attacker, a `select` run gets **its own mixer** back as
the permuter rather than silently taking slot 0 every step, which is not the
policy that was trained. The result screen names who handed each side its box.

The result screen also **recalculates**: `POST /api/recalc` hands the session's
own item stream back to a model you pick, under parameters you edit, one table
row per setting — so the model, the bin, the window, the reach, the action
space and the stability rule can each be varied against a fixed instance rather
than against a fresh sample. The model is the row's other axis and comes from
the request rather than the session, so a row can race a run the game never
faced; the panel offers the same policies and heuristics the setup form does,
and **Fill in that run's own parameters** pulls the selected run's geometry into
the fields it is allowed to move. Choosing a model does *not* fill them by
itself: holding the parameters still while the model changes is the controlled
comparison, and it would be a strange thing to undo for you.
`n_items`, `size_lo` and `size_hi` are held at the game's own values there and
shown inert — they are what `sample_items` draws from, so moving one deals a
different sequence and the row would be measuring the new boxes rather than the
field that was changed. The server takes those three from the session and not
from the request, which is what makes "the same boxes" a guarantee rather than
a promise the page is trusted to keep. Only the agent is re-run: your bin
stands at the parameters you packed it under, so the rows are read against each
other. Each row names what it moved, and warns when the move has walked the
agent off its own training configuration — the same drift the setup form marks,
for the same reason.

**`dashboard`** — polls `runs/*/log.jsonl` while the grid runs: held-out
nominal utilisation, the utilisation of whatever dynamics each algorithm is
actually training on, the losses, and the mixture model's distance loss. The
paper's number for that `(method, N_B)` cell is drawn as a dashed target line.
Overlay runs by clicking their pills.

**`report`** — one self-contained HTML page: Tables 1 and 2 with the paper's
numbers beside ours, training curves as inline SVG, rendered bins as base64 PNG.

**`replay3d`** — the bin built one box at a time, as PNG frames or a GIF.

**`heatmap`** — the debugging view for a single decision: height map, landing
height per loading position, which positions survive the EMS and stability
filters, `π_pack` over them, and `π_perm` over the conveyor with the promoted
item flagged.

**`attack`** — what the attacker learned: the size distribution of the items it
promotes against the nominal distribution, how far down the conveyor it reaches,
and the damage per `N_B`. This is the measurement behind the paper's claim that
the attacker prefers smaller items as `N_B` grows.

---

## Non-cubic bins

`S=` takes an int for a cube or an `(Lx, Ly, Lz)` triple, and `--bin 60x50x80`
/ `--size_hi 6x4x5` parse on the command line (`--size_hi` may also be a plain
int). Node features are divided by the single scale `max(Lx, Ly, Lz)` rather
than per axis: per-axis normalisation would map the bin to a unit cube and
distort item shape, which is exactly what the packer has to reason about.

`tests/test_env.py` pins six deliberately lopsided bins — 7x5x9, 5x9x6,
11x4x7, 6x6x13, 9x12x5, 4x13x11 — checking every feasibility cell and landing
height against brute force, both EMS enumerations against each other, and all
six heuristics for legal placement. They are never cubes, and `Lz` sits both
above and below the footprint sides, because an x/y swap or a height/side
confusion is silent on a square bin. That caught two real bugs: the EMS wall
sentinel was `max(Lx, Ly) + 1` when columns can reach `Lz`, so on a tall thin
bin a tall column was mistaken for a wall (40 wrong feasibility cells on
7x5x9, no exception raised); and routing item sampling through per-axis bounds
made `rng.integers` consume its stream in a different order, silently
re-rolling every stored dataset. The isotropic draw is now kept bit-identical,
and `scripts/ablate_env.py` reproduces the 10^3 table digit-for-digit.

Non-cubic bins work under either stability rule: contact area is counted from
the same windowed (max, count) sweep that produces the landing height, so
there is no longer a bin-height or item-side cap on it.

`max_c` is an initial capacity, not a limit. It allocates `C_t`, the packer's
memory of the bin; a 10^3 bin never holds more than ~30 boxes, but a large bin
of small items holds hundreds. `step` used to clamp the write slot to
`max_c - 1` while `n_packed` ran on, so box 81 silently overwrote box 80 --
a wrong *state*, not just a wrong count -- and any caller that indexed
`packed[n_packed - 1]` raised `index 80 is out of bounds for axis 1 with size
80`. It now doubles the array instead and keeps the larger capacity. Nothing
changes for any bin that stays under the capacity, so the 10^3 numbers are
untouched.

The viewers take the same extent. `ar2l.viz.agents.play` accepts an int or a
triple and de-normalises boxes by `env.scale` (it used to multiply by `S`,
which raised `operands could not be broadcast together with shapes (6,) (3,)`
on a triple); it also defaults `size_hi` to the sequence's own per-axis maximum
instead of the cube default, and takes `max_l`, `rot` and `ems` so the replay
uses the leaf cap, the orientation count and the action space the policy was
trained with. The game reads all of those out of the run itself, so racing a
policy under its training configuration is the default rather than something
you have to spell out; `--bin 60x50x80 --size_hi 30x25x40 --max_l 256` now only
seeds the form before you pick an opponent.

The game page sizes its top view to fit beside the 3D bin instead of at a fixed
14px per cell, which drew a 120-wide bin 1680px across and pushed the 3D card
out of the row. Below 7px per cell the per-cell rules are dropped (a 1px rule
around a 3px cell is just grey) and below 13px the stacked-height digits are,
so on a large bin the floor is read by colour alone. Conveyor thumbnails are
capped by pixel extent rather than a per-cell floor, for the same reason.

A fit that keeps a 120-wide bin on screen puts it at 3px per cell, which is too
small to aim at, so a `cell size` slider overrides the fit and the top view
scrolls inside its card instead of widening its grid track. The rules and the
height digits reappear on their own as the cell passes 7px and 13px. The choice
persists in `localStorage`, `Fit` returns to the automatic size, and `&cell=`
sets it from the query string so a board can be linked at the size it was
looked at.

## Deviations from the paper

* **Stability rule** — identified from the paper's own heuristic baselines, as
  described above. This is the one substantive difference.
* **Empty maximal spaces are enumerated, not maintained.** The official code
  runs a Lai & Chan difference process with inscribed-EMS elimination. Because
  every item here is dropped, the reachable free volume is exactly the region
  above the height map, so an EMS is a footprint whose floor cannot be lowered
  by widening it. At `S = 10` there are only 55 × 55 footprints per bin, so
  `_ems_exhaustive` scores all of them with running window maxima and keeps the
  maximal ones. That is the *exact* same set, with none of the bookkeeping or
  ordering artefacts — and `tests/test_env.py::test_ems_list_is_exact` checks it
  against brute force.

  Scoring every footprint is `O(S⁴)` per bin, which is 6.2M rectangles at
  `S = 70` to find ~18 spaces. `_ems_pruned` is a second exact enumeration for
  large bins: a space needs `left > floor`, and `floor` is at least the height
  of column `x0` anywhere in the y-interval, so some `y` in that interval has
  `hmap[x0-1, y] > hmap[x0, y]` — a left edge can only sit where the height map
  steps down going right. The other three sides give the same condition, so only
  those `O(items)` rows and columns can bound a space: 2.9k candidate footprints
  at `S = 70` instead of 6.2M. `EMS_PRUNE_S = 25` picks whichever is cheaper
  (measured crossover; see the constant's comment). Both read only the height
  map, so neither can carry stale spaces across an episode boundary, and
  `test_pruned_ems_matches_the_exhaustive_scan` pins them to the same set over
  bins packed full at seven bin sizes.

  The pruning is an optimisation, not an approximation, and this was measured
  rather than assumed: forcing the pruned path at `S = 10` leaves **every
  observation array bit-identical** over 2000 steps × 64 envs for all six
  heuristics, and reproduces the held-out `Uti./Std./Num.` of six trained
  checkpoints — nominal and attacked — to `max|diff| = 0`. What it buys is
  speed, and only on large bins: a whole env step at `S = 70, n_env = 64` drops
  from **4693 ms to 255 ms (18×)**, or ~115 → ~2100 PPO iterations in six hours.
  At `S = 10` it is 6× *slower* than the batched sweep, which is why it is
  behind a size dispatch rather than a replacement.
* **Reward** — AR2L defines a single terminal reward equal to the final space
  utilisation with `γ = 1`. We give the item's volume fraction at each step,
  which sums to exactly the same return with far lower variance.
* **`D_KL(π_mix ‖ 1{x=b_t+1,1})`** in Eq. 6 is infinite for any `π_mix` that is
  not a point mass. We use the finite standard surrogate, the cross-entropy to
  that indicator, `−log π_mix(0)`.
* **`μ₁, μ₂, λ`** in Eq. 18 are per-state functions in the paper; we optimise
  them per rollout batch (Panaganti et al.'s practical choice) rather than
  fitting two more networks.
* **Where the robust value enters.** A robust Bellman operator only says what
  value the *next* state takes, so we substitute the dual value for
  `V(s_{t+1})` inside the GAE residual rather than using it as a one-step
  target. Building a one-step target instead costs RfMDP and ApproxAR2L about
  seven points of utilisation here, for reasons that have nothing to do with
  robustness — over a 30-step episode they would be the only two methods doing
  single-step credit assignment.
* **CPPO** — the paper does not give the CVaR level; we use the worst half of
  the trajectories (`--cvar_q 0.5`).
* **BMF / OnlineBPH / MACS** score functions are cited, not stated; ours are
  documented in `ar2l/heuristics.py`.
* **Not implemented**: the continuous setting and its Intersection-Points
  candidate generator, the Mixed-Item dataset of Table 3, the CDRL baseline of
  Table 1 (reproduced in `../pallet_rl_v1`), and the `ρ` sweep of Figure 3(c,d).
* **Training budget** is well short of the paper's (see RESULTS).

## Install and run

See [RUN.md](RUN.md).

## Status and what is left

[TODO.md](TODO.md) tracks it: a read of the official implementation
(<https://github.com/panyxy/ar2l_bpp>, published after this reproduction) against
ours, where our numbers stand against the paper cell by cell, and the ordered
list of what would close the gap.
