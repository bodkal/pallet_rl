"""Configuration for the online 3D-BPP constrained-DRL reproduction.

Paper: "Online 3D Bin Packing with Constrained Deep Reinforcement Learning",
Zhao, She, Zhu, Yang, Xu (AAAI 2021).  arXiv:2006.14978

NEXT: markers below record what to try next and why, from measurements taken
2026-09-08 on a 20^3 bin.  Where we stand: 100M steps gives 41.4% on CUT-2
against the paper's Size-20 figure of 65.4%, the learning curve is still
log-linear (so extrapolating says billions of steps -- the budget is not the
limit), and 55-69% of episodes end on an ILLEGAL PLACEMENT rather than on a
full bin.  That last number is the dominant loss and most NEXT notes point at
it.  Flags marked DEVIATES make the result incomparable to the paper.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields


@dataclass
class Config:
    # ---- environment (paper Sec. 4 "Training and test set") -----------------
    # NEXT: 20 is the only resolution the paper reports (Fig. 14: RS 0.581,
    # CUT-1 0.634, CUT-2 0.654), so 20 is the size to compare on. 10 reproduces
    # the paper's headline tables. 15 compares to nothing published.
    L: int = 15                      # bin length  (X)
    W: int = 15                      # bin width   (Y)
    H: int = 15                      # bin height  (Z)
    item_min: int = 2                # item dims are drawn from {2,3,4,5}
    item_max: int = 5                # -> |I| = 4^3 = 64 pre-defined item types
                                     # build() derives this as min(L,W,H)//2 when
                                     # neither the preset nor the CLI pins it
    dataset: str = "CUT-2"           # RS | CUT-1 | CUT-2
    # NEXT: keep 1. The paper's Fig. 14 resolution study is 1-pose, so 2 makes
    # the comparison invalid. Measured at 10^3: 2 poses gained +8.3 pp on RS but
    # LOST 3.5 pp on CUT-1 (see TODO.md B).
    orientations: int = 1            # 1 = paper's main setting, 2 = re-orienting

    # NEXT (DEVIATES): "resample" is worth ONE diagnostic run. It replaces an
    # illegal choice with a random feasible LP instead of ending the episode, so
    # the 55-69% of episodes currently dying at ~9.6 items would run to a full
    # bin -- far more learning signal per rollout. But it rescues the agent from
    # its own mistakes, so train with it and always EVALUATE with "terminate".
    invalid_action_mode: str = "terminate"   # terminate | resample

    # Pre-generate this many training sequences once and cycle them, instead of
    # running the CUT cutting-stock recursion on every episode reset (measured
    # 5.7% of a PPO iteration at convergence, more early on when episodes are
    # short). 0 = off, generate a fresh sequence per reset (the paper's setting
    # and the default, since a finite pool is seen many times over a long run).
    # NEXT: set 40000. The only throughput knob measured free -- +11% steps/s
    # for -0.01 pp utilisation at matched steps (docs/ab_throughput.png). Keep it
    # large; a small pool is reused thousands of times over a long run.
    seq_pool: int = 0

    # ---- constrained-DRL scheme (paper Sec. 3.1/3.2) ------------------------
    use_mask_prediction: bool = True     # MP  - train the mask predictor
    use_mask_constraint: bool = True     # MC  - project action probs with mask
    use_feasibility_entropy: bool = True  # FE - entropy over feasible actions only
    # NEXT: low priority. Lowering to 1e-5 only shrinks the probability that
    # LEAKS onto LPs the predictor already knows are infeasible. Most illegal
    # moves come from false-feasible PREDICTIONS instead (invalid_rate 5.2% vs
    # mask_fpr 1.9% at 6M), which eps cannot touch -- fix w_mask first.
    mask_eps: float = 1e-3               # infeasible actions get prob * eps
    # NEXT (DEVIATES): the biggest single lever, and worth one run purely as a
    # CEILING measurement. The true mask is computable from the height map and
    # current item, both observed, so projecting with it removes essentially
    # every illegal move (only the mask_eps residual survives) and episodes run
    # to a full bin. The paper deliberately projects with the PREDICTED mask, so
    # a number from this is not a reproduction -- but it splits the 24 pp gap
    # into "mask error" and "bad packing", which nothing else does.
    use_true_mask_for_policy: bool = False  # if True, project with ground truth

    # ---- loss weights (paper Eq. 1) ----------------------------------------
    w_actor: float = 1.0        # alpha
    w_critic: float = 0.5       # beta
    # NEXT: try 2.0 -- the best faithful lever. It only re-weights the existing
    # loss, so the result stays comparable to the paper. The mask head is what
    # is failing at 20^3: false-feasible rate 1.9% against ~0.2% at 10^3, and
    # with 400 actions that is ~3 wrong cells per state, which over a ~15-step
    # episode is what kills half of them. The deep net's whole +1.5 pp came
    # through a better mask (fpr 2.36% -> 1.90%), not better packing.
    w_mask: float = 0.5         # lambda
    w_einf: float = 0.01        # omega
    w_entropy: float = 0.01     # psi

    # ---- network ------------------------------------------------------------
    # NOT the paper's Figure 9 stack any more.  The paper sizes its network for
    # a 10x10x10 bin (cnn_channels=64, cnn_layers=2, hidden=256 -> 1.07M params,
    # 5x5 receptive field); at 20^3 that has to drive 4x the actions through the
    # same 256-wide trunk, and 5x5 sees 6% of a 20x20 height map against 25% of
    # a 10x10 one.  Defaults below are the wider/deeper net.  For the paper's
    # own architecture pass --cnn-layers 2 --cnn-channels 64 --hidden 256.
    cnn_channels: int = 128
    cnn_layers: int = 4          # 3x3 conv layers before the 1x1 bottleneck;
                                 # each adds 2 cells of receptive field
                                 # (k layers -> 1+2k), so 4 -> 9x9
    cnn_out_channels: int = 4   # -> 4*L*W features, as in paper Fig. 9
    hidden: int = 1024
    # NEXT: unresolved, and expensive. 4/128/1024 beats the paper's 2/64/256 by
    # ~1.5 pp at matched steps (20^3 CUT-2) but costs 3.8x the compute per step,
    # so per GPU-HOUR it is currently behind. Nobody has separated depth from
    # width: run 2/64/256, 4/64/256 (depth only), 2/64/1024 (width only) and
    # 4/128/1024 at ~20M steps each and compare SLOPES. If depth alone carries
    # it, the cheap 1.14M net wins.

    # ---- PPO ----------------------------------------------------------------
    # NEXT: leave at 32. Raising it is a trap: 128 measured +71% steps/s for
    # -4.40 pp utilisation at matched steps, because the batch grows 1,280 ->
    # 5,120 and takes 4x fewer gradient steps per sample.
    num_envs: int = 32
    # NEXT: 2 if anything, and do not bother at all when running several jobs
    # at once. Measured on the deep net at 20^3: serial 1,268 fps, 2 workers
    # 1,436 (+13%, the peak), 4 workers 1,389, 8 workers 1,288 -- and with 3
    # concurrent runs the aggregate is identical at 0/2/4 workers (1,596-1,607
    # fps). The deep net moved the bottleneck onto the GPU, so there is little
    # env loop left to parallelise. It was worth 1.79x on the paper-sized net.
    env_workers: int = 0         # 0/1 = the serial env loop (one core, the
                                 # default); >1 spreads it over that many
                                 # processes -- identical gradients, just not
                                 # pinned to one core
    num_steps: int = 40          # rollout length per env (episodes are ~20 long)
    # NEXT: 100M is not enough at 20^3, but more steps is NOT the fix. Fitting
    # u = a + b*log2(steps) over the last 80% of the 100M runs and solving for
    # the paper's Size-20 numbers gives 4.5B (CUT-1), 9.6B (CUT-2) and 362B (RS)
    # steps -- 17 days to 3.7 years. Gain per doubling is flat or rising
    # (CUT-2: 2.75, 3.21, 3.48, 4.94 pp), i.e. no saturation at all, which says
    # the ceiling is the setup. Fix the illegal-move rate before buying steps.
    total_steps: int = 20_000_000
    max_hours: float = 16.0
    # NEXT: untested for the current network. 3e-4 was the paper's value for a
    # 1.07M-param net; the default here is 4.55M with 128-channel convs and
    # nobody has swept it. Note lr_schedule="linear" anneals to a 5% floor AT
    # total_steps, so every run flattens at its horizon by construction -- the
    # 15^3 runs all ended at exactly 1.50e-05 while still improving.
    lr: float = 3e-4
    lr_schedule: str = "linear"  # linear | constant
    gamma: float = 1.0           # paper sets gamma = 1
    gae_lambda: float = 0.96
    clip_range: float = 0.2
    # NEXT: leave both alone. 2 epochs x 4 minibatches measured +109% steps/s
    # for -14.69 pp utilisation at matched steps -- by far the worst trade of
    # any knob tested. Fewer gradient steps per sample is the wrong direction
    # for a problem that is already learning too slowly per sample.
    epochs: int = 4
    minibatches: int = 8
    max_grad_norm: float = 0.5
    clip_value_loss: bool = True

    # ---- MCTS / BPP-k (paper Sec. 3.3, Algorithm 1) -------------------------
    # NEXT: free upside on the REPORTED number, no retraining -- BPP-k reuses
    # the BPP-1 network and only searches harder at test time. Measured +7 to
    # +10 pp at 10^3 going BPP-1 -> BPP-5. The finished 20^3 runs have never
    # been evaluated this way: `src.evaluate --bppk 3 5`. Costs test-time
    # compute only (~3.8 s/episode at k=5 against 0.07 s for BPP-1).
    lookahead_k: int = 1
    mcts_simulations: int = 100
    mcts_c: float = 1.0
    mcts_last_item: str = "observed"   # observed | mean  (value of the leaf item)

    # ---- run bookkeeping -----------------------------------------------------
    seed: int = 0
    run_name: str = "bpp1_cut2"
    log_interval: int = 1
    save_interval: int = 50
    eval_interval: int = 100
    eval_episodes: int = 64
    device: str = "cuda"

    # ---- derived -------------------------------------------------------------
    @property
    def n_positions(self) -> int:
        return self.L * self.W

    @property
    def action_dim(self) -> int:
        return self.orientations * self.L * self.W

    @property
    def bin_volume(self) -> int:
        return self.L * self.W * self.H

    @property
    def obs_channels(self) -> int:
        return 4  # height map + 3 stretched item-dimension channels

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path) -> "Config":
        with open(path) as f:
            d = json.load(f)
        # A config written before a field existed has to keep the value that
        # field effectively had, NOT today's default -- otherwise the network we
        # rebuild stops matching the checkpoint saved next to it. cnn_layers was
        # added after the 10^3/15^3/20^3 runs were trained, and every checkpoint
        # written without it has the paper's 2 conv layers.
        d.setdefault("cnn_layers", 2)
        known = {f.name for f in fields(Config)}
        return Config(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# named presets
# ---------------------------------------------------------------------------
PRESETS = {
    # the paper's setting: 10x10x10, no re-orientation, CUT-2
    "paper": dict(),
    # fast end-to-end validation of the whole pipeline (~20-40 min)
    "smoke": dict(L=6, W=6, H=6, item_min=1, item_max=3,
                  num_envs=16, num_steps=32, total_steps=300_000,
                  max_hours=1.0, save_interval=20, eval_interval=25,
                  run_name="smoke"),
    # paper's re-orientation extension (Table 4)
    "orient": dict(orientations=2, run_name="bpp1_orient"),
    # Same algorithm as "paper", tuned for throughput on one GPU: measured
    # 7,257 vs 4,385 steps/s (1.65x, 6.3 h -> 3.8 h per 100M steps) on an
    # RTX 4070 Laptop. The env loop is serial and one core, so a bigger
    # num_envs is what amortises the per-step GPU round-trip and the update.
    # NOT a drop-in for a reproduction run: batch goes 1,280 -> 5,120, i.e.
    # 4x fewer gradient steps per sample. A/B it on utilisation-vs-step
    # before trusting it, and keep "paper" for the headline numbers.
    "fast": dict(num_envs=128, seq_pool=40_000, run_name="bpp1_fast"),
}


def build(preset: str = "paper", **overrides) -> Config:
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; have {sorted(PRESETS)}")
    kw = dict(PRESETS[preset])
    kw.update({k: v for k, v in overrides.items() if v is not None})
    # The paper constrains items to l <= L/2, w <= W/2, h <= H/2 "to avoid
    # over-simplified scenarios" (Sec. 4 and supplemental).  At its own 10^3 bin
    # that is exactly item_max=5, so deriving it changes nothing for the paper
    # preset -- it only matters once the bin size moves: a 15^3 bin gets items
    # up to 7 ({2..7}^3 = 216 types) rather than silently keeping the 10^3 range,
    # which would shrink items relative to the bin and change the problem.
    if "item_max" not in kw:
        n = min(kw.get("L", Config.L), kw.get("W", Config.W), kw.get("H", Config.H))
        kw["item_max"] = n // 2
    return Config(**kw)
