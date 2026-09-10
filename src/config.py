"""Configuration for the online 3D-BPP constrained-DRL reproduction.

Paper: "Online 3D Bin Packing with Constrained Deep Reinforcement Learning",
Zhao, She, Zhu, Yang, Xu (AAAI 2021).  arXiv:2006.14978

NEXT: markers below record what to try next and why, from measurements taken
2026-09-08/09 on a 20^3 bin.  Where we stand: 100M steps gives 41.4% on CUT-2
against the paper's Size-20 figure of 65.4%, the learning curve is still
log-linear (so extrapolating says billions of steps -- the budget is not the
limit), and 49-72% of episodes end on an ILLEGAL PLACEMENT rather than on a
full bin.  Flags marked DEVIATES make the result incomparable to the paper.

The 2026-09-08 lever sweep (7 runs, 10M steps each, 20^3 CUT-2, paper-sized
net, held out on 500 CUT-2 episodes) priced that illegal-move rate, and the
answer is that it is worth about a THIRD of the gap, not the whole of it:

  perfect mask, given to the trained control at test time      +4.2 pp
  perfect mask, trained under as well                          +7.4 pp more
  ------------------------------------------------------------ ----------
  all mask error                                              ~+11.6 pp
  remaining gap to the paper's 65.4%                            ~24 pp

So the mask levers are worth taking (--w-mask 5.0, +2.1 pp, faithful) and are
NOT where the reproduction is hiding.  The remaining ~24 pp is packing quality,
which points at the optimiser (ACKTR, TODO.md C.1) and the network, not at the
constrained-DRL scheme.  Read the note on use_true_mask_for_policy before
using it: a net trained that way is WORSE than the control when deployed.
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
    L: int = 20                      # bin length  (X)
    W: int = 20                      # bin width   (Y)
    H: int = 20                      # bin height  (Z)
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
    invalid_action_mode: str = "resample"   # terminate | resample
    # WARNING: "resample" is set. src.evaluate rebuilds its env from this same
    # config.json, so the EVALUATION will also rescue an illegal choice and its
    # utilisation will be inflated and not comparable to the paper. Pass
    # `--invalid-action-mode terminate` when evaluating (src.evaluate warns if
    # you forget).

    # Pre-generate this many training sequences once and cycle them, instead of
    # running the CUT cutting-stock recursion on every episode reset (measured
    # 5.7% of a PPO iteration at convergence, more early on when episodes are
    # short). 0 = off, generate a fresh sequence per reset (the paper's setting
    # and the default, since a finite pool is seen many times over a long run).
    # NEXT: set 40000. The only throughput knob measured free -- +11% steps/s
    # for -0.01 pp utilisation at matched steps (docs/ab_throughput.png). Keep it
    # large; a small pool is reused thousands of times over a long run.
    seq_pool: int = 40000

    # ---- constrained-DRL scheme (paper Sec. 3.1/3.2) ------------------------
    use_mask_prediction: bool = True     # MP  - train the mask predictor
    use_mask_constraint: bool = True     # MC  - project action probs with mask
    use_feasibility_entropy: bool = True  # FE - entropy over feasible actions only
    # MEASURED 2026-09-09: still low priority ON ITS OWN, for the reason below
    # -- most illegal moves come from false-feasible PREDICTIONS, which eps
    # cannot touch. But it is NOT negligible once the mask is exact: scored with
    # the ground-truth mask, 3.0% of held-out episodes STILL ended on an illegal
    # placement, and with an exact mask the eps leak is the only thing that can
    # cause that. So eps=0 belongs with use_true_mask_for_policy (run
    # `true_eps0`) and nowhere else. eps=0 is now supported: it takes the hard
    # -inf-equivalent path in PackNet.projected_logits.
    mask_eps: float = 1e-3               # infeasible actions get prob * eps
    # MEASURED 2026-09-08/09 (run `mask_true`, 10M steps, 20^3 CUT-2). It did
    # its job as a ceiling and produced one trap. Held out, 500 CUT-2 episodes:
    #
    #   control (kl_base), own predictor              29.91%
    #   control, handed the TRUE mask at test time    34.13%   +4.2 pp
    #   trained AND tested with the true mask         41.54%   +7.4 pp more
    #   trained with the true mask, own predictor     23.57%   -6.3 pp !!
    #
    # So mask error is worth about +11.6 pp in total -- roughly a third of the
    # 35 pp gap from the control to the paper's Size-20 65.4%. The other ~24 pp
    # is genuinely bad packing, and no mask lever will find it.
    #
    # THE TRAP: a net trained this way is WORSE THAN THE CONTROL when deployed
    # on its own predictor -- 23.57% against 29.91%, with 79.6% of episodes
    # dying on an illegal placement against the control's 48.6%. It never had to
    # hedge against a predictor mistake, so it does not. A common-state probe
    # says the policy is most of that, not the predictor: on kl_base's own
    # states mask_true's false-feasible rate is 6.9% against 4.1%, which is
    # worse but nowhere near enough to explain 6.3 pp.
    #
    # Therefore: DIAGNOSTIC ONLY. Never train a net you intend to deploy with
    # this, and never quote the TRAINING utilisation of such a run -- it is
    # measured with the mask in the loop and read 36.7% against a real 23.6%.
    # (Unlike invalid_action_mode="resample", this one IS physically deployable
    # -- the mask is a known function of the observed height map -- so 41.54% is
    # the number that matters for the UR20 cell. It is still not a reproduction.)
    use_true_mask_for_policy: bool = False  # if True, project with ground truth

    # ---- loss weights (paper Eq. 1) ----------------------------------------
    w_actor: float = 1.0        # alpha
    w_critic: float = 0.5       # beta
    # MEASURED 2026-09-08/09: 5.0 is the best DEPLOYABLE lever found so far, and
    # it stays faithful to the paper -- it only re-weights the existing loss.
    # Held out on CUT-2, 500 episodes: 0.5 -> 29.91%, 2.0 -> 31.03%,
    # 5.0 -> 31.98% (+2.1 pp), and episodes ending illegal fall 48.6% -> 42.4%.
    # A common-state probe also makes mask_w5's predictor the best of the three
    # on every state distribution tested (false-feasible 2.6-4.1%).
    #
    # But the gain is much smaller than the mask-head improvement suggests, and
    # that is the useful finding: w_mask=5 HALVED the training false-feasible
    # rate (1.98% -> 1.01%) and moved invalid_rate only 4.51% -> 4.00%. Per-cell
    # mask accuracy has stopped being the binding constraint -- the policy
    # concentrates on whatever false-feasible cells survive. Take 5.0 and stop
    # tuning it; the remaining ~24 pp is packing quality, not mask quality.
    #
    # Left at the paper's 0.5 deliberately: Eq. 1 specifies 0.5, and this repo's
    # headline numbers are reproduction numbers. Pass --w-mask 5.0 for the best
    # net, and say so when reporting it.
    w_mask: float = 0.5         # lambda (paper's value; --w-mask 5.0 scores +2.1 pp)
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
    # MEASURED 2026-09-09, the four-cell grid at 10M steps each on 20^3 CUT-2,
    # BPP-1 on 500 held-out episodes. The default is right, and DEPTH is why:
    #
    #   2/64/256   control     1,065,125 par   29.91%     --
    #   4/64/256   depth only  1,138,981 par   33.04%   +3.12 pp   (+7% params!)
    #   2/64/1024  width only  4,139,429 par   30.44%   +0.52 pp   (+289% params)
    #   4/128/1024 default     4,547,877 par   36.94%   +7.03 pp
    #
    # Depth is nearly free and width alone is nearly worthless. The mechanism is
    # the receptive field: 4 layers sees 9x9, i.e. 20% of a 20x20 height map,
    # against 5x5's 6%. The default adds another +3.9 pp over depth alone, but it
    # moves conv channels AND trunk width together, so that increment belongs to
    # neither -- a 4/128/256 cell would separate them, and is the only cell of
    # this grid still missing.
    #
    # It is also the biggest faithful lever found anywhere in the sweep, ahead of
    # w_mask (+2.1) and target_kl (+1.2). Sample efficiency: 36.94% at 10M steps
    # against the paper-sized net's 39.49% at 100M -- ~10x for under 2x the
    # per-step cost. CAVEAT: that ratio only holds with an ANNEALED lr; see the
    # lr note below, which is now the most important open item in this file.

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
    # MEASURED 2026-09-10 (run `deep_lr1e4`): KEEP 3e-4. Lowering the base lr to
    # 1e-4 on the default 4.55M-param net scored 35.36% held out against 3e-4's
    # 36.94% -- it is 1.59 pp WORSE, not better.
    #
    # This corrects a wrong reading recorded here on 09-09. Comparing the deep
    # net on a 100M horizon against the same net on a 10M horizon showed the 10M
    # one ahead by +3.98 -> +8.59 pp at 5.8M -> 9.0M, and that was mis-read as
    # "3e-4 is too high". What it actually shows is that the deep net's advantage
    # only materialises as the lr ANNEALS -- a 100M-horizon run is still near
    # 2.8e-4 at 9M, i.e. mid-flight, not crippled. A lower STARTING point does
    # not substitute for the anneal; it just learns more slowly the whole way.
    #
    # Consequences, both of which contradict the earlier note:
    #   - bpp1_cut2_20's 39.49% at a fully-annealed 100M was a fair number. The
    #     100M runs were never "LR-crippled".
    #   - The 100M deep run should use lr 3e-4, NOT 1e-4.
    # Comparing two architectures at matched steps is only meaningful when both
    # are at the same POINT IN THEIR ANNEAL; mid-flight comparisons across
    # different horizons mislead, which is what happened here.
    #
    # lr_schedule="linear" anneals to a 5% floor AT total_steps, so every run
    # flattens at its horizon by construction.
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
    # The reasoning still holds: PPO's clip_range bounds the per-action
    # probability RATIO and knows nothing about how many actions there are, so
    # the same parameter step moves a 400-action distribution (20^3) much
    # further in KL than a 100-action one (10^3). Measured on 20^3 CUT-2:
    # approx_kl 0.174 at 1M with peaks of 2.92, and clipfrac 0.258 rising to
    # 0.60 -- against the usual targets of ~0.01-0.02 and 0.1-0.2.
    #
    # But 0.02 WAS TRIED (runs `kl_tkl`, `kl_both`) and the result was worthless
    # for two reasons, both now fixed in ppo.py:
    #   1. 0.02 is an order of magnitude below the operating point, so it
    #      tripped on 99-100% of updates after a mean of ~3 of 32 minibatches.
    #   2. The trip did `break` on the WHOLE update, which also stopped the
    #      critic and the mask head -- supervised objectives with no trust
    #      region to respect. Those runs therefore took ~10% of the intended
    #      gradient steps, and mask_fpr went 2.0% -> 4.8%.
    # kl_tkl scored 22.87% held out against the control's 29.91%: that measures
    # "10x fewer gradient steps", not a trust region. ppo.py now freezes only
    # the actor (plus E_inf and the entropy bonus) and keeps training the heads,
    # and `kl_stop` is logged as the fraction of minibatches frozen rather than
    # the old unreadable 1/nb.
    #
    # NEXT: 0.2, which is where the measured approx_kl actually lives (run
    # `tkl_fixed`). At 0.02 the fixed code still freezes the actor on 89% of
    # minibatches, which is a learning-rate cut in disguise.
    # Still the cheap first-order stand-in for what ACKTR does properly; see the
    # NEXT note on the optimiser below.
    target_kl: float = 0.0
    max_grad_norm: float = 0.5
    clip_value_loss: bool = True
    # NEXT (the biggest open question): the paper uses ACKTR, we use PPO, and
    # its own Table 10 shows the optimiser is worth up to ~14 pp on CUT-2 at
    # 10^3 -- ACKTR 66.9%, RAINBOW 58.8%, A2C 53.0%, SAC 44.2%, DQN 35.3%,
    # all "with well-tuned parameters". PPO is not among them.
    #
    # This is NOT simply "PPO is worse": our PPO matched ACKTR at 10^3 (67.1%
    # vs 66.9%, far above A2C). The suspicion is that it degrades with the
    # action space. ACKTR's trust region is defined in KL and preconditioned by
    # the Fisher matrix, so it shrinks its step automatically as the action
    # distribution grows; PPO's ratio clip does not. That predicts ACKTR's edge
    # WIDENS with resolution, which matches what we see: the paper loses 1.5 pp
    # going 10^3 -> 20^3 (66.9 -> 65.4) while we lose 26 pp (67.1 -> 41.4).
    #
    # Order of attack, cheapest first: target_kl above, then a sweep of lr,
    # then implement ACKTR/K-FAC (TODO.md C.1).

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
