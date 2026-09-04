"""Configuration for the online 3D-BPP constrained-DRL reproduction.

Paper: "Online 3D Bin Packing with Constrained Deep Reinforcement Learning",
Zhao, She, Zhu, Yang, Xu (AAAI 2021).  arXiv:2006.14978
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # ---- environment (paper Sec. 4 "Training and test set") -----------------
    L: int = 10                      # bin length  (X)
    W: int = 10                      # bin width   (Y)
    H: int = 10                      # bin height  (Z)
    item_min: int = 2                # item dims are drawn from {2,3,4,5}
    item_max: int = 5                # -> |I| = 4^3 = 64 pre-defined item types
    dataset: str = "CUT-2"           # RS | CUT-1 | CUT-2
    orientations: int = 1            # 1 = paper's main setting, 2 = re-orienting

    # what happens when the agent picks a truly infeasible LP
    invalid_action_mode: str = "terminate"   # terminate | resample

    # ---- constrained-DRL scheme (paper Sec. 3.1/3.2) ------------------------
    use_mask_prediction: bool = True     # MP  - train the mask predictor
    use_mask_constraint: bool = True     # MC  - project action probs with mask
    use_feasibility_entropy: bool = True  # FE - entropy over feasible actions only
    mask_eps: float = 1e-3               # infeasible actions get prob * eps
    use_true_mask_for_policy: bool = False  # if True, project with ground truth

    # ---- loss weights (paper Eq. 1) ----------------------------------------
    w_actor: float = 1.0        # alpha
    w_critic: float = 0.5       # beta
    w_mask: float = 0.5         # lambda
    w_einf: float = 0.01        # omega
    w_entropy: float = 0.01     # psi

    # ---- network ------------------------------------------------------------
    cnn_channels: int = 64
    cnn_out_channels: int = 4   # -> 4*L*W = 400 features, matches paper Fig. 9
    hidden: int = 256

    # ---- PPO ----------------------------------------------------------------
    num_envs: int = 32
    num_steps: int = 40          # rollout length per env (episodes are ~20 long)
    total_steps: int = 20_000_000
    max_hours: float = 16.0
    lr: float = 3e-4
    lr_schedule: str = "linear"  # linear | constant
    gamma: float = 1.0           # paper sets gamma = 1
    gae_lambda: float = 0.96
    clip_range: float = 0.2
    epochs: int = 4
    minibatches: int = 8
    max_grad_norm: float = 0.5
    clip_value_loss: bool = True

    # ---- MCTS / BPP-k (paper Sec. 3.3, Algorithm 1) -------------------------
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
            return Config(**json.load(f))


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
}


def build(preset: str = "paper", **overrides) -> Config:
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; have {sorted(PRESETS)}")
    kw = dict(PRESETS[preset])
    kw.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**kw)
