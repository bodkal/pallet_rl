"""Actor-critic network with the independent mask predictor (paper Fig. 9)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _init(m, gain=np.sqrt(2)):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.constant_(m.bias, 0.0)
    return m


class PackNet(nn.Module):
    """state CNN -> {actor, critic}, plus an independent MLP mask predictor.

    The mask predictor consumes the same state-CNN features but is a separate
    head trained with the ground-truth feasibility mask as supervision
    (paper Sec. 3.1, "Feasibility constraints").
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        c, co, hid = cfg.cnn_channels, cfg.cnn_out_channels, cfg.hidden
        A = cfg.action_dim
        self.feat_dim = co * cfg.L * cfg.W          # 4*10*10 = 400 in the paper

        self.cnn = nn.Sequential(
            _init(nn.Conv2d(cfg.obs_channels, c, 3, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(c, c, 3, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(c, co, 1)), nn.ReLU(),
            nn.Flatten(),
        )
        self.trunk = nn.Sequential(_init(nn.Linear(self.feat_dim, hid)), nn.ReLU())
        self.actor = _init(nn.Linear(hid, A), gain=0.01)
        self.critic = _init(nn.Linear(hid, 1), gain=1.0)
        self.mask_head = nn.Sequential(
            _init(nn.Linear(self.feat_dim, hid)), nn.ReLU(),
            _init(nn.Linear(hid, A), gain=0.01),
        )

    def forward(self, x):
        f = self.cnn(x)
        h = self.trunk(f)
        return self.actor(h), self.critic(h).squeeze(-1), self.mask_head(f)

    # -- prediction and projection ----------------------------------------
    def projected_logits(self, logits, mask):
        """Multiply infeasible action probabilities by eps, in log space.

        Paper: "if the LP at (x, y) is infeasible ... P should be set to 0.
        However, we find that setting P to a small positive quantity like
        eps = 1e-3 works better in practice."
        """
        cfg = self.cfg
        if not cfg.use_mask_constraint:
            return logits
        return logits + torch.log(
            torch.where(mask > 0.5,
                        torch.ones_like(mask),
                        torch.full_like(mask, cfg.mask_eps)))

    @torch.no_grad()
    def act(self, x, true_mask=None, deterministic=False):
        """Sample an action.  Returns action, logprob, value, and the mask used."""
        cfg = self.cfg
        logits, value, mask_logits = self(x)
        pred_mask = (torch.sigmoid(mask_logits) > 0.5).float()
        used = true_mask if (cfg.use_true_mask_for_policy and true_mask is not None) \
            else pred_mask
        if not cfg.use_mask_prediction and true_mask is not None:
            used = true_mask                      # MP off -> fall back to oracle
        plogits = self.projected_logits(logits, used)
        dist = torch.distributions.Categorical(logits=plogits)
        a = plogits.argmax(-1) if deterministic else dist.sample()
        return a, dist.log_prob(a), value, pred_mask

    def evaluate(self, x, actions, true_mask):
        """Re-evaluate a batch for the PPO update (returns everything the loss needs)."""
        cfg = self.cfg
        logits, value, mask_logits = self(x)
        pred_mask = (torch.sigmoid(mask_logits) > 0.5).float().detach()
        used = true_mask if cfg.use_true_mask_for_policy or not cfg.use_mask_prediction \
            else pred_mask
        plogits = self.projected_logits(logits, used)
        logp_all = F.log_softmax(plogits, dim=-1)
        p_all = logp_all.exp()
        logp = logp_all.gather(1, actions[:, None]).squeeze(1)

        # E_inf : summed probability mass on genuinely infeasible LPs (paper Eq. 2)
        e_inf = (p_all * (1.0 - true_mask)).sum(-1)

        # feasibility-restricted entropy: only over LPs with M(x, y) = 1
        if cfg.use_feasibility_entropy:
            ent = -(p_all * logp_all * true_mask).sum(-1)
        else:
            ent = -(p_all * logp_all).sum(-1)

        return logp, value, ent, e_inf, torch.sigmoid(mask_logits)
