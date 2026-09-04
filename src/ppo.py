"""PPO with the paper's constrained-DRL loss (Eq. 1).

    L = alpha*L_actor + beta*L_critic + lambda*L_mask + omega*E_inf - psi*E_entropy

The paper uses ACKTR; we swap the optimiser for PPO but keep every term of the
constrained scheme (mask prediction, mask projection, infeasibility penalty and
the feasibility-restricted entropy) intact.  See TODO.txt item 1 for ACKTR.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .env import build_obs_tensor


class RolloutBuffer:
    def __init__(self, cfg, device):
        T, N, A = cfg.num_steps, cfg.num_envs, cfg.action_dim
        self.obs = torch.zeros(T, N, cfg.obs_channels, cfg.L, cfg.W, device=device)
        self.mask = torch.zeros(T, N, A, device=device)
        self.act = torch.zeros(T, N, dtype=torch.long, device=device)
        self.logp = torch.zeros(T, N, device=device)
        self.val = torch.zeros(T, N, device=device)
        self.rew = torch.zeros(T, N, device=device)
        self.done = torch.zeros(T, N, device=device)
        self.cfg, self.device = cfg, device

    def compute_gae(self, last_val, last_done):
        cfg = self.cfg
        adv = torch.zeros_like(self.rew)
        gae = torch.zeros(cfg.num_envs, device=self.device)
        for t in reversed(range(cfg.num_steps)):
            if t == cfg.num_steps - 1:
                nextnonterm, nextval = 1.0 - last_done, last_val
            else:
                nextnonterm, nextval = 1.0 - self.done[t + 1], self.val[t + 1]
            delta = self.rew[t] + cfg.gamma * nextval * nextnonterm - self.val[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nextnonterm * gae
            adv[t] = gae
        return adv, adv + self.val

    def flat(self, adv, ret):
        f = lambda x: x.reshape(-1, *x.shape[2:])
        return (f(self.obs), f(self.mask), f(self.act), f(self.logp),
                f(self.val), f(adv), f(ret))


class PPOTrainer:
    def __init__(self, cfg, net, envs, device):
        self.cfg, self.net, self.envs, self.device = cfg, net, envs, device
        self.opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
        self.buf = RolloutBuffer(cfg, device)
        self.obs = envs.reset()
        self.next_done = torch.zeros(cfg.num_envs, device=device)
        self.ep_stats = []          # rolling window of finished episodes

    def _to_tensor(self, obs):
        x = build_obs_tensor(obs["hmap"], obs["item"], self.cfg)
        return (torch.as_tensor(x, device=self.device),
                torch.as_tensor(obs["mask"], device=self.device))

    # -- rollout ------------------------------------------------------------
    @torch.no_grad()
    def collect(self):
        cfg, buf = self.cfg, self.buf
        n_invalid = n_act = 0
        for t in range(cfg.num_steps):
            x, m = self._to_tensor(self.obs)
            buf.obs[t], buf.mask[t] = x, m
            buf.done[t] = self.next_done
            a, logp, v, _ = self.net.act(x, m)
            buf.act[t], buf.logp[t], buf.val[t] = a, logp, v

            self.obs, r, d, infos = self.envs.step(a.cpu().numpy())
            buf.rew[t] = torch.as_tensor(r, device=self.device)
            self.next_done = torch.as_tensor(d.astype(np.float32), device=self.device)
            n_act += len(infos)
            for i in infos:
                n_invalid += bool(i.get("invalid"))
                if "final_utilization" in i:
                    self.ep_stats.append((i["final_utilization"],
                                          i["final_items"], i["reason"]))

        x, _ = self._to_tensor(self.obs)
        _, last_val, _ = self.net(x)
        return buf.compute_gae(last_val, self.next_done), n_invalid / max(1, n_act)

    # -- update -------------------------------------------------------------
    def update(self, adv, ret):
        cfg = self.cfg
        obs, mask, act, old_logp, old_val, adv, ret = self.buf.flat(adv, ret)
        B = obs.shape[0]
        mb = B // cfg.minibatches
        idx = np.arange(B)
        logs = {k: 0.0 for k in
                ("loss", "actor", "critic", "mask", "einf", "entropy",
                 "approx_kl", "clipfrac", "mask_acc", "mask_rec", "mask_fpr")}
        nb = 0
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for s in range(0, B, mb):
                j = idx[s:s + mb]
                logp, val, ent, einf, pmask = self.net.evaluate(obs[j], act[j], mask[j])

                a = adv[j]
                a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = (logp - old_logp[j]).exp()
                l_actor = -torch.min(
                    ratio * a,
                    torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * a
                ).mean()

                if cfg.clip_value_loss:
                    vclip = old_val[j] + torch.clamp(
                        val - old_val[j], -cfg.clip_range, cfg.clip_range)
                    l_critic = 0.5 * torch.max((val - ret[j]) ** 2,
                                               (vclip - ret[j]) ** 2).mean()
                else:
                    l_critic = 0.5 * ((val - ret[j]) ** 2).mean()

                # MSE mask-prediction loss (paper: "L_mask is the MSE loss")
                l_mask = F.mse_loss(pmask, mask[j]) if cfg.use_mask_prediction \
                    else torch.zeros((), device=obs.device)

                e_inf, e_ent = einf.mean(), ent.mean()
                loss = (cfg.w_actor * l_actor + cfg.w_critic * l_critic
                        + cfg.w_mask * l_mask + cfg.w_einf * e_inf
                        - cfg.w_entropy * e_ent)

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    pm = (pmask > 0.5).float()
                    logs["approx_kl"] += ((ratio - 1) - (logp - old_logp[j])).mean().item()
                    logs["clipfrac"] += ((ratio - 1).abs() > cfg.clip_range).float().mean().item()
                    logs["mask_acc"] += (pm == mask[j]).float().mean().item()
                    denom = mask[j].sum().clamp(min=1)
                    logs["mask_rec"] += ((pm * mask[j]).sum() / denom).item()
                    # false-feasible rate: predicted-feasible LPs that are not
                    # actually feasible -- the paper's "99.5% legit" diagnostic
                    logs["mask_fpr"] += ((pm * (1 - mask[j])).sum()
                                         / pm.sum().clamp(min=1)).item()
                logs["loss"] += loss.item(); logs["actor"] += l_actor.item()
                logs["critic"] += l_critic.item(); logs["mask"] += l_mask.item()
                logs["einf"] += e_inf.item(); logs["entropy"] += e_ent.item()
                nb += 1
        return {k: v / nb for k, v in logs.items()}

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def drain_episode_stats(self):
        s, self.ep_stats = self.ep_stats, []
        return s
