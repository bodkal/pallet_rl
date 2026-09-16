"""PPO, plus the value targets used by the robust variants.

AR2L trains the packing policy, the attacker and the mixture-dynamics model
with PPO (Sec. 4.2, last paragraph).  Two of the six algorithms replace the
plain TD target with a dual form of a robust Bellman operator:

  RfMDP        inf over a TV ball around the nominal next-state distribution
  ApproxAR2L   Eq. 18, the sup over the mixture set built from both the
               nominal and the attacked next state

Both duals are one- and two-dimensional convex problems in the Lagrange
multipliers; with a single next-state sample per state we optimise them over
the rollout batch, which is what Panaganti et al. do in practice.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F



def to_torch(obs, device):
    return {k: torch.as_tensor(v).to(device, non_blocking=True)
            for k, v in obs.items()}


def cat_obs(seq):
    return {k: torch.cat([o[k] for o in seq], 0) for k in seq[0]}


def gae(rew, val, done, last_val, gamma=1.0, lam=0.95, next_val=None):
    """rew/val/done: (T, N).  Returns advantages and value targets.

    `next_val[t]` overrides V(s_{t+1}) in the TD residual.  The robust
    algorithms pass the dual value of the next state there, so the Bellman
    operator they evaluate is the robust one while credit assignment stays
    multi-step -- a one-step target would otherwise handicap them for reasons
    that have nothing to do with robustness.
    """
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    nxt = last_val
    run = torch.zeros_like(last_val)
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        v_next = nxt if next_val is None else next_val[t]
        delta = rew[t] + gamma * v_next * nonterm - val[t]
        run = delta + gamma * lam * nonterm * run
        adv[t] = run
        nxt = val[t]
    return adv, adv + val


def _grid(lo, hi, n, device):
    return torch.linspace(float(lo), float(hi), n, device=device)


def sup_tv_dual(v_o, v_w, alpha, rho, v_all, n_grid=96, vmax=1.0):
    """Eq. 18: sup of E[V] over the mixture uncertainty set, per sample.

    `v_o`/`v_w` are the nominal and attacked next-state values; `v_all` stands
    in for the state values V(s) that define lambda.  mu1, mu2 are shared over
    the batch, lambda = max(V - mu1, V - mu2, 0).

    The operator is optimistic by up to rho * range(V) per application, and
    Theorem 2 only makes it a contraction for gamma < 1.  This MDP runs
    undiscounted, so without an anchor the value inflates every step and the
    range with it -- measured here, the critic ran away to V ~ 3400 where the
    return is a space utilisation in [0, 1].  Projecting onto [0, vmax], which
    contains the true fixed point, is non-expansive and restores a fixed point.
    """
    dev = v_o.device
    v_o, v_w, v_all = (x.clamp(0.0, vmax) for x in (v_o, v_w, v_all))
    lo = float(min(v_o.min(), v_w.min(), v_all.min())) - 0.05
    hi = float(max(v_o.max(), v_w.max(), v_all.max())) + 0.05
    g = _grid(lo, hi, n_grid, dev)
    a1 = F.relu(v_o[None, :] - g[:, None]).mean(1)              # (G,)
    a2 = F.relu(v_w[None, :] - g[:, None]).mean(1)
    m1 = F.relu(v_all[None, :] - g[:, None]).max(1).values      # (G,)
    obj = (a1[:, None] + alpha * a2[None, :]
           + g[:, None] + alpha * g[None, :]
           + rho * (1 + alpha) * torch.maximum(m1[:, None], m1[None, :]))
    k = int(obj.argmin())
    mu1, mu2 = g[k // n_grid], g[k % n_grid]
    lam = torch.maximum(F.relu(v_all - mu1).max(), F.relu(v_all - mu2).max())
    per = (F.relu(v_o - mu1) + alpha * F.relu(v_w - mu2)
           + mu1 + alpha * mu2 + rho * (1 + alpha) * lam)
    return (per / (1.0 + alpha)).clamp(0.0, vmax)


def inf_tv_dual(v_o, rho, v_all, n_grid=192, vmax=1.0):
    """RfMDP: inf of E[V] over a TV ball of radius rho around the nominal.

    Same anchoring as `sup_tv_dual`, in the other direction.
    """
    dev = v_o.device
    v_o, v_all = v_o.clamp(0.0, vmax), v_all.clamp(0.0, vmax)
    lo = float(min(v_o.min(), v_all.min())) - 0.05
    hi = float(max(v_o.max(), v_all.max())) + 0.05
    g = _grid(lo, hi, n_grid, dev)
    obj = g - F.relu(g[:, None] - v_o[None, :]).mean(1) \
            - rho * F.relu(g[:, None] - v_all[None, :]).max(1).values
    nu = g[int(obj.argmax())]
    return (nu - F.relu(nu - v_o)
            - rho * F.relu(nu - v_all).max()).clamp(0.0, vmax)


class PPO:
    """Clipped-surrogate update shared by all three networks."""

    def __init__(self, net, lr=3e-4, clip=0.2, epochs=4, minibatches=4,
                 vf_coef=0.5, ent_coef=0.01, max_grad=0.5):
        self.net = net
        self.opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
        self.clip, self.epochs, self.nmb = clip, epochs, minibatches
        self.vf_coef, self.ent_coef, self.max_grad = vf_coef, ent_coef, max_grad

    def update(self, obs, act, old_logp, adv, ret, weight=None, aux=None):
        """`aux(obs, idx) -> tensor` adds an extra per-sample loss term."""
        n = act.shape[0]
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        z = torch.zeros((), device=act.device)
        stats = {k: z.clone() for k in ("pg", "vf", "ent", "aux", "kl")}
        nb = 0
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=act.device)
            for mb in perm.chunk(self.nmb):
                o = {k: v[mb] for k, v in obs.items()}
                logits, val = self.net(o)
                logp_all = F.log_softmax(logits, -1)
                logp = logp_all.gather(1, act[mb, None]).squeeze(1)
                ent = -(logp_all.exp() * logp_all).sum(-1)

                ratio = (logp - old_logp[mb]).exp()
                a = adv_n[mb]
                w = 1.0 if weight is None else weight[mb]
                pg = -(torch.min(ratio * a,
                                 ratio.clamp(1 - self.clip, 1 + self.clip) * a) * w).mean()
                vf = F.mse_loss(val, ret[mb])
                loss = pg + self.vf_coef * vf - self.ent_coef * ent.mean()
                extra = torch.zeros((), device=act.device)
                if aux is not None:
                    extra = aux(o, mb, logp_all)
                    loss = loss + extra

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad)
                self.opt.step()

                with torch.no_grad():   # accumulate on device; sync once
                    stats["pg"] += pg.detach(); stats["vf"] += vf.detach()
                    stats["ent"] += ent.mean(); stats["aux"] += extra.detach()
                    stats["kl"] += (old_logp[mb] - logp).mean()
                nb += 1
        return {k: float(v) / max(nb, 1) for k, v in stats.items()}
