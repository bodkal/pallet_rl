"""Training loops for the six algorithms of the AR2L paper.

    pct      PCT baseline, nominal dynamics only                (Zhao et al. 2022a)
    cppo     CVaR-PPO: the policy update sees only the worst tail (Ying et al. 2022)
    rarl     packer trained purely on the attacker's dynamics    (Pinto et al. 2017)
    rfmdp    nominal samples, pessimistic TV-dual value target   (Ho / Panaganti)
    exact    exact AR2L   (Algorithm 1: attacker -> mixture model -> packer)
    approx   approximate AR2L (Algorithm 2: nominal rollout, Eq. 18 targets)
    attack   train an attacker against a frozen packer (Table 1 / test-set attacks)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import warnings

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

from .env import BPPBatch
from .evaluate import attack_score, nominal_score
from .heuristics import act as heur_act
from .model import PackNet, PermNet, sample
from .ppo import PPO, gae, inf_tv_dual, sup_tv_dual, to_torch


def move_to_front(b, b_mask, idx):
    """Reorder the observable items so that item `idx` comes first."""
    n, nb = b.shape[0], b.shape[1]
    ar = torch.arange(nb, device=b.device)[None, :].expand(n, nb)
    key = torch.where(ar == idx[:, None], -torch.ones_like(ar), ar)
    order = key.argsort(1)
    return torch.gather(b, 1, order[..., None].expand(-1, -1, b.shape[2])), \
        torch.gather(b_mask, 1, order)


class Tracker:
    """Rolling episode statistics."""

    def __init__(self, n=200):
        self.util, self.items, self.n = [], [], n

    def add(self, u, k):
        self.util += list(u); self.items += list(k)
        self.util, self.items = self.util[-self.n:], self.items[-self.n:]

    def stats(self):
        if not self.util:
            return 0.0, 0.0
        return float(np.mean(self.util)), float(np.mean(self.items))


class Runner:
    """Environment plus the networks that act on it."""

    def __init__(self, env, pack, device, heur=None, greedy_pack=False):
        self.env, self.pack, self.device, self.heur = env, pack, device, heur
        # a frozen packer is attacked greedily at test time, so training the
        # attacker against a sampling one would fit the wrong opponent
        self.greedy_pack = greedy_pack
        self.tracker = Tracker()
        self.ep_r = np.zeros(env.n_env, np.float32)

    def collect(self, T, permuter=None, greedy=False, keep_perm=False):
        """One rollout.  Returns the packer trajectory and, optionally, the
        permuter's.  `permuter` reorders the conveyor before the packer acts."""
        env, dev = self.env, self.device
        n = env.n_env
        P = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done", "ep")}
        M = {k: [] for k in ("obs", "act", "logp", "val")}
        # CVaR needs each sample tagged with the return of the episode it
        # belongs to, so remember where in this rollout each episode started
        ep_label = np.zeros((T, n), np.float32)
        ep_start = np.zeros(n, np.int64)

        for t in range(T):
            env.reset_done()
            if permuter is not None:
                ocb = to_torch(env.obs_cb(), dev)
                with torch.no_grad():
                    lg, v = permuter(ocb)
                idx, lp, _ = sample(lg, greedy)
                if keep_perm:
                    M["obs"].append(ocb); M["act"].append(idx)
                    M["logp"].append(lp); M["val"].append(v)
                env.permute(idx.cpu().numpy())

            if self.heur is not None:      # attacking a heuristic packer
                o = to_torch(env.obs(), dev)
                act = heur_act(env, self.heur)
                a = torch.as_tensor(act, device=dev)
                lp = v = torch.zeros(env.n_env, device=dev)
            else:
                o = to_torch(env.obs(), dev)
                with torch.no_grad():
                    lg, v = self.pack(o)
                a, lp, _ = sample(lg, greedy or self.greedy_pack)
                act = a.cpu().numpy()
            r, d = env.step(act)

            P["obs"].append(o); P["act"].append(a); P["logp"].append(lp)
            P["val"].append(v)
            P["rew"].append(torch.as_tensor(r, device=dev))
            P["done"].append(torch.as_tensor(d.astype(np.float32), device=dev))

            self.ep_r += r
            if d.any():
                self.tracker.add(env.utilization()[d], env.n_packed[d])
                for b in np.nonzero(d)[0]:
                    ep_label[ep_start[b]: t + 1, b] = self.ep_r[b]
                    ep_start[b] = t + 1
                    self.ep_r[b] = 0.0

        with torch.no_grad():
            ocb = to_torch(env.obs_cb(), dev)
            last_val = self.pack.value(ocb)
            perm_last = permuter.value(ocb) if keep_perm else None
        # episodes still running at the end of the rollout: what they have
        # banked so far plus the critic's estimate of the rest
        tail = self.ep_r + last_val.cpu().numpy()
        for b in range(n):
            if ep_start[b] < T:
                ep_label[ep_start[b]:, b] = tail[b]

        out = {k: (torch.stack(v) if k not in ("obs", "ep") else v)
               for k, v in P.items()}
        out["ep"] = torch.as_tensor(ep_label, device=dev)
        out["last_val"] = last_val
        if keep_perm:
            out["perm"] = {k: (torch.stack(v) if k != "obs" else v)
                           for k, v in M.items()}
            out["perm"]["last_val"] = perm_last
        return out


def flat_obs(obs_list):
    """Concatenate rollout observations, re-padding the ragged node axes."""
    out = {}
    for k in obs_list[0]:
        if obs_list[0][k].dim() < 2:
            out[k] = torch.cat([o[k] for o in obs_list], 0)
            continue
        n = max(o[k].shape[1] for o in obs_list)
        parts = []
        for o in obs_list:
            x = o[k]
            if x.shape[1] < n:
                w = n - x.shape[1]
                pad = (0, w) if x.dim() == 2 else (0, 0, 0, w)
                x = F.pad(x.to(torch.uint8), pad).bool() if x.dtype == torch.bool \
                    else F.pad(x, pad)
            parts.append(x)
        out[k] = torch.cat(parts, 0)
    return out


def flat(x):
    return x.reshape(-1)


def make_env(args, seed):
    return BPPBatch(args.n_env, S=args.bin, nb=args.nb, n_items=args.n_items,
                    size_lo=args.size_lo, size_hi=args.size_hi, seed=seed,
                    stability=args.stability, ems=bool(args.ems), rot=args.rot)


def build(args, device):
    kw = dict(d=args.width, n_head=args.heads, n_layer=args.layers)
    return PackNet(**kw).to(device), PermNet(**kw).to(device), PermNet(**kw).to(device)


def train(args):
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = os.path.join("runs", args.name)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out, "args.json"), "w"), indent=1)

    env = make_env(args, args.seed)
    pack, attacker, mixer = build(args, device)
    if args.compile:
        for net in (pack, attacker, mixer):
            net.forward = torch.compile(net.forward, dynamic=True)
    start = 1
    if args.resume and os.path.exists(os.path.join(out, "last.pt")):
        ck = torch.load(os.path.join(out, "last.pt"), map_location=device,
                        weights_only=False)
        pack.load_state_dict(ck["pack"]); attacker.load_state_dict(ck["attacker"])
        mixer.load_state_dict(ck["mixer"]); start = ck["it"] + 1
        print(f"[{args.name}] resuming at iteration {start}", flush=True)
    if args.init:
        pack.load_state_dict(torch.load(args.init, map_location=device,
                                        weights_only=False)["pack"])
    if args.freeze_pack:
        for p in pack.parameters():
            p.requires_grad_(False)

    opt_kw = dict(lr=args.lr, epochs=args.epochs, minibatches=args.minibatches,
                  ent_coef=args.ent_coef)
    ppo_pack = PPO(pack, **opt_kw)
    ppo_att = PPO(attacker, **opt_kw)
    ppo_mix = PPO(mixer, **opt_kw)
    runner = Runner(env, pack, device, heur=args.heur_pack,
                    greedy_pack=args.freeze_pack)

    log = open(os.path.join(out, "log.jsonl"), "a")
    nom = {"nom_util": 0.0, "nom_items": 0.0}
    best = -1.0
    t0 = time.time()

    for it in range(start, args.iters + 1):
        stats = {}
        if args.ent_final is not None:
            # the attacker is deployed greedily, so let it sharpen
            f = (it - 1) / max(args.iters - 1, 1)
            ppo_att.ent_coef = args.ent_coef + f * (args.ent_final - args.ent_coef)

        # ---- stage 1: the attacker, for every algorithm that needs one -----
        if args.algo in ("rarl", "exact", "approx", "attack"):
            tr = runner.collect(args.T, permuter=attacker, keep_perm=True)
            pm = tr["perm"]
            # the attacker's critic is fitted to the negated return, so its
            # bootstrap is already in the right units
            adv, ret = gae(-tr["rew"], pm["val"], tr["done"],
                           pm["last_val"], args.gamma, args.lam)
            stats["att"] = ppo_att.update(flat_obs(pm["obs"]), flat(pm["act"]),
                                          flat(pm["logp"]), flat(adv), flat(ret))
            if args.algo == "rarl":  # the packer learns from this same dynamics
                a2, r2 = gae(tr["rew"], tr["val"], tr["done"], tr["last_val"],
                             args.gamma, args.lam)
                stats["pack"] = ppo_pack.update(
                    flat_obs(tr["obs"]), flat(tr["act"]), flat(tr["logp"]),
                    flat(a2), flat(r2))

        # ---- stage 2: the packer ------------------------------------------
        if args.algo == "exact":
            tr = runner.collect(args.T, permuter=mixer, keep_perm=True)
            pm = tr["perm"]
            adv, ret = gae(tr["rew"], pm["val"], tr["done"],
                           pm["last_val"], args.gamma, args.lam)

            def aux(o, mb, logp_all, _a=args.alpha, _c=args.dist_coef):
                # Eq. 6: stay near the nominal order (index 0) and near pi_perm
                ce = -logp_all[:, 0]
                with torch.no_grad():
                    lq = F.log_softmax(attacker.logits(o), -1)
                kl = (logp_all.exp() * (logp_all - lq)).sum(-1)
                return _c * (ce.mean() + _a * kl.mean())

            stats["mix"] = ppo_mix.update(flat_obs(pm["obs"]), flat(pm["act"]),
                                          flat(pm["logp"]), flat(adv), flat(ret),
                                          aux=aux)
            a2, r2 = gae(tr["rew"], tr["val"], tr["done"], tr["last_val"],
                         args.gamma, args.lam)
            stats["pack"] = ppo_pack.update(flat_obs(tr["obs"]), flat(tr["act"]),
                                            flat(tr["logp"]), flat(a2), flat(r2))

        elif args.algo in ("pct", "cppo", "rfmdp", "approx"):
            tr = runner.collect(args.T)
            rew, val, done = tr["rew"], tr["val"], tr["done"]

            if args.algo in ("rfmdp", "approx"):
                obs = tr["obs"]
                nxt = [{k: o[k] for k in ("c", "c_mask", "b", "b_mask")}
                       for o in obs[1:]]
                with torch.no_grad():
                    last = to_torch(env.obs_cb(), device)
                nxt.append(last)
                with torch.no_grad():
                    v_o = torch.stack([pack.critic(n["c"], n["c_mask"],
                                                   n["b"], n["b_mask"]) for n in nxt])
                    if args.algo == "approx":
                        v_w = []
                        for n in nxt:
                            lg = attacker.logits(n)
                            idx, _, _ = sample(lg)
                            bb, bm = move_to_front(n["b"], n["b_mask"], idx)
                            v_w.append(pack.critic(n["c"], n["c_mask"], bb, bm))
                        v_w = torch.stack(v_w)
                        tgt = sup_tv_dual(flat(v_o), flat(v_w), args.alpha,
                                          args.rho, flat(val)).view_as(v_o)
                    else:
                        tgt = inf_tv_dual(flat(v_o), args.rho,
                                          flat(val)).view_as(v_o)
                adv, ret = gae(rew, val, done, tgt[-1], args.gamma, args.lam,
                               next_val=tgt)
            else:
                adv, ret = gae(rew, val, done, tr["last_val"], args.gamma, args.lam)

            w = None
            if args.algo == "cppo":
                ep = flat(tr["ep"])
                thr = torch.quantile(ep, args.cvar_q)
                w = (ep <= thr).float() / args.cvar_q

            stats["pack"] = ppo_pack.update(flat_obs(tr["obs"]), flat(tr["act"]),
                                            flat(tr["logp"]), flat(adv), flat(ret),
                                            weight=w)

        # ---- logging -------------------------------------------------------
        u, k = runner.tracker.stats()
        if it % args.eval_every == 0 or it == args.iters:
            ev = dict(device=args.device, stability=args.stability, rot=args.rot,
                      S=args.bin, size_hi=args.size_hi)
            if args.algo == "attack":
                au, ak = attack_score(args.heur_pack or pack, attacker, args.nb,
                                      args.eval_inst, **ev)
                nom = {"att_util": au, "att_items": ak}
            elif not args.heur_pack:
                nu, nk = nominal_score(pack, args.nb, args.eval_inst, **ev)
                nom = {"nom_util": nu, "nom_items": nk}
        if it % args.log_every == 0 or it == args.iters:
            rec = {"it": it, "util": u, "items": k, "t": time.time() - t0, **nom,
                   **{f"{a}_{b}": v for a, d in stats.items() for b, v in d.items()}}
            log.write(json.dumps(rec) + "\n"); log.flush()
            held = nom.get("att_util", nom.get("nom_util", 0.0))
            tag = "attacked" if args.algo == "attack" else "nominal"
            print(f"[{args.name}] it {it:6d}  util {u*100:5.2f}%  items {k:5.2f}"
                  f"  {tag} {held*100:5.2f}%"
                  f"  {(time.time()-t0)/it*1000:6.1f} ms/it", flush=True)
        if it % args.save_every == 0 or it == args.iters:
            ck = {"pack": pack.state_dict(), "attacker": attacker.state_dict(),
                  "mixer": mixer.state_dict(), "args": vars(args), "it": it}
            torch.save(ck, os.path.join(out, "last.pt"))
            # an attacker is best when the held-out packer does worst
            score = (-nom.get("att_util", 1.0) if args.algo == "attack"
                     else nom.get("nom_util", 0.0))
            if score > best:
                best = score
                torch.save(ck, os.path.join(out, "best.pt"))
    log.close()


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--algo", default="pct",
                   choices=["pct", "cppo", "rarl", "rfmdp", "exact", "approx", "attack"])
    p.add_argument("--nb", type=int, default=10, help="observable items N_B")
    p.add_argument("--alpha", type=float, default=1.0, help="robustness weight")
    p.add_argument("--rho", type=float, default=0.1, help="uncertainty radius")
    p.add_argument("--dist_coef", type=float, default=1.0)
    p.add_argument("--cvar_q", type=float, default=0.5)
    p.add_argument("--iters", type=int, default=8000)
    p.add_argument("--n_env", type=int, default=64)
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--bin", type=int, default=10)
    p.add_argument("--n_items", type=int, default=150)
    p.add_argument("--size_lo", type=int, default=1)
    p.add_argument("--size_hi", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--ent_final", type=float, default=None,
                   help="anneal the attacker's entropy bonus to this value")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--init", default=None, help="checkpoint to initialise the packer")
    p.add_argument("--freeze_pack", action="store_true")
    p.add_argument("--heur_pack", default=None,
                   help="attack a heuristic packer instead of a network")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    # torch.compile with dynamic node counts miscompiles the pointer head for
    # some N_B and shows up as an illegal memory access; off unless asked for
    p.add_argument("--compile", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stability", default="com", choices=["com", "cdrl"])
    p.add_argument("--rot", type=int, default=2,
                   help="item orientations offered: 1 = none, 2 = yaw by 90 deg")
    p.add_argument("--ems", type=int, default=1,
                   help="0 scores every loading position instead of EMS corners")
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_inst", type=int, default=256)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=200)
    return p


if __name__ == "__main__":
    train(get_parser().parse_args())
