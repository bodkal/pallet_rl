"""Training loops for the six algorithms of the AR2L paper.

    pct      PCT baseline, nominal dynamics only                (Zhao et al. 2022a)
    cppo     CVaR-PPO: the policy update sees only the worst tail (Ying et al. 2022)
    rarl     packer trained purely on the attacker's dynamics    (Pinto et al. 2017)
    rfmdp    nominal samples, pessimistic TV-dual value target   (Ho / Panaganti)
    exact    exact AR2L   (Algorithm 1: attacker -> mixture model -> packer)
    approx   approximate AR2L (Algorithm 2: nominal rollout, Eq. 18 targets)
    attack   train an attacker against a frozen packer (Table 1 / test-set attacks)
    select   cooperative selector: the permuter picks *for* the packer, not
             against it -- `rarl` with the advantage sign left alone.  With
             `--n_pick k` it chooses among the k items the cell can reach and
             the packer is trained on the stream it produces.

Every box carries a type and may only be stacked on its own type, so with a
reach of one the stream decides most of the packing for the agent and episodes
end where the conveyor happens to change type.  `--n_pick k` with a permuter --
`select` cooperatively, `rarl`/`exact`/`approx` adversarially -- is what lets a
run group boxes of one type together; `--n_types 1` turns the whole thing off.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import warnings

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except ImportError:                                  # optional dependency
    tqdm = None


class Progress:
    """A progress bar on stderr, so stdout stays a clean pipeable log.

    `write` routes the periodic log lines through the bar so they scroll
    normally instead of being overwritten by it.
    """

    def __init__(self, total, start, mode="auto"):
        on = (mode == "on" or (mode == "auto" and sys.stderr.isatty()))
        self.bar = (tqdm(total=total, initial=start - 1, unit="it",
                         dynamic_ncols=True, leave=True, file=sys.stderr)
                    if on and tqdm is not None else None)

    def step(self, **fields):
        if self.bar is not None:
            self.bar.set_postfix(fields, refresh=False)
            self.bar.update(1)

    def write(self, line):
        if self.bar is None:
            print(line, flush=True)
        else:
            self.bar.write(line, file=sys.stdout)

    def close(self):
        if self.bar is not None:
            self.bar.close()

from .config import CFG, load as load_config
from .env import BPPBatch
from .evaluate import attack_score, nominal_score
from .heuristics import act as heur_act
from .model import PackNet, PermNet, sample
from .ppo import PPO, gae, inf_tv_dual, sup_tv_dual, to_torch


#: the observable-item arrays a permutation has to carry along together
B_KEYS = ("b", "b_mask", "b_type", "b_pick")


def move_to_front(o, idx):
    """A copy of `o` with observable item `idx` moved to the front.

    Every array indexed by the conveyor slot moves as one -- the sizes, the
    validity mask, the reach mask and the type -- so that a permuted state is a
    state and not a size stream that has drifted away from its own types.
    """
    b = o["b"]
    n, nb = b.shape[0], b.shape[1]
    ar = torch.arange(nb, device=b.device)[None, :].expand(n, nb)
    key = torch.where(ar == idx[:, None], -torch.ones_like(ar), ar)
    order = key.argsort(1)
    out = dict(o)
    for k in B_KEYS:
        if k not in o:
            continue
        x = o[k]
        out[k] = (torch.gather(x, 1, order[..., None].expand(-1, -1, x.shape[2]))
                  if x.dim() == 3 else torch.gather(x, 1, order))
    return out


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


def _held(nom):
    """The held-out number this run is judged on, whatever produced it."""
    for k in ("att_util", "sel_util", "nom_util"):
        if k in nom:
            return nom[k]
    return 0.0


def make_env(args, seed):
    return BPPBatch(args.n_env, S=args.bin, nb=args.nb, n_items=args.n_items,
                    size_lo=args.size_lo, size_hi=args.size_hi, seed=seed,
                    max_l=args.max_l, stability=args.stability,
                    ems=bool(args.ems), rot=args.rot, max_c=args.max_c,
                    min_support=args.min_support, n_pick=args.n_pick,
                    n_types=args.n_types,
                    type_constraint=bool(args.type_constraint))


def build(args, device):
    kw = dict(d=args.width, n_head=args.heads, n_layer=args.layers,
              c_temp=args.c_temp, n_types=args.n_types,
              type_embed=args.type_embed)
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
    start, resumed, best = 1, None, -1.0
    if args.resume and os.path.exists(os.path.join(out, "last.pt")):
        resumed = torch.load(os.path.join(out, "last.pt"), map_location=device,
                             weights_only=False)
        pack.load_state_dict(resumed["pack"])
        attacker.load_state_dict(resumed["attacker"])
        mixer.load_state_dict(resumed["mixer"])
        start = resumed["it"] + 1
        # without the old high-water mark the first save after a resume always
        # "wins" and overwrites a better best.pt
        best = resumed.get("best", -1.0)
        print(f"[{args.name}] resuming at iteration {start} "
              f"(best so far {best:+.4f})", flush=True)
    if args.init:
        pack.load_state_dict(torch.load(args.init, map_location=device,
                                        weights_only=False)["pack"])
    if args.freeze_pack:
        for p in pack.parameters():
            p.requires_grad_(False)

    opt_kw = dict(lr=args.lr, epochs=args.epochs, minibatches=args.minibatches,
                  ent_coef=args.ent_coef, clip=args.clip,
                  vf_coef=args.vf_coef, max_grad=args.max_grad)
    ppo_pack = PPO(pack, **opt_kw)
    ppo_att = PPO(attacker, **opt_kw)
    ppo_mix = PPO(mixer, **opt_kw)
    if resumed is not None and "opt" in resumed:
        # Adam's moment estimates are part of the training state; dropping them
        # costs a visible transient every time a run is restarted
        for key, ppo in (("pack", ppo_pack), ("attacker", ppo_att),
                         ("mixer", ppo_mix)):
            ppo.opt.load_state_dict(resumed["opt"][key])
    runner = Runner(env, pack, device, heur=args.heur_pack,
                    greedy_pack=args.freeze_pack)

    log = open(os.path.join(out, "log.jsonl"), "a")
    nom = {"nom_util": 0.0, "nom_items": 0.0}
    t0, done_it = time.time(), 0
    prog = Progress(args.iters, start, args.progress)

    for it in range(start, args.iters + 1):
        done_it += 1
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

        elif args.algo == "select":
            # The mirror of `rarl`: same rollout structure, same two updates,
            # but the permuter is scored on the packer's own return rather
            # than its negation, so it learns to hand over the box that packs
            # best instead of the one that hurts most.  No attacker is built
            # or trained, which is the whole saving over `exact --dist_coef 0`.
            tr = runner.collect(args.T, permuter=mixer, keep_perm=True)
            pm = tr["perm"]
            adv, ret = gae(tr["rew"], pm["val"], tr["done"],
                           pm["last_val"], args.gamma, args.lam)
            stats["mix"] = ppo_mix.update(flat_obs(pm["obs"]), flat(pm["act"]),
                                          flat(pm["logp"]), flat(adv), flat(ret))
            a2, r2 = gae(tr["rew"], tr["val"], tr["done"], tr["last_val"],
                         args.gamma, args.lam)
            stats["pack"] = ppo_pack.update(flat_obs(tr["obs"]), flat(tr["act"]),
                                            flat(tr["logp"]), flat(a2), flat(r2))

        elif args.algo in ("pct", "cppo", "rfmdp", "approx"):
            tr = runner.collect(args.T)
            rew, val, done = tr["rew"], tr["val"], tr["done"]

            if args.algo in ("rfmdp", "approx"):
                obs = tr["obs"]
                nxt = [{k: o[k] for k in ("c", "c_mask", "c_type", "b",
                                          "b_mask", "b_type", "b_pick")}
                       for o in obs[1:]]
                with torch.no_grad():
                    last = to_torch(env.obs_cb(), device)
                nxt.append(last)
                with torch.no_grad():
                    v_o = torch.stack([pack.critic(n) for n in nxt])
                    if args.algo == "approx":
                        v_w = []
                        for n in nxt:
                            lg = attacker.logits(n)
                            idx, _, _ = sample(lg)
                            v_w.append(pack.critic(move_to_front(n, idx)))
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
        prog.step(util=f"{u*100:.1f}%", items=f"{k:.1f}",
                  **{{"attack": "att", "select": "sel"}.get(args.algo, "nom"):
                     f"{_held(nom)*100:.1f}%"})
        if it % args.eval_every == 0 or it == args.iters:
            ev = dict(device=args.device, stability=args.stability, rot=args.rot,
                      S=args.bin, size_hi=args.size_hi, max_l=args.max_l,
                      min_support=args.min_support, n_pick=args.n_pick,
                      n_types=args.n_types,
                      type_constraint=bool(args.type_constraint))
            if args.algo == "attack":
                au, ak = attack_score(args.heur_pack or pack, attacker, args.nb,
                                      args.eval_inst, **ev)
                nom = {"att_util": au, "att_items": ak}
            elif args.algo == "select":
                # scoring the packer on the plain conveyor would measure it
                # without the selector it was trained with, and would then
                # pick `best.pt` on a number the run is not optimising
                su, sk = attack_score(pack, mixer, args.nb, args.eval_inst, **ev)
                nom = {"sel_util": su, "sel_items": sk}
            elif not args.heur_pack:
                nu, nk = nominal_score(pack, args.nb, args.eval_inst, **ev)
                nom = {"nom_util": nu, "nom_items": nk}
        if it % args.log_every == 0 or it == args.iters:
            rec = {"it": it, "util": u, "items": k, "t": time.time() - t0, **nom,
                   **{f"{a}_{b}": v for a, d in stats.items() for b, v in d.items()}}
            log.write(json.dumps(rec) + "\n"); log.flush()
            held = _held(nom)
            tag = {"attack": "attacked",
                   "select": "selected"}.get(args.algo, "nominal")
            prog.write(
                f"[{args.name}] it {it:6d}  util {u*100:5.2f}%  items {k:5.2f}"
                f"  {tag} {held*100:5.2f}%"
                f"  {(time.time()-t0)/max(done_it, 1)*1000:6.1f} ms/it")
        if it % args.save_every == 0 or it == args.iters:
            # an attacker is best when the held-out packer does worst
            # the 1.0 default keeps an attacker from banking a best.pt on a
            # save that lands before its first eval; every other algorithm
            # scores 0.0 there and is beaten by the first real number
            score = (-nom.get("att_util", 1.0) if args.algo == "attack"
                     else _held(nom))
            improved = score > best
            best = max(best, score)
            ck = {"pack": pack.state_dict(), "attacker": attacker.state_dict(),
                  "mixer": mixer.state_dict(), "args": vars(args), "it": it,
                  "best": best,
                  "opt": {"pack": ppo_pack.opt.state_dict(),
                          "attacker": ppo_att.opt.state_dict(),
                          "mixer": ppo_mix.opt.state_dict()}}
            torch.save(ck, os.path.join(out, "last.pt"))
            if improved:
                torch.save(ck, os.path.join(out, "best.pt"))
    prog.close()
    log.close()


def extent(v):
    """`--bin 10` (a cube) or `--bin 60x50x80` (Lx, Ly, Lz)."""
    parts = str(v).lower().split("x")
    if len(parts) == 1:
        return int(parts[0])
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected an int or WxLxH, got {v!r}")
    return tuple(int(q) for q in parts)


def get_parser():
    """The CLI.  Every default is read from `config.yaml` at call time, so
    `--config other.yaml` (handled by `main`) is already in `CFG` here."""
    e, t, o, m, r = (CFG["env"], CFG["train"], CFG["ppo"], CFG["model"],
                     CFG["run"])
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--config", default=None,
                   help="YAML file of defaults to merge over config.yaml; "
                        "the same as AR2L_CONFIG=")
    p.add_argument("--algo", default=t["algo"],
                   choices=["pct", "cppo", "rarl", "rfmdp", "exact", "approx",
                            "attack", "select"])
    p.add_argument("--nb", type=int, default=t["nb"], help="observable items N_B")
    p.add_argument("--n_pick", type=int, default=t["n_pick"],
                   help="how many of the N_B are within reach; the rest are "
                        "preview only (default: all of them)")
    p.add_argument("--alpha", type=float, default=t["alpha"], help="robustness weight")
    p.add_argument("--rho", type=float, default=t["rho"], help="uncertainty radius")
    p.add_argument("--dist_coef", type=float, default=t["dist_coef"])
    p.add_argument("--cvar_q", type=float, default=t["cvar_q"])
    p.add_argument("--iters", type=int, default=t["iters"])
    p.add_argument("--n_env", type=int, default=t["n_env"])
    p.add_argument("--T", type=int, default=t["T"])
    p.add_argument("--bin", type=extent, default=e["bin"],
                   help="bin extent; an int for a cube or WxLxH, e.g. 60x50x80")
    p.add_argument("--n_items", type=int, default=e["n_items"])
    p.add_argument("--size_lo", type=int, default=e["size_lo"])
    p.add_argument("--max_l", type=int, default=e["max_l"],
                   help="leaf-node cap; raise it if the EMS corners hit it")
    p.add_argument("--max_c", type=int, default=e["max_c"],
                   help="initial packed-item capacity; it grows if exceeded")
    p.add_argument("--size_hi", type=extent, default=e["size_hi"],
                   help="item side cap; an int or WxLxH for per-axis caps")
    p.add_argument("--lr", type=float, default=o["lr"])
    p.add_argument("--gamma", type=float, default=o["gamma"])
    p.add_argument("--lam", type=float, default=o["lam"])
    p.add_argument("--epochs", type=int, default=o["epochs"])
    p.add_argument("--minibatches", type=int, default=o["minibatches"])
    p.add_argument("--ent_coef", type=float, default=o["ent_coef"])
    p.add_argument("--ent_final", type=float, default=o["ent_final"],
                   help="anneal the attacker's entropy bonus to this value")
    p.add_argument("--clip", type=float, default=o["clip"],
                   help="PPO ratio clip")
    p.add_argument("--vf_coef", type=float, default=o["vf_coef"],
                   help="value loss weight")
    p.add_argument("--max_grad", type=float, default=o["max_grad"],
                   help="gradient-norm clip")
    p.add_argument("--width", type=int, default=m["width"])
    p.add_argument("--heads", type=int, default=m["heads"])
    p.add_argument("--layers", type=int, default=m["layers"])
    p.add_argument("--c_temp", type=float, default=m["c_temp"],
                   help="pointer-head temperature c of Eq. 28")
    p.add_argument("--init", default=None, help="checkpoint to initialise the packer")
    p.add_argument("--freeze_pack", action="store_true")
    p.add_argument("--heur_pack", default=None,
                   help="attack a heuristic packer instead of a network")
    p.add_argument("--seed", type=int, default=r["seed"])
    p.add_argument("--device", default=r["device"])
    # torch.compile with dynamic node counts miscompiles the pointer head for
    # some N_B and shows up as an illegal memory access; off unless asked for
    p.add_argument("--compile", type=int, default=r["compile"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stability", default=e["stability"], choices=["com", "cdrl"])
    p.add_argument("--min_support", type=float, default=None,
                   help="fraction of an item's base that must rest on the "
                        "layer below for a placement to be offered; the "
                        "config.yaml value (%.2f) unless set, 0 for the bare "
                        "centre-of-mass rule (ignored by --stability cdrl)"
                        % e["min_support"])
    p.add_argument("--rot", type=int, default=e["rot"],
                   help="item orientations offered: 1 = none, 2 = yaw by 90 deg")
    p.add_argument("--n_types", type=int, default=e["n_types"],
                   help="box types; must match the `types:` size classes in "
                        "the config file, and 1 with `types: null` is the "
                        "untyped simulator")
    p.add_argument("--type_constraint", type=int, default=e["type_constraint"],
                   help="0 keeps type_id in the state but lets any box be "
                        "stacked on any other")
    p.add_argument("--type_embed", type=int, default=m["type_embed"],
                   help="width of the trainable box-type embedding")
    p.add_argument("--ems", type=int, default=e["ems"],
                   help="0 scores every loading position instead of EMS corners")
    p.add_argument("--eval_every", type=int, default=r["eval_every"])
    p.add_argument("--eval_inst", type=int, default=r["eval_inst"])
    p.add_argument("--log_every", type=int, default=r["log_every"])
    p.add_argument("--progress", choices=("auto", "on", "off"),
                   default=r["progress"],
                   help="progress bar on stderr; auto = only when attached "
                        "to a terminal")
    p.add_argument("--save_every", type=int, default=r["save_every"])
    return p


def main(argv=None):
    # --config has to be read before the parser is built, since it is what the
    # parser's defaults come from
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        load_config(known.config)
    args = get_parser().parse_args(argv)
    # resolve it here rather than in the env, so `args.json` records the rule
    # the run was trained under instead of a bare `null`
    if args.min_support is None:
        args.min_support = float(CFG["env"]["min_support"])
    train(args)


if __name__ == "__main__":
    main()
