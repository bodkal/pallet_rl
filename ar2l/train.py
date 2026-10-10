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
import hashlib
import json
import os
import signal
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
from .env import BPPBatch, OUTLINE_WHEN, TYPE_RULES
from .evaluate import attack_score, nominal_score
from .orders import add_cm_args, load_orders, orders_bin
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


class StopAsked:
    """SIGTERM, SIGHUP and Ctrl+C ask the loop to stop rather than kill it:
    the iteration under way finishes, last.pt is saved with everything
    `--resume` needs, and the run exits cleanly.  A second Ctrl+C quits on
    the spot, unsaved.  This is what the game's "Stop & save" sends, and what
    closing the terminal or stopping the game server sends too."""

    def __init__(self, name):
        self.name, self.asked = name, False
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, self)

    def __call__(self, sig, frame):
        if self.asked and sig == signal.SIGINT:
            raise KeyboardInterrupt
        if not self.asked:
            print(f"[{self.name}] stop asked ({signal.Signals(sig).name}): "
                  f"saving last.pt after this iteration", flush=True)
        self.asked = True


def save_atomic(obj, path):
    """torch.save through a temp file, so a kill mid-write leaves the old
    checkpoint whole rather than a truncated one."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def make_env(args, seed, pool=None):
    # a pool carries its own sizes and types, so the env draws no classes and
    # sizes its sweeps from the pool rather than from the config's envelope
    kw = ({} if pool is None else
          dict(pool=pool, pool_order_random=args.order_random, types=False,
               size_hi=pool[..., :3].reshape(-1, 3).max(0)))
    kw.setdefault("size_hi", args.size_hi)
    return BPPBatch(args.n_env, S=args.bin, nb=args.nb, n_items=args.n_items,
                    size_lo=args.size_lo, seed=seed,
                    max_l=args.max_l, stability=args.stability,
                    ems=args.ems, rot=args.rot, max_c=args.max_c,
                    min_support=args.min_support, n_pick=args.n_pick,
                    n_types=args.n_types,
                    type_constraint=bool(args.type_constraint),
                    type_rule=args.type_rule, soft_mix=bool(args.soft_mix),
                    outline_resort=bool(args.outline_resort),
                    outline_when=args.outline_when,
                    stack_cap=bool(args.stack_cap), **kw)


def load_data(args, out):
    """`--data` as (train pool, held-out set), or (None, None) without one.

    Resolves the bin from `--pallet_cm` for an orders CSV, splits off
    `--holdout` of the pallets with the run's seed, writes the split beside
    the run, and records the file's fingerprint in `args` so `--resume`
    can refuse a file that changed under it.
    """
    if not args.data:
        return None, None
    if args.data.lower().endswith(".csv"):
        if args.pallet_cm:
            args.bin = orders_bin(args.pallet_cm, args.cell_cm)
        seqs, ids = load_orders(args.data, args.cell_cm, args.bin,
                                rot=args.rot, box_scale=args.box_scale,
                                box_round=args.box_round,
                                box_pad_m=args.box_pad_m)
    else:
        seqs = np.load(args.data)
        ids = [str(i) for i in range(len(seqs))]
    with open(args.data, "rb") as f:
        args.data_sha1 = hashlib.sha1(f.read()).hexdigest()
    if not 0.0 <= args.holdout < 1.0:
        raise ValueError(f"--holdout is a share of the pallets in [0, 1), "
                         f"got {args.holdout}")
    order = np.random.default_rng(args.seed).permutation(len(seqs))
    n_held = int(round(len(seqs) * args.holdout))
    if args.holdout and n_held == 0:
        n_held = 1
    if n_held >= len(seqs):
        raise ValueError(f"--holdout {args.holdout} leaves no training pallets "
                         f"out of {len(seqs)}")
    held, trn = np.sort(order[:n_held]), np.sort(order[n_held:])
    for name, part in (("train_ids.txt", trn), ("holdout_ids.txt", held)):
        with open(os.path.join(out, name), "w") as f:
            f.write("".join(ids[i] + "\n" for i in part))
    args.n_train, args.n_holdout = len(trn), len(held)
    print(f"[{args.name}] data {args.data}: {len(trn)} training pallets, "
          f"{len(held)} held out, bin {args.bin}", flush=True)
    # with no hold-out the evals score the training pallets themselves
    return seqs[trn], (seqs[held] if len(held) else seqs[trn])


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
    pool, held_out = load_data(args, out)
    json.dump(vars(args), open(os.path.join(out, "args.json"), "w"), indent=1)

    env = make_env(args, args.seed, pool)
    pack, attacker, mixer = build(args, device)
    if args.compile:
        for net in (pack, attacker, mixer):
            net.forward = torch.compile(net.forward, dynamic=True)
    start, resumed, best = 1, None, -1.0
    if args.resume and os.path.exists(os.path.join(out, "last.pt")):
        resumed = torch.load(os.path.join(out, "last.pt"), map_location=device,
                             weights_only=False)
        # the last runs were two experiments glued together because the data
        # changed under a resume; never again for a data file
        was = resumed.get("args", {}).get("data_sha1")
        if was != args.data_sha1:
            raise SystemExit(
                f"[{args.name}] refusing to resume: the run was trained on "
                f"{resumed.get('args', {}).get('data') or 'the random generator'} "
                f"(sha1 {was}) but --data is now {args.data or 'the random generator'} "
                f"(sha1 {args.data_sha1}).  Start a new --name instead.")
        pack.load_state_dict(resumed["pack"])
        attacker.load_state_dict(resumed["attacker"])
        mixer.load_state_dict(resumed["mixer"])
        start = resumed["it"] + 1
        # without the old high-water mark the first save after a resume always
        # "wins" and overwrites a better best.pt
        best = resumed.get("best", -1.0)
        print(f"[{args.name}] resuming at iteration {start} "
              f"(best so far {best:+.4f})", flush=True)
        # rows past last.pt describe iterations a kill threw away and this run
        # is about to redo; left in, the log goes back in time at `start`
        p = os.path.join(out, "log.jsonl")
        if os.path.exists(p):
            with open(p) as f:
                lines = f.readlines()
            def before(ln):
                try:
                    return json.loads(ln).get("it", 0) < start
                except ValueError:
                    return False            # a line cut short by the kill
            keep = [ln for ln in lines if before(ln)]
            if len(keep) < len(lines):
                with open(p, "w") as f:
                    f.writelines(keep)
    if args.init and resumed is not None:
        print(f"[{args.name}] --init ignored: resuming from last.pt", flush=True)
    elif args.init:
        # every net the checkpoint has, so `select` keeps its trained selector;
        # optimiser and iteration count start fresh
        init = torch.load(args.init, map_location=device, weights_only=False)
        for key, net in (("pack", pack), ("attacker", attacker), ("mixer", mixer)):
            if key in init:
                net.load_state_dict(init[key])
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
    stop = StopAsked(args.name)

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
                      type_constraint=bool(args.type_constraint),
                      type_rule=args.type_rule,
                      soft_mix=bool(args.soft_mix),
                      outline_resort=bool(args.outline_resort),
                      outline_when=args.outline_when,
                      stack_cap=bool(args.stack_cap), seqs=held_out)
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
        due = it % args.save_every == 0 or it == args.iters
        if due or stop.asked:
            # an attacker is best when the held-out packer does worst
            # the 1.0 default keeps an attacker from banking a best.pt on a
            # save that lands before its first eval; every other algorithm
            # scores 0.0 there and is beaten by the first real number
            score = (-nom.get("att_util", 1.0) if args.algo == "attack"
                     else _held(nom))
            # a stop between save points saves last.pt only: best.pt is
            # chosen at the save points, on the evals that preceded them
            improved = due and score > best
            best = max(best, score) if due else best
            ck = {"pack": pack.state_dict(), "attacker": attacker.state_dict(),
                  "mixer": mixer.state_dict(), "args": vars(args), "it": it,
                  "best": best,
                  "opt": {"pack": ppo_pack.opt.state_dict(),
                          "attacker": ppo_att.opt.state_dict(),
                          "mixer": ppo_mix.opt.state_dict()}}
            save_atomic(ck, os.path.join(out, "last.pt"))
            if improved:
                save_atomic(ck, os.path.join(out, "best.pt"))
        if stop.asked:
            # the log row too, so the page and the dashboard end where it did
            if not (it % args.log_every == 0 or it == args.iters):
                rec = {"it": it, "util": u, "items": k, "t": time.time() - t0,
                       **nom, "stopped": True}
                log.write(json.dumps(rec) + "\n"); log.flush()
            prog.write(f"[{args.name}] stopped at iteration {it}; last.pt saved "
                       f"-- --resume carries on from {it + 1}")
            break
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
    p.add_argument("--name", required=True,
                   help="run folder under runs/: args.json, log.jsonl and the "
                        "best.pt / last.pt checkpoints go there")
    p.add_argument("--config", default=None,
                   help="YAML file of defaults to merge over config.yaml; "
                        "the same as AR2L_CONFIG=")
    p.add_argument("--algo", default=t["algo"],
                   choices=["pct", "cppo", "rarl", "rfmdp", "exact", "approx",
                            "attack", "select"],
                   help="what is trained: pct = packer alone; cppo = packer on "
                        "its worst episodes (CVaR); rarl = packer only against "
                        "an adversarial box order; rfmdp = packer with a "
                        "pessimistic value; exact / approx = AR2L, a mix of the "
                        "normal and the adversarial order; attack = only an "
                        "attacker against a fixed packer; select = a selector "
                        "picks the box *for* the packer, from the n_pick in reach")
    p.add_argument("--nb", type=int, default=t["nb"], help="window N_B: how many upcoming boxes the agent sees")
    p.add_argument("--n_pick", type=int, default=t["n_pick"],
                   help="how many of the N_B are within reach; the rest are "
                        "preview only (default: all of them)")
    p.add_argument("--alpha", type=float, default=t["alpha"], help="exact / approx: how much weight the adversarial order gets "
                        "against the normal one (0 = normal only, 1 = full AR2L)")
    p.add_argument("--rho", type=float, default=t["rho"], help="approx / rfmdp: radius of the uncertainty set around the "
                        "normal box order (total variation); bigger = more pessimistic")
    p.add_argument("--dist_coef", type=float, default=t["dist_coef"],
                   help="exact: weight of the loss that keeps the mixture order "
                        "close to both the normal and the adversarial order")
    p.add_argument("--data", default=t["data"],
                   help="train on these pallets (an orders .csv or an "
                        "instance .npy); empty = the random generator")
    p.add_argument("--holdout", type=float, default=t["holdout"],
                   help="with --data: share of pallets kept for the held-out "
                        "evals that select best.pt")
    p.add_argument("--order_random", type=float, default=t["order_random"],
                   help="with --data: randomise each drawn pallet's box order, "
                        "0 = file order, 1 = fully random")
    add_cm_args(p)
    p.add_argument("--cvar_q", type=float, default=t["cvar_q"],
                   help="cppo: the share of worst episodes the update learns "
                        "from, e.g. 0.5 = the worse half")
    p.add_argument("--iters", type=int, default=t["iters"],
                   help="PPO iterations to train: one rollout + update each")
    p.add_argument("--n_env", type=int, default=t["n_env"],
                   help="pallets packed in parallel per rollout; more = "
                        "steadier gradients, more GPU memory")
    p.add_argument("--T", type=int, default=t["T"],
                   help="rollout length: steps each pallet runs per iteration")
    p.add_argument("--bin", type=extent, default=e["bin"],
                   help="pallet size in grid cells, WxLxH (an int = a cube).  With an "
                        "orders .csv as --data, --pallet_cm sets it instead")
    p.add_argument("--n_items", type=int, default=e["n_items"],
                   help="boxes per episode from the random generator; the "
                        "episode usually ends earlier, when nothing fits")
    p.add_argument("--size_lo", type=int, default=e["size_lo"],
                   help="smallest box side, in grid cells")
    p.add_argument("--max_l", type=int, default=e["max_l"],
                   help="most candidate placements the packer is shown per step; "
                        "extra ones are dropped, so raise it if that happens")
    p.add_argument("--max_c", type=int, default=e["max_c"],
                   help="room reserved for packed boxes per pallet; grows by itself "
                        "if a pallet holds more, so it only affects speed")
    p.add_argument("--size_hi", type=extent, default=e["size_hi"],
                   help="largest box side in grid cells, an int or WxLxH per axis; "
                        "random generator only")
    p.add_argument("--lr", type=float, default=o["lr"],
                   help="learning rate of the Adam optimiser")
    p.add_argument("--gamma", type=float, default=o["gamma"],
                   help="discount factor; 1 = undiscounted, as in the paper")
    p.add_argument("--lam", type=float, default=o["lam"],
                   help="GAE lambda: 0 = one-step advantage, 1 = full return")
    p.add_argument("--epochs", type=int, default=o["epochs"],
                   help="passes PPO makes over each rollout")
    p.add_argument("--minibatches", type=int, default=o["minibatches"],
                   help="each rollout is split into this many minibatches per pass")
    p.add_argument("--ent_coef", type=float, default=o["ent_coef"],
                   help="entropy bonus: higher keeps the policy exploring longer")
    p.add_argument("--ent_final", type=float, default=o["ent_final"],
                   help="lower the attacker's entropy bonus linearly from --ent_coef "
                        "to this by the last iteration; empty = keep it fixed")
    p.add_argument("--clip", type=float, default=o["clip"],
                   help="PPO clip: how far one update may move the policy from the "
                        "one that collected the rollout (0.2 = 20%%)")
    p.add_argument("--vf_coef", type=float, default=o["vf_coef"],
                   help="weight of the value (critic) loss next to the policy loss")
    p.add_argument("--max_grad", type=float, default=o["max_grad"],
                   help="gradients are scaled down to at most this norm; guards "
                        "against one bad batch wrecking the weights")
    p.add_argument("--width", type=int, default=m["width"],
                   help="transformer embedding width")
    p.add_argument("--heads", type=int, default=m["heads"],
                   help="attention heads per block; must divide --width")
    p.add_argument("--layers", type=int, default=m["layers"],
                   help="attention blocks in the transformer")
    p.add_argument("--c_temp", type=float, default=m["c_temp"],
                   help="pointer-head temperature (c in Eq. 28): the logits are "
                        "squashed into [-c, c], so bigger = sharper choices")
    p.add_argument("--init", default=None, help="start a new run from this checkpoint's weights (packer, "
                        "selector, attacker); iterations and optimiser start fresh")
    p.add_argument("--freeze_pack", action="store_true",
                   help="do not train the packer: keep its weights (from "
                        "--init) fixed and let it act greedily")
    p.add_argument("--heur_pack", default=None,
                   help="attack a heuristic packer instead of a network: dbl, hmm, "
                        "lsah, bmf, onlinebph or macs")
    p.add_argument("--seed", type=int, default=r["seed"],
                   help="random seed for the boxes and the network init")
    p.add_argument("--device", default=r["device"],
                   help="cuda, cuda:1, ... or cpu (slow)")
    # torch.compile with dynamic node counts miscompiles the pointer head for
    # some N_B and shows up as an illegal memory access; off unless asked for
    p.add_argument("--compile", type=int, choices=(0, 1), default=r["compile"],
                   help="1 = torch.compile the nets; can miscompile the pointer "
                        "head, so 0 unless you are testing it")
    p.add_argument("--resume", action="store_true",
                   help="carry on the run of the same --name from its last.pt, "
                        "at the iteration where it stopped")
    p.add_argument("--stability", default=e["stability"], choices=["com", "cdrl"],
                   help="when a box rests stably: com = enough support area and "
                        "the centre of mass over it; cdrl = 60%% support or "
                        "all 4 corners supported")
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
    p.add_argument("--type_constraint", type=int, choices=(0, 1), default=e["type_constraint"],
                   help="0 keeps type_id in the state but lets any box be "
                        "stacked on any other")
    p.add_argument("--type_rule", choices=TYPE_RULES,
                   default=e.get("type_rule", "touch"),
                   help="touch: a box may not rest on a foreign type; "
                        "column: nor be anywhere over one, however deep; "
                        "mixed_touch / mixed_column: the cell's blue / white "
                        "/ brown rules on either reading")
    p.add_argument("--soft_mix", type=int, choices=(0, 1), default=e.get("soft_mix", 0),
                   help="1 under the mixed rules: a box's own fallback "
                        "onto another type becomes the station's last resort, "
                        "brown on blue gets one too, (c) is dropped")
    p.add_argument("--outline_resort", type=int, choices=(0, 1),
                   default=e.get("outline_resort", 0),
                   help="1 under the mixed rules: when every other step "
                        "leaves nothing, the rules again over every point "
                        "on a box's or the pallet's outline")
    p.add_argument("--outline_when", choices=OUTLINE_WHEN,
                   default=e.get("outline_when", "station"),
                   help="station: the outline step opens when no box within "
                        "reach has a place; box: for each box with none")
    p.add_argument("--stack_cap", type=int, choices=(0, 1), default=e["stack_cap"],
                   help="0 lifts the stack-height limit over small boxes of "
                        "env.stack_cap_types")
    p.add_argument("--type_embed", type=int, default=m["type_embed"],
                   help="size of the learned vector that tells the net each box's "
                        "type; 0 = the net does not see types")
    p.add_argument("--ems", type=int, choices=(0, 1, 2, 3), default=e["ems"],
                   help="candidate filter: 0 every loading position, 1 EMS "
                        "corners, 2 height-map corner cells, 3 EMS | corner")
    p.add_argument("--eval_every", type=int, default=r["eval_every"],
                   help="iterations between held-out evals; the best one is "
                        "saved as best.pt")
    p.add_argument("--eval_inst", type=int, default=r["eval_inst"],
                   help="pallets scored in each held-out eval")
    p.add_argument("--log_every", type=int, default=r["log_every"],
                   help="iterations between printed log lines")
    p.add_argument("--progress", choices=("auto", "on", "off"),
                   default=r["progress"],
                   help="progress bar on stderr; auto = only when attached "
                        "to a terminal")
    p.add_argument("--save_every", type=int, default=r["save_every"],
                   help="iterations between last.pt saves")
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
    # the arm filter and the stack cap read the cell's real size from
    # CFG["eval"]; without this a --box_scale/--cell_cm that differs from the
    # file has them see a pallet `box_scale` times too big
    CFG["eval"]["cell_cm"], CFG["eval"]["box_scale"] = args.cell_cm, args.box_scale
    CFG["eval"]["box_pad_m"] = args.box_pad_m
    # resolve it here rather than in the env, so `args.json` records the rule
    # the run was trained under instead of a bare `null`
    if args.min_support is None:
        args.min_support = float(CFG["env"]["min_support"])
    args.data = args.data or None      # "" in the config is the generator too
    args.data_sha1 = None
    train(args)


if __name__ == "__main__":
    main()
