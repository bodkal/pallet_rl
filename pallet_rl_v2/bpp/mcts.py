"""AlphaGo adaptation for the online 3D-BPP (Sec. V-C).

Differences from vanilla AlphaGo that are handled here:
  * single-player and score-based with intermediate rewards -> a node's value is
    the return of the *whole episode*; it is turned into a Go-like score signal
    by subtracting the return of a baseline policy (argmax of p_theta) that is
    evaluated on the sequence before the simulations start (Sec. V-C).
  * leaves are evaluated by rollout with action sampling rather than by the value
    network (Table IV: 83.4% vs 64.7% on CUT-1).

All `B` episodes of a batch advance in lock-step and share one node arena, so the
tree descent, the expansion, the backup and every network query are single
vectorised operations over the whole batch instead of per-game Python loops.
"""
import math
import numpy as np

from .env import (combo_dims, compute_masks, features, step_v,
                  sample_rows, new_state)

NEG = np.float32(-1e30)


class Arena:
    """Node storage for `nT` trees, `cap` nodes each (root + one per simulation)."""

    def __init__(self, nT, cap, cfg):
        self.cap, self.cfg = cap, cfg
        N = nT * cap
        A = cfg.A
        self.P = np.zeros((N, A), np.float32)
        self.Q = np.full((N, A), NEG, np.float32)
        self.Nv = np.zeros((N, A), np.float32)
        self.Wv = np.zeros((N, A), np.float32)
        self.sn = np.zeros(N, np.float32)
        self.child = np.full((N, A), -1, np.int32)
        self.hm = np.zeros((N, cfg.W, cfg.L), np.int16)
        self.buf = np.zeros((N, cfg.b, 3), np.int8)
        self.nxt = np.zeros(N, np.int32)
        self.G = np.zeros(N, np.float32)
        self.dims = np.zeros((N, cfg.M, 3), np.int8)
        self.zmap = np.zeros((N, cfg.M, cfg.W, cfg.L), np.int16)
        self.term = np.zeros(N, bool)
        self.free = None

    def reset(self, nT):
        n = nT * self.cap
        self.P[:n] = 0.0
        self.Q[:n] = NEG
        self.Nv[:n] = 0.0
        self.Wv[:n] = 0.0
        self.sn[:n] = 0.0
        self.child[:n] = -1
        self.term[:n] = False
        self.roots = np.arange(nT, dtype=np.int64) * self.cap
        self.free = self.roots + 1

    def alloc(self, trees):
        """One fresh slot for each tree id in `trees`."""
        s = self.free[trees].copy()
        self.free[trees] += 1
        return s


class Runner:
    def __init__(self, cfg, ev, n_sims=100, c_puct=2.5, dir_alpha=0.3,
                 dir_eps=0.25, scale=1.0, seed=0):
        self.cfg, self.ev = cfg, ev
        self.n_sims, self.c_puct = n_sims, c_puct
        self.dir_alpha, self.dir_eps, self.scale = dir_alpha, dir_eps, scale
        self.rng = np.random.default_rng(seed)
        self._arena = None

    # ---------------------------------------------------------------- rollouts
    def rollout(self, hm, buf, nxt, seqs, lens, dims, zmap, probs, greedy=False):
        """Lock-step batched rollouts to the end of every episode.
        Arrays are consumed (copies expected). Returns the (n,) rollout returns."""
        cfg, ev = self.cfg, self.ev
        n = hm.shape[0]
        tot = np.zeros(n, np.float32)
        live = np.nonzero(probs.sum(1) > 0)[0]
        if live.size == 0:
            return tot
        hm, buf, nxt = hm[live], buf[live], nxt[live]
        dims, zmap, probs = dims[live], zmap[live], probs[live]
        sq, ln = seqs[live], lens[live]
        while True:
            acts = probs.argmax(1) if greedy else sample_rows(probs, self.rng)
            tot[live] += step_v(hm, buf, nxt, sq, ln, dims, zmap, acts, cfg)
            dims = combo_dims(buf, cfg.k)
            mask, zmap = compute_masks(hm, dims, cfg.H)
            mf = mask.reshape(len(live), -1)
            ok = mf.any(1)
            if not ok.all():
                if not ok.any():
                    return tot
                live = live[ok]
                hm, buf, nxt = hm[ok], buf[ok], nxt[ok]
                dims, zmap, mf = dims[ok], zmap[ok], mf[ok]
                sq, ln = sq[ok], ln[ok]
            probs = ev.probs(features(hm, buf, cfg.H), mf)

    def greedy_return(self, seqs, lens):
        """Return of the baseline policy (argmax p_theta) on each sequence."""
        cfg = self.cfg
        hm, buf, nxt = new_state(seqs, lens, cfg)
        dims = combo_dims(buf, cfg.k)
        mask, zmap = compute_masks(hm, dims, cfg.H)
        mf = np.ascontiguousarray(mask.reshape(len(lens), -1))
        probs = np.zeros((len(lens), cfg.A), np.float32)
        ok = np.nonzero(mf.any(1))[0]
        if ok.size:
            probs[ok] = self.ev.probs(features(hm[ok], buf[ok], cfg.H), mf[ok])
        return self.rollout(hm, buf, nxt, seqs, lens, dims, zmap, probs, greedy=True)

    # ------------------------------------------------------------------- MCTS
    def _expand(self, ar, slots, hm, buf, nxt, G, seqs_g, lens_g, root_noise=False):
        """Fill `slots` with the given states; returns (terminal_mask, probs)."""
        cfg = self.cfg
        n = len(slots)
        dims = combo_dims(buf, cfg.k)
        mask, zmap = compute_masks(hm, dims, cfg.H)
        mf = np.ascontiguousarray(mask.reshape(n, -1))
        ok = mf.any(1)
        ar.hm[slots] = hm; ar.buf[slots] = buf; ar.nxt[slots] = nxt
        ar.G[slots] = G; ar.dims[slots] = dims; ar.zmap[slots] = zmap
        ar.term[slots] = ~ok
        probs = np.zeros((n, cfg.A), np.float32)
        sub = np.nonzero(ok)[0]
        if sub.size:
            probs[sub] = self.ev.probs(features(hm[sub], buf[sub], cfg.H), mf[sub])
            p = probs[sub]
            if root_noise and self.dir_eps > 0:
                for i in range(len(sub)):
                    v = np.nonzero(mf[sub[i]])[0]
                    if len(v) > 1:
                        nz = self.rng.dirichlet([self.dir_alpha] * len(v)).astype(np.float32)
                        p[i, v] = (1 - self.dir_eps) * p[i, v] + self.dir_eps * nz
            ar.P[slots[sub]] = p
            ar.Q[slots[sub]] = np.where(mf[sub], np.float32(0.0), NEG)
        return ok, probs, dims, zmap, mf

    def play(self, seqs, lens, temp=1.0, train=True, collect=True,
             temp_moves=10 ** 9, trace=False, belief=None, belief_lens=None,
             lookahead=None, item_lo=2, item_hi=5):
        """`belief` lets the search plan against a different future than the one
        the environment delivers: expansion and rollout consume `belief`, while
        the move actually played consumes `seqs`.  Default None = the two agree,
        which is the known-sequence setting the paper uses during training.
        The buffer is always filled from `seqs` at the root, so what the agent has
        genuinely been shown stays true - only what it *assumes* comes next changes.

        `lookahead=N` builds that belief automatically and per move: the next N
        items past the buffer are the real ones, everything beyond is resampled
        uniformly from [item_lo, item_hi]^3.  N=0 means the search knows nothing
        past the buffer; None (the default) means it knows the whole stream, which
        is the known-sequence setting the paper uses for training.
        """
        cfg, ev = self.cfg, self.ev
        B = seqs.shape[0]
        cap = self.n_sims + 1
        if self._arena is None or self._arena.cap != cap or len(self._arena.sn) < B * cap:
            self._arena = Arena(B, cap, cfg)
        ar = self._arena

        hm, buf, nxt = new_state(seqs, lens, cfg)
        base = self.greedy_return(seqs, lens)
        G = np.zeros(B, np.float32)
        items = np.zeros(B, np.int32)
        done = np.zeros(B, bool)
        rec = [{'hm': [], 'buf': [], 'pi': [], 'trace': []} for _ in range(B)]
        inv_scale = np.float32(1.0 / self.scale)
        cp = np.float32(self.c_puct)
        move = 0

        while not done.all():
            gidx = np.nonzero(~done)[0]
            nT = len(gidx)
            ar.reset(nT)
            roots = ar.roots[:nT]
            ok, _, dims0, zmap0, mf0 = self._expand(
                ar, roots, hm[gidx], buf[gidx], nxt[gidx], G[gidx],
                seqs[gidx], lens[gidx], root_noise=train)
            done[gidx[~ok]] = True
            if not ok.any():
                break
            if not ok.all():                      # drop finished games, re-run move
                gidx = gidx[ok]
                nT = len(gidx)
                ar.reset(nT)
                roots = ar.roots[:nT]
                ok, _, dims0, zmap0, mf0 = self._expand(
                    ar, roots, hm[gidx], buf[gidx], nxt[gidx], G[gidx],
                    seqs[gidx], lens[gidx], root_noise=train)
            bg = base[gidx]
            sq_g, ln_g = seqs[gidx], lens[gidx]
            if belief is not None:
                bl_g, bln_g = belief[gidx], belief_lens[gidx]
            elif lookahead is None:
                bl_g, bln_g = sq_g, ln_g
            else:
                # true up to the horizon, resampled past it.  The horizon moves
                # with the buffer, so this is rebuilt at every real move.
                T = sq_g.shape[1]
                horizon = nxt[gidx] + lookahead
                unknown = np.arange(T)[None, :] >= horizon[:, None]
                rnd = self.rng.integers(item_lo, item_hi + 1,
                                        size=(nT, T, 3)).astype(np.int8)
                bl_g = np.where(unknown[:, :, None], rnd, sq_g)
                bln_g = np.full(nT, T, np.int32)   # the guessed stream never runs dry
            tid = np.arange(nT)

            for _ in range(self.n_sims):
                # ---------------- descent (vectorised over trees) ------------
                cur = roots.copy()
                alive = np.ones(nT, bool)
                lv_t, lv_n, lv_a = [], [], []
                res_kind = np.zeros(nT, np.int8)   # 0 expand, 1 terminal
                res_node = np.zeros(nT, np.int64)
                res_act = np.zeros(nT, np.int64)
                while True:
                    ti = np.nonzero(alive)[0]
                    if ti.size == 0:
                        break
                    nn = cur[ti]
                    u = (ar.Q[nn] + (cp * np.sqrt(ar.sn[nn] + 1.0))[:, None]
                         * ar.P[nn] / (1.0 + ar.Nv[nn]))
                    a = u.argmax(1)
                    lv_t.append(ti); lv_n.append(nn); lv_a.append(a)
                    ch = ar.child[nn, a]
                    leaf = ch < 0
                    if leaf.any():
                        lt = ti[leaf]
                        res_kind[lt] = 0
                        res_node[lt] = nn[leaf]
                        res_act[lt] = a[leaf]
                        alive[lt] = False
                    go = ~leaf
                    if go.any():
                        gt, gc = ti[go], ch[go]
                        tm = ar.term[gc]
                        if tm.any():
                            res_kind[gt[tm]] = 1
                            res_node[gt[tm]] = gc[tm]
                            alive[gt[tm]] = False
                        cur[gt] = gc
                        alive[gt[tm]] = False
                # ---------------- expansion ---------------------------------
                vals = np.zeros(nT, np.float32)
                term_t = np.nonzero(res_kind == 1)[0]
                if term_t.size:
                    vals[term_t] = np.tanh((ar.G[res_node[term_t]] - bg[term_t]) * inv_scale)
                ex = np.nonzero(res_kind == 0)[0]
                if ex.size:
                    par, act = res_node[ex], res_act[ex]
                    nhm = ar.hm[par].copy()
                    nbuf = ar.buf[par].copy()
                    nnxt = ar.nxt[par].copy()
                    r = step_v(nhm, nbuf, nnxt, bl_g[ex], bln_g[ex],
                               ar.dims[par], ar.zmap[par], act, cfg)
                    nG = ar.G[par] + r
                    slots = ar.alloc(ex)
                    ar.child[par, act] = slots
                    okx, prx, dx, zx, mfx = self._expand(
                        ar, slots, nhm, nbuf, nnxt, nG, sq_g[ex], ln_g[ex])
                    tot = nG.copy()
                    sub = np.nonzero(okx)[0]
                    if sub.size:
                        tot[sub] += self.rollout(
                            nhm[sub].copy(), nbuf[sub].copy(), nnxt[sub].copy(),
                            bl_g[ex][sub], bln_g[ex][sub], dx[sub], zx[sub], prx[sub])
                    vals[ex] = np.tanh((tot - bg[ex]) * inv_scale)
                # ---------------- backup (vectorised per level) -------------
                for ti, nn, a in zip(lv_t, lv_n, lv_a):
                    v = vals[ti]
                    ar.Nv[nn, a] += 1.0
                    ar.Wv[nn, a] += v
                    ar.Q[nn, a] = ar.Wv[nn, a] / ar.Nv[nn, a]
                    ar.sn[nn] += 1.0

            # -------------------- pick the real move ------------------------
            Nroot = ar.Nv[roots]                                   # (nT, A)
            if temp <= 1e-6 or not train or move >= temp_moves:
                chosen = Nroot.argmax(1)
            else:
                w = np.where(mf0, Nroot ** (1.0 / temp), 0.0)
                s = w.sum(1, keepdims=True)
                w = np.where(s > 0, w / np.maximum(s, 1e-12), mf0 / mf0.sum(1, keepdims=True))
                chosen = sample_rows(w.astype(np.float32), self.rng)
            if collect:
                s = Nroot.sum(1, keepdims=True)
                pi = np.where(s > 0, Nroot / np.maximum(s, 1e-12),
                              mf0 / np.maximum(mf0.sum(1, keepdims=True), 1)).astype(np.float32)
                for t in range(nT):
                    g = gidx[t]
                    rec[g]['hm'].append(hm[g].copy())
                    rec[g]['buf'].append(buf[g].copy())
                    rec[g]['pi'].append(pi[t].copy())
            if trace:                       # geometry of the move, for the viewers
                mm = chosen // cfg.WL
                pp = chosen % cfg.WL
                tx, ty = pp // cfg.L, pp % cfg.L
                td = dims0[np.arange(nT), mm]
                tz = zmap0[np.arange(nT), mm, tx, ty]
                for t in range(nT):
                    rec[gidx[t]]['trace'].append(dict(
                        x=int(tx[t]), y=int(ty[t]), z=int(tz[t]),
                        w=int(td[t, 0]), l=int(td[t, 1]), h=int(td[t, 2]),
                        slot=int(mm[t]) // (cfg.k + 1), orient=int(mm[t]) % (cfg.k + 1),
                        item=[int(v) for v in buf[gidx[t], int(mm[t]) // (cfg.k + 1)]]))
            shm, sbuf, snxt = hm[gidx].copy(), buf[gidx].copy(), nxt[gidx].copy()
            r = step_v(shm, sbuf, snxt, sq_g, ln_g, dims0, zmap0, chosen, cfg)
            hm[gidx], buf[gidx], nxt[gidx] = shm, sbuf, snxt
            G[gidx] += r
            items[gidx] += 1
            move += 1

        for g in range(B):
            rec[g]['G'] = float(G[g])
            rec[g]['base'] = float(base[g])
            rec[g]['items'] = int(items[g])
            rec[g]['util'] = float(G[g]) / 10.0
        return rec
