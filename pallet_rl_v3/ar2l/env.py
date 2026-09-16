"""Batched online 3D-BPP environment with the PCT state representation.

AR2L Sec. 3 / A.4.  A state is the triple

    C_t  the N_C already packed items      (x, y, z, sx, sy, sz)
    B_t  the N_B observable incoming items (sx, sy, sz)
    L_t  the N_L feasible placements generated for the *first* item of B_t

and the packing action picks one l in L_t.  A placement is an (orientation,
x, y) triple: with `rot = 2` the leading item is offered both as (w, l, h) and
as (l, w, h), which is the discrete setting of the paper.  The attacker
instead acts on (C_t, B_t) and moves one observable item to the front of the
conveyor.

The whole batch of environments is advanced with array operations; nothing
here loops over the batch.  Feasibility for every loading position is obtained
from windowed max/sum over the height map, with the support count read off a
base-32 histogram (a window holds at most 25 cells and heights are < 32, so the
digits of sum(32**h) never carry).
"""
from __future__ import annotations

import numpy as np

# Two stability criteria, selected with `stability=`:
#
#   "com"  (default) the projection of the item's centre of mass must fall
#          inside the convex hull of its contact cells -- tested, as is
#          standard on a grid, by requiring the support to span the centre
#          along each axis.  This is the criterion that reproduces the
#          heuristic baselines of AR2L Table 1; see README.
#   "cdrl" the conservative area/corner rule written down by Zhao et al.
#          (AAAI 2021, Sec. 3.1).
SUPPORT_RULES = ((0.60, 4), (0.80, 3), (0.95, 0))
MIN_SUPPORT = 0.0  # optional extra contact-area floor on top of "com"

MAX_SIDE = 5  # item sides are <= S/2 = 5 in the discrete setting

# The support count is read off a base-32 histogram packed into one int64:
# a column of height h contributes 32**h, so the encoding needs 5*h <= 63.
# Past that `1 << 5*h` returns 0 and the stability test silently accepts
# unsupported placements, so refuse the bin rather than report nonsense.
MAX_HEIGHT = 63 // 5


def sample_items(rng, shape, lo=1, hi=5):
    """Item sizes, i.i.d. uniform on {lo..hi} per axis (125 types for 1..5)."""
    return rng.integers(lo, hi + 1, size=shape + (3,), dtype=np.int16)


def _win_max(m, axis, S, kmax=None):
    """Running window max of every width 1..kmax along one axis.

    `kmax` defaults to the full bin, which is what the EMS enumeration needs;
    the feasibility sweep only ever asks for widths up to the largest item
    side.
    """
    kmax = S if kmax is None else kmax
    out = np.empty((kmax,) + m.shape, m.dtype)
    out[0] = m
    for k in range(1, kmax):
        cur = out[k - 1].copy()
        if axis == 1:
            cur[:, : S - k] = np.maximum(out[k - 1][:, : S - k], m[:, k:])
        else:
            cur[:, :, : S - k] = np.maximum(out[k - 1][:, :, : S - k], m[:, :, k:])
        out[k] = cur
    return out


def _win_sum(c, axis, S, kmax):
    """Running window sum of every width 1..kmax along one axis."""
    out = np.empty((kmax,) + c.shape, c.dtype)
    out[0] = c
    for k in range(1, kmax):
        cur = out[k - 1].copy()
        if axis == 1:
            cur[:, : S - k] = out[k - 1][:, : S - k] + c[:, k:]
        else:
            cur[:, :, : S - k] = out[k - 1][:, :, : S - k] + c[:, :, k:]
        out[k] = cur
    return out


class BPPBatch:
    """`n_env` independent bins stepped in lockstep."""

    def __init__(self, n_env, S=10, nb=1, n_items=150, max_c=80, max_l=120,
                 size_lo=1, size_hi=5, seed=0, ems=True, stability="com",
                 rot=2):
        if stability != "com" and (S > MAX_HEIGHT or size_hi > MAX_SIDE):
            raise ValueError(
                f"stability={stability!r} counts contact area with a base-32 "
                f"histogram packed into an int64, which needs bin height <= "
                f"{MAX_HEIGHT} and item sides <= {MAX_SIDE}; got {S} and "
                f"{size_hi}. stability='com' compares window maxima instead "
                f"and has no size limit.")
        self.n_env, self.S, self.nb, self.ems = n_env, S, nb, ems
        self.stability, self.rot = stability, int(rot)
        self.n_items, self.max_c, self.max_l = n_items, max_c, max_l
        self.size_lo, self.size_hi = size_lo, size_hi
        self.bin_vol = float(S ** 3)
        self.rng = np.random.default_rng(seed)
        self._ar = np.arange(n_env)
        self.reset()

    # ---------------------------------------------------------------- reset
    def _invalidate(self, hmap=True):
        """Drop the cached sweeps.  `hmap=False` when only the item changed."""
        self._pos_dirty = True
        if hmap:
            self._sweep_cache = None
            self._ems_cache = None

    def reset(self, seqs=None):
        n, S = self.n_env, self.S
        self.hmap = np.zeros((n, S, S), np.int16)
        self.packed = np.zeros((n, self.max_c, 6), np.float32)
        self.n_packed = np.zeros(n, np.int32)
        self.volume = np.zeros(n, np.float32)
        self.done = np.zeros(n, bool)
        if seqs is None:
            self.seq = sample_items(self.rng, (n, self.n_items),
                                    self.size_lo, self.size_hi)
        else:
            self.seq = np.asarray(seqs, np.int16).copy()
        self.head = np.zeros(n, np.int32)
        # windows are only ever indexed by a footprint side, and `reset` may be
        # handed a sequence the constructor knew nothing about
        self.side = max(int(self.size_hi), int(self.seq[..., :2].max()))
        self._invalidate()
        return self.obs()

    def reset_done(self):
        """Re-roll only the finished bins (used during rollout collection)."""
        idx = np.nonzero(self.done)[0]
        if idx.size == 0:
            return
        self.hmap[idx] = 0
        self.packed[idx] = 0
        self.n_packed[idx] = 0
        self.volume[idx] = 0
        self.seq[idx] = sample_items(self.rng, (idx.size, self.n_items),
                                     self.size_lo, self.size_hi)
        self.side = max(self.side, int(self.seq[..., :2].max()))
        self.head[idx] = 0
        self.done[idx] = False
        self._invalidate()

    # ------------------------------------------------------------- conveyor
    def window(self):
        """(B, nb, 3) observable items and their validity mask."""
        off = self.head[:, None] + np.arange(self.nb)[None, :]
        valid = off < self.n_items
        items = self.seq[self._ar[:, None], np.minimum(off, self.n_items - 1)]
        return np.where(valid[..., None], items, 0), valid

    def permute(self, idx):
        """Move observable item `idx` (B,) to the front of the conveyor."""
        idx = np.asarray(idx, np.int64)
        if np.all(idx == 0):
            return
        off = self.head[:, None] + np.arange(self.nb)[None, :]
        off = np.minimum(off, self.n_items - 1)
        win = self.seq[self._ar[:, None], off].copy()          # (B, nb, 3)
        order = np.argsort(np.where(np.arange(self.nb)[None, :] == idx[:, None],
                                    -1, np.arange(self.nb)[None, :]), axis=1)
        self.seq[self._ar[:, None], off] = win[self._ar[:, None], order]
        self._invalidate(hmap=False)

    # ------------------------------------------------- empty maximal spaces
    def _ems_list(self):
        """Every empty maximal space, as one flat table over the whole batch.

        Items are always dropped, so the *reachable* free volume is exactly
        `{(x, y, z) : z >= hmap[x, y]}`.  Its maximal boxes are therefore the
        footprints `[x0, x0+wx) x [y0, y0+wy)` whose floor `h` -- the window
        max of the height map -- cannot be lowered by widening the footprint,
        each stacked up to the lid.  At `S = 10` there are only 55 x 55
        candidate footprints per bin, so scoring them all and keeping the
        maximal ones gives the exact EMS list with none of the bookkeeping
        (or the ordering artefacts) of an incremental difference process.

        Returns `(env, x0, y0, wx, wy, floor)`, one entry per space.
        """
        if self._ems_cache is not None:
            return self._ems_cache
        S, h = self.S, self.hmap
        big = np.int16(S + 1)                 # a wall is taller than anything
        ax = _win_max(h, 1, S)                # ax[wx-1, n, x0, y]  max over x
        ay = _win_max(h, 2, S)                # ay[wy-1, n, x, y0]  max over y
        g, wyr = np.arange(S), np.arange(1, S + 1)
        oky = g[None, :] + wyr[:, None] <= S               # (wy, y0)
        left = np.full_like(ay, big); left[:, :, 1:] = ay[:, :, :-1]
        parts = []
        for wx in range(1, S + 1):
            a = ax[wx - 1]                                 # (n, x0, y)
            flr = _win_max(a, 2, S)                        # (wy, n, x0, y0)
            # one-cell extensions: each must raise the floor, or hit a wall
            right = np.full_like(ay, big)
            if wx < S:
                right[:, :, : S - wx] = ay[:, :, wx:]
            back = np.full_like(a, big); back[:, :, 1:] = a[:, :, :-1]
            front = np.full((S,) + a.shape, big, a.dtype)
            for k in range(1, S):
                front[k - 1, :, :, : S - k] = a[:, :, k:]
            keep = ((left > flr) & (right > flr) & (back[None] > flr)
                    & (front > flr)
                    & (g[None, None, :, None] + wx <= S)
                    & oky[:, None, None, :])
            f = np.flatnonzero(keep)
            if f.size:
                iwy, env, x0, y0 = np.unravel_index(f, keep.shape)
                parts.append((env, x0, y0, np.full(f.size, wx, np.int32),
                              iwy + 1, flr.reshape(-1)[f].astype(np.int32)))
        self._ems_cache = tuple(np.concatenate(c) for c in zip(*parts))
        return self._ems_cache

    def _ems_corners(self, dims):
        """(n, S, S): the item fits flush into a bottom corner of some EMS.

        The four bottom corners of every empty maximal space wide enough and
        tall enough to take the item -- the leaf-node set of PCT and AR2L.
        """
        S = self.S
        env, x0, y0, wx, wy, floor = self._ems_list()
        sx, sy, sz = dims[env, 0], dims[env, 1], dims[env, 2]
        ok = (wx >= sx) & (wy >= sy) & (floor + sz <= S)
        env, x0, y0 = env[ok], x0[ok], y0[ok]
        xs = (x0, x0 + (wx - sx)[ok])
        ys = (y0, y0 + (wy - sy)[ok])
        out = np.zeros((self.n_env, S, S), bool)
        for x in xs:
            for y in ys:
                out[env, x, y] = True
        return out

    # --------------------------------------------------------- feasibility
    def _sweeps(self):
        """Height-map sweeps that depend on the bin but not on the item."""
        if self._sweep_cache is None:
            k = self.side
            self._sweep_cache = (_win_max(self.hmap, 2, self.S, k),
                                 _win_max(self.hmap, 1, self.S, k))
        return self._sweep_cache

    def _feas_one(self, dims):
        """Feasible (x, y) grid and landing height for one orientation."""
        S = self.S
        sx, sy, sz = dims[:, 0], dims[:, 1], dims[:, 2]
        ar, grid1 = self._ar, np.arange(S)
        my, mx = self._sweeps()

        myd = my[sy - 1, ar]                          # max over the y extent
        mxy = _win_max(myd, 1, S, self.side)          # ... and x sub-windows
        z = mxy[sx - 1, ar].astype(np.int32)

        if self.stability == "com":
            mxd = mx[sx - 1, ar]
            myx = _win_max(mxd, 2, S, self.side)      # sub-windows along y

            def span(stack, size, axis):
                """Support both at or before, and at or after, the centre.

                Every cell of a sub-window of the footprint is at most `z`, the
                max over the whole footprint, so "this sub-window rests on the
                contact layer" is exactly "its max equals `z`".
                """
                a = (size - 1) // 2                    # last index of the near half
                b = -((1 - size) // 2)                 # first index of the far half
                near = stack[a, ar] == z
                idx = np.minimum(grid1[None, :] + b[:, None], S - 1)
                far = stack[size - b - 1, ar]
                far = (far[ar[:, None], idx] if axis == 1
                       else far[ar[:, None, None], grid1[None, :, None],
                                idx[:, None, :]])
                return near & (far == z)

            stable = span(mxy, sx, 1) & span(myx, sy, 2)
            if MIN_SUPPORT > 0:
                stable &= self._support_ratio(dims, z) >= MIN_SUPPORT
        else:
            ratio = self._support_ratio(dims, z)
            ix = np.minimum(grid1[None, :] + (sx - 1)[:, None], S - 1)
            iy = np.minimum(grid1[None, :] + (sy - 1)[:, None], S - 1)
            hx = self.hmap[ar[:, None], ix]
            hy = self.hmap[ar[:, None, None], grid1[None, :, None], iy[:, None, :]]
            hxy = hx[ar[:, None, None], grid1[None, :, None], iy[:, None, :]]
            ncor = ((self.hmap == z).astype(np.int8) + (hx == z)
                    + (hy == z) + (hxy == z))
            stable = np.zeros_like(ratio, bool)
            for r_min, c_min in SUPPORT_RULES:
                stable |= (ratio > r_min) & (ncor >= c_min)

        inx = grid1[None, :, None] + sx[:, None, None] <= S
        iny = grid1[None, None, :] + sy[:, None, None] <= S
        fits = z + sz[:, None, None] <= S
        feas = stable & inx & iny & fits
        if self.ems:
            feas &= self._ems_corners(dims)
        return feas, z

    def _support_ratio(self, dims, z):
        """Fraction of the footprint that rests on the contact layer.

        Only the area-threshold rules need an actual count rather than a
        "does it touch" test, so this is the one place the base-32 histogram
        -- and its bin-height and item-side limits -- is still used.
        """
        S, ar = self.S, self._ar
        sx, sy = dims[:, 0], dims[:, 1]
        side = int(max(sx.max(), sy.max()))
        if S > MAX_HEIGHT or side > MAX_SIDE:
            raise ValueError(
                f"the area-threshold support count needs bin height <= "
                f"{MAX_HEIGHT} and item sides <= {MAX_SIDE}; this bin is {S} "
                f"with sides up to {side}. stability='com' needs no count and "
                f"has no such limit.")
        code = np.left_shift(np.int64(1), 5 * self.hmap.astype(np.int64))
        cy = _win_sum(code, 2, S, self.side)[sy - 1, ar]
        cxy = _win_sum(cy, 1, S, self.side)[sx - 1, ar]
        count = np.right_shift(cxy, 5 * z.astype(np.int64)) & np.int64(31)
        return count.astype(np.float32) / (sx * sy)[:, None, None]

    def _positions(self):
        """Feasibility and landing height per (orientation, x, y)."""
        if not self._pos_dirty:
            return self._pos_cache
        item = self.seq[self._ar, np.minimum(self.head, self.n_items - 1)]
        orients = [item] if self.rot < 2 else [item, item[:, [1, 0, 2]]]
        feas, zs = [], []
        for r, d in enumerate(orients):
            f, z = self._feas_one(d)
            if r:   # a square footprint is the same placement turned round
                f = f & (item[:, 0] != item[:, 1])[:, None, None]
            feas.append(f); zs.append(z)
        feas = np.stack(feas, 1)                      # (n, R, S, S)
        alive = ~self.done & (self.head < self.n_items)
        feas &= alive[:, None, None, None]
        self._pos_cache = (feas, np.stack(zs, 1), np.stack(orients, 1))
        self._pos_dirty = False
        return self._pos_cache

    # ------------------------------------------------------------ observation
    def _trim_c(self):
        """Padding tokens cost attention; keep only as many as the batch uses."""
        nc = max(int(self.n_packed.max()), 1)
        mask = np.arange(nc)[None, :] < self.n_packed[:, None]
        return self.packed[:, :nc], mask

    def obs_cb(self):
        """(C, B) only -- skips the feasibility sweep the attacker never uses."""
        win, wvalid = self.window()
        c, cmask = self._trim_c()
        return {"c": c, "c_mask": cmask,
                "b": win.astype(np.float32) / self.S, "b_mask": wvalid}

    def obs(self):
        feas, z, odims = self._positions()
        S, n = float(self.S), self.n_env
        flat = feas.reshape(n, -1)
        NL = int(min(self.max_l, flat.shape[1], max(flat.sum(1).max(), 1)))

        # keep the NL deepest candidates when more than NL are feasible
        order = np.lexsort((np.arange(flat.shape[1])[None, :].repeat(n, 0),
                            z.reshape(n, -1), ~flat), axis=1)[:, :NL]
        lmask = np.take_along_axis(flat, order, 1)
        zz = np.take_along_axis(z.reshape(n, -1), order, 1)
        rr, rem = order // (self.S * self.S), order % (self.S * self.S)
        xx, yy = rem // self.S, rem % self.S
        dims = odims[self._ar[:, None], rr]                   # (n, NL, 3)

        lnode = np.concatenate(
            [np.stack([xx, yy, zz], -1).astype(np.float32), dims], -1) / S
        lnode *= lmask[..., None]
        self._lxy = np.stack([xx, yy, rr], -1).astype(np.int32)
        self._lz = zz.astype(np.int32)
        self._ldim = dims.astype(np.int32)

        win, wvalid = self.window()
        c, cmask = self._trim_c()
        return {
            "c": c, "c_mask": cmask,
            "b": win.astype(np.float32) / S, "b_mask": wvalid,
            "l": lnode, "l_mask": lmask,
        }

    def candidates(self, b=0, k=None):
        """[(x, y, z, sx, sy, sz)] for one bin's live candidate list."""
        k = self._lxy.shape[1] if k is None else k
        return np.concatenate([self._lxy[b, :k, :2], self._lz[b, :k, None],
                               self._ldim[b, :k]], -1).tolist()

    def n_feasible(self):
        return self._positions()[0].reshape(self.n_env, -1).sum(1)

    # ------------------------------------------------------------------ step
    def step(self, action):
        """Place the leading item at L[action].  Returns (reward, done)."""
        act = np.asarray(action, np.int64)
        alive = ~self.done & (self.n_feasible() > 0)

        x, y = self._lxy[self._ar, act, 0], self._lxy[self._ar, act, 1]
        d = self._ldim[self._ar, act]
        sx, sy, sz = d[:, 0], d[:, 1], d[:, 2]
        zz = self._lz[self._ar, act]

        # raise the footprint; the per-env window makes a python loop cheapest
        grid = np.arange(self.S)
        inx = (grid[None, :] >= x[:, None]) & (grid[None, :] < (x + sx)[:, None])
        iny = (grid[None, :] >= y[:, None]) & (grid[None, :] < (y + sy)[:, None])
        foot = inx[:, :, None] & iny[:, None, :] & alive[:, None, None]
        self.hmap = np.where(foot, (zz + sz)[:, None, None].astype(np.int16),
                             self.hmap)

        S = float(self.S)
        slot = np.minimum(self.n_packed, self.max_c - 1)
        node = np.stack([x, y, zz, sx, sy, sz], 1).astype(np.float32) / S
        self.packed[self._ar, slot] = np.where(alive[:, None], node,
                                               self.packed[self._ar, slot])
        self.n_packed += alive
        vol = (sx * sy * sz).astype(np.float32)
        self.volume += vol * alive

        self.head += alive
        self._invalidate()

        reward = (vol / self.bin_vol) * alive
        # terminate when the new leading item has nowhere to go
        self.done |= ~alive
        self.done |= self.head >= self.n_items
        self.done |= self.n_feasible() == 0
        return reward.astype(np.float32), self.done.copy()

    # ----------------------------------------------------------------- stats
    def utilization(self):
        return self.volume / self.bin_vol
