"""One pallet, rebuilt from the boxes on it, and where the next box may go.

The state the policies read (AR2L Sec. 3 / A.4):

    C  the packed boxes          (x, y, z, sx, sy, sz) + type
    B  the conveyor window       (sx, sy, sz) + type, front first
    L  the legal placements of the front box, each with the type it rests on

Everything here is computed from the request: `Pallet(rules, arm, placed,
station)` lays the placed boxes down onto empty maps, then answers.  Nothing
is kept between requests.

Arrays keep a leading batch axis of size 1 so every rule stays one vectorised
sweep over the whole grid.  Results are cached per object: the bin caches
live as long as the object, the per-item caches are dropped (or carried) by
`permute`.

Height-map layers, all (1, Lx, Ly):
    hmap  top of each column           tmap  type of the top box (-1 floor)
    smap  bit per type in the column   cmap  stack-height cap per column
    gmap  column solid to the floor    omap  box outlines, on cell corners
"""
from __future__ import annotations

import contextlib

import numpy as np

from .rules import (BLUE, BROWN, EMS_BOTH, EMS_CORNER, EMS_OFF, EMS_PRUNE_S,
                    EMS_SPACES, IF, NO, NO_CAP, SUPPORT_EPS, SUPPORT_RULES,
                    TOP, TYPE_FLOOR, WHITE)
from .sweeps import covered, widest, win_max, win_maxcount

_UNSET = object()


class Pallet:
    """`placed`: (N, 7) int rows (x, y, z, sx, sy, sz, type) in cells, in the
    order the boxes were placed; z is the bottom as reported.
    `station`: (M, 4) int rows (sx, sy, sz, type), front of the conveyor
    first; at most `rules.nb` of them.

    `landed` holds the z each placed box was laid at (its rest height on the
    boxes below it), so the caller can compare it with what was reported.
    """

    #: arm-checked cells per (box, orientation) `_arm_any` tries first, and
    #: how much that grows each round, until one is clear
    ARM_PROBE = 4
    ARM_GROW = 8
    #: the caches one pass of the rules fills; the outline step keeps its own
    PASS_CACHES = ("_rule_pick", "_hard_pick", "_slot_fz", "_slot_geo",
                   "_slot_soft")

    def __init__(self, rules, arm, placed, station):
        R = self.R = rules
        self.arm = arm
        self.n = 1
        self._ar = np.arange(1)
        self.Lx, self.Ly, self.Lz = R.Lx, R.Ly, R.Lz
        self.nb, self.n_pick = R.nb, R.n_pick
        self.k_pick = min(R.n_pick, R.nb)

        # ---- the conveyor window
        station = np.asarray(station, np.int16).reshape(-1, 4)[: self.nb]
        if not len(station):
            raise ValueError("no boxes at the station")
        win = np.zeros((1, self.nb, 4), np.int16)
        win[0, : len(station)] = station
        self.win = win
        self.valid = (np.arange(self.nb) < len(station))[None]

        # sweeps are built for windows up to the widest side asked of them
        side = int(station[:, :2].max())
        if R.mixed:
            side = max(side, R.small_blue[0], R.small_blue[1])
        self.sidex, self.sidey = min(side, self.Lx), min(side, self.Ly)

        # ---- the pallet maps, then the placed boxes on them
        shape = (1, self.Lx, self.Ly)
        self.hmap = np.zeros(shape, np.int16)
        self.tmap = np.full(shape, TYPE_FLOOR, np.int8)
        self.smap = np.zeros(shape, np.int32)
        self.cmap = np.full(shape, NO_CAP, np.int16)
        self.gmap = np.ones(shape, bool)
        self.omap = np.zeros((1, self.Lx + 1, self.Ly + 1), bool)
        self.omap[0, [0, -1], :] = self.omap[0, :, [0, -1]] = True

        placed = np.asarray(placed, np.int64).reshape(-1, 7)
        n_c = len(placed)
        self.packed = np.zeros((1, max(n_c, 1), 6), np.float32)
        self.ptype = np.zeros((1, max(n_c, 1)), np.int8)
        self.n_packed = n_c
        self.landed = np.zeros(n_c, np.int64)
        # a box can only land on boxes laid before it, so laying them down
        # lowest first rebuilds every map exactly; C keeps the request order
        for i in np.argsort(placed[:, 2], kind="stable"):
            x, y, _, sx, sy, sz, t = (int(v) for v in placed[i])
            z = self._lay(x, y, sx, sy, sz, t)
            self.landed[i] = z
            self.packed[0, i] = np.array([x, y, z, sx, sy, sz],
                                         np.float32) / R.scale
            self.ptype[0, i] = t

        # ---- caches
        self._sweep = self._ems = self._under = self._cap = None
        self._ground = None
        self._open = _UNSET
        self._type_cache, self._on_cache = {}, {}
        # arm verdicts by (size, x, y): index of the first base that works
        self._arm_keys = np.zeros(0, np.int64)
        self._arm_move = np.zeros(0, np.int8)
        self._ol = False
        self._clear_item_caches()

    def _lay(self, x, y, sx, sy, sz, t):
        """Put a box on footprint [x, x+sx) x [y, y+sy); -> where it lands."""
        R = self.R
        if not (0 <= x and x + sx <= self.Lx and 0 <= y and y + sy <= self.Ly):
            raise ValueError(f"box at cell ({x}, {y}) size ({sx}, {sy}) is "
                             f"off the {self.Lx} x {self.Ly} pallet")
        if not 0 <= t < R.n_types:
            raise ValueError(f"box type {t} is not in 0..{R.n_types - 1}")
        foot = np.s_[0, x:x + sx, y:y + sy]
        h = self.hmap[foot]
        z = int(h.max())
        gap = h < z                     # air under the box: overhung cells
        self.gmap[foot] &= ~gap
        self.hmap[foot] = z + sz
        self.tmap[foot] = t
        self.smap[foot] |= np.int32(1) << t
        o = self.omap[0]
        o[x, y:y + sy + 1] = o[x + sx, y:y + sy + 1] = True
        o[x:x + sx + 1, y] = o[x:x + sx + 1, y + sy] = True
        if R.stack_cap:
            # a box bridging a gap starts a new stack there: no cap from below
            cap = self.cmap[foot]
            cap[gap] = NO_CAP
            allow = int(R.stack_allow[min(sx, sy, len(R.stack_allow) - 1)])
            if t in R.stack_types and allow < NO_CAP:
                np.minimum(cap, min(z + sz + allow, NO_CAP), out=cap)
            self.cmap[foot] = cap
        return z

    def _clear_item_caches(self):
        """Drop what depends on the boxes in the window (not on the pallet)."""
        self._pos = None
        self._pick_cache = self._rule_pick = self._hard_pick = None
        # per window slot: ([feas], [z]) per orientation, with / before the
        # arm filter, and by soft mix's last resort
        self._slot_fz, self._slot_geo, self._slot_soft = {}, {}, {}
        self._ol_caches = None

    # ============================================================ observation
    def obs_cb(self):
        """(C, B) for the station picker."""
        c, ctype, cmask = self._trim_c()
        return {"c": c, "c_mask": cmask, "c_type": self._emb_type(ctype),
                "b": self.win[..., :3].astype(np.float32) / self.R.scale,
                "b_mask": self.valid, "b_type": self._emb_type(self.win[..., 3]),
                "b_pick": self.pick_mask()}

    def obs(self):
        """(C, B, L) for the packer.  L is the `max_l` lowest legal
        placements of the front box; `self.cand` holds them as
        (x, y, orientation, z, sx, sy, sz) rows."""
        feas, z, odims, tunder = self._positions()
        scale = self.R.scale
        flat = feas.reshape(1, -1)
        NL = int(min(self.R.max_l, flat.shape[1], max(flat.sum(1).max(), 1)))
        # feasible first, then lowest, then grid order
        order = np.lexsort((np.arange(flat.shape[1])[None, :],
                            z.reshape(1, -1), ~flat), axis=1)[:, :NL]
        lmask = np.take_along_axis(flat, order, 1)
        zz = np.take_along_axis(z.reshape(1, -1), order, 1)
        tt = np.take_along_axis(tunder.reshape(1, -1), order, 1)
        cells = self.Lx * self.Ly
        rr, rem = order // cells, order % cells
        xx, yy = rem // self.Ly, rem % self.Ly
        dims = odims[self._ar[:, None], rr]
        lnode = np.concatenate(
            [np.stack([xx, yy, zz], -1).astype(np.float32), dims], -1) / scale
        lnode *= lmask[..., None]
        self.cand = np.concatenate([np.stack([xx, yy, rr, zz], -1), dims],
                                   -1)[0].astype(np.int64)
        lunder = np.where(lmask, tt, TYPE_FLOOR).astype(np.int8)
        out = self.obs_cb()
        out.update(l=lnode, l_mask=lmask, l_type=self._emb_type(lunder))
        return out

    def _trim_c(self):
        nc = max(self.n_packed, 1)
        mask = (np.arange(nc) < self.n_packed)[None]
        return self.packed[:, :nc], self.ptype[:, :nc], mask

    def _emb_type(self, t):
        """Type -> embedding row; the floor is the last row."""
        return np.where(t < 0, self.R.n_types, t).astype(np.int64)

    # =============================================================== conveyor
    def pick_mask(self):
        """(1, nb) boxes the picker may choose: within reach, and -- with
        `pick_feasible` -- with somewhere legal to go."""
        reach = self.valid & (np.arange(self.nb) < self.n_pick)[None]
        return reach & self._placeable() if self.R.pick_feasible else reach

    def permute(self, idx):
        """Bring station box `idx` to the front; the rest keep their order."""
        idx = int(idx)
        if idx == 0:
            return
        order = np.array([idx] + [j for j in range(self.nb) if j != idx])
        self.win = self.win[:, order]
        self.valid = self.valid[:, order]
        # the pallet is untouched, so each slot's answers move with its box
        pick, rule, hard = self._pick_cache, self._rule_pick, self._hard_pick
        slots, geo = self._slot_fz, self._slot_geo
        self._clear_item_caches()
        if idx >= self.k_pick or not self.valid[0, 0]:
            return
        if pick is not None:
            self._pick_cache = pick[:, order]
        if rule is not None:
            self._rule_pick = rule[:, order]
        if hard is not None:
            self._hard_pick = hard[:, order]
        k = self.k_pick
        self._slot_fz = {j: slots[order[j]] for j in range(k)
                         if order[j] in slots}
        self._slot_geo = {j: geo[order[j]] for j in range(k)
                          if order[j] in geo}

    def head_item(self):
        """(dims (1, 3), type (1,)) of the front box, int16 as stored."""
        return self.win[:, 0, :3], self.win[:, 0, 3]

    def _orients(self, dims):
        """As given, then turned 90 degrees (with `rot` 2)."""
        return [dims] if self.R.rot < 2 else [dims, dims[:, [1, 0, 2]]]

    # =========================================== which station boxes can go
    def _placeable(self):
        """(1, nb) does each box within reach have a placement?  By the
        rules, then -- where they leave nothing -- by the outline step."""
        if self._pick_cache is None:
            out = self._rule_placeable()
            gate = self._outline_gate()
            if gate.any():
                with self._outline():
                    out = out | (gate & self._rule_placeable())
            self._pick_cache = out
        return self._pick_cache

    def _rule_placeable(self):
        """The hard rules, then soft mix's last resort where the station is
        stuck."""
        if self._rule_pick is None:
            out = self._hard_placeable().copy()
            stuck = self._stuck()
            if stuck.any():
                k = self.k_pick
                out[:, :k] |= stuck[:, None] & self._soft_placeable(k)
            self._rule_pick = out
        return self._rule_pick

    def _hard_placeable(self):
        """By the normal rules or a box's own fallback.  Only existence is
        asked, so the arm checks as few cells as it takes (`_arm_any`)."""
        if self._hard_pick is not None:
            return self._hard_pick
        k = self.k_pick
        out = np.zeros((1, self.nb), bool)
        dims = np.maximum(self.win[:, :k, :3], 1)   # empty slots are masked
        todo = [i for i in range(k)
                if i not in self._slot_fz and i not in self._slot_geo]
        if todo:
            got = self._slot_grids([(dims[:, i], self.win[:, i, 3])
                                    for i in todo])
            self._slot_geo.update(zip(todo, got))
        grids, slot = [], []
        for i in range(k):
            if i in self._slot_fz:            # the arm-checked grid is known
                for f in self._slot_fz[i][0]:
                    out[:, i] |= f.any((1, 2))
                continue
            fs, zs = self._slot_geo[i]
            for d, f, z in zip(self._orients(dims[:, i]), fs, zs):
                grids.append((f, z, d))
                slot.append(i)
        if grids:
            hit = self._arm_any(grids)
            for g, i in enumerate(slot):
                out[:, i] |= hit[:, g]
        out[:, :k] &= self.valid[:, :k]
        self._hard_pick = out
        return out

    def _stuck(self):
        """(1,) soft mix steps in: no box within reach can go by the hard
        rules.  In the outline pass, only where the plain pass was stuck too."""
        if not self.R.soft_on:
            return np.zeros(1, bool)
        alive = self._ol_stuck.copy() if self._ol else np.ones(1, bool)
        if not alive.any():
            return alive
        return alive & ~self._hard_placeable()[:, :self.k_pick].any(1)

    def last_resort(self):
        """(1,) the front box goes by soft mix's last resort."""
        stuck = self._stuck()
        if not stuck.any():
            return stuck
        return stuck & self.R.soft_types[self.head_item()[1].astype(np.int64)]

    def _soft_slot(self, i, dims, types):
        """Slot i's grids (before the arm) by soft mix's last resort."""
        if i not in self._slot_soft:
            fz = [self._feas_one(d, types, arm=False, tier=3)
                  for d in self._orients(dims)]
            self._slot_soft[i] = ([f for f, _ in fz], [z for _, z in fz])
        return self._slot_soft[i]

    def _soft_placeable(self, k):
        """(1, k) placeable by soft mix's last resort, the arm included."""
        dims = np.maximum(self.win[:, :k, :3], 1)
        types = self.win[:, :k, 3].astype(np.int64)
        out = np.zeros((1, k), bool)
        grids, slot = [], []
        for i in range(k):
            if not self.R.soft_types[types[:, i]].any():
                continue
            fs, zs = self._soft_slot(i, dims[:, i], self.win[:, i, 3])
            for d, f, z in zip(self._orients(dims[:, i]), fs, zs):
                grids.append((f, z, d))
                slot.append(i)
        if grids:
            hit = self._arm_any(grids)
            for g, i in enumerate(slot):
                out[:, i] |= hit[:, g]
        return out & self.valid[:, :k] & self.R.soft_types[types]

    def _geo_slot(self, i, dims, types):
        if i not in self._slot_geo:
            self._slot_geo[i] = self._slot_grids([(dims, types)])[0]
        return self._slot_geo[i]

    def _slot_grids(self, items):
        """([feas], [z]) per orientation, before the arm, for each
        (dims, types).  A box with fallback rules goes by them where the
        normal rules leave it nothing in any orientation, the arm included."""
        out, wait, grids = [], {}, []
        for j, (dims, types) in enumerate(items):
            orients = self._orients(dims)
            if self._fallback_bins(types).any():
                geo = [self._feas_geo(d) for d in orients]
                fs = [g & self._rules_ok(d, z, types, 1, full)
                      for d, (g, z, full) in zip(orients, geo)]
                zs = [z for _, z, _ in geo]
                wait[j] = (len(grids), orients, types, geo)
            else:
                fz = [self._feas_one(d, types, arm=False) for d in orients]
                fs, zs = [f for f, _ in fz], [z for _, z in fz]
            out.append((fs, zs))
            grids += [(f, z, d) for d, f, z in zip(orients, fs, zs)]
        if not wait:
            return out
        # everything in one arm batch; the verdicts are kept for later asks
        hit = self._arm_any(grids)
        for j, (g0, orients, types, geo) in wait.items():
            if not hit[:, g0:g0 + len(orients)].any():
                fs, zs = out[j]
                out[j] = ([g & self._rules_ok(d, z, types, 2, full)
                           for d, (g, z, full) in zip(orients, geo)], zs)
        return out

    def _fallback_bins(self, types):
        """(1,) does a box of `types` have fallback rules: the mixed ones,
        or (b) for blue, (w) for white."""
        R, t = self.R, np.asarray(types, np.int64)
        fb = np.zeros(1, bool)
        if R.mixed and R.typed:
            fb = R.tables[1][t]
        if R.blue_on:
            fb = fb | (t == BLUE)
        if R.white_on:
            fb = fb | (t == WHITE)
        return fb

    # ======================================================== outline step
    def _outline_gate(self):
        """(1, nb) boxes within reach that go on to the outline step."""
        out = np.zeros((1, self.nb), bool)
        if not self.R.outline_on:
            return out
        k = self.k_pick
        none = ~self._rule_placeable()[:, :k]
        if self.R.outline_when == "station":
            none = none.all(1, keepdims=True)
        out[:, :k] = none & self.valid[:, :k]
        return out

    @contextlib.contextmanager
    def _outline(self):
        """Run the rules again with the outline positions added; the pass
        keeps its own caches.  Soft mix there waits for the station to be
        stuck in the plain pass too."""
        stuck = (~self._rule_placeable()[:, :self.k_pick].any(1)
                 if self.R.soft_on else np.zeros(1, bool))
        mine = {k: getattr(self, k) for k in self.PASS_CACHES}
        theirs = self._ol_caches or dict(_rule_pick=None, _hard_pick=None,
                                         _slot_fz={}, _slot_geo={},
                                         _slot_soft={})
        for k, v in theirs.items():
            setattr(self, k, v)
        self._ol, self._ol_stuck = True, stuck
        try:
            yield
        finally:
            self._ol_caches = {k: getattr(self, k) for k in self.PASS_CACHES}
            for k, v in mine.items():
                setattr(self, k, v)
            self._ol = False

    # ================================================= the front box's grid
    def _positions(self):
        """(feas, z, dims, type under), each (1, R, Lx, Ly) / (1, R, 3)."""
        if self._pos is not None:
            return self._pos
        item, types = self.head_item()
        orients = self._orients(item)
        raw = self._head_grids(item, types, orients)
        if self.R.outline_on:
            late = ~np.any([f.any((1, 2)) for f, _ in raw], 0)
            if late.any() and self.R.outline_when == "station":
                late &= self._outline_gate()[:, 0]
            if late.any():
                with self._outline():
                    ol = self._head_grids(item, types, orients)
                raw = [(np.where(late[:, None, None], g, f), z)
                       for (f, z), (g, _) in zip(raw, ol)]
        feas, zs, tu = [], [], []
        for r, (d, (f, z)) in enumerate(zip(orients, raw)):
            if r:       # a square box turned is the same placement
                f = f & (item[:, 0] != item[:, 1])[:, None, None]
            feas.append(f)
            zs.append(z)
            tu.append(self._type_under(d, z))
        self._pos = (np.stack(feas, 1), np.stack(zs, 1), np.stack(orients, 1),
                     np.stack(tu, 1))
        return self._pos

    def _head_grids(self, item, types, orients):
        """[(feas, z)] per orientation for the front box, arm included: its
        hard grids, or soft mix's last resort where the station is stuck."""
        if 0 in self._slot_fz:
            raw = list(zip(*self._slot_fz[0]))
        else:
            fs, zs = self._geo_slot(0, item, types)
            if self.R.arm_collision:
                fs = self._arm_clear_many(list(zip(orients, zs, fs)))
            raw = list(zip(fs, zs))
            self._slot_fz[0] = ([f for f, _ in raw], [z for _, z in raw])
        late = self.last_resort()
        if late.any():
            fs, zs = self._soft_slot(0, item, types)
            if self.R.arm_collision:
                fs = self._arm_clear_many(list(zip(orients, zs, fs)))
            raw = [(np.where(late[:, None, None], s, f), z)
                   for (f, z), s in zip(raw, fs)]
        return raw

    # ========================================================= feasibility
    def _feas_one(self, dims, types=None, arm=True, tier=1, outline=None):
        """(feas, z) for one orientation: geometry, then the type rules at
        `tier` (1 normal, 2 own fallback, 3 last resort), then the arm."""
        feas, z, full = self._feas_geo(dims, outline)
        if types is not None:
            feas &= self._rules_ok(dims, z, types, tier, full)
        if self.R.arm_collision and arm:
            feas = self._arm_clear_many([(dims, z, feas)])[0]
        return feas, z

    def _feas_geo(self, dims, outline=None):
        """-> (feas, z, full) by geometry alone: fits, stands (stability),
        is a candidate position (`ems`), under the stack cap.  `full`: every
        cell of the footprint is at z (None when no rule asks)."""
        R = self.R
        outline = self._ol if outline is None else outline
        Lx, Ly, Lz = self.Lx, self.Ly, self.Lz
        sx, sy, sz = dims[:, 0], dims[:, 1], dims[:, 2]
        ar, gx, gy = self._ar, np.arange(Lx), np.arange(Ly)
        my, myc, mx = self._sweeps()
        # a side too long for the axis still indexes a window that exists;
        # inx / iny below drop it
        cx = np.minimum(sx, self.sidex)
        cy = np.minimum(sy, self.sidey)
        kx, ky = widest(cx, self.sidex), widest(cy, self.sidey)

        myd = my[cy - 1, ar]
        if myc is None:
            mxy, cxy = win_max(myd, 1, Lx, kx), None
        else:
            mxy, cxy = win_maxcount(myd, myc[cy - 1, ar], 1, Lx, kx)
        z = mxy[cx - 1, ar].astype(np.int32)

        if R.stability == "com":
            # support on both sides of the centre along each axis
            myx = win_max(mx[cx - 1, ar], 2, Ly, ky)

            def span(stack, size, axis):
                L = Lx if axis == 1 else Ly
                g = gx if axis == 1 else gy
                a = (size - 1) // 2               # last cell of the near half
                b = -((1 - size) // 2)            # first cell of the far half
                near = stack[a, ar] == z
                idx = np.minimum(g[None, :] + b[:, None], L - 1)
                far = stack[size - b - 1, ar]
                far = (far[ar[:, None], idx] if axis == 1
                       else far[ar[:, None, None], gx[None, :, None],
                                idx[:, None, :]])
                return near & (far == z)

            stable = span(mxy, cx, 1) & span(myx, cy, 2)
            if R.min_support > 0:
                stable &= (self._support_ratio(dims, cxy, cx)
                           >= R.min_support - SUPPORT_EPS)
        else:
            ratio = self._support_ratio(dims, cxy, cx)
            ix = np.minimum(gx[None, :] + (sx - 1)[:, None], Lx - 1)
            iy = np.minimum(gy[None, :] + (sy - 1)[:, None], Ly - 1)
            hx = self.hmap[ar[:, None], ix]
            hy = self.hmap[ar[:, None, None], gx[None, :, None], iy[:, None, :]]
            hxy = hx[ar[:, None, None], gx[None, :, None], iy[:, None, :]]
            ncor = ((self.hmap == z).astype(np.int8) + (hx == z)
                    + (hy == z) + (hxy == z))
            stable = np.zeros_like(ratio, bool)
            for r_min, c_min in SUPPORT_RULES:
                stable |= (ratio > r_min) & (ncor >= c_min)
        if R.min_total_support > 0:
            stable &= self._ground_ok(dims, z)

        inx = gx[None, :, None] + sx[:, None, None] <= Lx
        iny = gy[None, None, :] + sy[:, None, None] <= Ly
        fits = z + sz[:, None, None] <= Lz
        feas = stable & inx & iny & fits
        if R.ems != EMS_OFF:
            feas &= self._space(dims, z, outline)
        if R.stack_cap:
            feas &= self._cap_ok(dims, z)
        full = None
        if R.mixed or R.blue_on:
            full = cxy[cx - 1, ar] == (sx.astype(np.int32) * sy)[:, None, None]
        return feas, z, full

    def _sweeps(self):
        """(my, myc, mx): y-window max, its count (None if unused), x-window
        max -- the item-independent half of every footprint query."""
        if self._sweep is None:
            if self.R.needs_count:
                my, myc = win_maxcount(self.hmap, None, 2, self.Ly, self.sidey)
            else:
                my, myc = win_max(self.hmap, 2, self.Ly, self.sidey), None
            self._sweep = (my, myc, win_max(self.hmap, 1, self.Lx, self.sidex))
        return self._sweep

    def _footprint_max(self, by, dims):
        """Max over each footprint of a map already swept along y (`by`)."""
        cx = np.minimum(dims[:, 0], self.sidex)
        cy = np.minimum(dims[:, 1], self.sidey)
        return win_max(by[cy - 1, self._ar], 1, self.Lx,
                       widest(cx, self.sidex))[cx - 1, self._ar]

    def _support_ratio(self, dims, cxy, cx):
        """Share of the footprint's cells at the landing height."""
        count = cxy[cx - 1, self._ar]
        return count.astype(np.float32) / (dims[:, 0] * dims[:, 1])[:, None, None]

    def _contact(self, dims):
        """How many footprint cells reach the landing height."""
        my, myc, _ = self._sweeps()
        cx = np.minimum(dims[:, 0], self.sidex)
        cy = np.minimum(dims[:, 1], self.sidey)
        _, cxy = win_maxcount(my[cy - 1, self._ar], myc[cy - 1, self._ar], 1,
                              self.Lx, widest(cx, self.sidex))
        return cxy[cx - 1, self._ar]

    def _full(self, dims):
        """Every footprint cell reaches the landing height."""
        base = dims[:, 0].astype(np.int32) * dims[:, 1]
        return self._contact(dims) == base[:, None, None]

    def _ground_ok(self, dims, z):
        """At least `min_total_support` of the base rests on columns solid
        to the floor.  Coded 2*h + g, the footprint max is 2z+1 exactly when
        such a cell is under it, and its count is how many."""
        if self._ground is None:
            code = self.hmap * np.int16(2) + self.gmap
            self._ground = win_maxcount(code, None, 2, self.Ly, self.sidey)
        by, byc = self._ground
        sx, sy, ar = dims[:, 0], dims[:, 1], self._ar
        cx = np.minimum(sx, self.sidex)
        cy = np.minimum(sy, self.sidey)
        m, c = win_maxcount(by[cy - 1, ar], byc[cy - 1, ar], 1, self.Lx,
                            widest(cx, self.sidex))
        m, c = m[cx - 1, ar], c[cx - 1, ar]
        grounded = np.where(m == 2 * np.asarray(z, np.int32) + 1, c, 0)
        ratio = grounded.astype(np.float32) / (sx * sy)[:, None, None]
        return ratio >= self.R.min_total_support - SUPPORT_EPS

    def _cap_ok(self, dims, z):
        """No cell the box rests on has reached its stack cap."""
        if self._cap is None:
            full = self.hmap >= self.cmap
            self._cap = (win_max(np.where(full, self.hmap, -1).astype(np.int16),
                                 2, self.Ly, self.sidey)
                         if full.any() else False)
        if self._cap is False:
            return True
        return self._footprint_max(self._cap, dims) < z

    def _type_under(self, dims, z):
        """(1, Lx, Ly) int8: the type a footprint lands on, or the floor.
        Coded h * (T+1) + (t+1), the footprint max carries the type of the
        highest cell in its remainder."""
        T = self.R.n_types
        if T == 1:
            return np.where(z > 0, np.int8(0), np.int8(TYPE_FLOOR))
        if self._under is None:
            code = np.where(self.hmap > 0,
                            self.hmap.astype(np.int32) * (T + 1) + self.tmap + 1,
                            -1)
            self._under = win_max(code, 2, self.Ly, self.sidey)
        m = self._footprint_max(self._under, dims)
        return np.where(m < 0, np.int8(TYPE_FLOOR),
                        (m % (T + 1) - 1).astype(np.int8))

    # ========================================================== type rules
    def _rules_ok(self, dims, z, types, tier=1, full=None):
        """The type rule at `tier`, and (b) and (w) under the normal rules."""
        R = self.R
        ok = True
        if R.typed:
            ok = self._type_ok(dims, z, types, tier)
        if tier == 1 and R.blue_on:
            ok = ok & self._blue_ok(dims, z, types, full)
        if tier == 1 and R.white_on:
            ok = ok & self._white_ok(dims, z, types)
        return ok

    def _type_sweep(self, types, tier=1, strict=False):
        """y-swept map of the cells a box of `types` may not rest on: their
        height (touch), or a height nothing reaches (column).  `strict` also
        shuts the near-the-lid (TOP) pairs."""
        key = (types.tobytes(), tier, strict)
        if key not in self._type_cache:
            R, t = self.R, types.astype(np.int64)
            if R.mixed:
                rows = R.tables[0][tier - 1][t]
                forbid = (rows == NO) | (strict & (rows == TOP))
            else:
                forbid = t[:, None] != np.arange(R.n_types)
            if R.column:
                bits = (forbid.astype(np.int64)
                        << np.arange(R.n_types)).sum(1).astype(np.int32)
                bad = np.where((self.smap & bits[:, None, None]) != 0,
                               np.int16(np.iinfo(np.int16).max), np.int16(-1))
            else:
                on = forbid[self._ar[:, None, None], np.maximum(self.tmap, 0)]
                bad = np.where((self.hmap > 0) & on, self.hmap, np.int16(-1))
            self._type_cache[key] = win_max(bad, 2, self.Ly, self.sidey)
        return self._type_cache[key]

    def _type_ok(self, dims, z, types, tier=1):
        R = self.R
        ok = self._footprint_max(self._type_sweep(types, tier), dims) < z
        if not R.mixed:
            return ok
        t = np.asarray(types, np.int64)
        rows = R.tables[0][tier - 1][t]
        # (a): the box's top ends within top_gap of the lid
        near = (z + dims[:, 2].astype(np.int32)[:, None, None]
                >= self.Lz - R.top_gap)
        if (rows == TOP).any():
            strict = self._type_sweep(types, tier, strict=True)
            ok &= near | (self._footprint_max(strict, dims) < z)
        # white or brown on blue: (a), a large brown (c, not under soft
        # mix), or blue that no small blue box could still go on (d)
        big = (t == BROWN) & self._big_brown(dims) & (not R.soft_mix)
        need = (rows[:, BLUE] == IF) & ~big
        if need.any():
            ok &= ~need[:, None, None] | near | ~self._on_open_blue(dims, z)
        return ok

    def _blue_ok(self, dims, z, types, full=None):
        """(b): blue on blue rests with all of its base."""
        blue = np.asarray(types) == BLUE
        if not blue.any():
            return True
        full = self._full(dims) if full is None else full
        return ~blue[:, None, None] | ~self._on_type(dims, z, BLUE) | full

    def _white_ok(self, dims, z, types):
        """(w): white on white rests on at least the small white's base."""
        white = np.asarray(types) == WHITE
        if not white.any():
            return True
        need = np.minimum(self.R.white_area,
                          dims[:, 0].astype(np.int32) * dims[:, 1])
        return (~white[:, None, None] | ~self._on_type(dims, z, WHITE)
                | (self._contact(dims) >= need[:, None, None]))

    def _on_type(self, dims, z, u):
        """Is the footprint on type u: resting on it (touch), or with one
        anywhere under it (column)."""
        if u not in self._on_cache:
            if self.R.column:
                m = np.where((self.smap >> u) & 1 != 0, np.int16(0),
                             np.int16(-1))
            else:
                m = np.where((self.hmap > 0) & (self.tmap == u), self.hmap,
                             np.int16(-1))
            self._on_cache[u] = win_max(m, 2, self.Ly, self.sidey)
        top = self._footprint_max(self._on_cache[u], dims)
        return top >= 0 if self.R.column else top == z

    def _big_brown(self, dims):
        """(c): footprint at least area_frac of the small blue's."""
        sb = self.R.small_blue
        area = dims[:, 0].astype(np.int64) * dims[:, 1]
        return area >= self.R.area_frac * (sb[0] * sb[1]) - SUPPORT_EPS

    def _on_open_blue(self, dims, z):
        """Is the footprint on a blue cell a small blue box could still go
        on (rule d)?"""
        by = self._open_sweep()
        if by is None:
            return np.zeros(np.shape(z), bool)
        top = self._footprint_max(by, dims)
        return top >= 0 if self.R.column else top == z

    def _open_sweep(self):
        """y-swept heights of the open blue cells (-1 elsewhere), None if
        none: blue cells some legal small-blue placement rests on, by the
        `ems` candidates and the arm."""
        if self._open is not _UNSET:
            return self._open
        R = self.R
        sb = np.asarray(R.small_blue, np.int16)[None]
        blue = np.full(1, BLUE, np.int16)
        grids = []
        for r, d in enumerate(self._orients(sb)):
            if r and sb[0, 0] == sb[0, 1]:
                break
            f, z = self._feas_one(d, blue, arm=False, outline=False)
            f = f & (z > 0)
            grids.append((d, z, f & self._full(d) if R.blue_on else f))
        fs = ([f for _, _, f in grids] if not R.arm_collision
              else self._arm_clear_many(grids))
        cover = np.zeros((1, self.Lx, self.Ly), bool)
        for (d, z, _), f in zip(grids, fs):
            sx, sy = int(d[0, 0]), int(d[0, 1])
            if R.blue_on:
                cover |= covered(f, sx, sy)
                continue
            # without (b), only the covered cells at its landing height
            for v in np.unique(z[f]):
                cover |= covered(f & (z == v), sx, sy) & (self.hmap == v)
        cover &= self.tmap == BLUE
        self._open = (win_max(np.where(cover, self.hmap, np.int16(-1)), 2,
                              self.Ly, self.sidey) if cover.any() else None)
        return self._open

    # ======================================================= candidate set
    def _space(self, dims, z, outline=False):
        """Positions the `ems` mode offers (plus outline ones in that pass)."""
        R = self.R
        if R.ems == EMS_OFF or (outline and R.outline_area == "all"):
            return np.ones((1, self.Lx, self.Ly), bool)
        if R.ems == EMS_SPACES:
            cand = self._ems_corners(dims)
        elif R.ems == EMS_CORNER:
            cand = self._corner_mask(dims, z)
        else:
            cand = self._ems_corners(dims) | self._corner_mask(dims, z)
        if outline:
            cand |= self._outline_mask(dims)
        return cand

    def _ems_corners(self, dims):
        """The four bottom corners of every empty maximal space the box
        fits in (the PCT / AR2L leaf nodes)."""
        env, x0, y0, wx, wy, floor = self._ems_list()
        sx, sy, sz = dims[env, 0], dims[env, 1], dims[env, 2]
        ok = (wx >= sx) & (wy >= sy) & (floor + sz <= self.Lz)
        env, x0, y0 = env[ok], x0[ok], y0[ok]
        out = np.zeros((1, self.Lx, self.Ly), bool)
        for x in (x0, x0 + (wx - sx)[ok]):
            for y in (y0, y0 + (wy - sy)[ok]):
                out[env, x, y] = True
        return out

    def _ems_list(self):
        """Every empty maximal space: (env, x0, y0, wx, wy, floor)."""
        if self._ems is None:
            self._ems = (self._ems_pruned()
                         if self.Lx * self.Ly >= EMS_PRUNE_S ** 2
                         else self._ems_exhaustive())
        return self._ems

    def _ems_pruned(self):
        """Only rows / columns where the height map steps can bound a
        space, so only those footprint edges are tried."""
        Lx, Ly, H = self.Lx, self.Ly, self.hmap
        big = np.int16(self.Lz + 1)
        tru = np.ones((1, 1), bool)
        x0c = np.concatenate([tru, (H[:, :-1] > H[:, 1:]).any(2)], 1)
        x1c = np.concatenate([(H[:, 1:] > H[:, :-1]).any(2), tru], 1)
        y0c = np.concatenate([tru, (H[:, :, :-1] > H[:, :, 1:]).any(1)], 1)
        y1c = np.concatenate([(H[:, :, 1:] > H[:, :, :-1]).any(1), tru], 1)
        pe, px0, px1 = np.nonzero(x0c[:, :, None] & x1c[:, None, :]
                                  & np.triu(np.ones((Lx, Lx), bool)))
        qe, qy0, qy1 = np.nonzero(y0c[:, :, None] & y1c[:, None, :]
                                  & np.triu(np.ones((Ly, Ly), bool)))
        if not len(pe) or not len(qe):
            z = np.zeros(0, np.int64)
            return (z,) * 6
        ax = win_max(H, 1, Lx)
        ay = win_max(H, 2, Ly)
        pwx = px1 - px0 + 1
        a = ax[pwx - 1, pe, px0]
        wa = win_max(a[None], 2, Ly)[:, 0]
        # every x-span paired with every y-span
        qcount = np.bincount(qe, minlength=1)
        qstart = np.cumsum(qcount) - qcount
        cnt = qcount[pe]
        p = np.repeat(np.arange(len(pe)), cnt)
        q = (qstart[pe][p] + np.arange(cnt.sum())
             - np.repeat(np.cumsum(cnt) - cnt, cnt))
        e, x0, x1, wx = pe[p], px0[p], px1[p], pwx[p]
        y0, y1 = qy0[q], qy1[q]
        wy = y1 - y0 + 1
        flr = wa[wy - 1, p, y0]
        # maximal: every one-cell extension raises the floor or hits a wall
        lo = np.where(x0 > 0, ay[wy - 1, e, np.maximum(x0 - 1, 0), y0], big)
        hi = np.where(x1 + 1 < Lx,
                      ay[wy - 1, e, np.minimum(x1 + 1, Lx - 1), y0], big)
        bk = np.where(y0 > 0, a[p, np.maximum(y0 - 1, 0)], big)
        ft = np.where(y1 + 1 < Ly, a[p, np.minimum(y1 + 1, Ly - 1)], big)
        f = np.flatnonzero((lo > flr) & (hi > flr) & (bk > flr) & (ft > flr))
        f = f[np.lexsort((x1[f], q[f], x0[f], e[f]))]
        return (e[f], x0[f], y0[f], wx[f], wy[f], flr[f].astype(np.int32))

    def _ems_exhaustive(self):
        """Every footprint scored; for small pallets."""
        Lx, Ly, h = self.Lx, self.Ly, self.hmap
        big = np.int16(self.Lz + 1)
        ax = win_max(h, 1, Lx)
        ay = win_max(h, 2, Ly)
        gx, gy = np.arange(Lx), np.arange(Ly)
        oky = gy[None, :] + np.arange(1, Ly + 1)[:, None] <= Ly
        left = np.full_like(ay, big)
        left[:, :, 1:] = ay[:, :, :-1]
        parts = []
        for wx in range(1, Lx + 1):
            a = ax[wx - 1]
            flr = win_max(a, 2, Ly)
            right = np.full_like(ay, big)
            if wx < Lx:
                right[:, :, : Lx - wx] = ay[:, :, wx:]
            back = np.full_like(a, big)
            back[:, :, 1:] = a[:, :, :-1]
            front = np.full((Ly,) + a.shape, big, a.dtype)
            for k in range(1, Ly):
                front[k - 1, :, :, : Ly - k] = a[:, :, k:]
            keep = ((left > flr) & (right > flr) & (back[None] > flr)
                    & (front > flr) & (gx[None, None, :, None] + wx <= Lx)
                    & oky[:, None, None, :])
            f = np.flatnonzero(keep)
            if f.size:
                iwy, env, x0, y0 = np.unravel_index(f, keep.shape)
                parts.append((env, x0, y0, np.full(f.size, wx, np.int32),
                              iwy + 1, flr.reshape(-1)[f].astype(np.int32)))
        if not parts:
            z = np.zeros(0, np.int64)
            return (z,) * 6
        return tuple(np.concatenate(c) for c in zip(*parts))

    def _corner_mask(self, dims, z):
        """Some footprint corner sits in a corner cell: bounded on an x side
        and a y side by a wall, a taller column, or the drop edge of what it
        rests on."""
        Lx, Ly = self.Lx, self.Ly
        sx, sy = dims[:, 0].astype(np.int64), dims[:, 1].astype(np.int64)
        kx, ky = int(sx.max()), int(sy.max())
        p = np.full((1, Lx + kx + 2, Ly + ky + 2), self.Lz + 1, np.int16)
        p[:, 1:Lx + 1, 1:Ly + 1] = self.hmap
        win = np.lib.stride_tricks.sliding_window_view(p, (Lx, Ly), axis=(1, 2))

        def at(ox, oy):
            if np.isscalar(ox) and np.isscalar(oy):
                return p[:, ox:ox + Lx, oy:oy + Ly]
            return win[self._ar, ox, oy]

        z = z.astype(np.int16)
        out = np.zeros((1, Lx, Ly), bool)
        for cx, nbx in ((1, 0), (sx, sx + 1)):
            for cy, nby in ((1, 0), (sy, sy + 1)):
                hc = at(cx, cy)
                nx, ny, nd = at(nbx, cy), at(cx, nby), at(nbx, nby)
                on = hc == z
                tx, ty = nx > z, ny > z
                ex = tx | (on & (nx < z)) | (ty & (nd <= z))
                ey = ty | (on & (ny < z)) | (tx & (nd <= z))
                out |= ex & ey
        out &= np.arange(Lx)[None, :, None] + sx[:, None, None] <= Lx
        out &= np.arange(Ly)[None, None, :] + sy[:, None, None] <= Ly
        return out

    def _outline_mask(self, dims):
        """Some footprint corner lies on a box outline or the pallet edge."""
        Lx, Ly = self.Lx, self.Ly
        sx = np.clip(dims[:, 0].astype(np.int64), 1, Lx)
        sy = np.clip(dims[:, 1].astype(np.int64), 1, Ly)
        p = np.zeros((1, Lx + int(sx.max()), Ly + int(sy.max())), bool)
        p[:, :Lx + 1, :Ly + 1] = self.omap
        win = np.lib.stride_tricks.sliding_window_view(p, (Lx, Ly), axis=(1, 2))
        ar = self._ar
        return (p[:, :Lx, :Ly] | win[ar, sx, 0] | win[ar, 0, sy]
                | win[ar, sx, sy])

    # ================================================================= arm
    def _arm_clear_many(self, grids):
        """Each (dims, z, feas) grid less the placements the arm cannot
        make, all asked in one batch."""
        cells = [np.nonzero(f) for _, _, f in grids]
        if not sum(len(c[0]) for c in cells):
            return [f for _, _, f in grids]
        ok = self._arm_moves(
            *(np.concatenate([c[k] for c in cells]) for k in range(3)),
            np.concatenate([d[c[0]] for (d, _, _), c in zip(grids, cells)]),
            np.concatenate([z[c] for (_, z, _), c in zip(grids, cells)])) >= 0
        out, o = [], 0
        for (_, _, f), c in zip(grids, cells):
            g = f.copy()
            g[c] = ok[o:o + len(c[0])]
            o += len(c[0])
            out.append(g)
        return out

    def _arm_moves(self, bb, xx, yy, sizes, zz):
        """Per placement, the first robot base it can be made from, -1 none.
        Each distinct (size, x, y) is asked once per request."""
        sizes = sizes.astype(np.int64)
        if (sizes >= 1024).any():
            raise ValueError(f"box side {int(sizes.max())} is too large")
        key = ((((sizes[:, 0] * 1024 + sizes[:, 1]) * 1024 + sizes[:, 2])
                * self.Lx + xx) * self.Ly + yy)
        key, first, inv = np.unique(key, return_index=True, return_inverse=True)
        bb, xx, yy, sizes, zz = (a[first] for a in (bb, xx, yy, sizes, zz))
        pos = np.searchsorted(self._arm_keys, key)
        found = pos < len(self._arm_keys)
        found[found] = self._arm_keys[pos[found]] == key[found]
        move = np.full(len(bb), -1, np.int8)
        move[found] = self._arm_move[pos[found]]
        todo = np.flatnonzero(~found)
        if len(todo):
            poses = np.stack([xx[todo], yy[todo], zz[todo]], 1)
            res = np.full(len(todo), -1, np.int8)
            left = np.arange(len(todo))
            for k, base in enumerate(self.arm.bases):
                hit = self.arm.collides(sizes[todo[left]], poses[left], base,
                                        self.hmap, bb[todo[left]])
                res[left[~hit]] = k
                left = left[hit]
                if not len(left):
                    break
            move[todo] = res
            keys = np.concatenate([self._arm_keys, key[todo]])
            order = np.argsort(keys, kind="stable")
            self._arm_keys = keys[order]
            self._arm_move = np.concatenate([self._arm_move, res])[order]
        return move[inv]

    def _arm_any(self, grids):
        """(1, G): does grid g leave a placement the arm can make?  Checks
        ARM_PROBE cells per grid first, then ARM_GROW times more, only
        where none so far was clear."""
        G = len(grids)
        out = np.zeros((1, G), bool)
        if not self.R.arm_collision:
            for g, (f, _, _) in enumerate(grids):
                out[:, g] = f.any((1, 2))
            return out
        parts = []
        for g, (f, z, d) in enumerate(grids):
            b, x, y = np.nonzero(f)
            parts.append((b * G + g, b, x, y, d[b], z[b, x, y]))
        grp, b, x, y, sz, zz = (np.concatenate(c) for c in zip(*parts))
        if not len(grp):
            return out
        o = np.argsort(grp, kind="stable")
        grp, b, x, y, sz, zz = (a[o] for a in (grp, b, x, y, sz, zz))
        rank = np.arange(len(grp)) - np.searchsorted(grp, grp)
        flat = out.reshape(-1)
        lo, hi, last = 0, self.ARM_PROBE, rank.max()
        while lo <= last:
            sel = (rank >= lo) & (rank < hi) & ~flat[grp]
            if sel.any():
                ok = self._arm_moves(b[sel], x[sel], y[sel], sz[sel],
                                     zz[sel]) >= 0
                flat[grp[sel][ok]] = True
            lo, hi = hi, hi * self.ARM_GROW
        return out
