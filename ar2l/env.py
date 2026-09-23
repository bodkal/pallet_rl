"""Batched online 3D-BPP environment with the PCT state representation.

AR2L Sec. 3 / A.4.  A state is the triple

    C_t  the N_C already packed items      (x, y, z, sx, sy, sz) + type_id
    B_t  the N_B observable incoming items (sx, sy, sz)        + type_id
    L_t  the N_L feasible placements generated for the *first* item of B_t,
         each carrying the type_id of whatever it would land on

Every box carries a discrete `type_id`, and a box may only be stacked on a box
of its own type; the bin floor takes any type.  `types:` in `config.yaml` gives
one item-size class per type, so the type is both a label and a size class.
See `_type_ok` for how the rule is enforced and `TYPE_FLOOR` for the sentinel
a candidate on the floor carries.

and the packing action picks one l in L_t.  A placement is an (orientation,
x, y) triple: with `rot = 2` the leading item is offered both as (w, l, h) and
as (l, w, h), which is the discrete setting of the paper.  The attacker
instead acts on (C_t, B_t) and moves one observable item to the front of the
conveyor.

`n_pick` splits B_t into a pickable prefix and a preview tail: with
`nb = 11, n_pick = 5` the permuter sees eleven items and may choose among the
first five, which is a cell that reaches five boxes at the pick station and
has sight of six more upstream.  The tail is masked out of the pointer only --
it is still attended to -- so it informs the choice without being choosable.
`n_pick = 1` is the FIFO conveyor of the paper, `n_pick = nb` the unrestricted
one; both are what the default `None` and a full-width value give.

The bin is `Lx x Ly x Lz` cells and need not be a cube; `S=` takes an int for
a cube or an `(Lx, Ly, Lz)` triple.  Node features are divided by the single
scale `max(Lx, Ly, Lz)` rather than per axis, so a cube still looks like a
cube to the network -- the packer reasons about geometric fit, and per-axis
normalisation would distort exactly that.

The whole batch of environments is advanced with array operations; nothing
here loops over the batch.  Feasibility for every loading position is obtained
from windowed maxima over the height map: one sweep per axis carries, for each
window, both the maximum and how many cells attain it, so the contact area of
a footprint comes out of the same pass as its landing height.
"""
from __future__ import annotations

import numpy as np

from .config import CFG

# Two stability criteria, selected with `stability=`:
#
#   "com"  (default) the item must rest on at least `min_support` of its own
#          base area, and the projection of its centre of mass must fall
#          inside the convex hull of the contact cells -- tested, as is
#          standard on a grid, by requiring the support to span the centre
#          along each axis.  The area floor is the binding half: at 80% it
#          already implies the centre-of-mass span, which cannot fail before
#          roughly half the base hangs free.
#   "cdrl" the conservative area/corner rule written down by Zhao et al.
#          (AAAI 2021, Sec. 3.1), kept verbatim as the reference rule: it
#          carries its own area thresholds and ignores `min_support`.
SUPPORT_RULES = ((0.60, 4), (0.80, 3), (0.95, 0))

# The contact-area floor for "com": a placement is offered only if at least
# this fraction of the item's base rests on the layer it lands on.  A real
# palletiser needs the load to sit on the box below it, not merely to balance
# on it, so the default is a physical requirement rather than a geometric one.
# `min_support=0.0` restores the bare centre-of-mass rule that reproduces the
# heuristic baselines of AR2L Table 1; see README.  The number itself, like
# every other default here, lives in `config.yaml`.
MIN_SUPPORT = float(CFG["env"]["min_support"])

# Ratios are counts over small integers, so the only ties that matter are
# exact ones (4/5 against 0.80); this absorbs the float32 rounding around them
# and is far smaller than the gap to the next attainable ratio.
SUPPORT_EPS = 1e-6

# Bin side at which pruning the EMS footprint edges starts to pay for its
# per-bin Python loop.  Whole-step times on bins packed full by DBL, sides up
# to S/2, relative to the batched sweep: 0.2x at S=10 and 0.6x at S=20 -- with
# small bins the height map steps down almost everywhere, so there is nothing
# to prune -- then 1.0x at S=24, 1.2x at S=26, 3.5x at S=40 and 18x at S=70
# (4693 ms -> 255 ms at n_env=64).  The ratio barely moves with n_env, both
# paths being linear in it.  They return the same set; see tests/test_env.py.
# The cost driver is the footprint count `Lx * Ly`, so a non-cubic bin is
# compared on the area rather than on either side alone.
EMS_PRUNE_S = 25

# What `ems=` filters the loading positions down to:
#
#   0  nothing: every stable position on the grid is a candidate
#   1  the bottom corners of the empty maximal spaces (PCT / AR2L)
#   2  corner cells of the height map (`_corner_mask`): the item's footprint
#      corner sits where it is bounded on one x side and one y side by a wall,
#      a taller column, or the drop edge of the surface it rests on
#   3  the union of 1 and 2
#
# 1 places items only against walls and taller boxes; over the top of an
# isolated box its maximal space runs out to the bin walls, so a box stacked
# exactly on a box is never offered.  2 offers that and costs a few grid
# comparisons instead of the space enumeration, but misses pockets that are
# bounded only as a whole space.  `True`/`False` still read as 1/0.
EMS_OFF, EMS_SPACES, EMS_CORNER, EMS_BOTH = 0, 1, 2, 3
EMS_NAMES = {"off": EMS_OFF, "ems": EMS_SPACES, "corner": EMS_CORNER,
             "ems|corner": EMS_BOTH, "both": EMS_BOTH}


def ems_mode(v):
    """`ems=` as one of the four modes: an int 0-3, a bool, or a name."""
    if isinstance(v, str) and not v.strip().lstrip("-").isdigit():
        if v.strip().lower() not in EMS_NAMES:
            raise ValueError(f"ems: expected 0-3 or one of {sorted(EMS_NAMES)}, "
                             f"got {v!r}")
        return EMS_NAMES[v.strip().lower()]
    m = int(v)
    if not EMS_OFF <= m <= EMS_BOTH:
        raise ValueError(f"ems: expected 0-3, got {m}")
    return m

# What `tmap` holds for a column no box has reached yet, and the type a
# candidate placement on the bin floor reports as the thing underneath it.
# The floor accepts every type, so it is a type of its own rather than one of
# the `n_types` box types; the networks give it its own embedding row, at
# index `n_types` (see `obs`, which maps the sentinel onto it).
TYPE_FLOOR = -1


def _triple(v):
    """An int (a cube) or an (x, y, z) triple, as three ints."""
    a = np.broadcast_to(np.asarray(v, np.int64), (3,))
    return int(a[0]), int(a[1]), int(a[2])


def type_classes(types, n_types=None, lo=1, hi=5):
    """Resolve the `types:` config block into (probs, lo, hi) arrays.

    `types` is a list of `{p, lo, hi}` mappings, one per type: `p` is that
    type's share of the stream (the shares are normalised) and `lo`/`hi` are
    its per-axis item-side bounds, an int for an isotropic class or a triple.
    `None` means "no size classes": `n_types` types all drawn from the single
    `lo`/`hi` envelope, which is the untyped stream of the original paper with
    a label glued on.

    Returns `(p, lo, hi)` shaped `(T,)`, `(T, 3)`, `(T, 3)`.
    """
    if types is None:
        T = 1 if n_types is None else int(n_types)
        return (np.full(T, 1.0 / T),
                np.tile(_triple(lo), (T, 1)), np.tile(_triple(hi), (T, 1)))
    T = len(types)
    if n_types is not None and int(n_types) != T:
        raise ValueError(f"n_types is {n_types} but `types` lists {T} classes")
    if T == 0:
        raise ValueError("`types` must list at least one size class")
    p = np.array([float(t.get("p", 1.0)) for t in types], np.float64)
    if (p < 0).any() or p.sum() <= 0:
        raise ValueError(f"type shares must be non-negative and not all zero, "
                         f"got {p.tolist()}")
    tlo = np.array([_triple(t.get("lo", lo)) for t in types], np.int64)
    thi = np.array([_triple(t.get("hi", hi)) for t in types], np.int64)
    if (tlo > thi).any():
        bad = int(np.argmax((tlo > thi).any(1)))
        raise ValueError(f"type {bad}: lo {tlo[bad].tolist()} is above "
                         f"hi {thi[bad].tolist()}")
    return p / p.sum(), tlo, thi


def sample_items(rng, shape, lo=1, hi=5, classes=None):
    """Items as `(sx, sy, sz, type_id)`, i.i.d. per axis and per item.

    `classes` is a `(probs, lo, hi)` triple from `type_classes`, so the type is
    drawn first and the sides from that type's own bounds -- one size class per
    type.  Without it every item is type 0 drawn from `lo`/`hi`.

    The untyped isotropic case keeps the original scalar draw: handing
    `rng.integers` array bounds consumes the stream in a different order, which
    would silently re-roll every stored dataset and invalidate every number
    already published against it.
    """
    if classes is None:
        classes = type_classes(None, 1, lo, hi)
    probs, tlo, thi = classes
    out = np.zeros(shape + (4,), np.int16)
    if len(probs) == 1:
        lo3, hi3 = tuple(tlo[0]), tuple(thi[0])
        if len(set(lo3)) == 1 and len(set(hi3)) == 1:
            out[..., :3] = rng.integers(lo3[0], hi3[0] + 1, size=shape + (3,),
                                        dtype=np.int16)
        else:
            out[..., :3] = rng.integers(np.asarray(lo3), np.asarray(hi3) + 1,
                                        size=shape + (3,)).astype(np.int16)
        return out
    t = rng.choice(len(probs), size=shape, p=probs)
    # one vectorised draw with per-item bounds: `rng.integers` broadcasts them,
    # so the whole typed stream costs the same single call the untyped one does
    out[..., :3] = rng.integers(tlo[t], thi[t] + 1).astype(np.int16)
    out[..., 3] = t
    return out


def _win_max(m, axis, L, kmax=None):
    """Running window max of every width 1..kmax along one axis.

    `L` is the length of the swept axis -- `Lx` for `axis=1`, `Ly` for
    `axis=2`.  `kmax` defaults to the whole axis, which is what the EMS
    enumeration needs; the feasibility sweep only ever asks for widths up to
    the largest item side.
    """
    S = L
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


def _win_maxcount(m, c, axis, L, kmax):
    """Running window max, and how many cells attain it, for widths 1..kmax.

    Same recurrence as `_win_max` -- width k+1 is width k with one more cell
    appended -- carried on the pair (max, count), which is associative over
    that append: the new cell either beats the running max, ties it, or loses.
    `c` counts the cells behind each entry of `m` (`None` for raw cells), so
    the y-sweep can be fed straight into the x-sweep and the counts of the
    disjoint columns add up without double counting.
    """
    S = L
    om = np.empty((kmax,) + m.shape, m.dtype)
    oc = np.empty((kmax,) + m.shape, np.int32)
    om[0], oc[0] = m, 1 if c is None else c
    for k in range(1, kmax):
        nm, nc = om[k - 1].copy(), oc[k - 1].copy()
        if axis == 1:
            head = np.s_[:, : S - k]
            am, ac = m[:, k:], 1 if c is None else c[:, k:]
        else:
            head = np.s_[:, :, : S - k]
            am, ac = m[:, :, k:], 1 if c is None else c[:, :, k:]
        hm, hc = om[k - 1][head], oc[k - 1][head]
        up = am > hm
        nm[head] = np.where(up, am, hm)
        nc[head] = np.where(up, ac, np.where(am == hm, hc + ac, hc))
        om[k], oc[k] = nm, nc
    return om, oc


class BPPBatch:
    """`n_env` independent bins stepped in lockstep."""

    def __init__(self, n_env, S=None, nb=1, n_items=None, max_c=None,
                 max_l=None, size_lo=None, size_hi=None, seed=0, ems=None,
                 stability=None, rot=None, min_support=None, n_pick=None,
                 n_types=None, types=None, type_constraint=None,
                 pick_feasible=None, pool=None, pool_order_random=0.0):
        # `None` means "whatever config.yaml says"; an explicit argument wins
        e = CFG["env"]
        S = e["bin"] if S is None else S
        n_items = e["n_items"] if n_items is None else n_items
        max_c = e["max_c"] if max_c is None else max_c
        max_l = e["max_l"] if max_l is None else max_l
        size_lo = e["size_lo"] if size_lo is None else size_lo
        size_hi = e["size_hi"] if size_hi is None else size_hi
        ems = e["ems"] if ems is None else ems
        stability = e["stability"] if stability is None else stability
        rot = e["rot"] if rot is None else rot
        # ---- box types -----------------------------------------------------
        # `types` is the per-type size class; `n_types` alone (with no classes)
        # is the untyped stream carrying a label, which `n_types = 1` makes the
        # exact pre-type simulator again.  An explicit `types=` beats the file,
        # and `types=False` asks for no classes at all rather than for the
        # file's.
        if types is None:
            types = e["types"]
        elif types is False:
            types = None
        n_types = e["n_types"] if n_types is None else n_types
        # a caller sizing the envelope from its own data (real orders of small
        # cartons) can bring `size_hi` under the config's `size_lo`; the
        # stream it would draw from must still be a valid range
        size_lo = tuple(min(a, b) for a, b in
                        zip(_triple(size_lo), _triple(size_hi)))
        self.classes = type_classes(types, n_types, size_lo, size_hi)
        self.n_types = len(self.classes[0])
        self.type_constraint = bool(e["type_constraint"]
                                    if type_constraint is None
                                    else type_constraint)
        if types is not None:
            # the classes *are* the stream, so the envelope has to cover them:
            # `_set_side` and every caller that asks the env how big an item
            # can get reads `size_lo`/`size_hi`
            size_lo = self.classes[1].min(0)
            size_hi = self.classes[2].max(0)
        self.Lx, self.Ly, self.Lz = _triple(S)
        self.size_lo, self.size_hi = _triple(size_lo), _triple(size_hi)
        self.n_env, self.nb, self.ems = n_env, nb, ems_mode(ems)
        # How many of the `nb` observable items are within the cell's reach.
        # `None` means all of them, which is the unrestricted conveyor the
        # paper assumes; a smaller number splits the window into a pickable
        # prefix and a preview tail that is seen but cannot be chosen.
        self.n_pick = nb if n_pick is None else int(n_pick)
        if not 1 <= self.n_pick <= nb:
            raise ValueError(f"n_pick is how many of the {nb} observable "
                             f"items can be reached, got {self.n_pick}")
        # Whether `b_pick` hides the reachable boxes that have nowhere to go.
        # With one type and a reach of one there is nothing to choose and the
        # episode simply ends, which is what the pre-type simulator did; once
        # the stacking rule can block a box while its neighbour in the station
        # is fine, the permuter has to be able to reach past it, so the mask
        # is filtered by default exactly there.  It costs one feasibility
        # sweep per reachable box, so it stays off when it cannot matter.
        self.pick_feasible = bool(
            (self.type_constraint and self.n_types > 1 and self.n_pick > 1)
            if pick_feasible is None else pick_feasible)
        self.stability, self.rot = stability, int(rot)
        # read from CFG rather than from MIN_SUPPORT, so a `--config` file
        # loaded after import is honoured too
        self.min_support = float(e["min_support"] if min_support is None
                                 else min_support)
        if not 0.0 <= self.min_support <= 1.0:
            raise ValueError(f"min_support is a fraction of the item base, "
                             f"got {self.min_support}")
        # the contact count costs a second sweep, so only pay for it when a
        # rule actually reads it
        self._needs_count = (self.stability != "com"
                             or self.min_support > 0.0)
        # `pool`: a fixed set of instances -- real orders -- that every episode
        # draws one of at random instead of sampling boxes from the size
        # classes, each draw reordered by `pool_order_random` (see
        # `orders.randomize_order`).  The episode is as long as the pool is wide.
        self.pool = self.pool_len = None
        self.pool_order_random = float(pool_order_random)
        if pool is not None:
            self.pool, self.pool_len = self._as_seq(pool)
            n_items = self.pool.shape[1]
        self.n_items, self.max_c, self.max_l = n_items, max_c, max_l
        # one divisor for every node feature, so item shape survives the
        # normalisation even when the bin is lopsided
        self.scale = float(max(self.Lx, self.Ly, self.Lz))
        self.bin_vol = float(self.Lx * self.Ly * self.Lz)
        self.rng = np.random.default_rng(seed)
        self._ar = np.arange(n_env)
        self.reset()

    # ---------------------------------------------------------------- reset
    def _invalidate(self, hmap=True):
        """Drop the cached sweeps.  `hmap=False` when only the item changed."""
        self._pos_dirty = True
        # which boxes in the station are placeable depends on the item at each
        # slot, so a permutation invalidates it even though the bin is untouched
        self._pick_cache = None
        if hmap:
            self._sweep_cache = None
            self._ems_cache = None
            # keyed by the incoming type vector, so a permutation that brings a
            # different type to the front reuses nothing it should not
            self._type_cache = {}
            self._under_cache = None

    def _set_side(self, seq):
        """Largest footprint extent the sweeps must cover, per axis.

        Rotation offers both (w, l) and (l, w), so either item side can land
        on either axis; the cap keeps `kmax` inside the axis, and any item too
        big for an axis is killed by the `inx`/`iny` masks anyway.
        """
        side = max(self.size_hi[0], self.size_hi[1], int(seq[..., :2].max()))
        self.sidex, self.sidey = min(side, self.Lx), min(side, self.Ly)

    def _as_seq(self, seqs):
        """A caller's sequence as the internal `(n, n_items, 4)` table.

        A three-column array is a pre-type dataset: every box is type 0, which
        is what it was drawn as.  Anything else has to carry its own types.

        Instances may be shorter than the table: trailing all-zero rows are
        padding, so real orders of different lengths share one array.  Returns
        the table and each instance's length.
        """
        a = np.asarray(seqs, np.int16)
        if a.shape[-1] == 3:
            a = np.concatenate([a, np.zeros(a.shape[:-1] + (1,), np.int16)], -1)
        elif a.shape[-1] != 4:
            raise ValueError(f"a sequence is (sx, sy, sz[, type_id]), got "
                             f"{a.shape[-1]} columns")
        t = a[..., 3]
        if t.size and (t.min() < 0 or t.max() >= self.n_types):
            raise ValueError(f"type_id must be in [0, {self.n_types}), got "
                             f"[{int(t.min())}, {int(t.max())}]")
        box = (a[..., :3] > 0).all(-1)
        pad = (a == 0).all(-1)
        if not (box | pad).all():
            raise ValueError("a box has a zero side but is not an all-zero "
                             "padding row")
        length = box.sum(-1).astype(np.int32)
        # padding only at the end: a box after a pad row would never be reached
        if (box != (np.arange(a.shape[-2]) < length[..., None])).any():
            raise ValueError("padding rows must all come after the last box")
        if (length == 0).any():
            raise ValueError("every instance needs at least one box")
        return a.copy(), length

    def _draw(self, k):
        """`k` instances from the pool, with replacement, and their lengths."""
        from .orders import randomize_order
        i = self.rng.integers(len(self.pool), size=k)
        rows = self.pool[i]
        if self.pool_order_random:
            # the env's own generator, so a seeded run replays the same orders
            rows = randomize_order(rows, self.pool_order_random, self.rng)
        return rows, self.pool_len[i].copy()

    def reset(self, seqs=None):
        n = self.n_env
        self.hmap = np.zeros((n, self.Lx, self.Ly), np.int16)
        # the type of the topmost box in each column, and so of whatever a
        # placement landing there would rest on
        self.tmap = np.full((n, self.Lx, self.Ly), TYPE_FLOOR, np.int8)
        self.packed = np.zeros((n, self.max_c, 6), np.float32)
        self.ptype = np.zeros((n, self.max_c), np.int8)
        self.n_packed = np.zeros(n, np.int32)
        self.volume = np.zeros(n, np.float32)
        self.done = np.zeros(n, bool)
        if seqs is None and self.pool is not None:
            self.seq, self.length = self._draw(n)
        elif seqs is None:
            self.seq = sample_items(self.rng, (n, self.n_items),
                                    self.size_lo, self.size_hi, self.classes)
            self.length = np.full(n, self.n_items, np.int32)
        else:
            self.seq, self.length = self._as_seq(seqs)
        self.head = np.zeros(n, np.int32)
        # windows are only ever indexed by a footprint side, and `reset` may be
        # handed a sequence the constructor knew nothing about
        self._set_side(self.seq)
        self._invalidate()
        return self.obs()

    def reset_done(self):
        """Re-roll only the finished bins (used during rollout collection)."""
        idx = np.nonzero(self.done)[0]
        if idx.size == 0:
            return
        self.hmap[idx] = 0
        self.tmap[idx] = TYPE_FLOOR
        self.packed[idx] = 0
        self.ptype[idx] = 0
        self.n_packed[idx] = 0
        self.volume[idx] = 0
        if self.pool is not None:
            self.seq[idx], self.length[idx] = self._draw(idx.size)
        else:
            self.seq[idx] = sample_items(self.rng, (idx.size, self.n_items),
                                         self.size_lo, self.size_hi,
                                         self.classes)
            self.length[idx] = self.n_items
        self._set_side(self.seq)
        self.head[idx] = 0
        self.done[idx] = False
        self._invalidate()

    # ------------------------------------------------------------- conveyor
    def window(self):
        """(B, nb, 4) observable items -- (sx, sy, sz, type_id) -- and validity."""
        off = self.head[:, None] + np.arange(self.nb)[None, :]
        valid = off < self.length[:, None]
        items = self.seq[self._ar[:, None], np.minimum(off, self.n_items - 1)]
        return np.where(valid[..., None], items, 0), valid

    def pick_mask(self, valid):
        """(B, nb) which observable items the cell can actually take.

        The first `n_pick` slots of the window are the pick station; the rest
        are visible further up the conveyor and are there to be reasoned about,
        not chosen.  `permute` keeps the split consistent by itself: moving
        slot i < n_pick to the front permutes the prefix among itself and
        leaves the tail alone, and the `head += 1` in `step` then slides one
        preview item into reach.  Validity is a prefix property, so a restricted
        window never runs out of pickable items while preview items remain.

        With `pick_feasible` the station is filtered again by whether each box
        has anywhere legal to go: under the stacking rule a box can be blocked
        while the one beside it in the station is not, and the permuter has to
        be able to reach past it rather than end the episode on it.  The
        episode ends when the whole station is blocked, which is then exactly
        `b_pick` being empty.
        """
        reach = (valid if self.n_pick >= self.nb
                 else valid & (np.arange(self.nb) < self.n_pick)[None, :])
        if not self.pick_feasible:
            return reach
        return reach & self._placeable()

    def _placeable(self):
        """(B, nb) does each box in the station have at least one placement?

        One feasibility sweep per reachable slot -- the sweeps that depend on
        the bin alone are shared, only the per-footprint windows are redone --
        and the preview tail is never asked, since it cannot be chosen anyway.
        """
        if self._pick_cache is not None:
            return self._pick_cache
        win, valid = self.window()
        k = min(self.n_pick, self.nb)
        out = np.zeros((self.n_env, self.nb), bool)
        alive = ~self.done & (self.head < self.length)
        for i in range(k):
            dims, types = win[:, i, :3], win[:, i, 3]
            # an invalid slot is zero-sized, and a zero side would index
            # window -1; the mask below drops it either way
            dims = np.maximum(dims, 1)
            any_pos = np.zeros(self.n_env, bool)
            for d in ([dims] if self.rot < 2 else [dims, dims[:, [1, 0, 2]]]):
                any_pos |= self._feas_one(d, types)[0].any((1, 2))
            out[:, i] = any_pos & valid[:, i] & alive
        self._pick_cache = out
        return out

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
        each stacked up to the lid.

        Two exact enumerations compute exactly this set; `EMS_PRUNE_S` picks
        the cheaper one.  Both are stateless -- they read only the height map
        -- so neither can carry stale spaces across an episode boundary the
        way an incremental difference process can, and neither imposes an
        order on the result.

        Returns `(env, x0, y0, wx, wy, floor)`, one entry per space.
        """
        if self._ems_cache is not None:
            return self._ems_cache
        self._ems_cache = (self._ems_pruned()
                           if self.Lx * self.Ly >= EMS_PRUNE_S ** 2
                           else self._ems_exhaustive())
        return self._ems_cache

    def _ems_pruned(self):
        """`_ems_exhaustive` restricted to footprint edges that can be maximal.

        A space needs `left > floor`, and `floor` is at least the height of
        column `x0` anywhere in the y-interval, so some `y` in that interval
        has `hmap[x0-1, y] > hmap[x0, y]`: a left edge can only sit where the
        height map steps down going right.  The other three sides give the
        same condition, so only those rows and columns -- O(items) of them,
        not O(S) -- can bound a space.  At S = 70 that is 2.9k candidate
        footprints per bin instead of 6.2M.
        """
        Lx, Ly, H = self.Lx, self.Ly, self.hmap
        big = np.int16(self.Lz + 1)           # taller than any column can be
        cols = []
        for e in range(self.n_env):
            h = H[e]
            x0c = np.flatnonzero(np.r_[True, (h[:-1] > h[1:]).any(1)])
            x1c = np.flatnonzero(np.r_[(h[1:] > h[:-1]).any(1), True])
            y0c = np.flatnonzero(np.r_[True, (h[:, :-1] > h[:, 1:]).any(0)])
            y1c = np.flatnonzero(np.r_[(h[:, 1:] > h[:, :-1]).any(0), True])
            ax = _win_max(h[None], 1, Lx)[:, 0]       # ax[wx-1, x0, y]
            ay = _win_max(h[None], 2, Ly)[:, 0]       # ay[wy-1, x, y0]
            yy0, yy1 = np.meshgrid(y0c, y1c, indexing="ij")
            m = yy1 >= yy0
            yy0, yy1, wy = yy0[m], yy1[m], (yy1 - yy0 + 1)[m]
            for x0 in x0c:
                x1 = x1c[x1c >= x0]
                if not x1.size:
                    continue
                wx = x1 - x0 + 1                      # (K,)
                a = ax[wx - 1, x0]                    # (K, Ly) max over the x-span
                wa = _win_max(a[None], 2, Ly)[:, 0]   # wa[wy-1, k, y0]
                k = np.arange(wx.size)
                flr = wa[wy[:, None] - 1, k[None, :], yy0[:, None]]        # (M, K)
                # the same four one-cell extensions as the exhaustive sweep;
                # `back`/`front` are the x-span max one row outside, which is
                # what `a` already holds
                lo = (np.full(wy.size, big) if x0 == 0
                      else ay[wy - 1, x0 - 1, yy0])[:, None]
                hi = np.where(x1 + 1 < Lx,
                              ay[wy[:, None] - 1,
                                 np.minimum(x1 + 1, Lx - 1)[None, :],
                                 yy0[:, None]], big)
                bk = np.where(yy0[:, None] > 0,
                              a[k[None, :], np.maximum(yy0 - 1, 0)[:, None]], big)
                ft = np.where(yy1[:, None] + 1 < Ly,
                              a[k[None, :], np.minimum(yy1 + 1, Ly - 1)[:, None]], big)
                f = np.flatnonzero((lo > flr) & (hi > flr) & (bk > flr) & (ft > flr))
                if f.size:
                    im, ik = np.unravel_index(f, flr.shape)
                    cols.append((np.full(f.size, e), np.full(f.size, x0), yy0[im],
                                 wx[ik], wy[im],
                                 flr.reshape(-1)[f].astype(np.int32)))
        if not cols:
            z = np.zeros(0, np.int64)
            return (z,) * 6
        return tuple(np.concatenate(c) for c in zip(*cols))

    def _ems_exhaustive(self):
        """Score every one of the `(Lx(Lx+1)/2)(Ly(Ly+1)/2)` footprints and
        keep the maximal ones.  Fully batched, and the reference the pruned
        enumeration is tested against."""
        Lx, Ly, h = self.Lx, self.Ly, self.hmap
        big = np.int16(self.Lz + 1)           # a wall is taller than anything
        ax = _win_max(h, 1, Lx)               # ax[wx-1, n, x0, y]  max over x
        ay = _win_max(h, 2, Ly)               # ay[wy-1, n, x, y0]  max over y
        gx, gy = np.arange(Lx), np.arange(Ly)
        oky = gy[None, :] + np.arange(1, Ly + 1)[:, None] <= Ly    # (wy, y0)
        left = np.full_like(ay, big); left[:, :, 1:] = ay[:, :, :-1]
        parts = []
        for wx in range(1, Lx + 1):
            a = ax[wx - 1]                                 # (n, x0, y)
            flr = _win_max(a, 2, Ly)                       # (wy, n, x0, y0)
            # one-cell extensions: each must raise the floor, or hit a wall
            right = np.full_like(ay, big)
            if wx < Lx:
                right[:, :, : Lx - wx] = ay[:, :, wx:]
            back = np.full_like(a, big); back[:, :, 1:] = a[:, :, :-1]
            front = np.full((Ly,) + a.shape, big, a.dtype)
            for k in range(1, Ly):
                front[k - 1, :, :, : Ly - k] = a[:, :, k:]
            keep = ((left > flr) & (right > flr) & (back[None] > flr)
                    & (front > flr)
                    & (gx[None, None, :, None] + wx <= Lx)
                    & oky[:, None, None, :])
            f = np.flatnonzero(keep)
            if f.size:
                iwy, env, x0, y0 = np.unravel_index(f, keep.shape)
                parts.append((env, x0, y0, np.full(f.size, wx, np.int32),
                              iwy + 1, flr.reshape(-1)[f].astype(np.int32)))
        return tuple(np.concatenate(c) for c in zip(*parts))

    def _ems_corners(self, dims):
        """(n, Lx, Ly): the item fits flush into a bottom corner of some EMS.

        The four bottom corners of every empty maximal space wide enough and
        tall enough to take the item -- the leaf-node set of PCT and AR2L.
        """
        env, x0, y0, wx, wy, floor = self._ems_list()
        sx, sy, sz = dims[env, 0], dims[env, 1], dims[env, 2]
        ok = (wx >= sx) & (wy >= sy) & (floor + sz <= self.Lz)
        env, x0, y0 = env[ok], x0[ok], y0[ok]
        xs = (x0, x0 + (wx - sx)[ok])
        ys = (y0, y0 + (wy - sy)[ok])
        out = np.zeros((self.n_env, self.Lx, self.Ly), bool)
        for x in xs:
            for y in ys:
                out[env, x, y] = True
        return out

    def _corner_mask(self, dims, z):
        """(n, Lx, Ly): some corner of the footprint sits in a corner cell.

        A footprint corner at column `c`, facing out along `dx` and `dy`, is
        bounded on a side when the neighbour `nb` across it is

            contact   taller than the landing height `z` -- a wall or a box
                      the item's side leans against
            drop      lower than `z` while `c` itself carries the item -- the
                      corner lines up with the edge of the surface below,
                      which is what puts a box exactly on top of a box
            aligned   (x side only when the y side is in contact, and vice
                      versa) the wall it leans against ends right there: the
                      diagonal column is no taller than `z`

        and it is a corner cell when both its x side and its y side are
        bounded.  The bin wall is a column taller than any `z`.  Stateless, like
        the EMS list, and item-dependent only through the footprint size and
        `z`, so it is a handful of gathers over the grid.
        """
        Lx, Ly = self.Lx, self.Ly
        p = np.full((self.n_env, Lx + 2, Ly + 2), self.Lz + 1, np.int32)
        p[:, 1:-1, 1:-1] = self.hmap
        ar = self._ar[:, None, None]
        gx = np.arange(Lx)[None, :, None]
        gy = np.arange(Ly)[None, None, :]
        sx, sy = dims[:, 0, None, None], dims[:, 1, None, None]
        # near and far column of the footprint along each axis, and which way
        # its outside lies; `+ 1` is the padding
        xs = ((gx + 1, -1), (np.minimum(gx + sx, Lx), 1))
        ys = ((gy + 1, -1), (np.minimum(gy + sy, Ly), 1))
        out = np.zeros((self.n_env, Lx, Ly), bool)
        for cx, dx in xs:
            for cy, dy in ys:
                hc = p[ar, cx, cy]
                nx, ny, nd = p[ar, cx + dx, cy], p[ar, cx, cy + dy], p[ar, cx + dx, cy + dy]
                on = hc == z
                tx, ty = nx > z, ny > z
                ex = tx | (on & (nx < z)) | (ty & (nd <= z))
                ey = ty | (on & (ny < z)) | (tx & (nd <= z))
                out |= ex & ey
        return out

    # --------------------------------------------------------- feasibility
    def _sweeps(self):
        """Height-map sweeps that depend on the bin but not on the item.

        Returns `(my, myc, mx)`: the y-window maxima, how many cells of each
        y-window attain them (`None` when no rule needs contact area), and the
        x-window maxima.
        """
        if self._sweep_cache is None:
            if self._needs_count:
                my, myc = _win_maxcount(self.hmap, None, 2, self.Ly, self.sidey)
            else:
                my, myc = _win_max(self.hmap, 2, self.Ly, self.sidey), None
            self._sweep_cache = (my, myc,
                                 _win_max(self.hmap, 1, self.Lx, self.sidex))
        return self._sweep_cache

    def _type_sweep(self, types):
        """The y-window maxima of the blocked-height map, per incoming type.

        `hbad[x, y]` is the height of column (x, y) when that column is topped
        by a box of some *other* type, and -1 when it is not -- either because
        the column is bare floor, which takes any type, or because the box on
        top of it is of the incoming type and may be stacked on.  A footprint
        is then type-legal exactly when the maximum of `hbad` over it is below
        the height `z` the item lands at: every cell the item actually touches
        is a cell attaining `z`, and any cell below `z` is a void the item
        bridges rather than rests on, so an unsupported overhang over a foreign
        type is legal and a contact patch over one is not.

        Cached on the type vector rather than dropped on every permutation: the
        map depends on the bin and on the incoming type, and a permutation
        changes only the latter.
        """
        key = types.tobytes()
        if key not in self._type_cache:
            bad = np.where((self.hmap > 0)
                           & (self.tmap != types[:, None, None].astype(np.int8)),
                           self.hmap, np.int16(-1))
            self._type_cache[key] = _win_max(bad, 2, self.Ly, self.sidey)
        return self._type_cache[key]

    def _type_ok(self, dims, z, types):
        """(n, Lx, Ly): may an item of this type rest on this footprint?"""
        cx = np.minimum(dims[:, 0], self.sidex)
        cy = np.minimum(dims[:, 1], self.sidey)
        by = self._type_sweep(types)[cy - 1, self._ar]
        return _win_max(by, 1, self.Lx, self.sidex)[cx - 1, self._ar] < z

    def _under_sweep(self):
        """y-window maxima of the height map with the column type packed in.

        `code = h * (n_types + 1) + (t + 1)` is monotone in the height, since
        `t + 1` never reaches the stride, so the window maximum still carries
        the landing height `z` in its quotient and, among the cells attaining
        it, the largest type in its remainder.  A bare column codes as -1, so
        an all-floor footprint reports `TYPE_FLOOR`.

        Under the stacking rule every cell a legal placement touches carries
        the same type, so the remainder *is* the type underneath; with the rule
        switched off it is the largest of the types the item straddles.
        """
        if self._under_cache is None:
            K = self.n_types + 1
            code = np.where(self.hmap > 0,
                            self.hmap.astype(np.int32) * K + self.tmap + 1, -1)
            self._under_cache = _win_max(code, 2, self.Ly, self.sidey)
        return self._under_cache

    def _type_under(self, dims, z):
        """(n, Lx, Ly) int8: the type a footprint would land on, or TYPE_FLOOR.

        `z` is the landing height from the same sweep, which the single-type
        case answers from directly: a footprint landing above the floor rests
        on the only type there is, whatever its own corner column happens to
        hold.  With more than one type the coded sweep has to say which.
        """
        if self.n_types == 1:
            return np.where(z > 0, np.int8(0), np.int8(TYPE_FLOOR))
        cx = np.minimum(dims[:, 0], self.sidex)
        cy = np.minimum(dims[:, 1], self.sidey)
        by = self._under_sweep()[cy - 1, self._ar]
        m = _win_max(by, 1, self.Lx, self.sidex)[cx - 1, self._ar]
        return np.where(m < 0, np.int8(TYPE_FLOOR),
                        (m % (self.n_types + 1) - 1).astype(np.int8))

    def _feas_one(self, dims, types=None):
        """Feasible (x, y) grid and landing height for one orientation."""
        Lx, Ly, Lz = self.Lx, self.Ly, self.Lz
        sx, sy, sz = dims[:, 0], dims[:, 1], dims[:, 2]
        ar, gx, gy = self._ar, np.arange(Lx), np.arange(Ly)
        my, myc, mx = self._sweeps()
        # an item too long for an axis still has to index a window that exists;
        # `inx`/`iny` below drop it, so the clamped sweep is never read out
        cx = np.minimum(sx, self.sidex)
        cy = np.minimum(sy, self.sidey)

        myd = my[cy - 1, ar]                          # max over the y extent
        if myc is None:                               # ... and x sub-windows
            mxy, cxy = _win_max(myd, 1, Lx, self.sidex), None
        else:
            mxy, cxy = _win_maxcount(myd, myc[cy - 1, ar], 1, Lx, self.sidex)
        z = mxy[cx - 1, ar].astype(np.int32)

        if self.stability == "com":
            mxd = mx[cx - 1, ar]
            myx = _win_max(mxd, 2, Ly, self.sidey)    # sub-windows along y

            def span(stack, size, axis):
                """Support both at or before, and at or after, the centre.

                Every cell of a sub-window of the footprint is at most `z`, the
                max over the whole footprint, so "this sub-window rests on the
                contact layer" is exactly "its max equals `z`".
                """
                L = Lx if axis == 1 else Ly
                g = gx if axis == 1 else gy
                a = (size - 1) // 2                    # last index of the near half
                b = -((1 - size) // 2)                 # first index of the far half
                near = stack[a, ar] == z
                idx = np.minimum(g[None, :] + b[:, None], L - 1)
                far = stack[size - b - 1, ar]
                far = (far[ar[:, None], idx] if axis == 1
                       else far[ar[:, None, None], gx[None, :, None],
                                idx[:, None, :]])
                return near & (far == z)

            stable = span(mxy, cx, 1) & span(myx, cy, 2)
            if self.min_support > 0:
                ratio = self._support_ratio(dims, cxy, cx)
                stable &= ratio >= self.min_support - SUPPORT_EPS
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

        inx = gx[None, :, None] + sx[:, None, None] <= Lx
        iny = gy[None, None, :] + sy[:, None, None] <= Ly
        fits = z + sz[:, None, None] <= Lz
        feas = stable & inx & iny & fits
        if self.ems == EMS_SPACES:
            feas &= self._ems_corners(dims)
        elif self.ems == EMS_CORNER:
            feas &= self._corner_mask(dims, z)
        elif self.ems == EMS_BOTH:
            feas &= self._ems_corners(dims) | self._corner_mask(dims, z)
        if self.type_constraint and self.n_types > 1 and types is not None:
            feas &= self._type_ok(dims, z, types)
        return feas, z

    def _support_ratio(self, dims, cxy, cx):
        """Fraction of the footprint that rests on the contact layer.

        A cell carries the item exactly when its column reaches `z`, the
        maximum over the whole footprint, so the contact area is the number of
        cells attaining that maximum -- which `cxy`, the count carried through
        both sweeps beside the maximum itself, already holds, at any bin
        height and any item side.
        """
        sx, sy = dims[:, 0], dims[:, 1]
        count = cxy[cx - 1, self._ar]
        return count.astype(np.float32) / (sx * sy)[:, None, None]

    def head_item(self):
        """(dims (n, 3), type (n,)) of the item the packer is being offered."""
        # int16, as the sequence stores them: the candidate node features are
        # built by concatenating these onto float32 coordinates, and a wider
        # integer would promote the whole block to float64 and hand the first
        # linear layer a dtype it does not have weights for
        row = self.seq[self._ar, np.minimum(self.head, self.length - 1)]
        return row[:, :3], row[:, 3]

    def _positions(self):
        """Feasibility, landing height and under-type per (orientation, x, y)."""
        if not self._pos_dirty:
            return self._pos_cache
        item, types = self.head_item()
        orients = [item] if self.rot < 2 else [item, item[:, [1, 0, 2]]]
        feas, zs, tu = [], [], []
        for r, d in enumerate(orients):
            f, z = self._feas_one(d, types)
            if r:   # a square footprint is the same placement turned round
                f = f & (item[:, 0] != item[:, 1])[:, None, None]
            feas.append(f); zs.append(z); tu.append(self._type_under(d, z))
        feas = np.stack(feas, 1)                      # (n, R, Lx, Ly)
        alive = ~self.done & (self.head < self.length)
        feas &= alive[:, None, None, None]
        self._pos_cache = (feas, np.stack(zs, 1), np.stack(orients, 1),
                           np.stack(tu, 1))
        self._pos_dirty = False
        return self._pos_cache

    def type_blocked(self):
        """(n,) is the leading item blocked *only* by the stacking rule?

        The same feasibility question with the type filter lifted: true where
        the item has somewhere to go geometrically but nowhere of its own type.
        Deliberately indifferent to `done`, since the question worth asking is
        why a bin stopped, and after `step` the head is still the box that
        could not be placed.  Diagnostic only -- nothing in `step` reads it --
        so it pays its two sweeps when a caller asks and never otherwise.
        """
        if not (self.type_constraint and self.n_types > 1):
            return np.zeros(self.n_env, bool)
        item, _ = self.head_item()
        orients = [item] if self.rot < 2 else [item, item[:, [1, 0, 2]]]
        free = np.zeros(self.n_env, bool)
        for r, d in enumerate(orients):
            f, _ = self._feas_one(d)
            if r:
                f = f & (item[:, 0] != item[:, 1])[:, None, None]
            free |= f.any((1, 2))
        return free & (self.head < self.length) & (self.n_feasible() == 0)

    # ------------------------------------------------------------ observation
    def _trim_c(self):
        """Padding tokens cost attention; keep only as many as the batch uses."""
        nc = max(int(self.n_packed.max()), 1)
        mask = np.arange(nc)[None, :] < self.n_packed[:, None]
        return self.packed[:, :nc], self.ptype[:, :nc], mask

    def _emb_type(self, t):
        """A type array as an embedding row index: TYPE_FLOOR is the last row."""
        return np.where(t < 0, self.n_types, t).astype(np.int64)

    def obs_cb(self):
        """(C, B) only -- skips the candidate list the attacker never uses.

        `b_pick` still costs a feasibility sweep per reachable box whenever the
        stacking rule can block one; that is the price of letting the permuter
        see which boxes in the station are dead.
        """
        win, wvalid = self.window()
        c, ctype, cmask = self._trim_c()
        return {"c": c, "c_mask": cmask, "c_type": self._emb_type(ctype),
                "b": win[..., :3].astype(np.float32) / self.scale,
                "b_mask": wvalid, "b_type": self._emb_type(win[..., 3]),
                "b_pick": self.pick_mask(wvalid)}

    def obs(self):
        feas, z, odims, tunder = self._positions()
        scale, n = self.scale, self.n_env
        flat = feas.reshape(n, -1)
        NL = int(min(self.max_l, flat.shape[1], max(flat.sum(1).max(), 1)))

        # keep the NL deepest candidates when more than NL are feasible
        order = np.lexsort((np.arange(flat.shape[1])[None, :].repeat(n, 0),
                            z.reshape(n, -1), ~flat), axis=1)[:, :NL]
        lmask = np.take_along_axis(flat, order, 1)
        zz = np.take_along_axis(z.reshape(n, -1), order, 1)
        tt = np.take_along_axis(tunder.reshape(n, -1), order, 1)
        cells = self.Lx * self.Ly
        rr, rem = order // cells, order % cells
        xx, yy = rem // self.Ly, rem % self.Ly
        dims = odims[self._ar[:, None], rr]                   # (n, NL, 3)

        lnode = np.concatenate(
            [np.stack([xx, yy, zz], -1).astype(np.float32), dims], -1) / scale
        lnode *= lmask[..., None]
        self._lxy = np.stack([xx, yy, rr], -1).astype(np.int32)
        self._lz = zz.astype(np.int32)
        self._ldim = dims.astype(np.int32)
        # a masked-out slot has no footprint and so no type underneath; giving
        # it the floor row keeps the index in range and the embedding is zeroed
        # by the mask anyway
        self._lunder = np.where(lmask, tt, TYPE_FLOOR).astype(np.int8)

        win, wvalid = self.window()
        c, ctype, cmask = self._trim_c()
        return {
            "c": c, "c_mask": cmask, "c_type": self._emb_type(ctype),
            "b": win[..., :3].astype(np.float32) / scale, "b_mask": wvalid,
            "b_type": self._emb_type(win[..., 3]),
            "b_pick": self.pick_mask(wvalid),
            "l": lnode, "l_mask": lmask, "l_type": self._emb_type(self._lunder),
        }

    def candidates(self, b=0, k=None):
        """[(x, y, z, sx, sy, sz, type_under)] for one bin's candidate list."""
        k = self._lxy.shape[1] if k is None else k
        return np.concatenate([self._lxy[b, :k, :2], self._lz[b, :k, None],
                               self._ldim[b, :k],
                               self._lunder[b, :k, None]], -1).tolist()

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
        tt = self.head_item()[1].astype(np.int8)

        # raise the footprint; the per-env window makes a python loop cheapest
        gx, gy = np.arange(self.Lx), np.arange(self.Ly)
        inx = (gx[None, :] >= x[:, None]) & (gx[None, :] < (x + sx)[:, None])
        iny = (gy[None, :] >= y[:, None]) & (gy[None, :] < (y + sy)[:, None])
        foot = inx[:, :, None] & iny[:, None, :] & alive[:, None, None]
        self.hmap = np.where(foot, (zz + sz)[:, None, None].astype(np.int16),
                             self.hmap)
        # the item is now the top of every column it covers, so it is what the
        # next placement over that footprint would rest on
        self.tmap = np.where(foot, tt[:, None, None], self.tmap)

        # `packed` is C_t, the packer's own memory of the bin.  `max_c` is an
        # initial capacity, not a limit: a large bin full of small items holds
        # far more boxes than a 10^3 one, and clamping the write slot would
        # silently overwrite the last box -- a wrong *state*, not just a wrong
        # count -- while `n_packed` ran past the array and raised out of any
        # caller that indexed it.  Grow instead, and keep the larger capacity
        # so the next reset starts big enough.
        if int(self.n_packed.max()) >= self.packed.shape[1]:
            self.max_c = self.packed.shape[1] * 2
            grown = np.zeros((self.n_env, self.max_c, 6), np.float32)
            grown[:, : self.packed.shape[1]] = self.packed
            gt = np.zeros((self.n_env, self.max_c), np.int8)
            gt[:, : self.ptype.shape[1]] = self.ptype
            self.packed, self.ptype = grown, gt
        node = (np.stack([x, y, zz, sx, sy, sz], 1).astype(np.float32)
                / self.scale)
        self.packed[self._ar, self.n_packed] = np.where(
            alive[:, None], node, self.packed[self._ar, self.n_packed])
        # the type rides alongside `packed` rather than inside it, so every
        # caller that unpacks a row as (x, y, z, sx, sy, sz) still can
        self.ptype[self._ar, self.n_packed] = np.where(
            alive, tt, self.ptype[self._ar, self.n_packed])
        self.n_packed += alive
        vol = (sx * sy * sz).astype(np.float32)
        self.volume += vol * alive

        self.head += alive
        self._invalidate()

        reward = (vol / self.bin_vol) * alive
        # terminate when the new leading item has nowhere to go
        self.done |= ~alive
        self.done |= self.head >= self.length
        self.done |= self.n_feasible() == 0
        return reward.astype(np.float32), self.done.copy()

    # ----------------------------------------------------------------- stats
    def utilization(self):
        return self.volume / self.bin_vol
