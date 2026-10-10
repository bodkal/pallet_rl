"""Batched online 3D-BPP environment with the PCT state representation.

AR2L Sec. 3 / A.4.  A state is the triple

    C_t  the N_C already packed items      (x, y, z, sx, sy, sz) + type_id
    B_t  the N_B observable incoming items (sx, sy, sz)        + type_id
    L_t  the N_L feasible placements generated for the *first* item of B_t,
         each carrying the type_id of whatever it would land on

Every box carries a discrete `type_id`, and a box may only be stacked on a box
of its own type; the bin floor takes any type.  `types:` in `config.yaml` gives
one item-size class per type, so the type is both a label and a size class.
`type_rule` says what "stacked on" means: `touch` tests only the cells the box
rests on, `column` every box anywhere under its footprint (`smap` holds the
types each column has seen).  See `_type_ok` for how the rule is enforced and
`TYPE_FLOOR` for the sentinel a candidate on the floor carries.
`mixed_touch` and `mixed_column` are the cell's own rule for its blue, white
and brown cartons, on either reading of "on", which lets some types onto
others -- near the lid, where no blue box can go any more, and as a fallback
where a box would otherwise have nowhere to go; see `BLUE`.

A small box carries only a short stack.  Once a box of one of the
`stack_cap_types` is placed, a later box may start over its footprint only
below its top plus an allowance that grows with its shorter footprint side --
the cell's own rule (its C++ `stable_grid`), set in real cm.  `cmap` holds
the limit per column, `stack_allowance` the allowance, and `_cap_ok` drops
the placements that break it.

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

import contextlib

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
# `type_rule` values: what a box of one type may not be over
TYPE_RULES = ("touch", "column", "mixed_touch", "mixed_column")
# the rules that read "on" as anything under the footprint, at any depth
COLUMN_RULES = ("column", "mixed_column")

# `type_rule: mixed_touch` / `mixed_column` is the cell's own rule for its
# three cartons, by the type ids the orders data gives them.  A box goes on
# the floor and on its own type -- blue on blue only with all of its base on
# blue -- and across types by these conditions:
#
#   (a)  its top ends within `mixed_top_gap_cm` of the lid
#   (c)  a brown footprint at least `mixed_area_frac` of the small blue's
#   (d)  no small blue box could still be put on the blue it would be on,
#        for want of height or of a pose the arm can make
#
#                      normally          else, soft mix off  else, soft mix on
#   blue on blue       all base on blue  fallback            fallback
#   blue on white      (a)               fallback            last resort
#   blue on brown      (a)               fallback            last resort
#   white on blue      (a) or (d)        fallback            last resort
#   white on brown     (a)               fallback            last resort
#   brown on blue      (a), (c) or (d)   fallback            last resort
#                      -- soft mix on: (a) or (d)
#   brown on white     never             never               never
#
# A fallback is a box's own: it opens when that box has no placement under
# the normal rules at all, in any orientation, the arm included -- blue on
# blue then needs only `min_support`.  The last resort is the station's: it
# opens when no box within reach has a placement by the rules before it
# (`_stuck`).
#
# "On" is read as `touch` reads it under `mixed_touch` -- only the cells the
# box rests on count -- and as `column` reads it under `mixed_column`: every
# box anywhere under the footprint, however deep and across any gap, so an
# overhang counts too.  A blue cell another box covers can take no small blue
# box any more, which is (d), so under `mixed_column` it is the blue cells on
# top of their columns that rule (d) asks about.  `mixed_tables` holds the
# pairs, `_type_ok` the conditions, `_open_sweep` the cells of rule (d).
MIXED_RULES = ("mixed_touch", "mixed_column")
BLUE, WHITE, BROWN = 0, 1, 2
# What a mixed rule says of a box of one type on a box of another: it may
# not, it may, it may if the pair's own condition holds, or it may if its top
# ends near the lid -- rule (a) alone (`mixed_tables`)
NO, YES, IF, TOP = 0, 1, 2, 3
# The pairs whose only normal condition is (a): a box near the lid may mix
TOP_PAIRS = ((BLUE, WHITE), (BLUE, BROWN), (WHITE, BROWN))
# What soft mix's last resort lets on, with no condition, once no box within
# reach has any other placement: every mix but brown on white.  Soft mix
# keeps no fallback of a box's own but blue on blue's.
SOFT_PAIRS = ((BLUE, WHITE), (BLUE, BROWN), (WHITE, BLUE), (WHITE, BROWN),
              (BROWN, BLUE))

# `outline_resort`, read by the mixed rules only: one more step after all of
# the above, when they leave nothing.  It runs the same rules again -- the
# normal ones, a box's own fallback, soft mix's last resort -- over a wider
# set of positions: the `ems` candidates, and every position that puts a
# corner of the footprint on an outline (`omap`).  The outlines are the edges
# of every packed box's footprint -- its top outline, where a box stands on it
# flush with an edge, and its bottom outline, where one stands beside it flush
# with a side, which seen from above are the same lines -- and the pallet's
# edges, at every cell along them, buried boxes' included.  `outline_when`
# says when it opens: `station`, when no box within reach has a place by the
# steps before it; `box`, for each box with none, whatever the others have.
OUTLINE_WHEN = ("station", "box")

# What `cmap` holds for a column no small box has limited: a later box may
# start over column (x, y) only below `cmap[x, y]`, and no bin is this tall.
NO_CAP = int(np.iinfo(np.int16).max)

# A float hair off an exact cm boundary -- a 28 cm side of 4 cm cells against
# a 28 cm threshold -- as `orders.EPS` is for the box sides themselves.
CM_EPS = 1e-6


def stack_allowance(n, side_cm, allow_cm, cell_cm):
    """(n + 1,) cells a later box may still start above a small box.

    Entry `k` is for a box whose shorter footprint side is `k` cells.  The
    allowance runs linearly from `allow_cm[0]` at a side of `side_cm[0]`, and
    stays there below it, to `allow_cm[1]` at `side_cm[1]`; from a side of
    `side_cm[1]` up there is no limit (`NO_CAP`).  It is rounded down to whole
    cells, so the grid never allows more than the rule in cm does.  The box's
    height plays no part, only its footprint.
    """
    s0, s1 = (float(v) for v in side_cm)
    a0, a1 = (float(v) for v in allow_cm)
    if not 0 < s0 < s1:
        raise ValueError(f"stack_cap_side_cm is two increasing sides in cm, "
                         f"got {list(side_cm)}")
    if min(a0, a1) < 0:
        raise ValueError(f"stack_cap_allow_cm is two heights in cm, got "
                         f"{list(allow_cm)}")
    if not cell_cm > 0:
        raise ValueError(f"a cell is a positive number of cm, got {cell_cm}")
    side = np.arange(n + 1) * float(cell_cm)
    allow = np.floor(np.interp(side, (s0, s1), (a0, a1)) / cell_cm + CM_EPS)
    return np.where(side < s1 - CM_EPS, allow, NO_CAP).astype(np.int64)


def mixed_tables(n_types, soft=False):
    """`(rule, fallback)` for the `MIXED_RULES` over `n_types` types.

    `rule[k, t, u]` is what a box of type `t` may do on one of type `u` under
    the normal rules (`k = 0`), a box's own fallback (`k = 1`) or the
    station's last resort (`k = 2`): `NO`, `YES`, `IF` the pair's own
    condition holds -- all of its base on blue for blue on blue; (a) or (d)
    for white on blue, and (a), (c) or (d) for brown on blue, (c) dropped
    under soft mix -- or `TOP`, if its top ends near the lid, (a);
    `_type_ok` tests the conditions.  Types past brown keep the `touch` rule.
    Without `soft` (`soft_mix`) blue and white fall back onto anything,
    brown onto blue, and there is no last resort beyond that; with it only
    blue on blue keeps a fallback, and the last resort opens `SOFT_PAIRS`.  `fallback[t]` is
    whether a box's own fallback gives type `t` anything the normal rules do
    not.
    """
    T = int(n_types)
    rule = np.stack([np.where(np.eye(T, dtype=bool), YES, NO)] * 3)
    rule = rule.astype(np.int8)

    def put(ks, pairs, v):
        for t, u in pairs:
            if max(t, u) < T:
                rule[ks, t, u] = v
    put([0], ((BLUE, BLUE), (WHITE, BLUE), (BROWN, BLUE)), IF)
    put([0], TOP_PAIRS, TOP)
    if soft:
        # a fallback for blue on blue alone, then the station's last resort
        rule[1] = rule[2] = rule[0]
        put([1, 2], ((BLUE, BLUE),), YES)
        put([2], SOFT_PAIRS, YES)
    else:
        # blue and white on anything, brown on blue; no last resort
        rule[1] = rule[2] = rule[0]
        put([1, 2], ((BLUE, BLUE), (BLUE, WHITE), (BLUE, BROWN),
                     (WHITE, BLUE), (WHITE, BROWN), (BROWN, BLUE)), YES)
    return rule, (rule[0] != rule[1]).any(1)


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
    m = np.ascontiguousarray(m)
    out = np.empty((kmax,) + m.shape, m.dtype)
    out[0] = m
    mf = m.reshape(-1)
    for k in range(1, kmax):
        # one contiguous max over the flattened array, shifted by k along the
        # swept axis; the tail -- windows the axis end cuts short, which the
        # flat shift fills from the next row -- is then carried over unchanged
        prev, cur = out[k - 1], out[k]
        sh = k if axis == 2 else k * m.shape[2]
        np.maximum(prev.reshape(-1)[:-sh], mf[sh:], out=cur.reshape(-1)[:-sh])
        if axis == 1:
            cur[:, S - k:] = prev[:, S - k:]
        else:
            cur[:, :, S - k:] = prev[:, :, S - k:]
    return out


def _covered(f, sx, sy):
    """(n, Lx, Ly): the cells some footprint `[x, x + sx) x [y, y + sy)` with
    `f[:, x, y]` set covers -- a box sum over the corners, by prefix sums."""
    n, Lx, Ly = f.shape
    c = np.zeros((n, Lx + 1, Ly + 1), np.int32)
    c[:, 1:, 1:] = f.cumsum(1, dtype=np.int32).cumsum(2)
    hx, hy = np.arange(1, Lx + 1)[:, None], np.arange(1, Ly + 1)
    lx, ly = np.maximum(hx - sx, 0), np.maximum(hy - sy, 0)
    return (c[:, hx, hy] - c[:, lx, hy] - c[:, hx, ly] + c[:, lx, ly]) > 0


def _widest(c, cap):
    """How many window widths a sweep has to produce when only widths `c`
    are read back out of it: their maximum, rather than every width up to the
    largest item in the sequence.  A zero side reads window -1, so it keeps
    the full `cap` and the window it always read."""
    return int(c.max()) if c.min() >= 1 else int(cap)


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
    m = np.ascontiguousarray(m)
    om = np.empty((kmax,) + m.shape, m.dtype)
    oc = np.empty((kmax,) + m.shape, np.int32)
    om[0], oc[0] = m, 1 if c is None else c
    mf = m.reshape(-1)
    cf = None if c is None else np.ascontiguousarray(c).reshape(-1)
    for k in range(1, kmax):
        # the flat shift of `_win_max`, then the tail carried over
        sh = k if axis == 2 else k * m.shape[2]
        pm, pc = om[k - 1].reshape(-1)[:-sh], oc[k - 1].reshape(-1)[:-sh]
        am = mf[sh:]
        # beats the running max: its own count; ties it: the two add up;
        # loses: the running count stands.  As arithmetic on the two
        # comparisons, which is several times faster than nested `np.where`.
        cnt = oc[k].reshape(-1)[:-sh]
        np.multiply(pc, am <= pm, out=cnt)
        cnt += (am >= pm) if cf is None else cf[sh:] * (am >= pm)
        np.maximum(pm, am, out=om[k].reshape(-1)[:-sh])
        tail = np.s_[:, S - k:] if axis == 1 else np.s_[:, :, S - k:]
        om[k][tail], oc[k][tail] = om[k - 1][tail], oc[k - 1][tail]
    return om, oc


class BPPBatch:
    """`n_env` independent bins stepped in lockstep."""

    #: the caches that hold one pass's answers -- the `ems` candidates', or
    #: the outline step's, which `_outline` keeps a second set of
    PASS_CACHES = ("_rule_pick", "_hard_pick", "_slot_fz", "_slot_geo",
                   "_slot_soft")

    def __init__(self, n_env, S=None, nb=1, n_items=None, max_c=None,
                 max_l=None, size_lo=None, size_hi=None, seed=0, ems=None,
                 stability=None, rot=None, min_support=None, n_pick=None,
                 n_types=None, types=None, type_constraint=None,
                 type_rule=None, pick_feasible=None, pool=None, pool_order_random=0.0,
                 arm_collision=None, arm_cell_m=None, arm_moves=None,
                 stack_cap=None, stack_types=None, stack_side_cm=None,
                 stack_allow_cm=None, stack_cell_cm=None,
                 mixed_top_gap_cm=None, mixed_area_frac=None,
                 mixed_small_blue=None, soft_mix=None, outline_resort=None,
                 outline_when=None):
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
        # `touch`: a box may not rest on a foreign type -- only the cells it
        # sits on count.  `column`: it may not be anywhere over one -- any
        # box under its footprint counts, however deep and across any gap.
        # `mixed_touch` / `mixed_column`: the cell's blue / white / brown
        # rules on either reading (see `BLUE`).
        self.type_rule = str(e.get("type_rule", "touch") if type_rule is None
                             else type_rule)
        if self.type_rule not in TYPE_RULES:
            raise ValueError(f"type_rule is one of {TYPE_RULES}, "
                             f"got {self.type_rule!r}")
        if self.n_types > 31:
            raise ValueError(f"smap holds one bit per type, so at most 31 "
                             f"types, got {self.n_types}")
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
        # Drop every placement the robot arm cannot make without one of its
        # links hitting the pack (or cannot reach at all) -- `ar2l.pack_collision`,
        # set up from `robot:` in config.yaml.  `arm_cell_m` is how many metres
        # one cell stands for; None derives it from `eval.cell_cm` and
        # `eval.box_scale`, the grid real orders are read onto.  `arm_moves`
        # is where the base may slide to rescue a placement, as
        # `robot.base_x_moves`; None reads that.
        self.arm_collision = bool(e.get("arm_collision", 0)
                                  if arm_collision is None else arm_collision)
        self.arm_cell_m = arm_cell_m
        from . import pack_collision as PC
        self.arm_moves = PC.base_x_moves(
            CFG["robot"] if arm_moves is None else {"base_x_moves": arm_moves})
        self._arm = None
        # The stack-height limit over small boxes (`stack_cap` in config.yaml):
        # a box of one of `stack_types` limits where a later box may start
        # over its footprint, by `stack_allowance`.  A type id the stream never
        # draws simply never sets a limit, so `n_types = 1` is untouched.
        # `stack_cell_cm` is how many cm one cell stands for; None takes the
        # arm filter's cell size -- `arm_cell_m`, or what `_arm_checker`
        # derives without it -- so both rules read the grid in the same real
        # units.  The table is built even when the rule is off, so a caller
        # may switch `stack_cap` on an existing env as the game does the arm.
        self.stack_cap = bool(e["stack_cap"] if stack_cap is None
                              else stack_cap)
        self.stack_types = np.asarray(
            (e["stack_cap_types"] or []) if stack_types is None
            else stack_types, np.int64).reshape(-1)
        cell_cm = (float(stack_cell_cm) if stack_cell_cm is not None else
                   100.0 * (arm_cell_m or PC.robot_cell_m(
                       CFG["robot"], CFG["eval"]["cell_cm"],
                       CFG["eval"]["box_scale"])))
        # indexed by the shorter footprint side, which for a box that fits is
        # at most the shorter side of the bin
        self.stack_allow = stack_allowance(
            min(self.Lx, self.Ly),
            e["stack_cap_side_cm"] if stack_side_cm is None else stack_side_cm,
            e["stack_cap_allow_cm"] if stack_allow_cm is None
            else stack_allow_cm, cell_cm)
        # The thresholds of the `MIXED_RULES`, in real units read onto the
        # grid with the same cell size: (a) the gap under the lid, rounded
        # down to whole cells; (c) the share of the small blue box's footprint
        # a brown box needs; (d) the small blue box itself, given in cm as an
        # order row is -- padded by `eval.box_pad_m` and rounded by
        # `eval.box_round` like every box of the data -- unless
        # `mixed_small_blue` hands it over in cells.
        self.cell_cm = cell_cm
        gap = float(e.get("mixed_top_gap_cm", 5) if mixed_top_gap_cm is None
                    else mixed_top_gap_cm)
        self.area_frac = float(e.get("mixed_area_frac", 0.8)
                               if mixed_area_frac is None else mixed_area_frac)
        if gap < 0 or self.area_frac < 0:
            raise ValueError(f"mixed_top_gap_cm and mixed_area_frac are not "
                             f"negative, got {gap} and {self.area_frac}")
        self.top_gap_cm = gap
        self.top_gap = int(np.floor(gap / cell_cm + CM_EPS))
        if mixed_small_blue is None:
            from .orders import box_cells, box_pad_cm, box_rounding
            cm = e.get("mixed_small_blue_cm", (50, 30, 18))
            if len(cm) != 3:
                raise ValueError(f"mixed_small_blue_cm is the length, width "
                                 f"and height of a box, got {cm}")
            mixed_small_blue = box_cells(
                dict(zip("lwh", cm)), cell_cm, "mixed_small_blue_cm", 1.0,
                box_rounding(), box_pad_cm(), "lwh")
        self.small_blue = _triple(mixed_small_blue)
        # `soft_mix`: the mixed rules trade a box's own fallback -- all but
        # blue on blue's -- for the station's last resort (`SOFT_PAIRS`,
        # `_stuck`), and drop (c).  Both tables are kept, so a caller may
        # switch it on an existing env.
        self.soft_mix = bool(e.get("soft_mix", 0) if soft_mix is None
                             else soft_mix)
        self._tables = {s: mixed_tables(self.n_types, s)
                        for s in (False, True)}
        # the types soft mix's last resort gives anything (`SOFT_PAIRS`)
        self._soft_types = (self._tables[True][0][2]
                            != self._tables[True][0][1]).any(1)
        # `outline_resort`: the outline step after every other (see
        # `OUTLINE_WHEN`), opening per station or per box.  `_ol` is whether
        # its pass is the one in force (`_outline`).
        self.outline_resort = bool(e.get("outline_resort", 0)
                                   if outline_resort is None
                                   else outline_resort)
        self.outline_when = str(e.get("outline_when", "station")
                                if outline_when is None else outline_when)
        if self.outline_when not in OUTLINE_WHEN:
            raise ValueError(f"outline_when is one of {OUTLINE_WHEN}, got "
                             f"{self.outline_when!r}")
        self._ol = False
        # the contact count costs a second sweep, so only pay for it when a
        # rule actually reads it -- the `MIXED_RULES` do too, for blue on blue,
        # which `_sweeps` asks as it goes, as a caller may switch the rule
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

    # Read off `type_rule` as asked, not stored, since a caller may switch it
    @property
    def _mixed(self):
        """Is the rule one of the `MIXED_RULES`?"""
        return self.type_rule in MIXED_RULES

    @property
    def _column(self):
        """Does the rule count every box under the footprint?  The
        `COLUMN_RULES`."""
        return self.type_rule in COLUMN_RULES

    # ---------------------------------------------------------------- reset
    def _invalidate(self, hmap=True):
        """Drop the cached sweeps.  `hmap=False` when only the item changed."""
        self._pos_dirty = True
        # which boxes in the station are placeable depends on the item at each
        # slot, so a change of item drops it even though the bin is untouched
        # (`permute`, which only reorders the items, reorders it instead);
        # `_rule_pick` is the same question short of the outline step, and
        # `_hard_pick` by the hard rules alone, which is what decides whether
        # soft mix steps in (`_stuck`)
        self._pick_cache = None
        self._rule_pick = None
        self._hard_pick = None
        # raw `_feas_one` grids per window slot, {slot: ([feas], [z]) per
        # orientation}, for the bin and items as they stand: `_positions`
        # fills slot 0, the only one anything needs in full
        self._slot_fz = {}
        # the same grids before the arm filter, which `_placeable` and the
        # end of `step` work from: they only ask whether a placement exists,
        # and arm-check as few cells as it takes to answer that
        self._slot_geo = {}
        # the same again by soft mix's last resort, for the slots that ask
        self._slot_soft = {}
        # the outline step's own set of the caches above (`PASS_CACHES`),
        # which `_outline` swaps in; None until it is first run
        self._ol_caches = None
        if hmap:
            self._sweep_cache = None
            self._ems_cache = None
            # keyed by the incoming type vector, so a permutation that brings a
            # different type to the front reuses nothing it should not
            self._type_cache = {}
            self._under_cache = None
            # `cmap` only ever changes with the height map, in `step`
            self._cap_cache = None
            # the blue cells a small blue box could still go on, per `ems`
            # mode -- rule (d) of the `MIXED_RULES` (`_open_sweep`)
            self._open_cache = {}
            # where a box would be on a given type, per reading (`_on_type`)
            self._on_cache = {}
            # arm verdicts keyed by (bin, (sx, sy, sz), x, y): the index into
            # `robot.base_x_moves` of the first base move that makes the
            # placement, -1 for none.  The landing height follows from the
            # height map, which is what resets them -- except in `step`, which
            # knows what changed and keeps the verdicts it cannot have touched
            # (`_keep_arm`).  `_arm_box` is the column range each one rests on.
            self._arm_keys = np.zeros(0, np.int64)
            self._arm_move = np.zeros(0, np.int8)
            self._arm_box = np.zeros((0, 4), np.int64)

    def _set_side(self, seq):
        """Largest footprint extent the sweeps must cover, per axis.

        Rotation offers both (w, l) and (l, w), so either item side can land
        on either axis; the cap keeps `kmax` inside the axis, and any item too
        big for an axis is killed by the `inx`/`iny` masks anyway.
        """
        side = max(self.size_hi[0], self.size_hi[1], int(seq[..., :2].max()))
        if self._mixed:
            # the small blue box of rule (d) is swept like any item
            side = max(side, self.small_blue[0], self.small_blue[1])
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

    def _rim(self):
        """(Lx + 1, Ly + 1): the pallet's own edges on the `omap` lattice."""
        r = np.zeros((self.Lx + 1, self.Ly + 1), bool)
        r[[0, -1], :] = r[:, [0, -1]] = True
        return r

    def reset(self, seqs=None):
        n = self.n_env
        self.hmap = np.zeros((n, self.Lx, self.Ly), np.int16)
        # the type of the topmost box in each column, and so of whatever a
        # placement landing there would rest on
        self.tmap = np.full((n, self.Lx, self.Ly), TYPE_FLOOR, np.int8)
        # a bit per type that has a box anywhere in the column, which the
        # `column` type rule tests against
        self.smap = np.zeros((n, self.Lx, self.Ly), np.int32)
        # how high a later box may start over each column: below this, which
        # a small box of a limited type lowers over its footprint
        self.cmap = np.full((n, self.Lx, self.Ly), NO_CAP, np.int16)
        # the outlines the outline step offers (`OUTLINE_WHEN`), on the
        # lattice of cell corners: point (i, j) is where cells (i - 1, j - 1)
        # and (i, j) meet.  The pallet's edges, then every packed box's.
        self.omap = np.repeat(self._rim()[None], n, 0)
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
        self.smap[idx] = 0
        self.cmap[idx] = NO_CAP
        self.omap[idx] = self._rim()
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
        # the arm verdicts of every bin that was not reset still hold: they
        # depend on that bin's height map alone, which is untouched
        keep = ~np.isin(self._arm_keys // self._arm_bin_stride(), idx)
        keys, move, box = (self._arm_keys[keep], self._arm_move[keep],
                           self._arm_box[keep])
        self._invalidate()
        self._arm_keys, self._arm_move, self._arm_box = keys, move, box

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

        By the rules (`_rule_placeable`) -- and, for the boxes they leave
        nothing where the outline step opens, by it (`_outline_gate`).
        """
        if self._pick_cache is None:
            out = self._rule_placeable()
            gate = self._outline_gate()
            if gate.any():
                with self._outline():
                    out = out | (gate & self._rule_placeable())
            self._pick_cache = out
        return self._pick_cache

    def _rule_placeable(self):
        """(B, nb) `_placeable` short of the outline step, over the positions
        of the pass in force: by the hard rules (`_hard_placeable`) -- and, in
        the bins where none of them has one and soft mix is on, by its last
        resort (`_stuck`)."""
        if self._rule_pick is not None:
            return self._rule_pick
        out = self._hard_placeable().copy()
        stuck = self._stuck()
        if stuck.any():
            k = min(self.n_pick, self.nb)
            out[:, :k] |= stuck[:, None] & self._soft_placeable(k)
        self._rule_pick = out
        return out

    def _outline_on(self):
        """Can the outline step give anything here at all?  Not with every
        position a candidate already (`ems` 0)."""
        return bool(self.outline_resort and self._mixed
                    and self.type_constraint and self.n_types > 1
                    and self.ems != EMS_OFF)

    def _outline_gate(self):
        """(B, nb) the boxes within reach that go on to the outline step:
        the rules leave every one of them nothing (`outline_when: station`),
        or this one (`box`)."""
        out = np.zeros((self.n_env, self.nb), bool)
        if not self._outline_on():
            return out
        k = min(self.n_pick, self.nb)
        none = ~self._rule_placeable()[:, :k]
        if self.outline_when == "station":
            none = none.all(1, keepdims=True)
        alive = ~self.done & (self.head < self.length)
        out[:, :k] = none & self.window()[1][:, :k] & alive[:, None]
        return out

    def _rules_stuck(self):
        """(n,) the bins where no box within reach has a placement by the
        rules, soft mix's last resort included -- what soft mix waits for in
        the outline step's pass.  All False without soft mix."""
        if not self._soft_on():
            return np.zeros(self.n_env, bool)
        k = min(self.n_pick, self.nb)
        alive = ~self.done & (self.head < self.length)
        return alive & ~self._rule_placeable()[:, :k].any(1)

    @contextlib.contextmanager
    def _outline(self):
        """Run the rules again in the outline step's pass: `_feas_geo` adds
        the outline positions to the `ems` candidates, and every per-pass
        cache (`PASS_CACHES`) is the pass's own, kept across calls until the
        bin or the items change.  Soft mix's last resort there waits for the
        station to be stuck at the `ems` candidates too (`_rules_stuck`)."""
        stuck = self._rules_stuck()
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

    def _hard_placeable(self):
        """(B, nb) `_placeable` by the hard rules: the normal ones, or a box's
        own fallback where they leave it nothing -- soft mix aside.

        One feasibility sweep per reachable slot -- the sweeps that depend on
        the bin alone are shared, only the per-footprint windows are redone --
        and the preview tail is never asked, since it cannot be chosen anyway.
        Only existence is asked, so the arm filter is not run over whole grids
        but through `_arm_any`, every slot and orientation in one batch.
        """
        if self._hard_pick is not None:
            return self._hard_pick
        win, valid = self.window()
        k = min(self.n_pick, self.nb)
        out = np.zeros((self.n_env, self.nb), bool)
        alive = ~self.done & (self.head < self.length)
        # an invalid slot is zero-sized, and a zero side would index window
        # -1; the mask below drops it either way
        dims = np.maximum(win[:, :k, :3], 1)
        todo = [i for i in range(k)
                if i not in self._slot_fz and i not in self._slot_geo]
        if todo:                  # every slot's grids in one go: `_slot_grids`
            got = self._slot_grids([(dims[:, i], win[:, i, 3]) for i in todo])
            self._slot_geo.update(zip(todo, got))
        grids, slot = [], []
        for i in range(k):
            if i in self._slot_fz:            # the full grid is known already
                for f in self._slot_fz[i][0]:
                    out[:, i] |= f.any((1, 2))
                continue
            fs, zs = self._slot_geo[i]
            for d, f, z in zip(self._orients(dims[:, i]), fs, zs):
                grids.append((f, z, d)); slot.append(i)
        if grids:
            hit = self._arm_any(grids)
            for g, i in enumerate(slot):
                out[:, i] |= hit[:, g]
        out[:, :k] &= valid[:, :k] & alive[:, None]
        self._hard_pick = out
        return out

    def _soft_on(self):
        """Can soft mix give anything here at all?"""
        return bool(self.soft_mix and self._mixed and self.type_constraint
                    and self.n_types > 1 and self._soft_types.any())

    def _stuck(self):
        """(n,) the bins where soft mix steps in: it is on, and no box within
        reach has a placement by the hard rules -- the normal ones or its own
        fallback -- the arm included.  Its last resort then opens to every
        box within reach: `SOFT_PAIRS`, with no condition.  In the outline
        step's pass only where nothing within reach could go by the `ems`
        candidates either."""
        if not self._soft_on():
            return np.zeros(self.n_env, bool)
        k = min(self.n_pick, self.nb)
        alive = ~self.done & (self.head < self.length)
        if self._ol:
            alive &= self._ol_stuck
            if not alive.any():           # spare the station's sweeps
                return alive
        return alive & ~self._hard_placeable()[:, :k].any(1)

    def last_resort(self):
        """(n,) does the leading item go by soft mix's last resort?  Its
        type is one soft mix gives anything, and the station is stuck."""
        stuck = self._stuck()
        if not stuck.any():
            return stuck
        return stuck & self._soft_types[self.head_item()[1].astype(np.int64)]

    def _soft_slot(self, i, dims, types):
        """Window slot `i`'s grids before the arm filter by soft mix's last
        resort -- the fallback, `SOFT_PAIRS` open -- cached as `_geo_slot`'s
        are.  `i=None` is the leading item, asked of no slot and kept."""
        if i is None or i not in self._slot_soft:
            fz = [self._feas_one(d, types, arm=False, tier=3)
                  for d in self._orients(dims)]
            got = ([f for f, _ in fz], [z for _, z in fz])
            if i is None:
                return got
            self._slot_soft[i] = got
        return self._slot_soft[i]

    def _soft_placeable(self, k):
        """(n, k) does each box within reach have a placement by soft mix's
        last resort, the arm included?  Only the types soft mix gives
        anything can, so only their slots are asked."""
        win, valid = self.window()
        dims = np.maximum(win[:, :k, :3], 1)
        types = win[:, :k, 3].astype(np.int64)
        out = np.zeros((self.n_env, k), bool)
        grids, slot = [], []
        for i in range(k):
            if not self._soft_types[types[:, i]].any():
                continue
            fs, zs = self._soft_slot(i, dims[:, i], win[:, i, 3])
            for d, f, z in zip(self._orients(dims[:, i]), fs, zs):
                grids.append((f, z, d)); slot.append(i)
        if grids:
            hit = self._arm_any(grids)
            for g, i in enumerate(slot):
                out[:, i] |= hit[:, g]
        return out & valid[:, :k] & self._soft_types[types]

    def _orients(self, dims):
        """The orientations an item is offered in: as given, then yawed."""
        return [dims] if self.rot < 2 else [dims, dims[:, [1, 0, 2]]]

    def _geo_slot(self, i, dims, types):
        """Window slot `i`'s grids before the arm filter, cached until the bin
        or the item there changes."""
        if i not in self._slot_geo:
            self._slot_geo[i] = self._slot_grids([(dims, types)])[0]
        return self._slot_geo[i]

    def _slot_grids(self, items):
        """`([feas], [z])` per orientation, before the arm filter, for each
        `(dims, types)` in `items` -- one box per bin each.

        Under the `MIXED_RULES` a box whose type has fallback rules goes by
        them in the bins where the normal rules leave it no placement at all
        -- in no orientation, the arm included.  `_arm_any` asks the arm that
        for every item in one batch, and only as far as it takes to find a
        placement; the geometry is swept once for both sets of rules.
        """
        out, wait, grids = [], {}, []
        for j, (dims, types) in enumerate(items):
            orients = self._orients(dims)
            fb = self._fallback_bins(types)
            if fb.any():
                geo = [self._feas_geo(d) for d in orients]
                fs = [g & self._type_ok(d, z, types, 1, full)
                      for d, (g, z, full) in zip(orients, geo)]
                zs = [z for _, z, _ in geo]
                wait[j] = (len(grids), orients, types, fb, geo)
            else:
                fz = [self._feas_one(d, types, arm=False) for d in orients]
                fs, zs = [f for f, _ in fz], [z for _, z in fz]
            out.append((fs, zs))
            grids += [(f, z, d) for d, f, z in zip(orients, fs, zs)]
        if not wait:
            return out
        # every item's grids go in the batch, not only the ones waiting on it:
        # the arm verdicts are kept, so whoever asks next about the rest --
        # `_placeable`, `_head_placeable` -- finds them answered
        hit = self._arm_any(grids)
        for j, (g0, orients, types, fb, geo) in wait.items():
            late = fb & ~hit[:, g0:g0 + len(orients)].any(1)
            if late.any():
                fs, zs = out[j]
                out[j] = ([np.where(late[:, None, None],
                                    g & self._type_ok(d, z, types, 2, full), f)
                           for d, f, (g, z, full) in zip(orients, fs, geo)],
                          zs)
        return out

    def _fallback_bins(self, types):
        """(n,) the bins whose box, of `types`, has fallback rules at all."""
        if not (self._mixed and self.type_constraint and self.n_types > 1):
            return np.zeros(self.n_env, bool)
        return self._tables[self.soft_mix][1][np.asarray(types, np.int64)]

    def fallback(self):
        """(n,) does the leading item go by the mixed rules' fallback?

        True where its type has them and the normal rules leave it no
        placement -- the choice `_slot_grids` makes, asked again for a caller
        that has to say why a cell is or is not legal.  False under every
        other rule.
        """
        item, types = self.head_item()
        fb = self._fallback_bins(types)
        if not fb.any():
            return fb
        grids = []
        for d in self._orients(item):
            f, z = self._feas_one(d, types, arm=False)
            grids.append((f, z, d))
        return fb & ~self._arm_any(grids).any(1)

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
        self._permute_caches(idx, order)

    def _permute_caches(self, idx, order):
        """Carry the per-item caches through `permute` instead of redoing them.

        The bin is untouched and every item is still in the window, only at
        another slot: new slot `j` of bin `b` holds what old slot
        `order[b, j]` held, so its placeability and its feasibility grids are
        that slot's.  A move from past the station, or one that reaches past
        the end of the sequence (where the window's clamped offsets alias),
        has nothing cached to carry and drops the caches as before.
        """
        pick, rule, hard = self._pick_cache, self._rule_pick, self._hard_pick
        slots, geo = self._slot_fz, self._slot_geo
        self._invalidate(hmap=False)
        moved = idx > 0
        k = min(self.n_pick, self.nb)
        if (idx >= k).any() or (self.head + idx >= self.length)[moved].any():
            return
        ar = self._ar
        if pick is not None:
            self._pick_cache = pick[ar[:, None], order]
        if rule is not None:
            self._rule_pick = rule[ar[:, None], order]
        if hard is not None:
            self._hard_pick = hard[ar[:, None], order]
        self._slot_fz = self._carry(slots, order, k)
        self._slot_geo = self._carry(geo, order, k)

    @staticmethod
    def _carry(slots, order, k):
        """{slot: ([feas], [z])} as it reads after the window is reordered."""
        out = {}
        for j in range(k):
            src = order[:, j]
            have = [s for s in np.unique(src).tolist() if s in slots]
            if len(have) < len(np.unique(src)):
                continue
            nf = len(slots[have[0]][0])
            fs = [slots[have[0]][0][r].copy() for r in range(nf)]
            zs = [slots[have[0]][1][r].copy() for r in range(nf)]
            for s in have[1:]:
                m = src == s
                for r in range(nf):
                    fs[r][m] = slots[s][0][r][m]
                    zs[r][m] = slots[s][1][r][m]
            out[j] = (fs, zs)
        return out

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

        The whole batch goes at once: every bin's candidate x-spans and
        y-spans are listed flat, each x-span is paired with the y-spans of its
        own bin, and the floor and the four one-cell extensions of every pair
        are a handful of gathers.  There are only ~10^4 pairs over 64 bins, so
        a Python loop over bins and left edges was nearly all overhead.
        """
        Lx, Ly, H = self.Lx, self.Ly, self.hmap
        n = H.shape[0]
        big = np.int16(self.Lz + 1)           # taller than any column can be
        tru = np.ones((n, 1), bool)
        x0c = np.concatenate([tru, (H[:, :-1] > H[:, 1:]).any(2)], 1)  # (n, Lx)
        x1c = np.concatenate([(H[:, 1:] > H[:, :-1]).any(2), tru], 1)
        y0c = np.concatenate([tru, (H[:, :, :-1] > H[:, :, 1:]).any(1)], 1)
        y1c = np.concatenate([(H[:, :, 1:] > H[:, :, :-1]).any(1), tru], 1)
        # every (bin, near edge, far edge) span, in (bin, near, far) order
        pe, px0, px1 = np.nonzero(x0c[:, :, None] & x1c[:, None, :]
                                  & np.triu(np.ones((Lx, Lx), bool)))
        qe, qy0, qy1 = np.nonzero(y0c[:, :, None] & y1c[:, None, :]
                                  & np.triu(np.ones((Ly, Ly), bool)))
        if not len(pe) or not len(qe):
            z = np.zeros(0, np.int64)
            return (z,) * 6
        ax = _win_max(H, 1, Lx)               # ax[wx-1, n, x0, y]  max over x
        ay = _win_max(H, 2, Ly)               # ay[wy-1, n, x, y0]  max over y
        pwx = px1 - px0 + 1
        a = ax[pwx - 1, pe, px0]              # (P, Ly) max over each x-span
        wa = _win_max(a[None], 2, Ly)[:, 0]   # wa[wy-1, p, y0]
        # pair each x-span with every y-span of its own bin
        qcount = np.bincount(qe, minlength=n)
        qstart = np.cumsum(qcount) - qcount
        cnt = qcount[pe]
        p = np.repeat(np.arange(len(pe)), cnt)
        q = (qstart[pe][p] + np.arange(cnt.sum())
             - np.repeat(np.cumsum(cnt) - cnt, cnt))
        e, x0, x1, wx = pe[p], px0[p], px1[p], pwx[p]
        y0, y1 = qy0[q], qy1[q]
        wy = y1 - y0 + 1
        flr = wa[wy - 1, p, y0]
        # the same four one-cell extensions as the exhaustive sweep;
        # `back`/`front` are the x-span max one row outside, which is what `a`
        # already holds
        lo = np.where(x0 > 0, ay[wy - 1, e, np.maximum(x0 - 1, 0), y0], big)
        hi = np.where(x1 + 1 < Lx,
                      ay[wy - 1, e, np.minimum(x1 + 1, Lx - 1), y0], big)
        bk = np.where(y0 > 0, a[p, np.maximum(y0 - 1, 0)], big)
        ft = np.where(y1 + 1 < Ly, a[p, np.minimum(y1 + 1, Ly - 1)], big)
        f = np.flatnonzero((lo > flr) & (hi > flr) & (bk > flr) & (ft > flr))
        # the order the per-bin loop produced: bin, left edge, y-span, right edge
        f = f[np.lexsort((x1[f], q[f], x0[f], e[f]))]
        return (e[f], x0[f], y0[f], wx[f], wy[f], flr[f].astype(np.int32))

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
        `z`, so it is a handful of window reads over the grid.

        Every column read is the height map shifted by an offset that is the
        same over the whole grid of one bin -- 0 or 1 at the near side, the
        footprint side (+1) at the far one -- so each is one block copy per bin
        out of a wall-padded map rather than a per-cell gather.  A footprint
        that overruns the bin is never a candidate, and reads False.
        """
        Lx, Ly, n = self.Lx, self.Ly, self.n_env
        sx, sy = dims[:, 0].astype(np.int64), dims[:, 1].astype(np.int64)
        kx, ky = int(sx.max()), int(sy.max())
        # wide enough that the far neighbour of an overrunning footprint is
        # still inside the array; `+ 1` is the near wall
        p = np.full((n, Lx + kx + 2, Ly + ky + 2), self.Lz + 1, np.int16)
        p[:, 1:Lx + 1, 1:Ly + 1] = self.hmap
        win = np.lib.stride_tricks.sliding_window_view(p, (Lx, Ly), axis=(1, 2))
        ar = self._ar

        def at(ox, oy):
            """`p[b, gx + ox[b], gy + oy[b]]` over the (Lx, Ly) grid."""
            if np.isscalar(ox) and np.isscalar(oy):
                return p[:, ox:ox + Lx, oy:oy + Ly]
            return win[ar, ox, oy]

        z = z.astype(np.int16)
        # near and far column of the footprint along each axis, as the offset
        # of that column and of its outside neighbour
        xs = ((1, 0), (sx, sx + 1))
        ys = ((1, 0), (sy, sy + 1))
        out = np.zeros((n, Lx, Ly), bool)
        for cx, nbx in xs:
            for cy, nby in ys:
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
        """(n, Lx, Ly): some corner of the footprint lies on an outline.

        The footprint `[x, x + sx) x [y, y + sy)` has its corners at lattice
        points (x, y), (x + sx, y), (x, y + sy) and (x + sx, y + sy) of
        `omap`, so each is the lattice shifted by an offset that is the same
        over the whole grid of one bin -- a block copy per bin, as in
        `_corner_mask`.  A footprint that overruns the bin reads whatever
        its clamped corner does; `inx`/`iny` drop it anyway.
        """
        Lx, Ly = self.Lx, self.Ly
        sx = np.clip(dims[:, 0].astype(np.int64), 1, Lx)
        sy = np.clip(dims[:, 1].astype(np.int64), 1, Ly)
        p = np.zeros((self.n_env, Lx + int(sx.max()), Ly + int(sy.max())),
                     bool)
        p[:, :Lx + 1, :Ly + 1] = self.omap
        win = np.lib.stride_tricks.sliding_window_view(p, (Lx, Ly),
                                                       axis=(1, 2))
        ar = self._ar
        return (p[:, :Lx, :Ly] | win[ar, sx, 0] | win[ar, 0, sy]
                | win[ar, sx, sy])

    # --------------------------------------------------------- feasibility
    def _sweeps(self):
        """Height-map sweeps that depend on the bin but not on the item.

        Returns `(my, myc, mx)`: the y-window maxima, how many cells of each
        y-window attain them (`None` when no rule needs contact area), and the
        x-window maxima.
        """
        if self._sweep_cache is None:
            if self._needs_count or self._mixed:
                my, myc = _win_maxcount(self.hmap, None, 2, self.Ly, self.sidey)
            else:
                my, myc = _win_max(self.hmap, 2, self.Ly, self.sidey), None
            self._sweep_cache = (my, myc,
                                 _win_max(self.hmap, 1, self.Lx, self.sidex))
        return self._sweep_cache

    def _type_sweep(self, types, tier=1, strict=False):
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

        Under the `COLUMN_RULES` a column is blocked outright -- at a height
        no landing reaches -- when any box in it, not only the top one, is of
        a type the incoming one may not be over, so a footprint that covers it
        at all is illegal.  Which types those are is every other type, except
        under the `MIXED_RULES`: the ones `mixed_tables` forbids at `tier`, the
        pairs that hang on a condition left open here for `_type_ok` -- but
        for `strict`, which shuts the `TOP` pairs too: the map for a box whose
        top does not end near the lid.

        Cached on the type vector rather than dropped on every permutation: the
        map depends on the bin and on the incoming type, and a permutation
        changes only the latter.
        """
        key = (types.tobytes(), tier, self.soft_mix, strict)
        if key not in self._type_cache:
            t = types.astype(np.int64)
            # `forbid[b, u]`: may bin b's incoming box not be on type u?
            if self._mixed:
                rows = self._tables[self.soft_mix][0][tier - 1][t]
                forbid = (rows == NO) | (strict & (rows == TOP))
            else:
                forbid = t[:, None] != np.arange(self.n_types)
            if self._column:
                bits = (forbid.astype(np.int64)
                        << np.arange(self.n_types)).sum(1).astype(np.int32)
                bad = np.where((self.smap & bits[:, None, None]) != 0,
                               np.int16(np.iinfo(np.int16).max), np.int16(-1))
            else:
                on = forbid[self._ar[:, None, None], np.maximum(self.tmap, 0)]
                bad = np.where((self.hmap > 0) & on, self.hmap, np.int16(-1))
            self._type_cache[key] = _win_max(bad, 2, self.Ly, self.sidey)
        return self._type_cache[key]

    def _type_ok(self, dims, z, types, tier=1, full=None):
        """(n, Lx, Ly): may an item of this type rest on this footprint?

        `tier` 2 asks a box's own fallback under the `MIXED_RULES` rather
        than their normal rules, and 3 soft mix's last resort.  `full` --
        every cell of the footprint reaches `z` -- is what blue on blue asks;
        `_feas_one` has it at hand, and without it it is swept here.
        """
        cx = np.minimum(dims[:, 0], self.sidex)
        cy = np.minimum(dims[:, 1], self.sidey)
        by = self._type_sweep(types, tier)[cy - 1, self._ar]
        ok = _win_max(by, 1, self.Lx, _widest(cx, self.sidex))[cx - 1, self._ar] < z
        if not self._mixed:
            return ok
        t = np.asarray(types, np.int64)
        rows = self._tables[self.soft_mix][0][tier - 1][t]
        # (a): the box's top ends within `top_gap` of the lid
        near = (z + dims[:, 2].astype(np.int32)[:, None, None]
                >= self.Lz - self.top_gap)
        if (rows == TOP).any():
            # a `TOP` pair holds near the lid, and is shut anywhere else
            by = self._type_sweep(types, tier, strict=True)[cy - 1, self._ar]
            ok &= near | (_win_max(by, 1, self.Lx, _widest(cx, self.sidex))
                          [cx - 1, self._ar] < z)
        # the bins whose box may be on blue only if its pair's condition holds
        cond = rows[:, BLUE] == IF
        blue = cond & (t == BLUE)
        if blue.any():
            # blue on blue rests on blue with all of its base; on the floor
            # the whole base rests anyway.  Where blue may be on nothing but
            # blue that is any blue box off the floor; with a `TOP` pair a
            # blue box near the lid may be on white or brown alone, which
            # asks nothing of its base, so ask where it is on blue
            full = self._full(dims) if full is None else full
            if (rows == TOP).any():
                full = full | ~self._on_type(dims, z, BLUE)
            ok &= ~blue[:, None, None] | full
        # white or brown on blue: its top near the lid (a), a large brown (c)
        # -- not under soft mix -- or on blue that no small blue box could
        # still go on (d), which with no blue under the footprint it is
        big = (t == BROWN) & self._big_brown(dims) & (not self.soft_mix)
        need = cond & (t != BLUE) & ~big
        if need.any():
            ok &= ~need[:, None, None] | near | ~self._on_open_blue(dims, z)
        return ok

    def _on_type(self, dims, z, u):
        """(n, Lx, Ly): would the footprint be on a box of type `u`?  Resting
        on one under the touch reading; under the column reading, with one
        anywhere under it -- `smap` -- however deep."""
        key = (u, self._column)
        if key not in self._on_cache:
            if self._column:
                m = np.where((self.smap >> u) & 1 != 0, np.int16(0),
                             np.int16(-1))
            else:
                m = np.where((self.hmap > 0) & (self.tmap == u), self.hmap,
                             np.int16(-1))
            self._on_cache[key] = _win_max(m, 2, self.Ly, self.sidey)
        cx = np.minimum(dims[:, 0], self.sidex)
        cy = np.minimum(dims[:, 1], self.sidey)
        top = _win_max(self._on_cache[key][cy - 1, self._ar], 1, self.Lx,
                       _widest(cx, self.sidex))[cx - 1, self._ar]
        # under the touch reading no cell of type `u` is above `z`, so one at
        # `z` is a cell the box rests on
        return top >= 0 if self._column else top == z

    def _full(self, dims):
        """(n, Lx, Ly): does every cell of the footprint reach the landing
        height -- is the count of cells attaining the maximum the whole
        base?"""
        my, myc, _ = self._sweeps()
        cx = np.minimum(dims[:, 0], self.sidex)
        cy = np.minimum(dims[:, 1], self.sidey)
        _, cxy = _win_maxcount(my[cy - 1, self._ar], myc[cy - 1, self._ar], 1,
                               self.Lx, _widest(cx, self.sidex))
        base = dims[:, 0].astype(np.int32) * dims[:, 1]
        return cxy[cx - 1, self._ar] == base[:, None, None]

    def _big_brown(self, dims):
        """(n,) is the footprint at least `area_frac` of the small blue's?
        Rule (c), which the box's orientation does not change."""
        blue = self.small_blue[0] * self.small_blue[1]
        area = dims[:, 0].astype(np.int64) * dims[:, 1]
        return area >= self.area_frac * blue - SUPPORT_EPS

    def _on_open_blue(self, dims, z):
        """(n, Lx, Ly): would the footprint be on a blue cell that a small
        blue box could still go on (`_open_sweep`)?

        Under `mixed_touch` that is a cell the box rests on: no open cell
        under the footprint is above `z`, so one at `z` is a contact cell.
        Under `mixed_column` it is any open cell under the footprint, one the
        box only hangs over included -- an open cell is never below the floor,
        so any at all reads 0 or more.
        """
        by = self._open_sweep()
        if by is None:
            return np.zeros(np.shape(z), bool)
        cx = np.minimum(dims[:, 0], self.sidex)
        cy = np.minimum(dims[:, 1], self.sidey)
        top = _win_max(by[cy - 1, self._ar], 1, self.Lx,
                       _widest(cx, self.sidex))[cx - 1, self._ar]
        return top >= 0 if self._column else top == z

    def _open_sweep(self):
        """y-window maxima of the height of every open blue cell, -1 for every
        other cell; None when no cell is open.

        A blue cell is open when the small blue box of rule (d) could still go
        on it: some placement the normal rules allow it -- every filter
        passed, the arm's included -- rests on the cell with all of its base.
        Blue on blue asks that anyway; near the lid it may go on white or
        brown too (`TOP_PAIRS`), and then only its blue cells open.  A cell closes when the small blue no longer fits under the lid
        there, or the arm cannot put one there.  It depends on the bin alone,
        so it is kept until the next `step` -- per `ems` mode, since the
        candidate filter decides which placements there are.
        """
        if self.ems not in self._open_cache:
            n = self.n_env
            sb = np.tile(np.asarray(self.small_blue, np.int16), (n, 1))
            blue = np.full(n, BLUE, np.int16)
            grids = []
            for r, d in enumerate(self._orients(sb)):
                if r and sb[0, 0] == sb[0, 1]:
                    break                     # a square turns into itself
                # by the `ems` candidates alone, in the outline step's pass too
                f, z = self._feas_one(d, blue, arm=False, outline=False)
                # all of its base rests, so the cells it covers are the cells
                # it rests on
                grids.append((d, z, f & (z > 0) & self._full(d)))
            fs = ([f for _, _, f in grids] if not self.arm_collision
                  else self._arm_clear_many(grids))
            cover = np.zeros((n, self.Lx, self.Ly), bool)
            for (d, _, _), f in zip(grids, fs):
                cover |= _covered(f, int(d[0, 0]), int(d[0, 1]))
            cover &= self.tmap == BLUE
            self._open_cache[self.ems] = (
                _win_max(np.where(cover, self.hmap, np.int16(-1)), 2, self.Ly,
                         self.sidey) if cover.any() else None)
        return self._open_cache[self.ems]

    def _under_sweep(self):
        """y-window maxima of the height map with the column type packed in.

        `code = h * (n_types + 1) + (t + 1)` is monotone in the height, since
        `t + 1` never reaches the stride, so the window maximum still carries
        the landing height `z` in its quotient and, among the cells attaining
        it, the largest type in its remainder.  A bare column codes as -1, so
        an all-floor footprint reports `TYPE_FLOOR`.

        Under `touch` and `column` every cell a legal placement touches carries
        the same type, so the remainder *is* the type underneath; under the
        `MIXED_RULES` or with the rule switched off it is the largest of the
        types the item straddles.
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
        m = _win_max(by, 1, self.Lx, _widest(cx, self.sidex))[cx - 1, self._ar]
        return np.where(m < 0, np.int8(TYPE_FLOOR),
                        (m % (self.n_types + 1) - 1).astype(np.int8))

    def _cap_ok(self, dims, z):
        """(n, Lx, Ly): may a box start at `z` over every cell of the footprint?

        The limit that binds is the lowest `cmap` under the footprint, cells
        the box only overhangs included, as on the cell.  The sweeps take
        maxima, so they run on `-cmap`; the y-sweep depends on the bin alone
        and is kept until the next `step`.  Until a small box has limited some
        column of the batch every answer is yes and nothing is swept.
        """
        if self._cap_cache is None:
            capped = bool((self.cmap < NO_CAP).any())
            self._cap_cache = (capped, _win_max(-self.cmap, 2, self.Ly,
                                                self.sidey) if capped else None)
        capped, by = self._cap_cache
        if not capped:
            return True
        cx = np.minimum(dims[:, 0], self.sidex)
        cy = np.minimum(dims[:, 1], self.sidey)
        lowest = -_win_max(by[cy - 1, self._ar], 1, self.Lx,
                           _widest(cx, self.sidex))[cx - 1, self._ar]
        return z < lowest

    def _feas_one(self, dims, types=None, arm=True, tier=1, outline=None):
        """Feasible (x, y) grid and landing height for one orientation.

        `arm=False` leaves out the arm filter, the last and dearest rule, for
        callers that only need to know whether a placement exists.  `tier` 2
        takes a box's own fallback under the `MIXED_RULES` (`_slot_grids`
        decides when), and 3 soft mix's last resort (`_stuck` decides when).
        `outline` as `_feas_geo` takes it.
        """
        feas, z, full = self._feas_geo(dims, outline)
        if self.type_constraint and self.n_types > 1 and types is not None:
            feas &= self._type_ok(dims, z, types, tier, full)
        if self.arm_collision and arm:
            feas = self._arm_clear(dims, z, feas)
        return feas, z

    def _feas_geo(self, dims, outline=None):
        """`_feas_one` short of the type rule and the arm: `(feas, z, full)`.

        `full` -- every cell of the footprint reaches `z` -- is what blue on
        blue asks under the `MIXED_RULES`, and None under the other rules.
        `outline` adds the outline positions to the `ems` candidates
        (`_outline_mask`); None reads it off the pass in force (`_outline`).
        """
        outline = self._ol if outline is None else outline
        Lx, Ly, Lz = self.Lx, self.Ly, self.Lz
        sx, sy, sz = dims[:, 0], dims[:, 1], dims[:, 2]
        ar, gx, gy = self._ar, np.arange(Lx), np.arange(Ly)
        my, myc, mx = self._sweeps()
        # an item too long for an axis still has to index a window that exists;
        # `inx`/`iny` below drop it, so the clamped sweep is never read out
        cx = np.minimum(sx, self.sidex)
        cy = np.minimum(sy, self.sidey)
        kx, ky = _widest(cx, self.sidex), _widest(cy, self.sidey)

        myd = my[cy - 1, ar]                          # max over the y extent
        if myc is None:                               # ... and x sub-windows
            mxy, cxy = _win_max(myd, 1, Lx, kx), None
        else:
            mxy, cxy = _win_maxcount(myd, myc[cy - 1, ar], 1, Lx, kx)
        z = mxy[cx - 1, ar].astype(np.int32)

        if self.stability == "com":
            mxd = mx[cx - 1, ar]
            myx = _win_max(mxd, 2, Ly, ky)            # sub-windows along y

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
        if self.ems != EMS_OFF:
            if self.ems == EMS_SPACES:
                cand = self._ems_corners(dims)
            elif self.ems == EMS_CORNER:
                cand = self._corner_mask(dims, z)
            else:
                cand = self._ems_corners(dims) | self._corner_mask(dims, z)
            if outline:
                cand |= self._outline_mask(dims)
            feas &= cand
        if self.stack_cap:
            feas &= self._cap_ok(dims, z)
        full = None
        if self._mixed:
            full = cxy[cx - 1, ar] == (sx.astype(np.int32) * sy)[:, None, None]
        return feas, z, full

    def _arm_checker(self):
        if self._arm is None:
            from . import pack_collision as PC
            rb = CFG["robot"]
            m = self.arm_cell_m or PC.robot_cell_m(
                rb, CFG["eval"]["cell_cm"], CFG["eval"]["box_scale"])
            self._arm = PC.checker_from_config(rb, m)
            self._arm_bases = PC.moved_bases(self._arm[1], self.arm_moves,
                                             self.Ly * m)
        return self._arm

    def _arm_bin_stride(self):
        """The arm-verdict key is `bin * this + (size, x, y)`."""
        return 1024 ** 3 * self.Lx * self.Ly

    def _arm_clear(self, dims, z, feas):
        """`feas` less the placements the arm collides on.

        Only the cells every other rule already allows are asked, all bins in
        one batched check (`ArmPackChecker.collides_batch`), and each answer
        is kept until the height map changes.  An unreachable pose counts as
        a collision, as it does on the cell.  A placement that collides from
        where the base stands is tried again from each of
        `robot.base_x_moves` in turn, and is dropped only if all of them fail.
        """
        return self._arm_clear_many([(dims, z, feas)])[0]

    def _arm_clear_many(self, grids):
        """`_arm_clear` for several `(dims, z, feas)` grids -- the
        orientations of one item, say -- asked of the arm in one batch."""
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
        """For each placement -- bin, x, y, item size, landing height -- the
        index of the first base move it can be made from, -1 for none.

        Answers come from the verdict cache where it has them; the rest are
        asked of the checker, each distinct placement once, and cached along
        with the columns the answer rests on (see `_keep_arm`).
        """
        sizes = sizes.astype(np.int64)
        # one int64 per (bin, item size, x, y), looked up in a sorted table
        if (sizes >= 1024).any():
            raise ValueError(f"item side {int(sizes.max())} is too large for "
                             f"the arm-verdict cache key")
        key = (bb * self._arm_bin_stride()
               + (((sizes[:, 0] * 1024 + sizes[:, 1]) * 1024 + sizes[:, 2])
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
            chk, _ = self._arm_checker()
            b, x, y = bb[todo], xx[todo], yy[todo]
            poses = np.stack([x, y, zz[todo]], 1)
            res = np.full(len(todo), -1, np.int8)
            left = np.arange(len(todo))   # still colliding from every base so far
            for k, B in enumerate(self._arm_bases):
                hit = chk.collides_batch(sizes[todo[left]], poses[left], B,
                                         self.hmap, b[left])
                res[left[~hit]] = k
                left = left[hit]
                if not len(left):
                    break
            move[todo] = res
            # what each verdict rests on: its own footprint, whose tallest
            # column is its landing height and so its pose, and the columns
            # the move that made it reads.  The moves before that one
            # collided, and stay collisions while heights only rise.
            st = sizes[todo]
            box = np.stack([x, x + st[:, 0] - 1, y, y + st[:, 1] - 1], 1)
            for k, B in enumerate(self._arm_bases):
                m = np.flatnonzero(res == k)
                if len(m):
                    r = chk.columns_read(st[m], poses[m], B)
                    box[m, 0::2] = np.minimum(box[m, 0::2], r[:, 0::2])
                    box[m, 1::2] = np.maximum(box[m, 1::2], r[:, 1::2])
            keys = np.concatenate([self._arm_keys, key[todo]])
            order = np.argsort(keys, kind="stable")
            self._arm_keys = keys[order]
            self._arm_move = np.concatenate([self._arm_move, res])[order]
            self._arm_box = np.concatenate([self._arm_box,
                                            box.astype(np.int64)])[order]
        return move[inv]

    #: candidate cells per (bin, grid) `_arm_any` arm-checks before it falls
    #: back to all of them; nearly every bin has a clear one among the first
    ARM_PROBE = 4

    def _arm_any(self, grids):
        """(n, G): does grid g leave bin b a placement the arm can make?

        `grids` is a list of (feas, z, dims) as `_feas_one(..., arm=False)`
        gives them.  The arm filter is a per-cell AND applied after every
        other rule, so this is `_arm_clear(dims, z, feas).any((1, 2))` per
        grid -- without arm-checking whole grids: `ARM_PROBE` cells of each
        (bin, grid) first, the remaining cells only where none of those was
        clear, every grid in the same batch.
        """
        n, G = self.n_env, len(grids)
        out = np.zeros((n, G), bool)
        if not self.arm_collision:
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
        flat = out.reshape(-1)                # a view: (bin, grid) -> b * G + g
        probe = rank < self.ARM_PROBE
        for sel in (probe, ~probe):
            sel = sel & ~flat[grp]
            if sel.any():
                ok = self._arm_moves(b[sel], x[sel], y[sel], sz[sel], zz[sel]) >= 0
                flat[grp[sel][ok]] = True
        return out

    def _keep_arm(self, keys, move, box, x, y, sx, sy, placed):
        """The arm verdicts a `step` cannot have changed.

        It put a box on footprint [x, x + sx) x [y, y + sy) of each `placed`
        bin.  Heights only rise between resets, and a collision is a column
        the arm reads reaching one of its capsules, so a move that collided,
        or could not be reached, still does.  A verdict can change only when
        the box raised a column it rests on (`_arm_box`); all others stay.
        """
        b = keys // self._arm_bin_stride()
        hit = (placed[b] & (box[:, 0] < x[b] + sx[b]) & (box[:, 1] >= x[b])
               & (box[:, 2] < y[b] + sy[b]) & (box[:, 3] >= y[b]))
        self._arm_keys, self._arm_move, self._arm_box = (
            keys[~hit], move[~hit], box[~hit])

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
        # the head item is window slot 0 exactly when every bin still has one;
        # a bin past its end reads a zero-sized slot there instead
        same = bool((self.head < self.length).all())
        raw = self._head_grids(item, types, orients, same)
        # where the rules leave the head nothing, the outline step's pass --
        # once it opens -- gives its grids instead; z is the same in both
        late = np.zeros(self.n_env, bool)
        if self._outline_on():
            alive = ~self.done & (self.head < self.length)
            late = alive & ~np.any([f.any((1, 2)) for f, _ in raw], 0)
            if late.any() and self.outline_when == "station":
                late &= self._outline_gate()[:, 0]
            if late.any():
                with self._outline():
                    ol = self._head_grids(item, types, orients, same)
                raw = [(np.where(late[:, None, None], g, f), z)
                       for (f, z), (g, _) in zip(raw, ol)]
        self._pos_ol = late
        feas, zs, tu = [], [], []
        for r, (d, (f, z)) in enumerate(zip(orients, raw)):
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

    def _head_grids(self, item, types, orients, same):
        """[(feas, z)] per orientation for the leading item, by the rules
        over the positions of the pass in force, the arm included: its hard
        grids, or soft mix's last resort where the station is stuck."""
        if same and 0 in self._slot_fz:
            raw = list(zip(*self._slot_fz[0]))
        else:
            # slot 0's grids before the arm filter are usually known from
            # `_placeable` or `step`; only the arm is left to apply
            fs, zs = (self._geo_slot(0, item, types) if same
                      else self._slot_grids([(item, types)])[0])
            if self.arm_collision:
                fs = self._arm_clear_many(list(zip(orients, zs, fs)))
            raw = list(zip(fs, zs))
            if same:
                self._slot_fz[0] = ([f for f, _ in raw], [z for _, z in raw])
        # where no box within reach can go by the hard rules, the head goes
        # by soft mix's last resort -- its hard grids are empty there
        late = self.last_resort()
        if late.any():
            fs, zs = self._soft_slot(0 if same else None, item, types)
            if self.arm_collision:
                fs = self._arm_clear_many(list(zip(orients, zs, fs)))
            raw = [(np.where(late[:, None, None], s, f), z)
                   for (f, z), s in zip(raw, fs)]
        return raw

    def outline(self):
        """(n,) does the leading item go by the outline step?  The rules
        leave it nothing over the `ems` candidates, and the step is open to
        it -- the choice `_positions` makes."""
        self._positions()
        return self._pos_ol.copy()

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
        self.smap = np.where(foot, self.smap | (np.int32(1) << tt.astype(
            np.int32))[:, None, None], self.smap)
        # its outline: the four edges of its footprint, on the corner lattice
        ox = np.arange(self.Lx + 1)[None, :]
        oy = np.arange(self.Ly + 1)[None, :]
        ix = (ox >= x[:, None]) & (ox <= (x + sx)[:, None])
        iy = (oy >= y[:, None]) & (oy <= (y + sy)[:, None])
        ex = (ox == x[:, None]) | (ox == (x + sx)[:, None])
        ey = (oy == y[:, None]) | (oy == (y + sy)[:, None])
        self.omap |= (((ex[:, :, None] & iy[:, None, :])
                       | (ix[:, :, None] & ey[:, None, :]))
                      & alive[:, None, None])
        if self.stack_cap:
            # a small box of a limited type carries only a short stack: from
            # here on nothing may start over its footprint at or above its top
            # plus its allowance.  The limit stays with these cells -- a box
            # laid over them later does not spread it over its own footprint.
            # A finished bin's slot is clamped into the table; `foot` drops it
            allow = self.stack_allow[np.minimum(np.minimum(sx, sy),
                                                len(self.stack_allow) - 1)]
            lim = np.where(np.isin(tt, self.stack_types) & (allow < NO_CAP),
                           np.minimum(zz + sz + allow, NO_CAP), NO_CAP)
            self.cmap = np.where(
                foot, np.minimum(self.cmap, lim.astype(np.int16)[:, None, None]),
                self.cmap)

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
        arm = self._arm_keys, self._arm_move, self._arm_box
        self._invalidate()
        # only the footprint just raised can have changed an arm verdict
        self._keep_arm(*arm, x, y, sx, sy, alive)

        reward = (vol / self.bin_vol) * alive
        # terminate when the new leading item has nowhere to go
        self.done |= ~alive
        self.done |= self.head >= self.length
        self.done |= ~self._head_placeable()
        return reward.astype(np.float32), self.done.copy()

    def _head_placeable(self):
        """(n,) `n_feasible() > 0`, without the full arm-checked grid that
        only `obs` needs: the leading item's grids before the arm filter, and
        `_arm_any` over them.  When every bin still has an item they are
        window slot 0's, and are kept there for `_placeable` and `obs`.
        Where the rules leave it nothing, the outline step's pass is asked
        the same, once the step is open to it."""
        ok = self._head_rules()
        if self._outline_on():
            late = ~self.done & (self.head < self.length) & ~ok
            if late.any() and self.outline_when == "station":
                late &= self._outline_gate()[:, 0]
            if late.any():
                with self._outline():
                    ok = ok | (late & self._head_rules())
        return ok

    def _head_rules(self):
        """`_head_placeable` by the rules over the positions of the pass in
        force, short of the outline step."""
        item, types = self.head_item()
        orients = self._orients(item)
        same = bool((self.head < self.length).all())
        if same:
            fs, zs = self._geo_slot(0, item, types)
        else:
            fs, zs = self._slot_grids([(item, types)])[0]
        grids = []
        for r, (d, f, z) in enumerate(zip(orients, fs, zs)):
            if r:   # as in `_positions`: a square footprint turned round
                f = f & (item[:, 0] != item[:, 1])[:, None, None]
            grids.append((f, z, d))
        alive = ~self.done & (self.head < self.length)
        ok = self._arm_any(grids).any(1) & alive
        # a head with nowhere to go by the hard rules may still have soft
        # mix's last resort -- if its type has one and the station is stuck
        soft = self._soft_types[types.astype(np.int64)]
        if self._soft_on() and (alive & ~ok & soft).any():
            late = alive & ~ok & self.last_resort()
            if late.any():
                fs, zs = self._soft_slot(0 if same else None, item, types)
                ok |= late & self._arm_any(list(zip(fs, zs, orients))).any(1)
        return ok

    # ----------------------------------------------------------------- stats
    def utilization(self):
        return self.volume / self.bin_vol
