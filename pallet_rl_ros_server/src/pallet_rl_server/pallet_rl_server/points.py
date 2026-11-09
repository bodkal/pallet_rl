"""The points the online view draws for the box in front of a `Pallet`.

`turned_down(pal)` is the server's twin of the game's `turned_down`
(`ar2l/viz/game.py`): every position the packer's action space offers the
front box (the `ems` positions that fit on the pallet) that the rules turn
down, each with the rule that does it, under the same names the game's 3D
views use -- 'arm', 'lid', 'cap', 'white', 'blue', 'type', 'total' and
'support' for what is left (it would not rest stably).

Only read for the snapshot; the decision never looks at it.  No ROS here.
"""
from __future__ import annotations

import contextlib

import numpy as np


def _grid(v, shape):
    """A rule's answer (a bool, or (1, Lx, Ly)) as one (Lx, Ly) grid."""
    return np.broadcast_to(np.asarray(v, bool), (1,) + shape)[0]


def _outline_used(pal):
    """Did the front box go on to the outline step (`Pallet._positions`):
    nothing by the plain rules, and -- with `outline_when` station -- nothing
    for any box within reach?"""
    R = pal.R
    if not R.outline_on:
        return False
    plain = pal._slot_fz.get(0)
    if plain is None or any(f.any() for f in plain[0]):
        return False
    return R.outline_when != "station" or bool(pal._outline_gate()[0, 0])


def turned_down(pal):
    """-> [[x, y, z, r, why]] for the box in front of `pal` (after
    `pal.obs()` or `pal._positions()`)."""
    R = pal.R
    feas, z, odims, _ = pal._positions()
    item, types = pal.head_item()
    shape = (pal.Lx, pal.Ly)
    gx, gy = np.arange(pal.Lx)[:, None], np.arange(pal.Ly)[None, :]
    orients = pal._orients(item)
    ol = _outline_used(pal)
    out = []
    with pal._outline() if ol else contextlib.nullcontext():
        soft = bool(pal.last_resort()[0])
        # the grids before the arm, by the rules the box goes by: the arm
        # alone turned down whatever is in them and not in `feas`
        pre = (pal._soft_slot(0, item, types) if soft
               else pal._geo_slot(0, item, types))[0]
        # its own fallback rules (tier 2) where they let it go further
        tier = 3 if soft else 1
        if not soft and pal._fallback_bins(types)[0]:
            for r, d in enumerate(orients):
                g, zg, full = pal._feas_geo(d)
                if (pre[r] & ~(g & pal._rules_ok(d, zg, types, 1, full))).any():
                    tier = 2
                    break
        for r in range(len(orients)):
            if r and item[0, 0] == item[0, 1]:     # a square box turned round
                break
            d, zr = odims[:, r], z[:, r]
            sx, sy, sz = (int(v) for v in d[0])
            space = _grid(pal._space(d, zr, ol), shape)
            bad = space & (gx + sx <= pal.Lx) & (gy + sy <= pal.Ly) & ~feas[0, r]
            xs, ys = np.nonzero(bad)
            if not len(xs):
                continue
            zero = np.zeros(shape, bool)
            arm = (pre[r][0] & ~feas[0, r]) if R.arm_collision else zero
            lid = zr[0] + sz > pal.Lz
            cap = ~_grid(pal._cap_ok(d, zr), shape) if R.stack_cap else zero
            typ = (~_grid(pal._type_ok(d, zr, types, tier), shape)
                   if R.typed else zero)
            wht = (~_grid(pal._white_ok(d, zr, types), shape)
                   if tier == 1 and R.white_on else zero)
            blu = (~_grid(pal._blue_ok(d, zr, types), shape)
                   if tier == 1 and R.blue_on else zero)
            tot = (~_grid(pal._ground_ok(d, zr), shape)
                   if R.min_total_support > 0 else zero)
            why = np.select(
                [arm, lid, cap, wht & ~typ, blu & ~typ, typ, tot],
                ["arm", "lid", "cap", "white", "blue", "type", "total"],
                "support")
            zi = zr[0]
            out += [[int(x), int(y), int(zi[x, y]), r, str(why[x, y])]
                    for x, y in zip(xs, ys)]
    return out
