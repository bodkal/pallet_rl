"""Sliding-window maxima over a height map.

Every rule in `pallet.py` asks "what is the highest cell under this footprint,
and how many cells reach it?" for every (x, y) at once.  These helpers answer
that with one running max per axis: width k + 1 is width k plus one more cell.

Arrays are `(n, Lx, Ly)`; axis 1 is x, axis 2 is y.  The result of a sweep is
indexed `[width - 1, n, x, y]`.
"""
from __future__ import annotations

import numpy as np


def win_max(m, axis, L, kmax=None):
    """Running max of every window width 1..kmax along `axis` (length `L`).

    A window the axis end cuts short keeps the max of what is left of it.
    """
    kmax = L if kmax is None else kmax
    m = np.ascontiguousarray(m)
    out = np.empty((kmax,) + m.shape, m.dtype)
    out[0] = m
    flat = m.reshape(-1)
    for k in range(1, kmax):
        prev, cur = out[k - 1], out[k]
        shift = k if axis == 2 else k * m.shape[2]
        np.maximum(prev.reshape(-1)[:-shift], flat[shift:],
                   out=cur.reshape(-1)[:-shift])
        tail = np.s_[:, L - k:] if axis == 1 else np.s_[:, :, L - k:]
        cur[tail] = prev[tail]
    return out


def win_maxcount(m, c, axis, L, kmax):
    """`win_max` plus how many cells attain the max in each window.

    `c` counts the cells behind each entry of `m` (None for raw cells), so a
    y-sweep can be fed into an x-sweep and the counts add up.
    """
    m = np.ascontiguousarray(m)
    om = np.empty((kmax,) + m.shape, m.dtype)
    oc = np.empty((kmax,) + m.shape, np.int32)
    om[0], oc[0] = m, 1 if c is None else c
    mf = m.reshape(-1)
    cf = None if c is None else np.ascontiguousarray(c).reshape(-1)
    for k in range(1, kmax):
        shift = k if axis == 2 else k * m.shape[2]
        pm = om[k - 1].reshape(-1)[:-shift]
        pc = oc[k - 1].reshape(-1)[:-shift]
        am = mf[shift:]
        # the new cell beats the max (its own count), ties it (counts add),
        # or loses (count stands) -- as arithmetic, faster than np.where
        cnt = oc[k].reshape(-1)[:-shift]
        np.multiply(pc, am <= pm, out=cnt)
        cnt += (am >= pm) if cf is None else cf[shift:] * (am >= pm)
        np.maximum(pm, am, out=om[k].reshape(-1)[:-shift])
        tail = np.s_[:, L - k:] if axis == 1 else np.s_[:, :, L - k:]
        om[k][tail], oc[k][tail] = om[k - 1][tail], oc[k - 1][tail]
    return om, oc


def widest(c, cap):
    """How many widths a sweep must produce to read widths `c` back out.
    A zero side reads window -1, so it keeps the full `cap`."""
    return int(c.max()) if c.min() >= 1 else int(cap)


def covered(f, sx, sy):
    """(n, Lx, Ly): cells covered by some footprint [x, x+sx) x [y, y+sy)
    whose corner has `f[:, x, y]` set -- by prefix sums."""
    n, Lx, Ly = f.shape
    c = np.zeros((n, Lx + 1, Ly + 1), np.int32)
    c[:, 1:, 1:] = f.cumsum(1, dtype=np.int32).cumsum(2)
    hx, hy = np.arange(1, Lx + 1)[:, None], np.arange(1, Ly + 1)
    lx, ly = np.maximum(hx - sx, 0), np.maximum(hy - sy, 0)
    return (c[:, hx, hy] - c[:, lx, hy] - c[:, hx, ly] + c[:, lx, ly]) > 0
