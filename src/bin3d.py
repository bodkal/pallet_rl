"""Height-map bin representation, placement and feasibility mask.

Paper Sec. 3.1.  The bin bottom is discretised into an L x W grid; H[x, y] is the
current stacked height at that cell.  An action places the item's front-left-bottom
(FLB) corner at loading position (LP) (x, y), giving action index a = x + L * y.

A LP is feasible iff it provides room for the item AND the placement is physically
stable under the paper's conservative support criterion:
    1) > 60% of the bottom area supported and all 4 bottom corners supported, or
    2) > 80% of the bottom area supported and >= 3 of 4 bottom corners supported, or
    3) > 95% of the bottom area supported.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SUPPORT_RULES = ((0.60, 4), (0.80, 3), (0.95, 0))


def orientations_of(item, n_orient: int):
    """The axis-aligned horizontal poses of an item (rotation about Z only)."""
    l, w, h = item
    if n_orient == 1 or l == w:
        return [(l, w, h)] * n_orient if n_orient > 1 else [(l, w, h)]
    return [(l, w, h), (w, l, h)]


class Bin3D:
    """A single bin, tracked as an integer height map."""

    __slots__ = ("L", "W", "H", "hmap", "placed", "volume")

    def __init__(self, L: int, W: int, H: int):
        self.L, self.W, self.H = L, W, H
        self.hmap = np.zeros((L, W), dtype=np.int32)
        self.placed: list[tuple] = []   # (x, y, z, l, w, h)
        self.volume = 0

    def copy(self) -> "Bin3D":
        b = Bin3D.__new__(Bin3D)
        b.L, b.W, b.H = self.L, self.W, self.H
        b.hmap = self.hmap.copy()
        b.placed = list(self.placed)
        b.volume = self.volume
        return b

    # -- feasibility ------------------------------------------------------
    def pose_mask(self, dims, hmap=None) -> np.ndarray:
        """Feasibility mask (L, W) for one concrete pose (l, w, h)."""
        l, w, h = dims
        hm = self.hmap if hmap is None else hmap
        m = np.zeros((self.L, self.W), dtype=np.float32)
        if l > self.L or w > self.W or h > self.H:
            return m

        win = sliding_window_view(hm, (l, w))          # (L-l+1, W-w+1, l, w)
        z = win.max(axis=(2, 3))                        # resting height per LP
        eq = (win == z[:, :, None, None])               # cells actually touching
        ratio = eq.mean(axis=(2, 3))
        ncorner = (eq[:, :, 0, 0].astype(np.int8) + eq[:, :, -1, 0] +
                   eq[:, :, 0, -1] + eq[:, :, -1, -1])

        stable = np.zeros_like(ratio, dtype=bool)
        for r_min, c_min in SUPPORT_RULES:
            stable |= (ratio > r_min) & (ncorner >= c_min)

        fits = (z + h) <= self.H
        m[:self.L - l + 1, :self.W - w + 1] = (stable & fits).astype(np.float32)
        return m

    def feasibility_mask(self, item, n_orient: int = 1, hmap=None) -> np.ndarray:
        """Flat mask over the whole action space (n_orient * L * W).

        Action a = o * (L*W) + x + L * y  for pose o at LP (x, y).
        """
        poses = orientations_of(item, n_orient)
        out = np.empty(n_orient * self.L * self.W, dtype=np.float32)
        for o in range(n_orient):
            # duplicate poses (square footprint) -> only the first is usable
            if o > 0 and poses[o] == poses[0]:
                out[o * self.L * self.W:(o + 1) * self.L * self.W] = 0.0
            else:
                out[o * self.L * self.W:(o + 1) * self.L * self.W] = \
                    self.pose_mask(poses[o], hmap).T.reshape(-1)
        return out

    # -- placement ---------------------------------------------------------
    def decode(self, action: int, item, n_orient: int = 1):
        """action -> (x, y, l, w, h)."""
        n = self.L * self.W
        o, rest = divmod(int(action), n)
        y, x = divmod(rest, self.L)          # a = x + L*y
        l, w, h = orientations_of(item, n_orient)[o]
        return x, y, l, w, h

    def place(self, action: int, item, n_orient: int = 1):
        """Stack the item; returns (z, l, w, h).  Assumes the action is feasible."""
        x, y, l, w, h = self.decode(action, item, n_orient)
        z = int(self.hmap[x:x + l, y:y + w].max())
        self.hmap[x:x + l, y:y + w] = z + h
        self.placed.append((x, y, z, l, w, h))
        self.volume += l * w * h
        return z, l, w, h

    def is_feasible(self, action: int, item, n_orient: int = 1) -> bool:
        x, y, l, w, h = self.decode(action, item, n_orient)
        if x + l > self.L or y + w > self.W:
            return False
        sub = self.hmap[x:x + l, y:y + w]
        z = int(sub.max())
        if z + h > self.H:
            return False
        eq = (sub == z)
        ratio = float(eq.mean())
        nc = int(eq[0, 0]) + int(eq[-1, 0]) + int(eq[0, -1]) + int(eq[-1, -1])
        return any(ratio > r and nc >= c for r, c in SUPPORT_RULES)

    # -- misc --------------------------------------------------------------
    @property
    def utilization(self) -> float:
        return self.volume / (self.L * self.W * self.H)

    def blocked_hmap(self, footprints) -> np.ndarray:
        """Height map with the given (x, y, l, w) footprints raised to H.

        Used by BPP-k to enforce order dependence: an item may never be packed
        on top of a later-arriving item that has already been virtually placed.
        """
        hm = self.hmap.copy()
        for (x, y, l, w) in footprints:
            hm[x:x + l, y:y + w] = self.H
        return hm
