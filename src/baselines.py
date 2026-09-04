"""Non-learning baselines.

* boundary_rule -- the paper's heuristic (supplemental D).  It scores a candidate
  placement by the "regularity" of the resulting bin, measured through maximal
  spare cuboids: RS_c = |I_valid| + c_volume, with a +10 bonus when a cuboid
  admits every item type.  p_best = argmax_p  (1/|C|) * sum_c RS_c.
* dbl -- deepest-bottom-left, the classic online placement rule.
* random_feasible -- uniform over legal LPs (sanity floor).
"""
from __future__ import annotations

import numpy as np

from .bin3d import Bin3D
from .items import item_set


# ---------------------------------------------------------------------------
def _maximal_spare_cuboids(hmap, H, max_n=64):
    """Maximal axis-aligned empty cuboids sitting on the height-map surface.

    For every distinct base height z we take the free region {hmap <= z} and
    enumerate maximal all-free rectangles in it; each gives a cuboid of
    height H - z.  This is the standard height-map approximation of the
    spare-cuboid decomposition drawn in the paper's Figure 12.
    """
    L, W = hmap.shape
    out = []
    for z in np.unique(hmap):
        free = (hmap <= z)
        if not free.any():
            continue
        # maximal rectangles via the largest-rectangle-in-histogram sweep
        heights = np.zeros(W, dtype=np.int32)
        for x in range(L):
            heights = np.where(free[x], heights + 1, 0)
            stack = []
            for y in range(W + 1):
                h = heights[y] if y < W else 0
                start = y
                while stack and stack[-1][1] >= h:
                    sy, sh = stack.pop()
                    out.append((int(sh), int(y - sy), int(H - z)))
                    start = sy
                stack.append((start, h))
    if not out:
        return []
    # keep the distinct largest ones
    out = sorted(set(out), key=lambda c: -(c[0] * c[1] * c[2]))[:max_n]
    return out


def _regularity(hmap, H, types, n_types):
    cuboids = _maximal_spare_cuboids(hmap, H)
    if not cuboids:
        return 0.0
    total = 0.0
    for (cl, cw, ch) in cuboids:
        valid = sum(1 for (l, w, h) in types
                    if (l <= cl and w <= cw or l <= cw and w <= cl) and h <= ch)
        vol = cl * cw * ch
        total += (n_types + vol + 10) if valid == n_types else (valid + vol)
    return total / len(cuboids)


class BoundaryRule:
    name = "boundary rule"

    def __init__(self, cfg):
        self.cfg = cfg
        self.types = item_set(cfg.item_min, cfg.item_max)
        self.n_types = len(self.types)

    def __call__(self, bin_: Bin3D, obs):
        mask = obs["mask"]
        feas = np.flatnonzero(mask)
        if len(feas) == 0:
            return None
        item = tuple(int(v) for v in obs["item"])
        best, best_s = None, -1e18
        for a in feas:
            b = bin_.copy()
            b.place(int(a), item, self.cfg.orientations)
            s = _regularity(b.hmap, self.cfg.H, self.types, self.n_types)
            if s > best_s:
                best, best_s = int(a), s
        return best


class DeepestBottomLeft:
    """Lowest resting height, ties broken by smallest (y, x)."""
    name = "deepest-bottom-left"

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, bin_, obs):
        feas = np.flatnonzero(obs["mask"])
        if len(feas) == 0:
            return None
        item = tuple(int(v) for v in obs["item"])
        best, key = None, None
        for a in feas:
            x, y, l, w, h = bin_.decode(int(a), item, self.cfg.orientations)
            z = int(bin_.hmap[x:x + l, y:y + w].max())
            k = (z, y, x)
            if key is None or k < key:
                best, key = int(a), k
        return best


class RandomFeasible:
    name = "random feasible"

    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(0)

    def __call__(self, bin_, obs):
        feas = np.flatnonzero(obs["mask"])
        return int(self.rng.choice(feas)) if len(feas) else None


BASELINES = {"boundary": BoundaryRule, "dbl": DeepestBottomLeft,
             "random": RandomFeasible}
