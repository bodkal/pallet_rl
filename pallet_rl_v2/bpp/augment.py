"""Model-agnostic data augmentation (Sec. V-D): the 4 rotations x horizontal flip
symmetries of the bin, up to 8x more experience per sample.

Rotating/flipping the height map is trivial, but the action is not: the FLB is the
item's origin of coordinates, so it has to be relocated to the min-corner of the
transformed footprint, and for 90/270 degrees the item itself is rotated w.r.t. z
(its w and l swap).  The exact FLB relocation is precomputed here per
(transform, w, l) as an index permutation.
"""
import numpy as np


def _apply(A, t):
    f, r = divmod(t, 4)
    if f:
        A = A[::-1]
    return np.rot90(A, r)


class Augmenter:
    N_T = 8

    def __init__(self, cfg):
        assert cfg.W == cfg.L, 'non-square bins only admit 180 deg / flips'
        self.cfg = cfg
        W, L = cfg.W, cfg.L
        idx = np.arange(W * L).reshape(W, L)
        self.perm = []
        for t in range(8):
            it = _apply(idx, t)
            inv = np.empty(W * L, np.int64)
            inv[it.ravel()] = np.arange(W * L)
            self.perm.append(inv)
        self.flb = {}
        for t in range(8):
            for w in range(1, max(W, 6)):
                for l in range(1, max(L, 6)):
                    if w > W or l > L:
                        continue
                    m = np.full(W * L, -1, np.int64)
                    pm = self.perm[t]
                    for x in range(W - w + 1):
                        for y in range(L - l + 1):
                            p1 = pm[x * L + y]
                            p2 = pm[(x + w - 1) * L + (y + l - 1)]
                            m[x * L + y] = (min(p1 // L, p2 // L) * L
                                            + min(p1 % L, p2 % L))
                    self.flb[(t, w, l)] = m

    def transform(self, hm, buf, pi, t):
        """hm (W,L), buf (b,3), pi (A,) -> transformed copies."""
        if t == 0:
            return hm, buf, pi
        cfg = self.cfg
        r = t % 4
        hm2 = np.ascontiguousarray(_apply(hm, t))
        buf2 = buf[:, [1, 0, 2]].copy() if r % 2 else buf.copy()
        piM = pi.reshape(cfg.M, cfg.WL)
        out = np.zeros_like(piM)
        kk1 = cfg.k + 1
        for m in range(cfg.M):
            src = np.nonzero(piM[m])[0]
            if src.size == 0:
                continue
            j, o = m // kk1, m % kk1
            w, l = int(buf[j, 0]), int(buf[j, 1])
            if o == 1:
                w, l = l, w
            if w <= 0 or l <= 0:
                continue
            dst = self.flb[(t, w, l)][src]
            ok = dst >= 0
            out[m, dst[ok]] = piM[m, src[ok]]
        return hm2, buf2, out.reshape(-1)
