"""The six heuristic packing policies benchmarked in AR2L Table 1.

Every heuristic scores the *same* candidate list the learned policy sees, so
nothing here can drift from the simulator the agent was trained in.  Each one
is a primary score with a deepest-bottom-left tie-break, and the placement is
the arg-max.

    dbl         deepest-bottom-left                (Karabulut & Inceoglu 2005)
    bmf         best-match-first                   (Li & Zhang 2015)
    lsah        least surface area of the pile     (Hu et al. 2017)
    onlinebph   deepest-bottom-left-with-fill      (Ha et al. 2017)
    hmm         height-map minimisation            (Wang & Hauser 2019)
    macs        maximise accessible convex space   (Hu et al. 2020)
"""
from __future__ import annotations

import numpy as np

NAMES = ("dbl", "bmf", "lsah", "onlinebph", "hmm", "macs")
NEG = -1e18
MACS_TOPK = 24  # MACS rescores only the most promising candidates (see README)


def _run_all(m, axis, S):
    """Running window max of every width 1..S along one axis."""
    out = np.empty((S,) + m.shape, m.dtype)
    out[0] = m
    for k in range(1, S):
        cur = out[k - 1].copy()
        if axis == 1:
            cur[:, : S - k] = np.maximum(out[k - 1][:, : S - k], m[:, k:])
        else:
            cur[:, :, : S - k] = np.maximum(out[k - 1][:, :, : S - k], m[:, :, k:])
        out[k] = cur
    return out


def _free_extent(hmap, z, sx, sy, S, ar):
    """How far the item could grow in x and in y before hitting a taller column.

    Returns the pair stacked on a leading axis, so callers can hold one entry
    per orientation.
    """
    ay = _run_all(hmap, 2, S)[sy - 1, ar]          # window of depth sy in y ...
    mx = _run_all(ay, 1, S)                        # ... and of every width in x
    ax = _run_all(hmap, 1, S)[sx - 1, ar]
    my = _run_all(ax, 2, S)
    w = np.arange(1, S + 1)[:, None, None, None]
    gx = np.arange(S)[None, None, :, None]
    gy = np.arange(S)[None, None, None, :]
    ex = ((mx <= z[None]) & (gx + w <= S)).sum(0)
    ey = ((my <= z[None]) & (gy + w <= S)).sum(0)
    return np.stack([ex, ey], 0)


def _largest_cuboid(hmap, S):
    """Volume of the biggest empty axis-aligned box above the height map."""
    M = hmap.shape[0]
    ax = _run_all(hmap, 1, S)                              # (S, M, S, S)
    w = np.arange(1, S + 1)
    gx = np.arange(S)[None, None, :, None]
    gy = np.arange(S)[None, None, None, :]
    best = np.zeros(M, np.int64)
    for i in range(S):
        axy = _run_all(ax[i], 2, S)                        # (S, M, S, S)
        vol = (S - axy.astype(np.int64)) * (w[i] * w[:, None, None, None])
        keep = (gx + w[i] <= S) & (gy + w[:, None, None, None] <= S)
        best = np.maximum(best, np.where(keep, vol, 0).max(axis=(0, 2, 3)))
    return best


def scores(env, name):
    """(B, NL) score per candidate placement; higher is better.

    Every candidate carries its own oriented footprint, so each heuristic is
    evaluated per placement rather than per (x, y) cell.
    """
    mask = env.obs()["l_mask"]          # also refreshes the candidate table
    _, zgrid, odims = env._positions()
    lxy, S, ar = env._lxy, env.S, env._ar
    n, NL = env.n_env, lxy.shape[1]
    x, y, r = lxy[..., 0], lxy[..., 1], lxy[..., 2]
    z = env._lz
    sx, sy, sz = env._ldim[..., 0], env._ldim[..., 1], env._ldim[..., 2]
    dbl = -(z.astype(np.float64) * 1e4 + y * 1e2 + x)   # tie-break, in (-1e6, 0]

    if name == "dbl":
        s = dbl

    elif name == "hmm":
        # increase of the volume under the height map = item volume + sealed void
        cs = np.cumsum(np.cumsum(env.hmap.astype(np.int64), 1), 2)
        cs = np.pad(cs, ((0, 0), (1, 0), (1, 0)))
        x2, y2 = np.minimum(x + sx, S), np.minimum(y + sy, S)
        tot = (cs[ar[:, None], x2, y2] - cs[ar[:, None], x, y2]
               - cs[ar[:, None], x2, y] + cs[ar[:, None], x, y])
        void = (sx * sy) * z - tot
        s = -void.astype(np.float64) * 1e7 + dbl

    elif name == "lsah":
        pk = env.packed[:, : max(int(env.n_packed.max()), 1)] * S
        m = (np.arange(pk.shape[1])[None, :] < env.n_packed[:, None])[..., None]
        lo = np.where(m, pk[..., :3], np.inf).min(1)
        hi = np.where(m, pk[..., :3] + pk[..., 3:], -np.inf).max(1)
        p0 = np.stack([x, y, z], -1).astype(np.float64)
        p1 = p0 + np.stack([sx, sy, sz], -1)
        d = np.maximum(hi[:, None, :], p1) - np.minimum(lo[:, None, :], p0)
        area = 2.0 * (d[..., 0] * d[..., 1] + d[..., 1] * d[..., 2]
                      + d[..., 2] * d[..., 0])
        s = -area * 1e7 + dbl

    elif name in ("bmf", "onlinebph"):
        # the free extent depends on the footprint, so it is swept once per
        # orientation and then read off at each candidate's own orientation
        ex = np.stack([_free_extent(env.hmap, zgrid[:, k], odims[:, k, 0],
                                    odims[:, k, 1], S, ar)
                       for k in range(odims.shape[1])], 0)
        ey = ex[:, 1]; ex = ex[:, 0]
        ex = ex[r, ar[:, None], x, y]; ey = ey[r, ar[:, None], x, y]
        ez = S - z
        waste = (ex * ey * ez - sx * sy * sz).astype(np.float64)
        if name == "bmf":     # how many sides of the free space the item matches
            match = ((ex == sx).astype(np.float64) + (ey == sy) + (ez == sz))
            s = match * 1e13 - waste * 1e7 + dbl
        else:                 # deepest first, the tightest space among equals
            s = -z.astype(np.float64) * 1e13 - waste * 1e7 + dbl

    elif name == "macs":
        rank = np.where(mask, dbl, NEG)
        K = min(MACS_TOPK, NL)
        top = np.argsort(-rank, 1)[:, :K]
        tx, ty, tz = (np.take_along_axis(v, top, 1) for v in (x, y, z))
        dx, dy, dz = (np.take_along_axis(v, top, 1) for v in (sx, sy, sz))
        g = np.arange(S)[None, None, :]
        inx = (g >= tx[..., None]) & (g < (tx + dx)[..., None])
        iny = (g >= ty[..., None]) & (g < (ty + dy)[..., None])
        hm = np.where(inx[..., None] & iny[..., None, :],
                      (tz + dz)[..., None, None].astype(np.int16),
                      env.hmap[:, None])
        vol = _largest_cuboid(np.ascontiguousarray(hm.reshape(-1, S, S)), S)
        s = np.full((n, NL), NEG)
        np.put_along_axis(s, top, vol.reshape(n, K).astype(np.float64) * 1e7
                          + np.take_along_axis(dbl, top, 1), 1)
    else:
        raise ValueError(name)

    return np.where(mask, s, NEG)


def act(env, name):
    return scores(env, name).argmax(1)
