"""Py3DBP - preventive 3D bin-packing simulator (Sec. III / V-A of the paper).

State
-----
  hm   : (B, W, L) int16   height map  H_t in Z^{WxL}
  buf  : (B, b, 3) int8    the b items currently in the buffer (w,l,h); zeros = empty
  nxt  : (B,)      int32   index of the next item of the sequence to enter the buffer

Actions
-------
  flat index a in [0, M*W*L) with M = b*(k+1)
  m = a // (W*L)  ->  slot j = m // (k+1), orientation o = m % (k+1)
  p = a %  (W*L)  ->  x = p // L, y = p % L        (x,y) is the FLB loading position

Reward  r_i = 10 * V_i / V_B   (episode return = 10 * space utilisation)
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Cfg:
    __slots__ = ('W', 'L', 'H', 'b', 'k', 'VB', 'M', 'A', 'WL', 'CIN')

    def __init__(self, W=10, L=10, H=10, b=1, k=0):
        self.W, self.L, self.H, self.b, self.k = W, L, H, b, k
        self.VB = W * L * H
        self.M = b * (k + 1)
        self.WL = W * L
        self.A = self.M * self.WL
        self.CIN = 1 + 3 * b

    def as_dict(self):
        return dict(W=self.W, L=self.L, H=self.H, b=self.b, k=self.k)


def combo_dims(buf, k):
    """(B,b,3) -> (B,M,3) laid out as m = j*(k+1) + o."""
    B, b, _ = buf.shape
    if k == 0:
        return buf.reshape(B, b, 3).copy()
    d = np.stack([buf, buf[:, :, [1, 0, 2]]], axis=2)      # (B,b,2,3)
    return d.reshape(B, b * 2, 3)


def compute_masks(hm, dims, H):
    """Feasibility mask + landing height for every (game, combo, x, y).

    Stability rules (Sec. V-A), disjunctive for a given FLB:
      1) >60% of the item area supported *and* all four corners supported
      2) >80% supported *and* three or more corners supported
      3) >95% supported
    plus the item must stay inside the walls and below the ceiling.

    Support count is obtained from a single radix-32 integral image: encoding a
    cell of height v as 32**v makes the window sum a base-32 histogram of the
    heights inside the window (a window holds at most 25 cells < 32, so the
    digits never carry), and the digit at position h_max is exactly the number
    of supporting cells. h_max itself comes from a separable running max.

    Returns mask (B,M,W,L) bool and zmap (B,M,W,L) int16 (landing z = h_max).
    """
    B, W, L = hm.shape
    M = dims.shape[1]
    N = B * M
    w = dims[:, :, 0].reshape(N).astype(np.int32)
    l = dims[:, :, 1].reshape(N).astype(np.int32)
    h = dims[:, :, 2].reshape(N).astype(np.int32)
    g = np.repeat(np.arange(B), M)

    mask = np.zeros((N, W, L), dtype=bool)
    zmap = np.zeros((N, W, L), dtype=np.int16)

    pw = np.left_shift(np.int64(1), 5 * hm.astype(np.int64))
    integ = np.zeros((B, W + 1, L + 1), np.int64)
    np.cumsum(np.cumsum(pw, axis=1), axis=2, out=integ[:, 1:, 1:])

    key = w * 256 + l
    for kv in np.unique(key):
        ww, ll = int(kv // 256), int(kv % 256)
        if ww <= 0 or ll <= 0 or ww > W or ll > L:
            continue
        sel = np.nonzero(key == kv)[0]
        gs = g[sel]
        sub = hm[gs]                                          # (n,W,L)
        nx, ny = W - ww + 1, L - ll + 1

        mx = sub[:, :nx, :].copy()                            # running max over x
        for i in range(1, ww):
            np.maximum(mx, sub[:, i:i + nx, :], out=mx)
        hmax = mx[:, :, :ny].copy()                           # then over y
        for j in range(1, ll):
            np.maximum(hmax, mx[:, :, j:j + ny], out=hmax)

        it = integ[gs]
        S = (it[:, ww:, ll:] - it[:, :nx, ll:] - it[:, ww:, :ny] + it[:, :nx, :ny])
        cnt = np.right_shift(S, 5 * hmax.astype(np.int64)) & 31

        cor = (sub[:, :nx, :ny] == hmax).astype(np.int8)
        cor += (sub[:, :nx, ll - 1:] == hmax)
        cor += (sub[:, ww - 1:, :ny] == hmax)
        cor += (sub[:, ww - 1:, ll - 1:] == hmax)

        area = ww * ll
        ok = ((hmax + h[sel][:, None, None]) <= H) & (
            ((cnt > 0.60 * area) & (cor == 4))
            | ((cnt > 0.80 * area) & (cor >= 3))
            | (cnt > 0.95 * area))
        ix = np.ix_(sel, np.arange(nx), np.arange(ny))
        mask[ix] = ok
        zmap[ix] = hmax

    return mask.reshape(B, M, W, L), zmap.reshape(B, M, W, L)


def features(hm, buf, H, out=None):
    """(B,1+3b,W,L) float32 - height map and the 3 stretched dims per buffer item,
    all normalised by the bin height H."""
    B, W, L = hm.shape
    b = buf.shape[1]
    if out is None:
        out = np.empty((B, 1 + 3 * b, W, L), dtype=np.float32)
    inv = np.float32(1.0 / H)
    out[:, 0] = hm * inv
    for j in range(b):
        for d in range(3):
            out[:, 1 + 3 * j + d] = buf[:, j, d, None, None] * inv
    return out


def apply_actions(hm, buf, nxt, seq, lens, dims, zmap, acts, idx, cfg):
    """Apply one action per entry of `idx` (indices into the batch). In-place.
    `dims`/`zmap`/`acts` are indexed by position within `idx`.
    Returns rewards (len(idx),) float32."""
    W, L, WL, kk1, VB = cfg.W, cfg.L, cfg.WL, cfg.k + 1, cfg.VB
    rew = np.empty(len(idx), dtype=np.float32)
    for t in range(len(idx)):
        i = idx[t]
        a = int(acts[t])
        m, p = a // WL, a % WL
        x, y = p // L, p % L
        ww, ll, hh = dims[t, m]
        z = zmap[t, m, x, y]
        hm[i, x:x + ww, y:y + ll] = z + hh
        rew[t] = 10.0 * float(ww) * float(ll) * float(hh) / VB
        j = m // kk1
        ni = nxt[i]
        if ni < lens[i]:
            buf[i, j] = seq[i, ni]
            nxt[i] = ni + 1
        else:
            buf[i, j] = 0
    return rew


def sample_rows(p, rng):
    """Sample one column index per row of a (n,A) probability matrix."""
    c = np.cumsum(p, axis=1)
    c /= c[:, -1:]
    u = rng.random((p.shape[0], 1))
    return np.minimum((c < u).sum(axis=1), p.shape[1] - 1)


def new_state(seqs, lens, cfg):
    """Fresh batch of episodes for the given sequences."""
    B = seqs.shape[0]
    hm = np.zeros((B, cfg.W, cfg.L), dtype=np.int16)
    buf = np.zeros((B, cfg.b, 3), dtype=np.int8)
    nxt = np.zeros(B, dtype=np.int32)
    for i in range(B):
        n = min(cfg.b, int(lens[i]))
        buf[i, :n] = seqs[i, :n]
        nxt[i] = n
    return hm, buf, nxt


def step_v(hm, buf, nxt, seqs, lens, dims, zmap, acts, cfg):
    """Fully vectorised transition for a compact batch (mutates hm/buf/nxt).
    Every array is indexed 0..n-1 and describes one state. Returns rewards."""
    n = hm.shape[0]
    ar = np.arange(n)
    WL, L, kk1 = cfg.WL, cfg.L, cfg.k + 1
    m = acts // WL
    p = acts % WL
    x = (p // L).astype(np.int64)
    y = (p % L).astype(np.int64)
    d = dims[ar, m]
    w = d[:, 0].astype(np.int64)
    l = d[:, 1].astype(np.int64)
    h = d[:, 2].astype(np.int64)
    z = zmap[ar, m, x, y].astype(np.int64)
    xs = np.arange(cfg.W)[None, :, None]
    ys = np.arange(cfg.L)[None, None, :]
    fp = ((xs >= x[:, None, None]) & (xs < (x + w)[:, None, None])
          & (ys >= y[:, None, None]) & (ys < (y + l)[:, None, None]))
    np.copyto(hm, (z + h).astype(np.int16)[:, None, None], where=fp)
    j = (m // kk1).astype(np.int64)
    has = nxt < lens
    if has.any():
        hi = ar[has]
        buf[hi, j[has]] = seqs[hi, nxt[has]]
    if (~has).any():
        ni = ar[~has]
        buf[ni, j[~has]] = 0
    nxt += has
    return (10.0 * w * l * h / cfg.VB).astype(np.float32)
