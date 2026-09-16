"""CUT-1 / CUT-2 / RS sequence generation (Sec. IV of the paper).

CUT-1 and CUT-2 are built by cutting-stock: a bin-sized cuboid is recursively
'cut' until every slice respects the item size constraints ([2,5] per axis).
Both admit a 100% space-utilisation packing.
  CUT-1 : items sorted by their z coordinate, ascending.
  CUT-2 : items sorted by stacking dependency (an item may appear only once all
          the items supporting it are already in the sequence).
  ties  : broken at random.
RS    : items drawn uniformly at random, no packing guarantee.
"""
import numpy as np

LO, HI = 2, 5


def cut_pieces(rng, W=10, L=10, H=10):
    """Recursive guillotine cutting-stock. Returns list of (x,y,z,w,l,h)."""
    pieces = [(0, 0, 0, W, L, H)]
    while True:
        cand = [i for i, p in enumerate(pieces) if max(p[3], p[4], p[5]) > HI]
        if not cand:
            break
        i = int(rng.choice(cand))
        x, y, z, w, l, h = pieces.pop(i)
        axes = [ax for ax, d in enumerate((w, l, h)) if d > HI]
        ax = int(rng.choice(axes))
        d = (w, l, h)[ax]
        c = int(rng.integers(LO, d - LO + 1))          # both halves >= LO
        o = [x, y, z]
        s = [w, l, h]
        a = list(o), list(s)
        s1 = list(s); s1[ax] = c
        o2 = list(o); o2[ax] = o[ax] + c
        s2 = list(s); s2[ax] = d - c
        pieces.append((o[0], o[1], o[2], s1[0], s1[1], s1[2]))
        pieces.append((o2[0], o2[1], o2[2], s2[0], s2[1], s2[2]))
    return pieces


def supporters(pieces):
    """sup[i] = set of indices whose top face touches the bottom face of i."""
    n = len(pieces)
    sup = [set() for _ in range(n)]
    for i, (xi, yi, zi, wi, li, hi) in enumerate(pieces):
        if zi == 0:
            continue
        for j, (xj, yj, zj, wj, lj, hj) in enumerate(pieces):
            if i == j or zj + hj != zi:
                continue
            if xj < xi + wi and xi < xj + wj and yj < yi + li and yi < yj + lj:
                sup[i].add(j)
    return sup


def order_cut1(pieces, rng):
    z = np.array([p[2] for p in pieces], dtype=float)
    return np.lexsort((rng.random(len(pieces)), z))


def order_cut2(pieces, rng):
    sup = supporters(pieces)
    n = len(pieces)
    remaining = [len(s) for s in sup]
    dependents = [[] for _ in range(n)]
    for i, s in enumerate(sup):
        for j in s:
            dependents[j].append(i)
    ready = [i for i in range(n) if remaining[i] == 0]
    out = []
    while ready:
        t = int(rng.integers(len(ready)))
        i = ready.pop(t)
        out.append(i)
        for d in dependents[i]:
            remaining[d] -= 1
            if remaining[d] == 0:
                ready.append(d)
    assert len(out) == n, "cycle in support graph"
    return np.array(out)


def gen_cut(n_seq, kind, seed, W=10, L=10, H=10):
    rng = np.random.default_rng(seed)
    seqs, lens = [], []
    for _ in range(n_seq):
        pieces = cut_pieces(rng, W, L, H)
        order = order_cut1(pieces, rng) if kind == 1 else order_cut2(pieces, rng)
        s = np.array([[pieces[i][3], pieces[i][4], pieces[i][5]] for i in order],
                     dtype=np.int8)
        seqs.append(s)
        lens.append(len(s))
    T = max(lens)
    out = np.zeros((n_seq, T, 3), dtype=np.int8)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out, np.array(lens, dtype=np.int32)


def gen_rs(n_seq, seed, length=80):
    rng = np.random.default_rng(seed)
    seqs = rng.integers(LO, HI + 1, size=(n_seq, length, 3)).astype(np.int8)
    return seqs, np.full(n_seq, length, dtype=np.int32)


def build_all(root, n_train=2000, n_test=100, n_val=200, seed=0):
    """train / val / test, from three disjoint seed offsets.

    `val` exists so hyperparameter search never selects on `test`: the headline
    numbers are only meaningful if nothing was tuned against those sequences.
    """
    import os, json
    os.makedirs(root, exist_ok=True)
    meta = {}
    for name, fn in (('cut1', lambda n, s: gen_cut(n, 1, s)),
                     ('cut2', lambda n, s: gen_cut(n, 2, s)),
                     ('rs',   lambda n, s: gen_rs(n, s))):
        for split, n, off in (('train', n_train, 0), ('test', n_test, 10_000),
                              ('val', n_val, 20_000)):
            seqs, lens = fn(n, seed + off + hash(name) % 1000)
            np.savez_compressed(f'{root}/{name}_{split}.npz', seqs=seqs, lens=lens)
            vol = (seqs[..., 0].astype(int) * seqs[..., 1] * seqs[..., 2])
            vol = np.array([vol[i, :lens[i]].sum() for i in range(n)])
            meta[f'{name}_{split}'] = dict(n=int(n), T=int(seqs.shape[1]),
                                           mean_len=float(lens.mean()),
                                           mean_total_vol=float(vol.mean()))
    with open(f'{root}/meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    return meta


if __name__ == '__main__':
    import sys, json
    print(json.dumps(build_all(sys.argv[1] if len(sys.argv) > 1 else 'data'), indent=2))
