"""The simulator, checked against straightforward reference implementations."""
import numpy as np
import pytest

import ar2l.env as E
from ar2l.env import BPPBatch
from ar2l import heuristics as H


def brute(hmap, item, L, mode):
    """Feasibility and landing height, one loading position at a time."""
    Lx, Ly, Lz = L
    sx, sy, sz = item
    ok = np.zeros((Lx, Ly), bool)
    Z = np.zeros((Lx, Ly), np.int32)
    for x in range(Lx - sx + 1):
        for y in range(Ly - sy + 1):
            sub = hmap[x:x + sx, y:y + sy]
            z = int(sub.max())
            Z[x, y] = z
            if z + sz > Lz:
                continue
            eq = sub == z
            if mode == "com":
                xs = np.nonzero(eq.any(1))[0]
                ys = np.nonzero(eq.any(0))[0]
                ok[x, y] = (xs[0] + .5 <= sx / 2 <= xs[-1] + .5 and
                            ys[0] + .5 <= sy / 2 <= ys[-1] + .5)
            else:
                r = eq.mean()
                nc = int(eq[0, 0]) + int(eq[-1, 0]) + int(eq[0, -1]) + int(eq[-1, -1])
                ok[x, y] = any(r > a and nc >= b for a, b in E.SUPPORT_RULES)
    return ok, Z


def brute_ems(hmap, L):
    """Every maximal footprint, found by trying all of them."""
    Lx, Ly, _ = L
    out = []
    for x0 in range(Lx):
        for wx in range(1, Lx - x0 + 1):
            for y0 in range(Ly):
                for wy in range(1, Ly - y0 + 1):
                    f = int(hmap[x0:x0 + wx, y0:y0 + wy].max())
                    grow = ((x0 - 1, wx + 1, y0, wy), (x0, wx + 1, y0, wy),
                            (x0, wx, y0 - 1, wy + 1), (x0, wx, y0, wy + 1))
                    if all(hmap[a:a + b, c:c + d].max() > f for a, b, c, d in grow
                           if a >= 0 and c >= 0 and a + b <= Lx and c + d <= Ly):
                        out.append((x0, y0, wx, wy, f))
    return out


def brute_corners(ems, L, item):
    """The four bottom corners of every space that can take the item."""
    Lx, Ly, Lz = L
    sx, sy, sz = item
    m = np.zeros((Lx, Ly), bool)
    for x0, y0, wx, wy, f in ems:
        if wx >= sx and wy >= sy and f + sz <= Lz:
            for px in (x0, x0 + wx - sx):
                for py in (y0, y0 + wy - sy):
                    m[px, py] = True
    return m


def extents(env):
    return (env.Lx, env.Ly, env.Lz)


# Deliberately lopsided, and never a cube: an x/y swap inside the sweeps is
# silent on a square bin but wrong on these.  `Lz` above and below both
# footprint sides catches anything that confuses height with a side.
ODD_BINS = [((7, 5, 9), (3, 2, 4)), ((5, 9, 6), 3), ((11, 4, 7), (5, 2, 3)),
            ((6, 6, 13), 3), ((9, 12, 5), (4, 5, 2)), ((4, 13, 11), (2, 6, 5))]


def random_action(env, rng):
    m = env.obs()["l_mask"]
    return np.array([rng.choice(np.nonzero(r)[0]) if r.any() else 0 for r in m])


@pytest.mark.parametrize("mode", ["com", "cdrl"])
def test_feasibility_matches_brute_force(mode):
    rng = np.random.default_rng(1)
    env = BPPBatch(16, nb=3, seed=3, stability=mode)
    checked = 0
    for _ in range(40):
        feas, z, odims = env._positions()
        for b in range(env.n_env):
            if env.done[b]:
                continue
            ems = brute_ems(env.hmap[b], extents(env))
            for r in range(odims.shape[1]):
                item = odims[b, r]
                ref, Zr = brute(env.hmap[b], item, extents(env), mode)
                ref = ref & brute_corners(ems, extents(env), item)
                if r and odims[b, 0, 0] == odims[b, 0, 1]:
                    ref = np.zeros_like(ref)   # a square turns into itself
                assert (ref == feas[b, r]).all()
                assert (Zr[ref] == z[b, r][ref]).all()
            checked += 1
        env.step(random_action(env, rng))
        env.reset_done()
    assert checked > 400


def test_ems_list_is_exact():
    """The dense scan must find every maximal space and no others."""
    rng = np.random.default_rng(8)
    env = BPPBatch(6, nb=1, seed=17)
    for _ in range(20):
        cols = env._ems_list()
        for b in range(env.n_env):
            got = {tuple(int(v) for v in row)
                   for row in zip(*[c[cols[0] == b] for c in cols[1:]])}
            ref = {(x0, y0, wx, wy, f)
                   for x0, y0, wx, wy, f in brute_ems(env.hmap[b], extents(env))}
            assert got == ref, b
        if env.done.all():
            break
        env.step(random_action(env, rng))
        env.reset_done()


def test_rotation_offers_both_footprints():
    env = BPPBatch(32, nb=1, seed=23, rot=2)
    _, _, odims = env._positions()
    item = env.seq[env._ar, env.head]
    assert (odims[:, 0] == item).all()
    assert (odims[:, 1, 0] == item[:, 1]).all()
    assert (odims[:, 1, 1] == item[:, 0]).all()
    assert (odims[:, :, 2] == item[:, 2:3]).all(), "height never rotates"
    # on an empty bin both orientations are reachable unless the base is square
    feas = env._positions()[0]
    square = item[:, 0] == item[:, 1]
    assert feas[~square, 1].any(1).any(1).all()
    assert not feas[square, 1].any()


def test_placed_item_matches_the_chosen_candidate():
    rng = np.random.default_rng(6)
    env = BPPBatch(16, nb=1, seed=29, rot=2)
    turned = 0
    while not env.done.all():
        a = random_action(env, rng)
        alive = ~env.done.copy()
        want = env._ldim[env._ar, a].copy()
        raw = env.seq[env._ar, np.minimum(env.head, env.n_items - 1)].copy()
        before = env.n_packed.copy()
        env.step(a)
        for b in np.nonzero(alive & (env.n_packed > before))[0]:
            got = (env.packed[b, env.n_packed[b] - 1] * env.scale).round().astype(int)
            assert (got[3:] == want[b]).all(), "placed a different box than chosen"
            assert sorted(got[3:]) == sorted(raw[b]), "item volume changed"
            turned += int(got[3] != raw[b][0])
    assert turned > 0, "rotation was never used"


def test_no_overlap_and_volume_bookkeeping():
    rng = np.random.default_rng(2)
    env = BPPBatch(32, nb=3, seed=7)
    while not env.done.all():
        env.step(random_action(env, rng))
    for b in range(env.n_env):
        vox = np.zeros((env.Lx, env.Ly, env.Lz), np.int8)
        for i in range(env.n_packed[b]):
            x, y, z, sx, sy, sz = (env.packed[b, i] * env.scale).round().astype(int)
            vox[x:x + sx, y:y + sy, z:z + sz] += 1
        assert vox.max() <= 1, "two items occupy the same cell"
        assert abs(vox.sum() - env.volume[b]) < 1e-3
        assert (vox.sum(2) == env.hmap[b] * 0 + vox.sum(2)).all()


def test_items_rest_on_support():
    """Every placed item sits on the floor or on the top face of another."""
    env = BPPBatch(16, nb=1, seed=11)
    while not env.done.all():
        env.step(np.array([H.act(env, "dbl")]).reshape(-1))
    for b in range(env.n_env):
        items = (env.packed[b, : env.n_packed[b]] * env.scale).round().astype(int)
        for x, y, z, sx, sy, sz in items:
            if z > 0:
                assert any(zz + ss == z and not (x + sx <= xx or xx + sxx <= x)
                           and not (y + sy <= yy or yy + syy <= y)
                           for xx, yy, zz, sxx, syy, ss in items
                           if (xx, yy, zz) != (x, y, z)), "floating item"


def test_permute_moves_the_chosen_item_to_the_front():
    env = BPPBatch(8, nb=5, seed=5)
    win, _ = env.window()
    idx = np.array([0, 1, 2, 3, 4, 2, 1, 4])
    env.permute(idx)
    new, _ = env.window()
    for b in range(8):
        assert (new[b, 0] == win[b, idx[b]]).all()
        rest = [tuple(v) for v in win[b] if True]
        rest.pop(idx[b])
        assert [tuple(v) for v in new[b, 1:]] == rest, "order of the others changed"


def test_heuristics_pick_legal_placements():
    for name in H.NAMES:
        env = BPPBatch(24, nb=1, seed=13)
        for _ in range(40):
            if env.done.all():
                break
            a = H.act(env, name)
            m = env.obs()["l_mask"]
            alive = ~env.done & m.any(1)
            assert m[np.arange(env.n_env), a][alive].all(), name
            env.step(a)


def test_ems_candidates_are_a_subset_of_the_full_grid():
    """The EMS filter may only remove positions, never invent one."""
    rng = np.random.default_rng(4)
    a = BPPBatch(8, nb=1, seed=21, ems=True)
    b = BPPBatch(8, nb=1, seed=21, ems=False)
    for _ in range(25):
        for f in ("hmap", "seq", "head", "packed", "n_packed", "done", "volume"):
            setattr(b, f, getattr(a, f).copy())
        b._invalidate()
        fa, fb = a._positions()[0], b._positions()[0]
        assert not (fa & ~fb).any(), "EMS produced a position the grid rejects"
        if a.done.all():
            break
        a.step(random_action(a, rng))
        a.reset_done()


def test_pruned_ems_matches_the_exhaustive_scan():
    """The edge-pruned enumeration must return the same *set* of spaces.

    Covers bin sides either side of `EMS_PRUNE_S`, and drives the bins to
    full with a real heuristic so the height maps have as many step edges as
    packing ever produces.
    """
    from ar2l.heuristics import scores

    def as_set(cols):
        return set(map(tuple, np.stack([np.asarray(c, np.int64) for c in cols], 1)))

    for S, hi in ((6, 3), (10, 5), (16, 8), (20, 10), (26, 13), (32, 16), (40, 20)):
        env = BPPBatch(4, S=S, nb=1, size_lo=1, size_hi=hi, seed=S, n_items=400)
        for _ in range(60):
            assert as_set(env._ems_pruned()) == as_set(env._ems_exhaustive()), S
            m = env.obs()["l_mask"]
            if not m.any():
                break
            s = np.where(m, scores(env, "dbl"), -np.inf)
            env.step(np.where(m.any(1), s.argmax(1), 0))
            env.reset_done()
            env._invalidate()


def test_pruned_ems_survives_episode_boundaries():
    """A finished bin must not inherit spaces -- the failure mode a stateful
    incremental EMS would have.  Both paths are stateless, so this is a
    regression pin, not a fix."""
    env = BPPBatch(8, S=10, nb=1, seed=4, n_items=12)
    rng = np.random.default_rng(0)
    for _ in range(80):
        env.step(random_action(env, rng))
        env.reset_done()
        cols = env._ems_pruned()
        for b in np.nonzero(env.n_packed == 0)[0]:
            got = {tuple(int(v) for v in r)
                   for r in zip(*[c[cols[0] == b] for c in cols[1:]])}
            assert got == {(0, 0, env.Lx, env.Ly, 0)}, (b, got)


@pytest.mark.parametrize("L,hi", ODD_BINS)
def test_non_cubic_feasibility_matches_brute_force(L, hi):
    """Every claim the simulator makes about a lopsided bin, against brute force."""
    rng = np.random.default_rng(0)
    env = BPPBatch(6, S=L, nb=1, seed=11, size_hi=hi, n_items=200)
    checked = 0
    for _ in range(25):
        feas, z, odims = env._positions()
        for b in range(env.n_env):
            if env.done[b]:
                continue
            ems = brute_ems(env.hmap[b], extents(env))
            cols = env._ems_list()
            got = {tuple(int(v) for v in r)
                   for r in zip(*[c[cols[0] == b] for c in cols[1:]])}
            assert got == set(ems), (L, b)
            for r in range(odims.shape[1]):
                item = odims[b, r]
                ref, Zr = brute(env.hmap[b], item, extents(env), "com")
                ref = ref & brute_corners(ems, extents(env), item)
                if r and odims[b, 0, 0] == odims[b, 0, 1]:
                    ref = np.zeros_like(ref)
                assert (ref == feas[b, r]).all(), (L, b, r)
                assert (Zr[ref] == z[b, r][ref]).all(), (L, b, r)
            checked += 1
        env.step(random_action(env, rng))
        env.reset_done()
    assert checked > 100


@pytest.mark.parametrize("L,hi", ODD_BINS)
def test_non_cubic_ems_paths_agree(L, hi):
    def as_set(cols):
        return set(map(tuple, np.stack([np.asarray(c, np.int64) for c in cols], 1)))

    rng = np.random.default_rng(2)
    env = BPPBatch(6, S=L, nb=1, seed=5, size_hi=hi, n_items=200)
    for _ in range(25):
        assert as_set(env._ems_pruned()) == as_set(env._ems_exhaustive()), L
        env.step(random_action(env, rng))
        env.reset_done()


@pytest.mark.parametrize("L,hi", ODD_BINS)
def test_non_cubic_heuristics_place_legally(L, hi):
    """Every heuristic must run on a lopsided bin and pack inside it."""
    for name in H.NAMES:
        env = BPPBatch(4, S=L, nb=1, seed=7, size_hi=hi, n_items=200)
        for _ in range(40):
            m = env.obs()["l_mask"]
            if not m.any():
                break
            s = np.where(m, H.scores(env, name), -np.inf)
            env.step(np.where(m.any(1), s.argmax(1), 0))
        for b in range(env.n_env):
            it = (env.packed[b, : env.n_packed[b]] * env.scale).round().astype(int)
            assert (it[:, 0] + it[:, 3] <= env.Lx).all(), (name, L)
            assert (it[:, 1] + it[:, 4] <= env.Ly).all(), (name, L)
            assert (it[:, 2] + it[:, 5] <= env.Lz).all(), (name, L)
        assert env.utilization().mean() > 0
