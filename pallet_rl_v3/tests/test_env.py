"""The simulator, checked against straightforward reference implementations."""
import numpy as np
import pytest

import ar2l.env as E
from ar2l.env import BPPBatch
from ar2l import heuristics as H


def brute(hmap, item, S, mode):
    """Feasibility and landing height, one loading position at a time."""
    sx, sy, sz = item
    ok = np.zeros((S, S), bool)
    Z = np.zeros((S, S), np.int32)
    for x in range(S - sx + 1):
        for y in range(S - sy + 1):
            sub = hmap[x:x + sx, y:y + sy]
            z = int(sub.max())
            Z[x, y] = z
            if z + sz > S:
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


def brute_ems(hmap, S):
    """Every maximal footprint, found by trying all of them."""
    out = []
    for x0 in range(S):
        for wx in range(1, S - x0 + 1):
            for y0 in range(S):
                for wy in range(1, S - y0 + 1):
                    f = int(hmap[x0:x0 + wx, y0:y0 + wy].max())
                    grow = ((x0 - 1, wx + 1, y0, wy), (x0, wx + 1, y0, wy),
                            (x0, wx, y0 - 1, wy + 1), (x0, wx, y0, wy + 1))
                    if all(hmap[a:a + b, c:c + d].max() > f for a, b, c, d in grow
                           if a >= 0 and c >= 0 and a + b <= S and c + d <= S):
                        out.append((x0, y0, wx, wy, f))
    return out


def brute_corners(ems, S, item):
    """The four bottom corners of every space that can take the item."""
    sx, sy, sz = item
    m = np.zeros((S, S), bool)
    for x0, y0, wx, wy, f in ems:
        if wx >= sx and wy >= sy and f + sz <= S:
            for px in (x0, x0 + wx - sx):
                for py in (y0, y0 + wy - sy):
                    m[px, py] = True
    return m


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
            ems = brute_ems(env.hmap[b], env.S)
            for r in range(odims.shape[1]):
                item = odims[b, r]
                ref, Zr = brute(env.hmap[b], item, env.S, mode)
                ref = ref & brute_corners(ems, env.S, item)
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
            ref = {(x0, y0, wx, wy, f) for x0, y0, wx, wy, f in brute_ems(env.hmap[b], env.S)}
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
            got = (env.packed[b, env.n_packed[b] - 1] * env.S).round().astype(int)
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
        vox = np.zeros((env.S,) * 3, np.int8)
        for i in range(env.n_packed[b]):
            x, y, z, sx, sy, sz = (env.packed[b, i] * env.S).round().astype(int)
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
        items = (env.packed[b, : env.n_packed[b]] * env.S).round().astype(int)
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
