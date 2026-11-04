"""Total support: the share of a box's base on columns solid to the floor.

`min_support` counts a cell once the box right below covers it; the total
reading also asks that nothing further down hangs over air there.  The
reference rebuilds that from the packed boxes alone: a column is solid when
the boxes over it, bottom to top, start where the one below ends, from 0.
"""
import numpy as np
import pytest

from ar2l.env import BPPBatch


def plain(*a, **kw):
    """One box type, no type rule, no arm: the geometry alone."""
    kw.setdefault("S", 10)
    kw.setdefault("size_lo", 1)
    kw.setdefault("size_hi", 5)
    kw.setdefault("n_types", 1)
    kw.setdefault("types", False)
    kw.setdefault("arm_collision", False)
    kw.setdefault("ems", 1)
    return BPPBatch(*a, **kw)


def solid(env, b):
    """(Lx, Ly) the columns of bin `b` with no air under any box in them."""
    boxes = np.rint(env.packed[b, :env.n_packed[b]] * env.scale).astype(int)
    ok = np.ones((env.Lx, env.Ly), bool)
    for x in range(env.Lx):
        for y in range(env.Ly):
            top = 0
            for z, h in sorted((bz, bh) for bx, by, bz, bl, bw, bh in boxes
                               if bx <= x < bx + bl and by <= y < by + bw):
                ok[x, y] &= z == top
                top = z + h
    return ok


def ratio(env, b, dims, x, y):
    """The share of the footprint at (x, y) resting on solid columns."""
    sx, sy = int(dims[0]), int(dims[1])
    sub = env.hmap[b, x:x + sx, y:y + sy]
    on = (sub == sub.max()) & solid(env, b)[x:x + sx, y:y + sy]
    return on.mean()


def put(env, x, dims):
    """Step the box in front to (x, 0) in its first orientation."""
    env.obs()
    hit = np.nonzero((env._lxy[0, :, 0] == x) & (env._lxy[0, :, 1] == 0)
                     & (env._ldim[0] == dims).all(1))[0]
    assert hit.size, f"{dims} cannot go at x = {x}"
    env.step(np.array([hit[0]]))


def test_the_picture():
    """A box overhangs air; what sits on its overhang has full direct support
    and none in total, and what sits on its whole top some of each."""
    seq = np.array([[[4, 2, 2, 0], [4, 2, 2, 0], [1, 2, 2, 0], [4, 2, 2, 0]]])
    env = plain(1, S=(10, 2, 10), ems=0, rot=1, min_support=0.0,
                min_total_support=0.0, max_l=256)
    env.reset(seq)
    put(env, 0, (4, 2, 2))            # A on the floor, x 0..3
    put(env, 1, (4, 2, 2))            # B on A's x 1..3, air under x 4
    assert env.gmap[0, :4].all() and not env.gmap[0, 4].any()
    assert env.gmap[0, 5:].all()      # bare floor is solid
    C, D = np.array([[1, 2, 2]]), np.array([[4, 2, 2]])
    z = np.full((1, env.Lx, env.Ly), 4)
    for thr, c_ok, d_ok in ((0.75, False, True), (0.8, False, False),
                            (0.01, False, True)):
        env.min_total_support = thr
        # C on B's overhang alone: 100% direct, 0% total
        assert env._ground_ok(C, z)[0, 4, 0] == c_ok
        # D on B's whole top: 100% direct, x 1..3 of its 4 cells in total
        assert env._ground_ok(D, z)[0, 1, 0] == d_ok
    env.min_total_support = 0.0
    put(env, 4, (1, 2, 2))            # C: its whole column hangs over air
    assert not env.gmap[0, 4].any() and env.hmap[0, 4, 0] == 6


@pytest.mark.parametrize("mode", ["com", "cdrl"])
def test_feasibility_matches_brute_force(mode):
    """Every position the rule leaves is one the rule-off env offers and the
    reference finds solid enough, and every such one is left."""
    rng = np.random.default_rng(5)
    env = plain(8, nb=2, seed=2, stability=mode, min_support=0.4,
                min_total_support=0.6, ems=0)
    checked = 0
    for _ in range(30):
        on = env._positions()[0].copy()
        env.min_total_support = 0.0
        env._invalidate(hmap=False)
        off, z, odims, _ = env._positions()
        off = off.copy()
        env.min_total_support = 0.6
        env._invalidate(hmap=False)
        assert not (on & ~off).any()
        for b in range(env.n_env):
            if env.done[b]:
                continue
            for r in range(odims.shape[1]):
                for x, y in zip(*np.nonzero(off[b, r])):
                    want = ratio(env, b, odims[b, r], x, y) >= 0.6 - 1e-6
                    assert on[b, r, x, y] == want
                    checked += 1
        m = env.obs()["l_mask"]
        env.step(np.array([rng.choice(np.nonzero(row)[0]) if row.any() else 0
                           for row in m]))
        env.reset_done()
    assert checked > 1000


def test_gmap_matches_the_packed_boxes():
    """`gmap` is the reference's solid columns after every step."""
    rng = np.random.default_rng(0)
    env = plain(4, seed=1, min_support=0.3, ems=0)
    for _ in range(40):
        m = env.obs()["l_mask"]
        env.step(np.array([rng.choice(np.nonzero(row)[0]) if row.any() else 0
                           for row in m]))
        for b in range(env.n_env):
            assert (env.gmap[b] == solid(env, b)).all()


def test_off_by_default_and_checked():
    assert BPPBatch(1, arm_collision=False).min_total_support == 0.0
    with pytest.raises(ValueError):
        BPPBatch(1, arm_collision=False, min_total_support=1.5)


def test_the_command_lines_hand_it_to_the_env():
    from ar2l import evaluate, train
    base = ["--name", "t", "--type_rule", "touch", "--n_env", "2"]
    a = train.get_parser().parse_args(base)
    assert a.min_total_support == 0.0
    assert train.make_env(a, seed=0).min_total_support == 0.0
    a = train.get_parser().parse_args(base + ["--min_total_support", "0.7"])
    assert train.make_env(a, seed=0).min_total_support == 0.7
    assert evaluate.get_parser().parse_args(
        ["--min_total_support", "0.7"]).min_total_support == 0.7


def test_the_game_hands_it_to_the_env():
    """Every tab's form carries the field through `validate` to the env, and
    the board marks what the rule alone turns down."""
    from ar2l.viz import game as G
    from ar2l.viz import agents as A
    base = {"bin": [31, 27, 40], "size_lo": [2, 2, 2], "size_hi": [12, 12, 10],
            "n_items": 30, "nb": 4, "n_pick": 2, "max_l": 256, "rot": 2,
            "ems": 3, "stability": "com", "min_support": 0.8,
            "source": "random", "type_rule": "touch", "arm_collision": 0}
    for v in (0.0, 0.7):
        p, err = G.validate(dict(base, min_total_support=v))
        assert not err, err
        assert p["min_total_support"] == v
        env = A.episode_env(np.array([[12, 8, 4, 0]]), S=p["bin"],
                            **G.stack_kw(p))
        assert env.min_total_support == v
        board = G.board(G.new_game(3, "none", p, human_pick=True)[1])
        assert board["min_total_support"] == v
        assert not np.asarray(board["totalblock"]).any()   # empty bin
    assert G.validate(dict(base, min_total_support=2))[1]
    assert G.defaults()["min_total_support"] == 0.0
    assert "min_total_support" in G.GAME_KEYS
    assert "min_total_support" in G.EXP_KEYS
    assert "gmap" in G.STATE
