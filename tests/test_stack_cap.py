"""The stack-height limit over small boxes.

Once a box of a limited type is placed, a later box may start over its
footprint only below the box's top plus an allowance set by its shorter
footprint side.  The reference here is the packed list itself: replaying it
box by box rebuilds every limit independently of `cmap` and the sweeps.

Every env below pins the rule's numbers rather than reading `config.yaml`,
so editing the shipped values cannot move these tests.
"""
import numpy as np
import pytest

from ar2l import pack_collision as PC
from ar2l.config import CFG
from ar2l.env import BPPBatch, NO_CAP, stack_allowance


CLASSES = [{"p": 1, "lo": [1, 1, 1], "hi": [3, 3, 3]},
           {"p": 1, "lo": [2, 1, 1], "hi": [4, 3, 2]},
           {"p": 1, "lo": [1, 2, 1], "hi": [3, 4, 3]}]

# the shipped rule on 4 cm cells: a short side of 1-3 cells gets 2 cells,
# 4 -> 9, 5 -> 16, 6 -> 24, 7 -> 31, and 8 cells (32 cm) and up no limit
RULE = dict(stack_side_cm=(12, 30), stack_allow_cm=(8, 140), stack_cell_cm=4.0)

# one instance's width, as `capped` is told to expect -- `reset_done` deals
# that many boxes into a restarted bin
N = 8


def capped(n_env=1, **kw):
    """Only the limit in play: every grid position is offered, no rotation,
    no type rule, no support floor, no arm."""
    kw.setdefault("S", (8, 8, 20))
    for k, v in dict(nb=1, n_items=N, rot=1, ems=0, stability="com",
                     min_support=0.0, n_types=3, types=CLASSES,
                     type_constraint=False, arm_collision=False,
                     stack_cap=True, stack_types=[2]).items():
        kw.setdefault(k, v)
    return BPPBatch(n_env, **{**RULE, **kw})


def seq(*boxes):
    """One instance: the boxes in order, then 1x1x1 type-0 filler."""
    return np.array([list(boxes) + [(1, 1, 1, 0)] * (N - len(boxes))], np.int16)


def offered(env, x, y, r=0, b=0):
    return bool(env._positions()[0][b, r, x, y])


def offered_without_the_rule(env, x, y):
    """The same question with the limit lifted on the same bin."""
    env.stack_cap = False
    env._invalidate(hmap=False)
    try:
        return offered(env, x, y)
    finally:
        env.stack_cap = True
        env._invalidate(hmap=False)


def place(env, x, y, r=0):
    """Step bin 0 with the candidate at (x, y) in orientation `r`."""
    m, lxy = env.obs()["l_mask"][0], env._lxy[0]
    i = np.flatnonzero(m & (lxy[:, 0] == x) & (lxy[:, 1] == y)
                       & (lxy[:, 2] == r))
    assert i.size, f"({x}, {y}) is not on offer"
    env.step(np.array([i[0]]))


def test_the_allowance_follows_the_rule_in_cm():
    a = stack_allowance(9, (12, 30), (8, 140), 4.0)
    assert a.tolist() == [2, 2, 2, 2, 9, 16, 24, 31, NO_CAP, NO_CAP]
    # finer cells: 28 cm is 14 of them and still limited, 30 cm is not
    b = stack_allowance(15, (12, 30), (8, 140), 2.0)
    assert b[14] == 62 and b[15] == NO_CAP


@pytest.mark.parametrize("side, allow, cell", [((30, 12), (8, 140), 4.0),
                                               ((0, 30), (8, 140), 4.0),
                                               ((12, 30), (-1, 140), 4.0),
                                               ((12, 30), (8, 140), 0.0)])
def test_a_broken_rule_is_rejected(side, allow, cell):
    with pytest.raises(ValueError):
        stack_allowance(8, side, allow, cell)


def test_a_small_box_takes_a_box_on_top_but_no_tall_stack():
    """A 2x2 brown box (8 cm) has 2 cells of allowance: its top is 2, so a
    box may start at 2 or 3 over it and never at 4."""
    env = capped()
    env.reset(seq((2, 2, 2, 2), (2, 2, 1, 0), (2, 2, 1, 0), (2, 2, 1, 0)))
    place(env, 0, 0)
    assert (env.cmap[0, :2, :2] == 4).all()
    assert (env.cmap[0, 2:] == NO_CAP).all() and (env.cmap[0, :, 2:] == NO_CAP).all()
    place(env, 0, 0)                        # starts at 2
    place(env, 0, 0)                        # starts at 3, the last cell allowed
    assert not offered(env, 0, 0), "a box started at the limit"
    assert offered_without_the_rule(env, 0, 0)
    assert offered(env, 2, 0), "a column beside the small box was limited"


def test_lifting_the_rule_offers_the_same_placement():
    env = capped(stack_cap=False)
    env.reset(seq((2, 2, 2, 2), (2, 2, 1, 0), (2, 2, 1, 0), (2, 2, 1, 0)))
    for _ in range(3):
        place(env, 0, 0)
    assert offered(env, 0, 0)
    assert (env.cmap == NO_CAP).all()


def test_only_the_listed_types_set_a_limit():
    env = capped()
    env.reset(seq((2, 2, 2, 0), (2, 2, 6, 1), (2, 2, 1, 0)))
    place(env, 0, 0)
    place(env, 0, 0)
    assert (env.cmap == NO_CAP).all()
    assert offered(env, 0, 0), "a box of an unlimited type limited its stack"


def test_a_wide_small_box_is_not_limited():
    """The shorter side decides: 8 x 8 cells is 32 cm, past the 30 cm end."""
    env = capped()
    env.reset(seq((8, 8, 2, 2), (8, 8, 9, 0), (2, 2, 1, 0)))
    place(env, 0, 0)
    assert (env.cmap == NO_CAP).all()
    place(env, 0, 0)
    assert offered(env, 0, 0)


def test_the_allowance_grows_with_the_side():
    """A 4-cell side (16 cm) allows 9 cells: a box may start up to 4 + 8."""
    env = capped(S=(8, 8, 30))
    env.reset(seq((4, 4, 4, 2), (4, 4, 8, 0), (4, 4, 1, 0), (4, 4, 1, 0)))
    place(env, 0, 0)
    assert (env.cmap[0, :4, :4] == 4 + 9).all()
    place(env, 0, 0)                        # starts at 4, top 12
    place(env, 0, 0)                        # starts at 12 < 13, top 13
    assert not offered(env, 0, 0)


def test_the_limit_stays_with_the_small_box():
    """A slab bridging the small box and a neighbour does not spread the
    limit over its own footprint: above the neighbour any height is fine."""
    env = capped()
    env.reset(seq((2, 2, 2, 2), (2, 2, 2, 0), (4, 2, 2, 0), (2, 2, 1, 0)))
    place(env, 0, 0)                        # the small box, limit 4
    place(env, 2, 0)                        # its neighbour, as tall
    place(env, 0, 0)                        # the slab over both, starts at 2
    assert offered(env, 2, 0), "the limit spread to the neighbour's cells"
    assert not offered(env, 0, 0)
    # half over the small box, fully supported by the slab: only the limit
    # can be what rules it out
    assert not offered(env, 1, 0), "one limited cell under the box let it pass"
    assert offered_without_the_rule(env, 1, 0)


def test_a_restart_clears_only_the_bins_it_restarts():
    env = capped(2)
    env.reset(np.concatenate([seq((2, 2, 2, 2))] * 2))
    o = env.obs()
    act = [int(np.flatnonzero(o["l_mask"][b] & (env._lxy[b, :, 0] == 0)
                              & (env._lxy[b, :, 1] == 0))[0]) for b in range(2)]
    env.step(np.array(act))
    assert (env.cmap[:, :2, :2] == 4).all()
    env.done[0] = True
    env.reset_done()
    assert (env.cmap[0] == NO_CAP).all()
    assert (env.cmap[1, :2, :2] == 4).all()
    env.reset()
    assert (env.cmap == NO_CAP).all()


def random_action(env, rng):
    m = env.obs()["l_mask"]
    return np.array([rng.choice(np.flatnonzero(r)) if r.any() else 0 for r in m])


def test_the_mask_is_exactly_the_rule_on_every_candidate():
    """Every placement the rule-free twin offers, checked by hand: kept
    exactly when it starts below the lowest limit under its footprint."""
    rng = np.random.default_rng(0)
    kw = dict(S=(8, 8, 12), rot=2, n_items=80, seed=1, stack_types=[0, 1, 2])
    on, off = capped(6, **kw), capped(6, stack_cap=False, **kw)
    removed = kept = 0
    for _ in range(40):
        for f in ("hmap", "tmap", "cmap", "seq", "head", "length", "packed",
                  "ptype", "n_packed", "done", "volume"):
            setattr(off, f, getattr(on, f).copy())
        off._invalidate()
        fa, z, dims, _ = on._positions()
        fb = off._positions()[0]
        assert not (fa & ~fb).any(), "the rule invented a placement"
        for b, r, x, y in zip(*np.nonzero(fb)):
            sx, sy = dims[b, r, :2]
            want = z[b, r, x, y] < on.cmap[b, x:x + sx, y:y + sy].min()
            assert fa[b, r, x, y] == want, (b, r, x, y)
            kept += int(want)
            removed += int(not want)
        if on.done.all():
            break
        on.step(random_action(on, rng))
        on.reset_done()
    assert removed > 50 and kept > 50, (removed, kept)


def test_no_box_ever_starts_above_a_limit():
    """Replayed from the packed list alone, which also rebuilds `cmap`."""
    rng = np.random.default_rng(2)
    env = capped(8, S=(8, 8, 16), rot=2, n_items=100, seed=3)
    while not env.done.all():
        env.step(random_action(env, rng))
    allow, checked = env.stack_allow, 0
    for b in range(env.n_env):
        lim = np.full((env.Lx, env.Ly), NO_CAP, np.int64)
        items = (env.packed[b, : env.n_packed[b]] * env.scale).round().astype(int)
        for i, (x, y, z, sx, sy, sz) in enumerate(items):
            assert z < lim[x:x + sx, y:y + sy].min(), (b, i)
            a = allow[min(sx, sy)]
            if env.ptype[b, i] == 2 and a < NO_CAP:
                lim[x:x + sx, y:y + sy] = np.minimum(lim[x:x + sx, y:y + sy],
                                                     z + sz + a)
            checked += 1
        assert (env.cmap[b] == lim).all(), f"bin {b}: cmap drifted from the boxes"
    assert checked > 100 and (env.cmap < NO_CAP).any()


def test_one_type_never_sets_a_limit():
    """`n_types = 1` is the pre-type simulator, whatever types are limited."""
    rng = np.random.default_rng(4)
    env = BPPBatch(4, S=(8, 8, 12), nb=1, n_items=60, n_types=1, types=False,
                   size_lo=1, size_hi=3, stack_cap=True, stack_types=[2],
                   arm_collision=False, seed=0, **RULE)
    while not env.done.all():
        env.step(random_action(env, rng))
    assert (env.cmap == NO_CAP).all()


def test_the_cell_size_defaults_to_the_arm_filters():
    cell = 100 * PC.robot_cell_m(CFG["robot"], CFG["eval"]["cell_cm"],
                                 CFG["eval"]["box_scale"])
    a = capped(stack_cell_cm=None).stack_allow
    b = capped(stack_cell_cm=cell).stack_allow
    assert (a == b).all()
    # the game hands the arm its own cell size; the limit follows it
    c = capped(stack_cell_cm=None, arm_cell_m=0.02).stack_allow
    assert (c == capped(stack_cell_cm=2.0).stack_allow).all()


def test_the_game_and_the_experiment_take_the_limit_from_their_forms():
    from ar2l.viz import game as G
    from ar2l.viz import experiment as X
    form = dict(G.defaults(), source="random", bin=[8, 8, 12], arm_collision=0,
                size_lo=[1, 1, 1], size_hi=[3, 3, 3],
                stack_cap=1, stack_cap_types="1, 2", stack_cap_side_cm="10, 20",
                stack_cap_allow_cm=[4, 40])
    p, err = G.validate(form)
    assert not err, err
    assert p["stack_cap_types"] == [1, 2]
    assert p["stack_cap_side_cm"] == [10.0, 20.0]
    assert p["stack_cap_allow_cm"] == [4.0, 40.0]
    env = G.new_game(0, None, p)[1]["env"]
    assert env.stack_types.tolist() == [1, 2]
    want = stack_allowance(min(env.Lx, env.Ly), (10, 20), (4, 40),
                           100 * G.cell_m(p))
    assert (env.stack_allow == want).all()
    assert "stack_cap_side_cm=10x20" in X.params_text("heur:dbl", p, (8, 8, 12))[0]
    q, err = G.exp_params(dict(form, n_types=3))
    assert not err and q["stack_cap_allow_cm"] == [4.0, 40.0]


@pytest.mark.parametrize("bad", [dict(stack_cap_types="-1"),
                                 dict(stack_cap_side_cm="30, 12"),
                                 dict(stack_cap_side_cm="12"),
                                 dict(stack_cap_allow_cm="8, -1"),
                                 dict(stack_cap_allow_cm="a, b")])
def test_the_forms_refuse_a_broken_limit(bad):
    from ar2l.viz import game as G
    p, err = G.validate(dict(G.defaults(), source="random", **bad))
    assert err and any("stack" in e or "limited" in e for e in err), err
