"""`type_rule: mixed_touch` / `mixed_column`: the cell's rules for its blue,
white and brown cartons.

Blue (0), white (1) and brown (2) each go on the floor and on their own type --
blue on blue only with all of its base on blue -- and across types:

                       normally          else, soft mix off  else, soft mix on
    blue on blue       all base on blue  fallback            fallback
    blue on white      (a)               fallback            last resort
    blue on brown      (a)               fallback            last resort
    white on blue      (a) or (d)        fallback            last resort
    white on brown     (a)               fallback            last resort
    brown on blue      (a), (c) or (d)   fallback            last resort
                       -- soft mix on: (a) or (d)
    brown on white     never             never               never

(a) its top within `mixed_top_gap_cm` of the lid; (c) a brown footprint at
least `mixed_area_frac` of the small blue's; (d) no small blue box could still
be put on that blue.  A fallback is a box's own, when it has no normal place;
the last resort is the station's, when no box within reach has any place.

"On" is what the box rests on under `mixed_touch`, and everything under its
footprint at any depth under `mixed_column`.  Most tests hold for both and run
under both (the `rule` fixture); the ones that tell them apart say so.

The bins here are tiny and set up by hand: an 8 x 8 x 10 grid of 1 cm cells,
so `mixed_top_gap_cm=1` is a one-cell gap, every position offered (`ems=0`),
no rotation, no arm and no stack cap unless a test says otherwise.  The last
test plays random episodes against a brute-force reading of the rule.
"""
import copy

import numpy as np
import pytest

import ar2l.env as E
from ar2l.env import (BLUE, BROWN, IF, NO, TOP, WHITE, YES, MIXED_RULES,
                      BPPBatch, mixed_tables)

CLASSES = [{"p": 1, "lo": [1, 1, 1], "hi": [3, 3, 3]},
           {"p": 1, "lo": [2, 1, 1], "hi": [4, 3, 2]},
           {"p": 1, "lo": [1, 2, 1], "hi": [3, 4, 3]}]

RULE = dict(mixed_small_blue=(3, 2, 2), mixed_top_gap_cm=1,
            mixed_area_frac=0.8, stack_cell_cm=1.0)


@pytest.fixture(params=MIXED_RULES)
def rule(request):
    return request.param



def mixed(rule, n_env=1, **kw):
    for k, v in dict(S=(8, 8, 10), nb=1, n_items=2, rot=1, ems=0,
                     min_support=0.5, n_types=3, types=CLASSES,
                     arm_collision=False, stack_cap=False, **RULE).items():
        kw.setdefault(k, v)
    return BPPBatch(n_env, type_rule=rule, **kw)


def bin_with(rule, item, cols, below=None, **kw):
    """One bin with the box `item` (sx, sy, sz, type) in front and the columns
    `cols` -- {(x0, x1, y0, y1): (height, type)} -- built by hand.  `below`
    -- {(x0, x1, y0, y1): type} -- buries a box of that type deeper in those
    columns, which only `smap` remembers."""
    env = mixed(rule, **kw)
    env.reset(np.array([[item, (1, 1, 1, 0)]], np.int16))
    for (x0, x1, y0, y1), (h, t) in cols.items():
        env.hmap[0, x0:x1, y0:y1] = h
        env.tmap[0, x0:x1, y0:y1] = t
        env.smap[0, x0:x1, y0:y1] = 1 << t
    for (x0, x1, y0, y1), t in (below or {}).items():
        env.smap[0, x0:x1, y0:y1] |= 1 << t
    env._invalidate()
    return env


def offered(env, x, y, r=0):
    return bool(env._positions()[0][0, r, x, y])


EVERYWHERE = (0, 8, 0, 8)


# ------------------------------------------------------------------- tables
def test_the_tables_say_who_may_go_on_whom():
    rule, fallback = mixed_tables(3)
    # normal rules: own type -- blue on blue if all of its base is on blue --
    # white and brown on blue under their conditions, the rest near the lid
    assert rule[0].tolist() == [[IF, TOP, TOP],
                                [IF, YES, TOP],
                                [IF, NO, YES]]
    # a box's own fallback: blue and white on anything, brown on blue; and
    # no last resort beyond it
    assert rule[1].tolist() == [[YES, YES, YES],
                                [YES, YES, YES],
                                [YES, NO, YES]]
    assert (rule[2] == rule[1]).all()
    assert fallback.tolist() == [True, True, True]
    # soft mix: the same normal rules, a fallback for blue on blue alone, and
    # the station's last resort on every mix but brown on white
    soft, soft_fb = mixed_tables(3, soft=True)
    assert (soft[0] == rule[0]).all()
    assert soft[1].tolist() == [[YES, TOP, TOP],
                                [IF, YES, TOP],
                                [IF, NO, YES]]
    assert soft[2].tolist() == [[YES, YES, YES],
                                [YES, YES, YES],
                                [YES, NO, YES]]
    assert soft_fb.tolist() == [True, False, False]
    # a fourth type keeps the `touch` rule, and two types still work
    r4, fb4 = mixed_tables(4)
    assert (r4[:, 3] == [NO, NO, NO, YES]).all()
    assert (r4[:, :3, 3] == NO).all() and not fb4[3]
    assert mixed_tables(2)[0].shape == (3, 2, 2)


def test_the_small_blue_box_is_read_like_an_order_row():
    """50 x 30 x 18 cm with the shipped 4 cm padding on length and width, on
    4 cm cells rounded to the nearest (a half going down): 13 x 8 x 4 cells."""
    kw = dict(type_rule="mixed_touch", mixed_top_gap_cm=5,
              arm_collision=False)
    env = BPPBatch(1, stack_cell_cm=4.0, **kw)
    assert env.small_blue == (13, 8, 4)
    assert env.top_gap == 1                     # 5 cm on 4 cm cells
    assert BPPBatch(1, stack_cell_cm=5.0, **kw).top_gap == 1
    assert BPPBatch(1, stack_cell_cm=2.0, **kw).top_gap == 2


# ------------------------------------------------------------- blue on blue
def test_blue_on_blue_needs_all_of_its_base_on_blue(rule):
    env = bin_with(rule, (3, 2, 1, BLUE), {(0, 3, 0, 2): (2, BLUE)})
    assert offered(env, 0, 0), "a blue box exactly on a blue one"
    assert not offered(env, 1, 0), "2/3 of its base on blue is not enough"
    assert offered(env, 4, 4), "the floor"
    assert not env.fallback()[0]


def test_blue_goes_anywhere_once_nothing_else_is_left(rule):
    """No full-blue support and no floor anywhere: the fallback opens, and
    blue goes on white, and on part blue."""
    env = bin_with(rule, (3, 2, 1, BLUE), {EVERYWHERE: (2, WHITE),
                                           (0, 2, 0, 2): (2, BLUE)})
    assert env.fallback()[0]
    assert offered(env, 0, 0), "part blue, part white"
    assert offered(env, 4, 4), "all white"


# --------------------------------------------------- white and brown on blue
def test_white_on_blue_near_the_lid(rule):
    """(a): its top at most one cell under the lid."""
    on_blue = {(0, 3, 0, 2): (6, BLUE)}
    assert offered(bin_with(rule, (3, 2, 3, WHITE), on_blue), 0, 0), \
        "top 9 of 10"
    assert offered(bin_with(rule, (3, 2, 4, WHITE), on_blue), 0, 0), \
        "top 10 of 10"
    assert not offered(bin_with(rule, (3, 2, 2, WHITE), on_blue), 0, 0), \
        "top 8 of 10, and a small blue box still fits on that blue"


def test_white_on_blue_where_no_small_blue_box_fits_any_more(rule):
    """(d) by height: a 3 x 2 x 4 small blue fits on blue at 6, not at 7."""
    tall = dict(mixed_small_blue=(3, 2, 4))
    assert offered(bin_with(rule, (3, 2, 1, WHITE),
                            {(0, 3, 0, 2): (7, BLUE)}, **tall), 0, 0)
    assert not offered(bin_with(rule, (3, 2, 1, WHITE),
                                {(0, 3, 0, 2): (6, BLUE)}, **tall), 0, 0)


def test_rule_d_is_per_spot(rule):
    """A blue column the small blue no longer fits on takes white, while the
    blue next to it, which still takes one, does not."""
    env = bin_with(rule, (3, 2, 1, WHITE), {(0, 3, 0, 2): (7, BLUE),
                                            (4, 7, 0, 2): (3, BLUE)},
                   mixed_small_blue=(3, 2, 4))
    assert offered(env, 0, 0)
    assert not offered(env, 4, 0)


def test_rule_d_asks_the_arm(rule):
    """A blue spot the arm cannot put a small blue box on takes white; the
    same spot does not once the arm can."""
    def env_with_arm(blocked):
        env = bin_with(rule, (3, 2, 1, WHITE), {(0, 3, 0, 2): (3, BLUE)},
                       arm_collision=True)
        sb = np.asarray(env.small_blue)

        def moves(bb, xx, yy, sizes, zz):
            small = (np.asarray(sizes) == sb).all(1)
            return np.where(small & blocked, -1, 0).astype(np.int8)
        env._arm_moves = moves
        return env
    assert offered(env_with_arm(True), 0, 0)
    assert not offered(env_with_arm(False), 0, 0)


def test_a_large_brown_goes_on_blue(rule):
    """(c): a 3 x 2 brown is all of the small blue's footprint, a 2 x 2 one
    is 4/6 of it, under 80%.  Soft mix drops (c)."""
    on_blue = {(0, 3, 0, 2): (2, BLUE)}
    assert offered(bin_with(rule, (3, 2, 1, BROWN), on_blue), 0, 0)
    assert not offered(bin_with(rule, (2, 2, 1, BROWN), on_blue), 0, 0)
    assert not offered(bin_with(rule, (3, 2, 1, BROWN), on_blue,
                                soft_mix=True), 0, 0)


def test_white_on_blue_is_also_a_fallback(rule):
    """Nothing but blue a small blue box still fits on, and far from the lid:
    white has no placement under the normal rules, so it falls back."""
    env = bin_with(rule, (3, 2, 1, WHITE), {EVERYWHERE: (2, BLUE)})
    assert env.fallback()[0]
    assert offered(env, 0, 0)


# ------------------------------------------------------------- near the lid
@pytest.mark.parametrize("soft", [False, True], ids=["hard", "soft"])
@pytest.mark.parametrize("t, u", [(BLUE, WHITE), (BLUE, BROWN),
                                  (WHITE, BROWN)])
def test_mixes_near_the_lid(rule, t, u, soft):
    """(a): blue on white or brown, white on brown, as normal placements --
    the floor free -- when the box's top ends within a cell of the lid, soft
    mix or not; lower down they are not."""
    box = (3, 2, 2, t)
    near, low = {(0, 3, 0, 2): (8, u)}, {(0, 3, 0, 2): (4, u)}
    assert offered(bin_with(rule, box, near, soft_mix=soft), 0, 0)
    assert not offered(bin_with(rule, box, low, soft_mix=soft), 0, 0)


@pytest.mark.parametrize("soft", [False, True], ids=["hard", "soft"])
def test_brown_never_goes_on_white(rule, soft):
    """Nothing but white, near the lid or not, soft mix or not: the brown
    box has nowhere to go -- its fallback, or the last resort, opens blue
    to it and nothing else."""
    for h in (2, 8):
        env = bin_with(rule, (3, 2, 2, BROWN), {EVERYWHERE: (h, WHITE)},
                       soft_mix=soft)
        assert bool(env.fallback()[0]) == (not soft)
        assert bool(env.last_resort()[0]) == soft
        assert env.n_feasible()[0] == 0, h
        assert env.done[0] or not env._head_placeable()[0]


def test_column_counts_a_deep_box_near_the_lid_too():
    """Blue all of whose base is on blue, but white buried under that blue:
    `mixed_column` shuts it -- unless its top ends near the lid."""
    deep = dict(below={(0, 3, 0, 2): WHITE})
    for h, want in ((8, True), (4, False)):
        env = bin_with("mixed_column", (3, 2, 2, BLUE),
                       {(0, 3, 0, 2): (h, BLUE)}, **deep)
        assert offered(env, 0, 0) is want, h


# ------------------------------------------------------- the fallback, hard
@pytest.mark.parametrize("t", [BLUE, WHITE])
def test_blue_and_white_go_on_brown_only_as_a_fallback(rule, t):
    one = {(0, 3, 0, 2): (2, BROWN)}
    assert not offered(bin_with(rule, (3, 2, 1, t), one), 0, 0), \
        "the floor is free, so brown is off limits"
    env = bin_with(rule, (3, 2, 1, t), {EVERYWHERE: (2, BROWN)})
    assert env.fallback()[0] and offered(env, 0, 0)


def test_blue_goes_on_white_only_as_a_fallback(rule):
    one = {(0, 3, 0, 2): (2, WHITE)}
    assert not offered(bin_with(rule, (3, 2, 1, BLUE), one), 0, 0)
    assert offered(bin_with(rule, (3, 2, 1, BLUE), {EVERYWHERE: (2, WHITE)}),
                   0, 0)


def test_a_box_on_two_types_must_be_allowed_on_both(rule):
    """White far from the lid, on blue no small blue box fits on, (d): with
    white beside that blue it may go, with brown it may not."""
    for u, want in ((WHITE, True), (BROWN, False)):
        env = bin_with(rule, (3, 2, 1, WHITE), {(0, 2, 0, 2): (4, BLUE),
                                                (2, 3, 0, 2): (4, u)})
        assert offered(env, 0, 0) is want, u


def test_the_fallback_is_per_box(rule):
    """A white box in front with nowhere normal to go falls back even though
    the blue box behind it in the station has a normal placement."""
    env = mixed(rule, nb=2, n_pick=2)
    env.reset(np.array([[(3, 2, 1, WHITE), (3, 2, 1, BLUE)]], np.int16))
    env.hmap[0] = 2
    env.tmap[0] = BLUE
    env.smap[0] = 1 << BLUE
    env._invalidate()
    assert env.fallback()[0]
    assert env.pick_mask(env.window()[1])[0].tolist() == [True, True]


# ----------------------------------------------- soft mix: the last resort
def station(rule, boxes, cols, **kw):
    """`bin_with` for a station of `boxes`, all of them within reach."""
    env = mixed(rule, nb=len(boxes), n_pick=len(boxes), n_items=len(boxes),
                **kw)
    env.reset(np.array([boxes], np.int16))
    for (x0, x1, y0, y1), (h, t) in cols.items():
        env.hmap[0, x0:x1, y0:y1] = h
        env.tmap[0, x0:x1, y0:y1] = t
        env.smap[0, x0:x1, y0:y1] = 1 << t
    env._invalidate()
    return env


def picks(env):
    return env.pick_mask(env.window()[1])[0].tolist()


def test_soft_mix_turns_a_fallback_into_the_last_resort(rule):
    """A white box with only open blue far from the lid, beside a blue box
    that can go on blue: without soft mix the white box falls back onto blue
    by itself; with it, it waits while the blue box can go."""
    every_blue = {EVERYWHERE: (2, BLUE)}
    boxes = [(3, 2, 1, WHITE), (3, 2, 1, BLUE)]
    hard = station(rule, boxes, every_blue)
    assert hard.fallback()[0] and picks(hard) == [True, True]
    soft = station(rule, boxes, every_blue, soft_mix=True)
    assert not soft.fallback()[0] and not soft.last_resort()[0]
    assert picks(soft) == [False, True]


def test_soft_mix_keeps_blue_off_white_until_nothing_else_can_go(rule):
    every_white = {EVERYWHERE: (2, WHITE)}
    boxes = [(3, 2, 1, BLUE), (3, 2, 1, WHITE)]
    assert picks(station(rule, boxes, every_white)) == [True, True]
    assert picks(station(rule, boxes, every_white,
                         soft_mix=True)) == [False, True]
    env = station(rule, [(3, 2, 1, BLUE), (3, 2, 1, BLUE)], every_white,
                  soft_mix=True)
    assert env.last_resort()[0] and picks(env) == [True, True]


def test_soft_mix_keeps_blue_on_blue_a_fallback_of_its_own(rule):
    """Blue strips two cells wide, a lower cell between: a 3-wide blue box
    can rest on part blue only.  That stays its own fallback under soft mix
    -- open while the white box beside it can go too."""
    strips = {EVERYWHERE: (2, BLUE), (2, 3, 0, 8): (1, BLUE),
              (5, 6, 0, 8): (1, BLUE)}
    env = station(rule, [(3, 2, 1, BLUE), (1, 1, 1, WHITE)], strips,
                  soft_mix=True)
    assert env.fallback()[0] and not env.last_resort()[0]
    assert picks(env) == [True, True] and env.n_feasible()[0] > 0


def test_brown_on_blue_without_a_c_d(rule):
    """Nothing but blue a small blue box still fits on, far from the lid, and
    a small brown box: no (a), (c) or (d).  It goes on blue all the same, by
    its own fallback -- or, under soft mix, as the last resort."""
    every_blue = {EVERYWHERE: (2, BLUE)}
    env = bin_with(rule, (2, 2, 1, BROWN), every_blue)
    assert env.fallback()[0] and not env.last_resort()[0]
    assert offered(env, 0, 0)
    env = bin_with(rule, (2, 2, 1, BROWN), every_blue, soft_mix=True)
    assert env.last_resort()[0] and not env.fallback()[0]
    assert offered(env, 0, 0)
    # with the floor free it has a normal place, so blue a small blue box
    # still fits on (3 x 2) stays off limits either way
    for soft in (False, True):
        assert not offered(bin_with(rule, (2, 2, 1, BROWN),
                                    {(0, 3, 0, 2): (2, BLUE)},
                                    soft_mix=soft), 0, 0)


def test_soft_mix_waits_for_the_whole_station(rule):
    """A small brown box in front with only blue to go on, and a blue box
    beside it that can go: the brown box is out of reach.  Make it two brown
    boxes and nothing within reach can go: soft mix lets both onto blue."""
    every_blue = {EVERYWHERE: (2, BLUE)}
    env = station(rule, [(2, 2, 1, BROWN), (3, 2, 1, BLUE)], every_blue,
                  soft_mix=True)
    assert picks(env) == [False, True]
    assert not env.last_resort()[0] and env.n_feasible()[0] == 0
    env = station(rule, [(2, 2, 1, BROWN), (1, 2, 1, BROWN)], every_blue,
                  soft_mix=True)
    assert picks(env) == [True, True]
    assert env.last_resort()[0] and env.n_feasible()[0] > 0


def test_soft_mix_is_read_by_the_mixed_rules_only():
    """Under `touch` brown on blue stays out, soft mix or not."""
    env = BPPBatch(1, S=(8, 8, 10), nb=1, n_items=2, rot=1, ems=0,
                   n_types=3, types=CLASSES, arm_collision=False,
                   stack_cap=False, type_rule="touch", soft_mix=True)
    env.reset(np.array([[(2, 2, 1, BROWN), (1, 1, 1, 0)]], np.int16))
    env.hmap[0], env.tmap[0], env.smap[0] = 2, BLUE, 1 << BLUE
    env._invalidate()
    assert not env.fallback()[0] and env.n_feasible()[0] == 0


# ------------------------------------------------ touch against column: "on"
def offered_by(rules, *a, at=(0, 0), **kw):
    """{rule: is the same placement offered in the same bin under it}"""
    return {r: offered(bin_with(r, *a, **kw), *at) for r in rules}


def test_only_column_counts_a_foreign_box_the_box_hangs_over():
    """A large brown resting on blue, (c), and hanging over a lower white:
    `mixed_touch` does not see the white, `mixed_column` does."""
    got = offered_by(MIXED_RULES, (3, 2, 1, BROWN),
                     {(0, 2, 0, 2): (4, BLUE), (2, 3, 0, 2): (2, WHITE)})
    assert got == {"mixed_touch": True, "mixed_column": False}


def test_only_column_counts_a_foreign_box_deeper_in_the_column():
    """A blue box with all of its base on blue -- blue that sits on a white
    box: under `mixed_column` the white below rules it out."""
    got = offered_by(MIXED_RULES, (3, 2, 1, BLUE), {(0, 3, 0, 2): (4, BLUE)},
                     below={(0, 3, 0, 2): WHITE})
    assert got == {"mixed_touch": True, "mixed_column": False}


def test_only_column_counts_open_blue_the_box_hangs_over():
    """(d): white resting on blue the small blue no longer fits on (7 + 4
    over the lid), and hanging over blue it still fits on (3 + 4)."""
    got = offered_by(MIXED_RULES, (3, 2, 1, WHITE),
                     {(0, 3, 0, 2): (7, BLUE), (3, 6, 0, 2): (3, BLUE)},
                     at=(1, 0), mixed_small_blue=(3, 2, 4))
    assert got == {"mixed_touch": True, "mixed_column": False}


def test_column_blue_under_another_box_is_closed():
    """Blue buried under white can take no small blue box any more, so (d)
    lets white over it -- and over the white on top, white is at home."""
    got = offered_by(MIXED_RULES, (3, 2, 1, WHITE), {(0, 3, 0, 2): (4, WHITE)},
                     below={(0, 3, 0, 2): BLUE})
    assert got == {"mixed_touch": True, "mixed_column": True}


# ----------------------------------------------------------------- the game
def test_the_game_explains_the_mixed_rule_without_contradicting_itself(rule):
    """A cell you may click is never shown as blocked by the type rule, and
    the board carries what the hover note quotes."""
    from ar2l.viz import game as G
    p, err = G.validate({"bin": [31, 27, 40], "size_lo": [2, 2, 2],
                         "size_hi": [12, 12, 10], "n_items": 30, "nb": 4,
                         "n_pick": 2, "max_l": 256, "rot": 2, "ems": 3,
                         "stability": "com", "min_support": 0.8,
                         "source": "random", "type_rule": rule,
                         "soft_mix": int(rule == "mixed_column"),
                         "arm_collision": 0})
    assert not err, err
    st = G.new_game(3, "none", p, human_pick=True)[1]
    rng = np.random.default_rng(0)
    for _ in range(8):
        b = G.board(st)
        if b["done"]:
            break
        assert b["type_rule"] == rule
        assert b["mixed"]["small_blue"] == list(st["env"].small_blue)
        assert b["mixed"]["soft"] is st["env"].soft_mix \
            is (rule == "mixed_column")
        free, tb = (np.array(b[k], bool) for k in ("free", "typeblock"))
        assert not (free & tb).any()
        r, x, y = np.argwhere(free)[rng.integers(int(free.sum()))]
        assert G.place(st["env"], int(r), int(x), int(y))


def test_the_command_lines_hand_soft_mix_to_the_env():
    from ar2l import evaluate, train
    a = train.get_parser().parse_args(["--name", "t", "--type_rule",
                                       "mixed_column", "--soft_mix", "1",
                                       "--n_env", "2"])
    env = train.make_env(a, seed=0)
    assert env.type_rule == "mixed_column" and env.soft_mix is True
    e = evaluate.get_parser().parse_args(["--soft_mix", "1"])
    assert e.soft_mix == 1


# ------------------------------------------------------- against brute force
def brute(env, b):
    """(R, Lx, Ly) what the mixed rule lets the head box of bin `b` do, read
    from the env's own geometry with the type rule lifted and every condition
    tested cell by cell -- on what the box rests on (`mixed_touch`), or on
    every type each column under it has held (`mixed_column`, from `smap`)."""
    env.type_constraint = False
    env._invalidate(hmap=False)
    try:
        geo, _, odims, _ = (a.copy() for a in env._positions())
    finally:
        env.type_constraint = True
        env._invalidate(hmap=False)
    column, soft = env.type_rule == "mixed_column", env.soft_mix
    H, T, Sm, Lz = env.hmap[b], env.tmap[b], env.smap[b], env.Lz
    sbx, sby, sbz = env.small_blue
    # a small blue box goes where it rests with all of its base on what the
    # normal rules let blue onto -- blue, and near the lid white or brown too
    # -- read by touch or by column; its blue cells open
    open_ = np.zeros_like(H, bool)
    for sx, sy in {(sbx, sby), (sby, sbx)}:
        for x in range(env.Lx - sx + 1):
            for y in range(env.Ly - sy + 1):
                h, tt = H[x:x + sx, y:y + sy], T[x:x + sx, y:y + sy]
                top = h.max()
                if top == 0 or not (h == top).all() or top + sbz > Lz:
                    continue
                lid = top + sbz >= Lz - env.top_gap
                if column:
                    bits = int(np.bitwise_or.reduce(Sm[x:x + sx, y:y + sy],
                                                    None))
                    under = {u for u in range(env.n_types) if bits >> u & 1}
                else:
                    under = set(tt.ravel().tolist())
                if all(u == BLUE or (lid and (BLUE, u) in E.TOP_PAIRS)
                       for u in under):
                    open_[x:x + sx, y:y + sy] |= tt == BLUE
    t = int(env.head_item()[1][b])

    def legal(tier, x, y, sx, sy, sz):
        h, tt = H[x:x + sx, y:y + sy], T[x:x + sx, y:y + sy]
        z = h.max()
        if z == 0:
            return True
        contact = h == z
        near = z + sz >= Lz - env.top_gap
        if column:
            bits = int(np.bitwise_or.reduce(Sm[x:x + sx, y:y + sy], None))
            on = {u for u in range(env.n_types) if bits >> u & 1}
            seen = np.ones_like(contact)
        else:
            on, seen = set(tt[contact].tolist()), contact
        for u in on:
            if u == t and t != BLUE:
                continue
            if t == BLUE and u == BLUE:
                if tier >= 2 or contact.all():
                    continue
                return False
            if tier == 3 and (t, u) in E.SOFT_PAIRS:     # the last resort
                continue
            if (t, u) in E.TOP_PAIRS and near:
                continue
            if t in (WHITE, BROWN) and u == BLUE:
                big = (not soft and t == BROWN
                       and sx * sy >= env.area_frac * sbx * sby - 1e-6)
                closed = not (open_[x:x + sx, y:y + sy] & seen).any()
                if near or big or closed:
                    continue
            if tier == 2 and not soft and (t in (BLUE, WHITE)
                                           or (t, u) == (BROWN, BLUE)):
                continue                                    # fallback
            return False
        return True

    def grid(tier):
        out = np.zeros_like(geo[b])
        for r, (sx, sy, sz) in enumerate(odims[b]):
            for x, y in zip(*np.nonzero(geo[b, r])):
                out[r, x, y] = legal(tier, x, y, sx, sy, sz)
        return out

    one = grid(1)
    if one.any():
        return one, False
    # its own fallback: every type's, or blue's alone under soft mix
    own = t == BLUE or not soft
    hard = grid(2) if own else one
    if not soft or hard.any():
        return hard, own
    # the station is just this box: nothing else within reach can go
    return grid(3), True


def play_against_brute(env, steps=80, seed=7):
    """Random play, every head box's placements checked against `brute` cell
    for cell; returns how often each part of the rule came up."""
    rng = np.random.default_rng(seed)
    seen = dict(fallback=0, cross=0, brown_late=0, brown_blue_late=0,
                brown_on_white=0, top_mix=0)
    for _ in range(steps):
        if env.done.all():
            env.reset_done()
        feas, _, _, under = env._positions()
        # the station is just the head here, so soft mix's last resort is
        # that box's own, as `brute` reads it
        fb = env.fallback() | env.last_resort()
        for b in range(env.n_env):
            if env.done[b]:
                continue
            want, late = brute(env, b)
            assert (feas[b] == want).all(), (b, seen)
            assert bool(fb[b]) == late
            t = int(env.head_item()[1][b])
            seen["fallback"] += late
            # placements that rest on another type, the rule's whole point
            seen["cross"] += int((feas[b] & (under[b] >= 0)
                                  & (under[b] != t)).sum())
            if not late:
                for a, u in E.TOP_PAIRS:
                    if a == t:
                        seen["top_mix"] += int((feas[b] & (under[b] == u))
                                               .sum())
            if t == BROWN:
                seen["brown_late"] += late
                seen["brown_on_white"] += int((feas[b]
                                               & (under[b] == WHITE)).sum())
                if late:
                    seen["brown_blue_late"] += int((feas[b]
                                                    & (under[b] == BLUE)).sum())
        m = env.obs()["l_mask"]
        env.step(np.array([rng.choice(np.nonzero(r)[0]) if r.any() else 0
                           for r in m]))
    return seen


@pytest.mark.parametrize("soft", [False, True], ids=["hard", "soft"])
def test_random_play_matches_brute_force(rule, soft):
    """Cell for cell, under rotation, over whole episodes -- and every rule
    gets exercised, or the comparison proves little."""
    seen = play_against_brute(mixed(rule, 6, seed=7, rot=2, nb=1, n_items=60,
                                    soft_mix=soft))
    assert seen["fallback"] > 5 and seen["cross"] > 50, seen
    assert seen["top_mix"] > 0 and seen["brown_on_white"] == 0, seen


# mostly blue, and browns too small for (c): brown runs out of normal places
# often enough for soft mix's brown on blue to come up
BLUE_HEAVY = [{"p": 6, "lo": [3, 3, 1], "hi": [4, 4, 2]},
              {"p": 1, "lo": [2, 1, 1], "hi": [4, 3, 2]},
              {"p": 3, "lo": [1, 1, 1], "hi": [2, 2, 1]}]


@pytest.mark.parametrize("soft", [False, True], ids=["hard", "soft"])
def test_brown_falls_onto_blue_in_random_play(rule, soft):
    """Brown on blue without (a), (c) or (d) -- its own fallback, or soft
    mix's last resort -- against `brute` too."""
    seen = play_against_brute(mixed(rule, 6, seed=7, rot=2, nb=1, n_items=60,
                                    soft_mix=soft, types=BLUE_HEAVY))
    assert seen["brown_late"] > 0 and seen["brown_blue_late"] > 0, seen
    assert seen["brown_on_white"] == 0, seen


def test_the_last_resort_waits_while_a_box_within_reach_can_go(rule):
    """Soft mix with its last resort against soft mix without it, station by
    station over random play: they agree wherever some box within reach has
    a place by the other rules -- the near-lid mixes included.  (A stuck
    station is rare here -- a blocked box in front ends the bin first -- so
    `test_soft_mix_steps_in_only_while_nothing_else_can_go` builds one.)"""
    env = mixed(rule, 6, seed=3, rot=2, nb=4, n_pick=3, n_items=60,
                soft_mix=True, types=BLUE_HEAVY)
    rng = np.random.default_rng(3)
    compared = 0
    for _ in range(60):
        if env.done.all():
            env.reset_done()
        off = copy.deepcopy(env)
        off._stuck = lambda: np.zeros(off.n_env, bool)   # no last resort
        off._invalidate()
        valid = env.window()[1]
        pick_on, pick_off = env.pick_mask(valid), off.pick_mask(valid)
        f_on, f_off = env._positions()[0], off._positions()[0]
        alive = ~env.done & (env.head < env.length)
        stuck = env._stuck()
        assert (stuck == alive & ~pick_off.any(1)).all()
        for b in np.nonzero(alive & ~stuck)[0]:
            assert (pick_on[b] == pick_off[b]).all(), b
            assert (f_on[b] == f_off[b]).all(), b
            compared += 1
        # stand in for a permuter: hand over a box that can go
        env.permute(np.where(pick_on.any(1), pick_on.argmax(1), 0))
        m = env.obs()["l_mask"]
        env.step(np.array([rng.choice(np.nonzero(r)[0]) if r.any() else 0
                           for r in m]))
    assert compared > 100, compared


def test_soft_mix_steps_in_only_while_nothing_else_can_go(rule):
    """Two small brown boxes and nothing but open blue far from the lid:
    nothing within reach can go, so soft mix puts the first on blue -- and
    its top is brown, which the second may go on by the normal rules, so
    soft mix steps back out."""
    env = station(rule, [(2, 2, 1, BROWN), (2, 2, 1, BROWN), (2, 2, 1, BROWN)],
                  {EVERYWHERE: (2, BLUE)}, soft_mix=True)
    assert env.last_resort()[0] and env._stuck()[0]
    m = env.obs()["l_mask"][0]
    env.step(np.array([int(np.nonzero(m)[0][0])]))
    assert not env.done[0], "the second brown box had somewhere to go"
    assert not env._stuck()[0] and not env.last_resort()[0]
    feas, z, _, under = env._positions()
    assert feas[0].any() and (under[0][feas[0]] == BROWN).all(), \
        "with brown to go on, soft mix still put brown on blue"


# ------------------------------------------------------- the outline step
def outlined(env, cols):
    """Draw the outline of every block of `cols` on `omap`, as `step` would
    for a box of that footprint."""
    for x0, x1, y0, y1 in cols:
        env.omap[0, [x0, x1], y0:y1 + 1] = True
        env.omap[0, x0:x1 + 1, [y0, y1]] = True
    env._invalidate()
    return env


# a white floor two cells up, flat, so the only `ems` candidates are the bin
# corners -- and brown never goes on white
WHITE_FLOOR = {EVERYWHERE: (2, WHITE)}
# a blue box of the brown box's own footprint: no small blue box (3 x 2) fits
# on it, so brown may go on it by (d), and only flush with its edges
SMALL_BLUE = (3, 5, 3, 5)


def ol_station(rule, boxes, cols, **kw):
    kw.setdefault("ems", 3)
    return outlined(station(rule, boxes, cols, **kw), cols)


@pytest.mark.parametrize("soft", [False, True])
def test_the_outline_step_finds_a_place_the_corners_miss(rule, soft):
    cols = {**WHITE_FLOOR, SMALL_BLUE: (2, BLUE)}
    off = ol_station(rule, [(2, 2, 1, BROWN)], cols, soft_mix=soft)
    assert off.n_feasible()[0] == 0 and not off.outline()[0]
    on = ol_station(rule, [(2, 2, 1, BROWN)], cols, soft_mix=soft,
                    outline_resort=True)
    assert on.outline()[0]
    assert np.argwhere(on._positions()[0][0, 0]).tolist() == [[3, 3]]
    # and the episode goes on rather than ending on the brown box
    assert on._head_placeable()[0]


def test_the_outline_step_opens_per_station_or_per_box(rule):
    """A brown box with only the outline place, and a white box behind it
    that can go at a corner: per station the brown box waits, per box it
    goes."""
    cols = {**WHITE_FLOOR, SMALL_BLUE: (2, BLUE)}
    boxes = [(2, 2, 1, BROWN), (2, 2, 1, WHITE)]
    env = ol_station(rule, boxes, cols, soft_mix=True, outline_resort=True)
    assert picks(env) == [False, True] and not env.outline()[0]
    assert env.n_feasible()[0] == 0
    env = ol_station(rule, boxes, cols, soft_mix=True, outline_resort=True,
                     outline_when="box")
    assert picks(env) == [True, True] and env.outline()[0]
    assert np.argwhere(env._positions()[0][0, 0]).tolist() == [[3, 3]]


@pytest.mark.parametrize("soft", [False, True])
def test_the_outline_step_runs_the_rules_in_order(rule, soft):
    """On the outline points the normal rules come first.  A raised blue
    block a small blue box still fits on, whose corners the arm cannot reach
    with a brown box: the brown box goes on it along its edge, by its own
    fallback or soft mix's last resort -- but not while a brown box gives it
    a normal outline place."""
    cols = {**WHITE_FLOOR, (3, 6, 0, 4): (3, BLUE)}
    corners = {(3, 0), (4, 0), (3, 2), (4, 2)}

    def arm(env):
        def moves(b, x, y, sizes, z):
            brown = (sizes[:, 0] == 2) & (sizes[:, 1] == 2)
            hit = np.array([(int(i), int(j)) in corners for i, j in zip(x, y)],
                           bool)
            return np.where(brown & hit, -1, 0).astype(np.int8)
        env.arm_collision, env._arm_moves = True, moves
        env._invalidate()
        return env

    env = arm(ol_station(rule, [(2, 2, 1, BROWN)], cols, soft_mix=soft,
                         outline_resort=True))
    assert np.argwhere(env._positions()[0][0, 0]).tolist() == [[3, 1], [4, 1]]
    assert env.outline()[0]
    with env._outline():
        assert env.last_resort()[0] if soft else env.fallback()[0]
    # brown cells level with the floor: a brown box flush on them, by the
    # normal rules -- the blue waits
    cols[(0, 2, 5, 7)] = (2, BROWN)
    env = arm(ol_station(rule, [(2, 2, 1, BROWN)], cols, soft_mix=soft,
                         outline_resort=True))
    assert np.argwhere(env._positions()[0][0, 0]).tolist() == [[0, 5]]
    with env._outline():
        assert not env.fallback()[0] and not env.last_resort()[0]


def test_the_outline_step_slides_along_an_edge(rule):
    """A box flush beside another, anywhere along its side: the corners
    give only the ends of that side, the outline every cell along it."""
    tall = (0, 8, 0, 2)                       # a white wall along y = 0..2
    cols = {**WHITE_FLOOR, tall: (6, WHITE)}
    env = ol_station(rule, [(2, 2, 1, WHITE)], cols, outline_resort=True)
    # the corners have it -- white on white -- so the step stays shut
    assert not env.outline()[0]
    corners = {tuple(c) for c in np.argwhere(env._positions()[0][0, 0])}
    with env._outline():
        f, _ = env._feas_one(np.array([[2, 2, 1]], np.int16),
                             np.array([WHITE], np.int16), arm=False)
    along = {tuple(c) for c in np.argwhere(f[0])}
    assert corners < along and {(x, 2) for x in range(7)} <= along


def test_the_outline_lattice_follows_the_boxes():
    env = mixed("mixed_touch", n_items=3, ems=0, rot=1)
    env.reset(np.array([[(3, 2, 1, WHITE), (1, 1, 1, WHITE),
                         (1, 1, 1, WHITE)]], np.int16))
    rim = env.omap[0].copy()
    assert rim[0].all() and rim[-1].all() and rim[:, 0].all() \
        and rim[:, -1].all() and not rim[1:-1, 1:-1].any()
    env.obs()
    a = env.candidates(0).index(next(c for c in env.candidates(0)
                                     if c[:2] == [2, 3]))
    env.step(np.array([a]))
    want = rim.copy()
    want[[2, 5], 3:6] = True
    want[2:6, [3, 5]] = True
    assert (env.omap[0] == want).all()
    env.done[:] = True
    env.reset_done()
    assert (env.omap[0] == rim).all()


def test_the_outline_step_is_read_by_the_mixed_rules_only(rule):
    cols = {**WHITE_FLOOR, SMALL_BLUE: (2, BLUE)}
    # under `touch` brown goes on nothing but brown, outline or not
    env = ol_station(rule, [(2, 2, 1, BROWN)], cols, outline_resort=True)
    env.type_rule = "touch"
    env._invalidate()
    assert env.n_feasible()[0] == 0 and not env.outline()[0]
    # every position a candidate already: nothing for it to add
    env = ol_station(rule, [(2, 2, 1, BROWN)], cols, outline_resort=True,
                     ems=0)
    assert not env._outline_on() and env.n_feasible()[0] == 1


def test_the_command_lines_hand_the_outline_step_to_the_env():
    from ar2l import evaluate, train
    a = train.get_parser().parse_args(["--name", "t", "--type_rule",
                                       "mixed_column", "--outline_resort", "1",
                                       "--outline_when", "box",
                                       "--n_env", "2"])
    env = train.make_env(a, seed=0)
    assert env.outline_resort is True and env.outline_when == "box"
    e = evaluate.get_parser().parse_args(["--outline_resort", "1"])
    assert e.outline_resort == 1 and e.outline_when == "station"
    with pytest.raises(ValueError):
        mixed("mixed_touch", outline_when="pallet")


def test_the_game_reads_the_outline_step():
    from ar2l.viz import game as G
    p, err = G.validate({"type_rule": "mixed_column", "outline_resort": 1,
                         "outline_when": "box"})
    assert not err, err
    assert G.stack_kw(p)["outline_resort"] is True
    assert G.stack_kw(p)["outline_when"] == "box"
    assert G.validate({"outline_when": "pallet"})[1]
