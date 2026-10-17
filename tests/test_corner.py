"""Corner first: a blue or white box is held to x = 0, turned.

While the turned box has a legal floor placement somewhere along x = 0, the
packer is offered only its legal turned placements at x = 0 -- every y, any
height; with none on the floor it is offered everything.  A brown box, and every other spot, are left alone.  A
human placing through the game's `place` is never narrowed.

Every env below pins its rules rather than reading `config.yaml`, so editing
the shipped values cannot move these tests.
"""
import numpy as np

from ar2l.env import BPPBatch
from ar2l.viz import game as G

BLUE, WHITE, BROWN = [12, 7, 7, 0], [12, 7, 6, 1], [6, 5, 4, 2]


def env(seq, corner_first=True, rot=2, S=(31, 27, 40)):
    e = BPPBatch(1, S=list(S), nb=3, n_items=len(seq), n_types=3,
                 types=False, rot=rot, arm_collision=False, ems=0,
                 max_l=2 * S[0] * S[1],  # every position, none cut off
                 corner_first=corner_first)
    e.reset(np.asarray(seq, np.int16)[None])
    return e


def spots(e):
    """{(x, y, z, r)} of the packer's candidate list."""
    k = int(e.obs()["l_mask"][0].sum())
    return {(c[0], c[1], c[2], int(r)) for c, r in
            zip(e.candidates(0, k), e._lxy[0, :k, 2])}


def edge(c, r):
    return {(x, y) for x, y, z, rr in c if x == 0 and z == 0 and rr == r}


def test_blue_and_white_get_only_turned_x0():
    for box in (BLUE, WHITE):
        on, off = spots(env([box, BROWN])), spots(env([box, BROWN], False))
        assert on == {c for c in off if c[0] == 0 and c[3] == 1}
        assert len(off) > len(on) > 0


def test_turned_x0_above_the_floor_counts_too():
    # a turned blue at (0, 0) leaves the floor at x = 0 open from y = 12 on;
    # the next blue then also gets the turned spot on top of the first
    e = env([BLUE, BLUE, BROWN])
    assert G.place(e, 1, 0, 0)
    on = spots(e)
    assert on and all(c[0] == 0 and c[3] == 1 for c in on)
    assert {c[2] for c in on} == {0, 7}


def test_unturned_stays_where_turned_cannot_go():
    # 11 cells wide in y: turned (7 x 12) fits nowhere, unturned (12 x 7)
    # at (0, y) for y <= 4
    on = spots(env([BLUE, BROWN], S=(31, 11, 40)))
    assert not edge(on, 1)
    assert edge(on, 0) == {(0, y) for y in range(5)}


def test_brown_and_off_leave_the_list_alone():
    assert edge(spots(env([BROWN, BLUE])), 0)
    assert edge(spots(env([BLUE, BROWN], rot=1)), 0)
    assert not env([BLUE, BROWN], rot=1).corner().any()


def test_no_turned_floor_spot_frees_the_box():
    # three unturned blues at y = 0, 7, 14 fill x = 0 up to y = 21: turned
    # (12 wide in y) no longer fits on the floor there
    e = env([BLUE, BLUE, BLUE, WHITE], corner_first=False)
    assert all(G.place(e, 0, 0, y) for y in (0, 7, 14))
    e.corner_first = True
    e._invalidate(hmap=False)
    assert not e.corner()[0]
    assert {c for c in spots(e) if c[0]}


def test_a_human_places_freely():
    e = env([BLUE, BROWN])
    assert G.place(e, 0, 0, 3)                         # unturned at x = 0
    assert (e.packed[0, 0] * e.scale).round().tolist() == [0, 3, 0, 12, 7, 7]
    assert e.corner_first          # switched back on after the placement
