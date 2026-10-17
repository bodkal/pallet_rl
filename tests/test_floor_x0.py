"""Floor only at x = 0: the model's floor placement, kept to the x = 0 edge.

A blue or white box the policy would put on the floor at x > 0 goes to the
likeliest placement left once every floor placement with x > 0 is dropped --
while the box has any legal placement at x = 0.  Brown boxes, a choice at
x = 0 or off the floor, and a box with nothing at x = 0 are left alone.
"""
import numpy as np

from ar2l.env import BPPBatch
from ar2l.viz import agents as A
from ar2l.viz import game as G

BLUE, WHITE, BROWN = [12, 7, 7, 0], [12, 7, 6, 1], [6, 5, 4, 2]


def env(seq, first=None):
    e = BPPBatch(1, S=[31, 27, 40], nb=3, n_items=len(seq), n_types=3,
                 types=False, arm_collision=False)
    e.reset(np.asarray(seq, np.int16)[None])
    for r, x, y in first or []:
        assert G.place(e, r, x, y)
    return e


def likes(want):
    """A policy whose favourite is the first candidate `want(x, y, z)` holds
    (0.6), the rest sharing what is left."""
    def fn(e):
        lm = e.obs()["l_mask"][0]
        x, y, z = e._lxy[0, :, 0], e._lxy[0, :, 1], e._lz[0]
        w = lm * 0.4 / lm.sum()
        w[np.flatnonzero(lm & want(x, y, z))[0]] += 0.6
        return w.argmax()[None], w[None]
    return fn


def go(e, want):
    act, p = A.floor_x0(likes(want))(e)
    e.obs()
    c = e.candidates(0)[int(act[0])]
    return c[0], c[2], p


FLOOR_FAR = lambda x, y, z: (z == 0) & (x > 0)


def test_far_floor_goes_to_x0_or_up():
    for box in (BLUE, WHITE):
        x, z, _ = go(env([box, BROWN]), FLOOR_FAR)
        assert x == 0 or z > 0


def test_brown_is_left_alone():
    x, z, _ = go(env([BROWN, BLUE]), FLOOR_FAR)
    assert x > 0 and z == 0


def test_choice_at_x0_or_up_is_left_alone():
    x, z, _ = go(env([BLUE, BROWN]), lambda x, y, z: (x == 0) & (z == 0))
    assert (x, z) == (0, 0)
    e = env([BLUE, BLUE, BROWN], first=[(0, 10, 10)])
    x, z, _ = go(e, lambda x, y, z: z > 0)
    assert z > 0


def test_nothing_at_x0_frees_the_floor():
    # three unturned blues fill x = 0 up to y = 21, the 6 cells left are too
    # narrow for any blue, and a 7-cell lid leaves no room on top of them
    e = BPPBatch(1, S=[31, 27, 7], nb=3, n_items=6, n_types=3, types=False,
                 arm_collision=False)
    e.reset(np.asarray([BLUE] * 6, np.int16)[None])
    assert all(G.place(e, 0, 0, y) for y in (0, 7, 14))
    e.obs()
    assert not (e._lmask[0] & (e._lxy[0, :, 0] == 0)).any()
    x, z, _ = go(e, FLOOR_FAR)
    assert x > 0 and z == 0


def test_label():
    pol = lambda e: (np.zeros(1, int), np.ones((1, 1)))
    _, lab = A.stack_rule(pol, "m", {"floor_x0": 1})
    assert lab == "m, floor only at x = 0"
    _, lab = A.stack_rule(pol, "m", {"floor_x0": 1, "prefer_stack": 1,
                                    "prefer_stack_pct": 30})
    assert lab == "m, stacks within 30 points, floor only at x = 0"
