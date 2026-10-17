"""Bigger box on ties: the model's box picker, with near-ties to the biggest.

Boxes whose probability is at most `tol` below the picker's top choice tie
with it, and the biggest of them by x * y + z goes; an equal size goes to the
likelier box, and a box the picker gave nothing is never taken.
"""
import numpy as np

from ar2l.env import BPPBatch
from ar2l.viz import agents as A

SMALL, WHITE, BLUE = [6, 5, 4, 2], [12, 7, 6, 1], [12, 7, 7, 0]


def env(seq):
    e = BPPBatch(1, S=[31, 27, 40], nb=3, n_items=len(seq), n_types=3,
                 types=False, arm_collision=False)
    e.reset(np.asarray(seq, np.int16)[None])
    return e


def pick(seq, probs, tol=0.01):
    fixed = lambda e: (np.argmax(probs)[None], np.asarray([probs], float))
    idx, p = A.prefer_big(fixed, tol)(env(seq))
    assert p[0].tolist() == probs          # the picker's scores are kept
    return int(idx[0])


def test_a_tie_goes_to_the_bigger_box():
    # 20% ties with 19% and 21%: blue (12 * 7 + 7 = 91) beats white (90)
    assert pick([WHITE, BLUE, SMALL], [0.41, 0.40, 0.19]) == 1
    assert pick([SMALL, WHITE, BLUE], [0.20, 0.21, 0.59]) == 2
    assert pick([BLUE, SMALL, WHITE], [0.20, 0.60, 0.20]) == 1


def test_footprint_counts_before_height():
    # 12 x 7 x 6 (90) against 6 x 5 x 4 (34) at 20% / 21%
    assert pick([SMALL, WHITE, BLUE], [0.21, 0.20, 0.59]) == 2
    assert pick([SMALL, WHITE, BLUE], [0.395, 0.40, 0.205]) == 1


def test_no_tie_and_threshold():
    assert pick([WHITE, BLUE, SMALL], [0.43, 0.40, 0.17]) == 0
    assert pick([WHITE, BLUE, SMALL], [0.43, 0.40, 0.17], tol=0.05) == 1
    assert pick([WHITE, BLUE, SMALL], [0.41, 0.40, 0.19], tol=0.0) == 0


def test_never_an_unscored_box():
    assert pick([SMALL, BLUE, BLUE], [1.0, 0.0, 0.0], tol=1.0) == 0
