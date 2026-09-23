"""Real orders: the CSV loader, and pallets of different lengths in one batch.

The contract is that padding is invisible: a pallet padded out to the batch
width plays exactly as it would alone at its own length, and it ends when its
last box is placed rather than running on into the zeros.
"""
import numpy as np
import pytest

from ar2l.env import BPPBatch
from ar2l.evaluate import metrics, run
from ar2l.orders import (box_divisor, load_instances, load_orders, orders_bin,
                         randomize_order)

S = (30, 24, 40)


@pytest.fixture(autouse=True)
def reference_eval_config(monkeypatch):
    """Pin the CSV settings, so a user's config.yaml cannot move the tests."""
    from ar2l.config import CFG
    for k, v in {"cell_cm": 2.0, "pallet_cm": None, "box_scale": 0,
                 "box_round": "up", "order_random": 0.0}.items():
        monkeypatch.setitem(CFG["eval"], k, v)


def write(tmp_path, text, name="orders.csv"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def env_for(seqs, **kw):
    kw.setdefault("S", S)
    kw.setdefault("nb", 1)
    return BPPBatch(len(seqs), n_items=seqs.shape[1], n_types=3, types=False,
                    size_hi=seqs[..., :3].reshape(-1, 3).max(0), **kw)


def greedy_first(env):
    """The first feasible candidate: deterministic, so two runs can be compared."""
    return env.obs()["l_mask"].argmax(1)


def play(env, seqs):
    env.reset(seqs)
    placed = [[] for _ in range(len(seqs))]
    while not env.done.all():
        before = env.n_packed.copy()
        env.step(greedy_first(env))
        for b in np.nonzero(env.n_packed > before)[0]:
            placed[b].append(tuple(env.packed[b, env.n_packed[b] - 1]))
    return placed


# ------------------------------------------------------------------- loader
def test_csv_cells_order_qty_and_padding(tmp_path):
    p = write(tmp_path, "pallet_id,seq,length_cm,width_cm,height_cm,type,qty\n"
                        "B,1,10,10,10,2,1\n"
                        "A,2,24,14,12,0,2\n"
                        "A,1,16.5,12,9.9,1,\n")
    seqs, ids = load_orders(p, cell_cm=2.0, S=S)
    assert ids == ["B", "A"]                      # first appearance
    assert seqs.shape == (2, 3, 4) and seqs.dtype == np.int16
    # seq orders within a pallet; 16.5 cm and 9.9 cm round *up* to cells
    assert seqs[1].tolist() == [[9, 6, 5, 1], [12, 7, 6, 0], [12, 7, 6, 0]]
    assert seqs[0].tolist() == [[5, 5, 5, 2], [0, 0, 0, 0], [0, 0, 0, 0]]


def test_csv_optional_columns_default(tmp_path):
    p = write(tmp_path, "pallet_id,length_cm,width_cm,height_cm\n"
                        "X,20,20,20\nX,10,10,10\n")
    seqs, _ = load_orders(p, S=S)
    assert seqs[0].tolist() == [[10, 10, 10, 0], [5, 5, 5, 0]]  # file order


def test_csv_exact_multiple_is_not_rounded_up(tmp_path):
    # 0.6 / 0.2 is 2.9999999999999996 and 24 / 0.2 is 119.99999999999999
    p = write(tmp_path, "pallet_id,length_cm,width_cm,height_cm\nX,0.6,0.4,24\n")
    got = load_orders(p, cell_cm=0.2, S=(200, 200, 200))[0][0, 0, :3]
    assert got.tolist() == [3, 2, 120]
    p = write(tmp_path, "pallet_id,length_cm,width_cm,height_cm\nX,24,14,12\n")
    assert load_orders(p, S=S)[0][0, 0, :3].tolist() == [12, 7, 6]


@pytest.mark.parametrize("body,msg", [
    ("pallet_id,length_cm,width_cm\nX,1,1\n", "missing column"),
    ("pallet_id,length_cm,width_cm,height_cm\nX,1,abc,1\n", "not a number"),
    ("pallet_id,length_cm,width_cm,height_cm\nX,1,0,1\n", "positive"),
    ("pallet_id,length_cm,width_cm,height_cm\n,1,1,1\n", "empty pallet_id"),
    ("pallet_id,length_cm,width_cm,height_cm\nX,61,10,10\n", "does not fit"),
    ("pallet_id,length_cm,width_cm,height_cm\nX,10,10,81\n", "does not fit"),
    ("pallet_id,length_cm,width_cm,height_cm\n", "no boxes"),
])
def test_csv_errors(tmp_path, body, msg):
    with pytest.raises(ValueError, match=msg):
        load_orders(write(tmp_path, body), S=S)


def test_csv_rotation_decides_fit(tmp_path):
    # 20 x 60 cm = 10 x 30 cells: only fits the 30 x 24 bin turned round
    p = write(tmp_path, "pallet_id,length_cm,width_cm,height_cm\nX,20,60,10\n")
    assert load_orders(p, S=S, rot=2)[0].shape == (1, 1, 4)
    with pytest.raises(ValueError, match="does not fit"):
        load_orders(p, S=S, rot=1)


def test_load_instances_dispatch(tmp_path):
    p = write(tmp_path, "pallet_id,length_cm,width_cm,height_cm\nX,20,20,20\n")
    a = load_instances(p)
    np.save(tmp_path / "o.npy", a)
    assert (load_instances(str(tmp_path / "o.npy")) == a).all()


def test_pallet_cm_rounds_down_and_defaults_to_env_bin(monkeypatch):
    assert orders_bin([120, 80, 180], 2.0) == (60, 40, 90)
    assert orders_bin([121.9, 80, 180], 2.0) == (60, 40, 90)   # never bigger
    assert orders_bin([0.6, 0.4, 1.0], 0.2) == (3, 2, 5)        # float-exact
    from ar2l.config import CFG
    # whatever the user's config says: a set pallet is used, none is env.bin
    monkeypatch.setitem(CFG["eval"], "pallet_cm", [100, 60, 50])
    assert orders_bin(cell_cm=2.0) == (50, 30, 25)
    monkeypatch.setitem(CFG["eval"], "pallet_cm", None)
    assert orders_bin() == tuple(CFG["env"]["bin"])


def test_interleaved_pallet_ids_are_gathered_in_file_order(tmp_path):
    p = write(tmp_path, "pallet_id,length_cm,width_cm,height_cm\n"
                        "A,10,10,10\nB,20,20,20\nA,12,12,12\nB,8,8,8\n")
    seqs, ids = load_orders(p, S=S)
    assert ids == ["A", "B"]
    assert seqs[:, :, 0].tolist() == [[5, 6], [10, 4]]


def test_too_big_box_points_at_pallet_cm(tmp_path):
    p = write(tmp_path, "pallet_id,length_cm,width_cm,height_cm\nX,62,40,25\n")
    with pytest.raises(ValueError, match="--pallet_cm"):
        load_orders(p, S=S)
    assert load_orders(p, S=orders_bin([120, 80, 180]))[0].shape == (1, 1, 4)


# --------------------------------------------------------------- box_scale
def test_box_scale_divides_every_side_before_gridding(tmp_path):
    p = write(tmp_path, "pallet_id,length_cm,width_cm,height_cm\nX,50,30,28\n")
    side = lambda k: load_orders(p, S=(60, 40, 90), box_scale=k)[0][0, 0, :3].tolist()
    assert side(0) == side(1) == [25, 15, 14]
    assert side(2) == [13, 8, 7]          # 25 x 15 x 14 cm, rounded up to cells
    assert side(5) == [5, 3, 3]           # 10 x 6 x 5.6 cm
    assert side(2.5) == [10, 6, 6]        # 20 x 12 x 11.2 cm


def test_box_scale_lets_a_big_box_fit_and_says_so(tmp_path):
    # the 60 x 48 x 80 cm test bin; the box is 130 cm tall
    p = write(tmp_path, "pallet_id,length_cm,width_cm,height_cm\nX,62,40,130\n")
    with pytest.raises(ValueError, match="cm / 1.5 is"):
        load_orders(p, S=S, box_scale=1.5)       # 86.7 cm tall: still too big
    assert load_orders(p, S=S, box_scale=2)[0].shape == (1, 1, 4)   # 65 cm


def test_box_scale_default_and_range():
    assert box_divisor() == 1.0                  # box_scale 0 is no scaling
    assert box_divisor(3) == 3.0
    for bad in (-1, 0.5):
        with pytest.raises(ValueError, match="box_scale"):
            box_divisor(bad)


# --------------------------------------------------------------- box_round
@pytest.mark.parametrize("cm,up,down,nearest", [
    (0.4, 1, 1, 1),            # never below one cell
    (1.0, 1, 1, 1), (1.1, 2, 1, 1), (1.5, 2, 1, 1), (1.51, 2, 1, 2), (1.9, 2, 1, 2),
    (2.5, 3, 2, 2), (2.51, 3, 2, 3),
    (2.0, 2, 2, 2), (2.1, 3, 2, 2)])
def test_box_round_modes(tmp_path, cm, up, down, nearest):
    p = write(tmp_path, f"pallet_id,length_cm,width_cm,height_cm\nX,{cm},{cm},{cm}\n")
    got = {r: int(load_orders(p, cell_cm=1.0, S=S, box_round=r)[0][0, 0, 0])
           for r in ("up", "down", "nearest")}
    assert got == {"up": up, "down": down, "nearest": nearest}


def test_box_round_is_exact_on_float_boundaries(tmp_path):
    # 0.6 / 0.2 is 2.9999999999999996: down must still say 3, not 2
    p = write(tmp_path, "pallet_id,length_cm,width_cm,height_cm\nX,0.6,0.4,24\n")
    for r in ("up", "down", "nearest"):
        got = load_orders(p, cell_cm=0.2, S=(200, 200, 200), box_round=r)[0][0, 0, :3]
        assert got.tolist() == [3, 2, 120]


def test_box_round_after_box_scale_and_default(tmp_path, monkeypatch):
    from ar2l.config import CFG
    p = write(tmp_path, "pallet_id,length_cm,width_cm,height_cm\nX,30.2,23.2,19.3\n")
    monkeypatch.setitem(CFG["eval"], "box_round", "down")    # the config default
    got = load_orders(p, cell_cm=1.0, S=S, box_scale=4)[0][0, 0, :3]
    assert got.tolist() == [7, 5, 4]         # 7.55 x 5.8 x 4.825 cm, down
    with pytest.raises(ValueError, match="box_round"):
        load_orders(p, S=S, box_round="sideways")


# ------------------------------------------------------------ order_random
def ragged(n=200, width=60, seed=0):
    """Instances of assorted lengths whose boxes are all distinct: the box's
    arrival index is its first side, so a reorder can be read straight off."""
    rng = np.random.default_rng(seed)
    seqs = np.zeros((n, width, 4), np.int16)
    for b, L in enumerate(rng.integers(1, width + 1, n)):
        seqs[b, :L, 0] = np.arange(1, L + 1)
        seqs[b, :L, 1:3] = 3
        seqs[b, :L, 3] = rng.integers(0, 3, L)
    return seqs


def drift(seqs, out):
    """Mean |new position - old position| over real boxes, as a pallet share."""
    real = out[..., 0] > 0
    L = real.sum(-1, keepdims=True)
    moved = np.abs(out[..., 0].astype(int) - 1 - np.arange(seqs.shape[1]))
    return float((moved / np.maximum(L, 1))[real].mean())


def test_order_random_zero_is_the_data_order():
    seqs = ragged()
    out = randomize_order(seqs, 0.0, seed=3)
    assert (out == seqs).all() and out is not seqs


@pytest.mark.parametrize("amount", [0.05, 0.3, 1.0])
def test_order_random_only_reorders_real_boxes(amount):
    seqs = ragged()
    out = randomize_order(seqs, amount, seed=1)
    for a, b in zip(seqs, out):
        L = int((a[:, 0] > 0).sum())
        assert (b[L:] == 0).all()                           # padding stays last
        assert sorted(map(tuple, b[:L])) == sorted(map(tuple, a[:L]))
    assert (seqs == ragged()).all()                         # input untouched


def test_order_random_is_seeded():
    seqs = ragged()
    assert (randomize_order(seqs, 0.4, 7) == randomize_order(seqs, 0.4, 7)).all()
    assert (randomize_order(seqs, 0.4, 7) != randomize_order(seqs, 0.4, 8)).any()


def test_order_random_grows_with_amount_and_one_is_uniform():
    seqs = ragged(400)
    d = [drift(seqs, randomize_order(seqs, r, 0)) for r in (0.02, 0.1, 0.5, 1.0)]
    assert d[0] < d[1] < d[2] < d[3]
    assert d[1] < 0.1                        # 0.1 is local jitter, not a reshuffle
    # a uniform permutation moves a box a third of the pallet on average
    assert abs(d[3] - 1 / 3) < 0.03
    # and every position is equally likely for the first box
    full = np.zeros((2000, 10, 4), np.int16)
    full[:, :, 0] = np.arange(1, 11)
    full[:, :, 1:3] = 3
    first = (randomize_order(full, 1.0, 0)[:, :, 0] == 1).argmax(1)
    assert np.bincount(first, minlength=10).min() > 150      # ~200 each


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_order_random_range(bad):
    with pytest.raises(ValueError, match="0 .data order. to 1"):
        randomize_order(ragged(3), bad)


# ---------------------------------------------------------------------- env
def test_padding_is_invisible():
    rng = np.random.default_rng(0)
    full = np.zeros((4, 50, 4), np.int16)
    full[..., :3] = rng.integers(4, 11, (4, 50, 3))
    full[..., 3] = rng.integers(0, 3, (4, 50))
    lengths = [50, 7, 23, 1]
    padded = full.copy()
    for b, n in enumerate(lengths):
        padded[b, n:] = 0
    together = play(env_for(padded), padded)
    for b, n in enumerate(lengths):
        alone = play(env_for(full[b:b + 1, :n]), full[b:b + 1, :n])[0]
        assert together[b] == alone
        assert len(together[b]) <= n


def test_short_pallet_ends_at_its_last_box():
    seqs = np.zeros((2, 10, 4), np.int16)
    seqs[:, :, :3] = 2                          # tiny boxes: nothing blocks
    seqs[1, 3:] = 0
    env = env_for(seqs, nb=5)
    env.reset(seqs)
    assert env.length.tolist() == [10, 3]
    _, valid = env.window()
    assert valid[1].tolist() == [True, True, True, False, False]
    while not env.done.all():
        env.step(greedy_first(env))
    assert env.n_packed.tolist() == [10, 3]


def test_pick_station_never_offers_padding():
    seqs = np.zeros((1, 6, 4), np.int16)
    seqs[0, :2] = [[4, 4, 4, 0], [4, 4, 4, 1]]
    env = env_for(seqs, nb=5, n_pick=5)
    env.reset(seqs)
    assert env.obs()["b_pick"][0].tolist() == [True, True, False, False, False]


@pytest.mark.parametrize("bad,msg", [
    ([[4, 4, 4, 0], [0, 0, 0, 0], [4, 4, 4, 0]], "after the last box"),
    ([[4, 0, 4, 0]], "zero side"),
    ([[0, 0, 0, 0]], "at least one box"),
])
def test_bad_padding_is_refused(bad, msg):
    seqs = np.array([bad], np.int16)
    env = env_for(np.full((1, 3, 4), 1, np.int16))
    with pytest.raises(ValueError, match=msg):
        env.reset(seqs)


def test_evaluate_all_packed_metric():
    seqs = np.zeros((2, 8, 4), np.int16)
    seqs[:, :, :3] = 2
    seqs[1, 5:] = 0
    u, k = run(seqs, "dbl", nb=1, S=S, device="cpu", n_types=3)
    length = (seqs[..., :3] > 0).all(-1).sum(-1)
    assert k.tolist() == [8, 5]
    assert metrics(u, k, length)["all"] == 100.0
