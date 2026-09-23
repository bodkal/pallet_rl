"""Training on real pallets: `--data`, the hold-out split, and resume safety.

Without `--data` nothing may change -- the random generator is the default and
every earlier run depends on it drawing exactly what it drew before.
"""
import json
import os

import numpy as np
import pytest

from ar2l import train as T
from ar2l.env import BPPBatch

S = (30, 24, 40)


@pytest.fixture(autouse=True)
def reference_eval_config(monkeypatch):
    """Pin the CSV settings, so a user's config.yaml cannot move the tests."""
    from ar2l.config import CFG
    for k, v in {"cell_cm": 2.0, "pallet_cm": None, "box_scale": 0,
                 "box_round": "up", "order_random": 0.0}.items():
        monkeypatch.setitem(CFG["eval"], k, v)


def pool(n=12, width=9, seed=0):
    rng = np.random.default_rng(seed)
    p = np.zeros((n, width, 4), np.int16)
    for b, L in enumerate(rng.integers(1, width + 1, n)):
        p[b, :L, :3] = rng.integers(3, 9, (L, 3))
        p[b, :L, 3] = rng.integers(0, 3, L)
    return p


def pool_env(p, n_env=16, **kw):
    return BPPBatch(n_env, S=S, nb=3, n_pick=3, n_types=3, types=False,
                    size_hi=p[..., :3].reshape(-1, 3).max(0), pool=p, **kw)


def is_pool_row(row, L, p):
    return any((row[:L] == q[:L]).all() and (q[L:] == 0).all() for q in p)


# ---------------------------------------------------------------------- env
def test_generator_is_untouched_without_a_pool():
    a = BPPBatch(8, S=S, seed=3)
    b = BPPBatch(8, S=S, seed=3, pool=None)
    assert (a.seq == b.seq).all() and (a.length == a.n_items).all()


def test_every_episode_is_a_pool_pallet():
    p = pool()
    env = pool_env(p, seed=1)
    assert env.n_items == p.shape[1]
    seen = set()
    for _ in range(40):                       # play through many resets
        for b in range(env.n_env):
            L = int(env.length[b])
            assert is_pool_row(env.seq[b], L, p)
            seen.add(env.seq[b, :L].tobytes())
        m = env.obs()["l_mask"]
        env.step(m.argmax(1))
        env.reset_done()
    assert len(seen) == len(p)                # all pallets get drawn


def test_pool_order_random_keeps_the_boxes():
    p = pool()
    env = pool_env(p, seed=2, pool_order_random=1.0)
    reordered = False
    for b in range(env.n_env):
        L = int(env.length[b])
        row = env.seq[b, :L]
        match = [q for q in p if int((q[:, 0] > 0).sum()) == L
                 and sorted(map(tuple, q[:L])) == sorted(map(tuple, row))]
        assert match and (env.seq[b, L:] == 0).all()
        reordered |= not any((q[:L] == row).all() for q in match)
    assert reordered


def test_pool_is_seeded():
    p = pool()
    a, b = pool_env(p, seed=5, pool_order_random=0.5), pool_env(p, seed=5, pool_order_random=0.5)
    assert (a.seq == b.seq).all()


# ---------------------------------------------------------------- load_data
def args_for(tmp_path, data, **kw):
    a = T.get_parser().parse_args(["--name", "t", "--data", str(data),
                                   "--device", "cpu"])
    a.data_sha1 = None
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def write_csv(tmp_path, n=10):
    lines = ["pallet_id,length_cm,width_cm,height_cm,type"]
    for i in range(n):
        lines += [f"P{i},{20 + i},16,12,{i % 3}"] * (i % 4 + 1)
    p = tmp_path / "orders.csv"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_holdout_split_is_disjoint_seeded_and_written(tmp_path):
    csv = write_csv(tmp_path)
    out = tmp_path / "run"; out.mkdir()
    a = args_for(tmp_path, csv, holdout=0.3, pallet_cm=None, bin=S)
    trn, held = T.load_data(a, str(out))
    assert (a.n_train, a.n_holdout) == (7, 3) and len(trn) == 7 and len(held) == 3
    tids = (out / "train_ids.txt").read_text().split()
    hids = (out / "holdout_ids.txt").read_text().split()
    assert not set(tids) & set(hids) and len(tids) + len(hids) == 10
    b = args_for(tmp_path, csv, holdout=0.3, pallet_cm=None, bin=S)
    out2 = tmp_path / "run2"; out2.mkdir()
    T.load_data(b, str(out2))
    assert (out2 / "holdout_ids.txt").read_text().split() == hids
    assert a.data_sha1 and a.data_sha1 == b.data_sha1


def test_pallet_cm_sets_the_bin(tmp_path):
    out = tmp_path / "run"; out.mkdir()
    a = args_for(tmp_path, write_csv(tmp_path), pallet_cm=[120, 80, 180],
                 cell_cm=2.0)
    T.load_data(a, str(out))
    assert tuple(a.bin) == (60, 40, 90)


def test_no_data_is_the_generator(tmp_path):
    a = args_for(tmp_path, "")
    a.data = None
    assert T.load_data(a, str(tmp_path)) == (None, None)
    assert a.data_sha1 is None


@pytest.mark.parametrize("h", [-0.1, 1.0])
def test_holdout_range(tmp_path, h):
    with pytest.raises(ValueError, match="holdout"):
        T.load_data(args_for(tmp_path, write_csv(tmp_path), holdout=h,
                             pallet_cm=None, bin=S), str(tmp_path))


# ------------------------------------------------------------------ training
def run_train(tmp_path, monkeypatch, data, name="t", iters=2, extra=()):
    monkeypatch.chdir(tmp_path)
    T.main(["--name", name, "--data", str(data), "--device", "cpu",
            "--algo", "select", "--nb", "3", "--n_pick", "3", "--bin", "30x24x40",
            "--iters", str(iters), "--n_env", "4", "--T", "4",
            "--eval_every", "1", "--save_every", "1", "--log_every", "1",
            "--progress", "off", "--holdout", "0.3", *extra])
    return tmp_path / "runs" / name


def test_trains_on_csv_and_scores_the_holdout(tmp_path, monkeypatch):
    run = run_train(tmp_path, monkeypatch, write_csv(tmp_path))
    args = json.load(open(run / "args.json"))
    assert args["n_train"] == 7 and args["n_holdout"] == 3 and args["data_sha1"]
    log = [json.loads(l) for l in open(run / "log.jsonl")]
    assert log[-1]["it"] == 2 and log[-1]["sel_util"] > 0
    assert (run / "best.pt").exists()


def test_resume_refuses_a_changed_data_file(tmp_path, monkeypatch):
    csv = write_csv(tmp_path)
    run_train(tmp_path, monkeypatch, csv, iters=1)
    with open(csv, "a") as f:
        f.write("P99,30,20,10,0\n")
    with pytest.raises(SystemExit, match="refusing to resume"):
        run_train(tmp_path, monkeypatch, csv, iters=2, extra=("--resume",))
