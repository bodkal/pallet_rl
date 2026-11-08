"""TODO D.2: the reverse KL of Eq. 6 (`kl_reverse`).  It is off by default and
must then leave training exactly as it was."""
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ar2l import train as T
from ar2l.model import NEG


@pytest.fixture(autouse=True)
def no_arm(monkeypatch):
    """The arm filter is the dearest rule and beside the point here."""
    from ar2l.config import CFG
    monkeypatch.setitem(CFG["env"], "arm_collision", 0)


# ----------------------------------------------------------------- reverse KL
def test_eq6_kl_directions():
    g = torch.Generator().manual_seed(0)
    mask = torch.ones(5, 4, dtype=torch.bool)
    mask[:, 3] = False                              # one masked slot
    def lsm(x):
        return F.log_softmax(x + torch.where(mask, 0.0, NEG), -1)
    lm, lq = lsm(torch.randn(5, 4, generator=g)), lsm(torch.randn(5, 4, generator=g))
    pm, pq = lm.exp()[:, :3], lq.exp()[:, :3]
    fwd = (pm * (pm.log() - pq.log())).sum(-1)
    rev = (pq * (pq.log() - pm.log())).sum(-1)
    assert torch.allclose(T.eq6_kl(lm, lq), fwd, atol=1e-5)
    assert torch.allclose(T.eq6_kl(lm, lq, reverse=True), rev, atol=1e-5)
    assert not torch.allclose(fwd, rev)
    assert torch.allclose(T.eq6_kl(lm, lm, True), torch.zeros(5), atol=1e-6)
    # the gradient reaches the mixture in both directions, and is finite
    for r in (False, True):
        x = torch.randn(5, 4, generator=g, requires_grad=True)
        T.eq6_kl(lsm(x), lq, r).sum().backward()
        assert torch.isfinite(x.grad).all() and x.grad[:, :3].abs().sum() > 0


# ------------------------------------------------------------- end to end
def test_train_runs_with_every_option(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for algo, extra in (("exact", ["--kl_reverse", "1", "--nb", "3"]),):
        name = f"t_{algo}"
        T.main(["--name", name, "--device", "cpu", "--algo", algo, "--data", "",
                "--nb", "3", "--n_pick", "3", "--bin", "10x10x10",
                "--size_lo", "2", "--size_hi", "5x5x5", "--n_items", "30",
                "--iters", "2", "--n_env", "4", "--T", "4",
                "--eval_every", "1", "--eval_inst", "4", "--save_every", "1",
                "--log_every", "1", "--progress", "off", "--width", "16",
                *extra])
        out = tmp_path / "runs" / name
        a = json.load(open(out / "args.json"))
        log = [json.loads(l) for l in open(out / "log.jsonl")]
        assert log[-1]["it"] == 2
        assert a["kl_reverse"] == 1 and np.isfinite(log[-1]["mix_aux"])
