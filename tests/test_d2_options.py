"""TODO D.2: the turned-down candidates shown to the packer (`show_infeasible`),
the reverse KL of Eq. 6 (`kl_reverse`) and the pointer saturation penalty
(`sat_coef_sel` / `sat_coef_pack`).  Every one is off by default and must then
leave training exactly as it was."""
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ar2l import train as T
from ar2l.env import BPPBatch
from ar2l.evaluate import load_nets, run
from ar2l.model import NEG, PackNet, PermNet, Pointer
from ar2l.ppo import PPO, to_torch


@pytest.fixture(autouse=True)
def no_arm(monkeypatch):
    """The arm filter is the dearest rule and beside the point here."""
    from ar2l.config import CFG
    monkeypatch.setitem(CFG["env"], "arm_collision", 0)


def pair(n=8, seed=3, **kw):
    a = BPPBatch(n, seed=seed, **kw)
    b = BPPBatch(n, seed=seed, show_infeasible=True, **kw)
    a.reset(); b.reset()
    return a, b


# ------------------------------------------------------------ show_infeasible
def test_off_by_default_and_no_new_key():
    env = BPPBatch(4, seed=0)
    env.reset()
    assert not env.show_infeasible and "l_seen" not in env.obs()


def test_legal_candidates_come_first_and_unchanged():
    a, b = pair()
    grew = False
    for _ in range(40):
        oa, ob = a.obs(), b.obs()
        k = oa["l_mask"].shape[1]
        # the legal prefix is the list the env gives without the option
        assert np.array_equal(ob["l_mask"][:, :k], oa["l_mask"])
        assert not ob["l_mask"][:, k:].any()
        m = oa["l_mask"]
        assert np.array_equal(ob["l"][:, :k][m], oa["l"][m])
        assert np.array_equal(ob["l_type"][:, :k][m], oa["l_type"][m])
        # every legal one is seen; padding past the seen ones is zero
        assert (ob["l_seen"] | ~ob["l_mask"]).all()
        assert (ob["l"][~ob["l_seen"]] == 0).all()
        grew |= bool((ob["l_seen"].sum(1) > ob["l_mask"].sum(1)).any())
        act = m.argmax(1)
        a.step(act); b.step(act)
        if a.done.all():
            break
    assert grew, "no turned-down candidate was ever listed"
    # the same actions placed the same boxes
    assert np.array_equal(a.hmap, b.hmap)


def test_turned_down_candidates_are_really_illegal():
    """A seen-but-not-legal slot must not be a placement the rules allow."""
    _, b = pair(n=4, seed=7)
    for _ in range(15):
        o = b.obs()
        feas = b._positions()[0]
        for e in range(b.n_env):
            for i in np.flatnonzero(o["l_seen"][e] & ~o["l_mask"][e]):
                x, y, r = b._lxy[e, i]
                assert not feas[e, r, x, y]
        b.step(o["l_mask"].argmax(1))
        if b.done.all():
            break


def test_flagged_packer_only_picks_legal_and_needs_l_seen():
    _, b = pair(n=6)
    net = PackNet(feas_flag=True)
    assert net.enc.embed[2].in_features == 7 + net.enc.type_embed
    for _ in range(10):
        o = b.obs()
        lg = net.logits(to_torch(o, "cpu"))
        assert (lg[torch.as_tensor(~o["l_mask"])] < NEG / 2).all()
        b.step(o["l_mask"].argmax(1))
    o = to_torch(b.obs(), "cpu")
    o.pop("l_seen")
    with pytest.raises(ValueError, match="show_infeasible"):
        net.logits(o)


def test_unflagged_packer_reads_a_flagged_env_as_before():
    """A net trained without the flag ignores the extra key."""
    a, b = pair(n=4)
    net = PackNet()
    oa, ob = to_torch(a.obs(), "cpu"), to_torch(b.obs(), "cpu")
    k = oa["l_mask"].shape[1]
    # only the legal list reaches it, padded further in the flagged env
    la, lb = net.logits(oa), net.logits(ob)
    m = oa["l_mask"]
    assert torch.allclose(la[m], lb[:, :k][m], atol=1e-5)


def test_checkpoint_rebuilds_the_flag_and_eval_turns_it_on(tmp_path):
    kw = dict(d=32, n_head=1, n_layer=1)
    ck = {"pack": PackNet(**kw, feas_flag=True).state_dict(),
          "args": {"width": 32, "heads": 1, "layers": 1, "show_infeasible": 1}}
    torch.save(ck, tmp_path / "f.pt")
    net = load_nets(tmp_path / "f.pt", "cpu", ("pack",))["pack"]
    assert net.feas_flag
    seqs = BPPBatch(3, seed=0)
    seqs.reset()
    u, k = run(seqs.seq.copy(), net, nb=1, device="cpu", batch=3)
    assert (k > 0).all()
    # an old checkpoint has no such key and is rebuilt without the flag
    ck["args"].pop("show_infeasible")
    ck["pack"] = PackNet(**kw).state_dict()
    torch.save(ck, tmp_path / "o.pt")
    assert not load_nets(tmp_path / "o.pt", "cpu", ("pack",))["pack"].feas_flag


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


# ------------------------------------------------------- saturation penalty
def test_penalty_zero_inside_the_tanh_range():
    p = Pointer(d=4, c_temp=10.0, norm=2.0)        # |y| <= 2 < SAT
    p(torch.randn(3, 4), torch.randn(3, 5, 4), torch.ones(3, 5, dtype=torch.bool))
    assert float(p.pen.detach()) == 0.0


def test_penalty_counts_live_candidates_only_and_has_a_gradient():
    torch.manual_seed(0)
    p = Pointer(d=8, c_temp=10.0, norm=0.0, center=2)
    with torch.no_grad():
        p.q.weight.mul_(30); p.k.weight.mul_(30)
    mask = torch.ones(4, 6, dtype=torch.bool)
    mask[:, 4:] = False
    xbar, nodes = torch.randn(4, 8), torch.randn(4, 6, 8)
    p(xbar, nodes, mask)
    assert float(p.pen.detach()) > 0
    p.pen.backward()
    assert p.q.weight.grad.abs().sum() > 0
    # what a masked slot scores changes nothing
    nodes2 = nodes.clone(); nodes2[:, 4:] *= 100
    pen = float(p.pen.detach())
    p(xbar, nodes2, mask)
    assert abs(float(p.pen.detach()) - pen) < 1e-4


def test_ppo_penalty_pulls_scores_back_and_is_off_by_default():
    torch.manual_seed(1)
    env = BPPBatch(8, seed=1, nb=3, n_pick=3)
    env.reset()
    o = to_torch(env.obs_cb(), "cpu")
    def fresh():
        torch.manual_seed(2)
        net = PermNet(d=16, n_head=1, n_layer=1, ptr_norm=0.0, ptr_center=2)
        with torch.no_grad():
            net.ptr.q.weight.mul_(40); net.ptr.k.weight.mul_(40)
        return net
    n = o["b"].shape[0]
    act = torch.zeros(n, dtype=torch.long)
    zero = torch.zeros(n)
    def train(coef):
        net = fresh()
        ppo = PPO(net, lr=1e-2, epochs=20, minibatches=1, sat_coef=coef)
        with torch.no_grad():
            lp = F.log_softmax(net.logits(o), -1)[:, 0]
        s = ppo.update(o, act, lp, zero, zero)
        net.logits(o)
        return float(net.ptr.pen.detach()), s
    off, s = train(0.0)
    on, _ = train(1.0)
    assert "pen" in s and PPO(fresh()).sat_coef == 0.0
    assert on < off * 0.5, (on, off)


# ------------------------------------------------------------- end to end
def test_train_runs_with_every_option(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for algo, extra in (("select", ["--show_infeasible", "1",
                                    "--sat_coef_sel", "0.1",
                                    "--sat_coef_pack", "0.1"]),
                        ("exact", ["--kl_reverse", "1", "--nb", "3"])):
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
        if algo == "select":
            assert a["show_infeasible"] == 1 and a["sat_coef_pack"] == 0.1
            assert log[-1]["pack_pen"] >= 0 and log[-1]["mix_pen"] >= 0
            # the checkpoint plays back with the flag on
            net = load_nets(out / "last.pt", "cpu", ("pack",))["pack"]
            assert net.feas_flag
        else:
            assert a["kl_reverse"] == 1 and np.isfinite(log[-1]["mix_aux"])
