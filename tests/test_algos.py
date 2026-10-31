"""The robust value targets and the pieces of the training loop."""
import torch

from ar2l.model import PackNet, PermNet, sample
from ar2l.ppo import gae, inf_tv_dual, sup_tv_dual
from ar2l.train import flat_obs, move_to_front


def test_sup_dual_collapses_to_the_mixture_mean_at_rho_zero():
    # values are expected remaining utilisations, so they live in [0, 1];
    # the duals project onto that range, which is what anchors them
    g = torch.Generator().manual_seed(0)
    vo = torch.rand(256, generator=g)
    vw = (vo - 0.2 * torch.rand(256, generator=g)).clamp(0.0, 1.0)
    for alpha in (0.3, 0.5, 1.0):
        got = sup_tv_dual(vo, vw, alpha, 0.0, vo).mean()
        want = ((vo + alpha * vw) / (1 + alpha)).mean()
        assert torch.allclose(got, want, atol=2e-3), (alpha, got, want)


def test_duals_move_in_the_right_direction():
    g = torch.Generator().manual_seed(1)
    vo = torch.rand(256, generator=g)
    vw = (vo - 0.2 * torch.rand(256, generator=g)).clamp(0.0, 1.0)
    base = ((vo + vw) / 2).mean()
    for rho in (0.05, 0.1, 0.3):
        assert sup_tv_dual(vo, vw, 1.0, rho, vo).mean() >= base - 1e-4
        assert inf_tv_dual(vo, rho, vo).mean() <= vo.mean() + 1e-4
    # a wider ball can only be more optimistic / more pessimistic
    s = [float(sup_tv_dual(vo, vw, 1.0, r, vo).mean()) for r in (0.05, 0.2, 0.5)]
    i = [float(inf_tv_dual(vo, r, vo).mean()) for r in (0.05, 0.2, 0.5)]
    assert s[0] <= s[1] + 1e-4 <= s[2] + 1e-4
    assert i[0] >= i[1] - 1e-4 >= i[2] - 1e-4


def test_duals_stay_inside_the_return_range():
    """A runaway critic must not be able to inflate the target without bound."""
    g = torch.Generator().manual_seed(2)
    runaway = torch.rand(256, generator=g) * 4000
    for rho in (0.1, 0.4):
        t = sup_tv_dual(runaway, runaway, 1.0, rho, runaway)
        assert float(t.max()) <= 1.0 + 1e-5 and float(t.min()) >= -1e-5
        r = inf_tv_dual(runaway, rho, runaway)
        assert float(r.max()) <= 1.0 + 1e-5 and float(r.min()) >= -1e-5


def test_gae_with_gamma_one_sums_the_rewards():
    T, N = 6, 4
    rew = torch.ones(T, N) * 0.1
    val = torch.zeros(T, N)
    done = torch.zeros(T, N); done[-1] = 1
    adv, ret = gae(rew, val, done, torch.zeros(N), gamma=1.0, lam=1.0)
    assert torch.allclose(ret[0], torch.full((N,), 0.6), atol=1e-5)


def test_move_to_front_matches_the_environment():
    b = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    o = {"b": b, "b_mask": torch.ones(2, 5, dtype=torch.bool),
         "b_type": torch.arange(10).reshape(2, 5),
         "c": torch.randn(2, 4, 6)}
    idx = torch.tensor([3, 0])
    out = move_to_front(o, idx)
    nb, nm, nt = out["b"], out["b_mask"], out["b_type"]
    assert torch.equal(nb[0, 0], b[0, 3]) and torch.equal(nb[1, 0], b[1, 0])
    assert torch.equal(nb[0, 1:], torch.cat([b[0, :3], b[0, 4:]]))
    assert nm.all()
    # the type has to travel with the box it belongs to, not stay in its slot
    assert nt[0].tolist() == [3, 0, 1, 2, 4] and nt[1].tolist() == [5, 6, 7, 8, 9]
    assert torch.equal(out["c"], o["c"]), "a permutation touched C_t"


def test_flat_obs_pads_ragged_rollouts():
    a = {"l": torch.randn(4, 3, 6), "l_mask": torch.ones(4, 3, dtype=torch.bool)}
    b = {"l": torch.randn(4, 7, 6), "l_mask": torch.ones(4, 7, dtype=torch.bool)}
    o = flat_obs([a, b])
    assert o["l"].shape == (8, 7, 6)
    assert torch.equal(o["l"][:4, :3], a["l"]) and (o["l"][:4, 3:] == 0).all()
    assert o["l_mask"][:4, :3].all() and not o["l_mask"][:4, 3:].any()


def test_networks_only_ever_choose_unmasked_nodes():
    torch.manual_seed(0)
    n, nl, nb = 6, 9, 4
    o = {"c": torch.randn(n, 5, 6), "c_mask": torch.ones(n, 5, dtype=torch.bool),
         "c_type": torch.randint(0, 4, (n, 5)),
         "b": torch.randn(n, nb, 3), "b_mask": torch.ones(n, nb, dtype=torch.bool),
         "b_type": torch.randint(0, 3, (n, nb)),
         "l": torch.randn(n, nl, 6),
         "l_type": torch.randint(0, 4, (n, nl)),
         "l_mask": torch.rand(n, nl) > 0.5}
    o["l_mask"][:, 0] = True
    o["c_mask"][:, 3:] = False
    lg, v = PackNet()(o)
    assert torch.isfinite(lg).all() and torch.isfinite(v).all()
    for _ in range(20):
        a, _, _ = sample(lg)
        assert o["l_mask"][torch.arange(n), a].all()
    lg2, _ = PermNet()(o)
    assert lg2.shape == (n, nb) and torch.isfinite(lg2).all()


def test_pointer_norm_zero_is_eq28_as_written():
    torch.manual_seed(0)
    from ar2l.model import Pointer
    p = Pointer(d=16, c_temp=10.0, norm=0.0)
    xbar, nodes = torch.randn(3, 16), torch.randn(3, 5, 16)
    mask = torch.ones(3, 5, dtype=torch.bool)
    y = torch.einsum("bd,bnd->bn", p.q(xbar), p.k(nodes)) / 4.0
    assert torch.allclose(p(xbar, nodes, mask), 10.0 * torch.tanh(y))


def test_pointer_norm_keeps_a_gradient_where_eq28_saturates():
    """The state the 256-wide selector died in: q and k dominated by a shared
    direction, so every candidate scores far past tanh's range.  Unbounded,
    they all get logit +c and no gradient; bounded, the policy can still move."""
    torch.manual_seed(0)
    from ar2l.model import Pointer
    xbar, nodes = torch.randn(4, 16), torch.randn(4, 6, 16)
    mask = torch.ones(4, 6, dtype=torch.bool)
    for norm, dead in ((0.0, True), (3.0, False)):
        p = Pointer(d=16, c_temp=10.0, norm=norm)
        with torch.no_grad():
            for lin in (p.q, p.k):
                lin.bias.fill_(3.0)
        lg = p(xbar, nodes, mask)
        lg.log_softmax(-1)[:, 0].sum().backward()
        g = p.q.weight.grad.abs().max().item()
        assert (p.sat.item() > 0.9) == dead
        assert (g < 1e-6) == dead, (norm, g)
    assert lg.abs().max() <= 10.0 * torch.tanh(torch.tensor(3.0)) + 1e-5


def test_checkpoints_without_ptr_norm_load_unbounded(tmp_path):
    from ar2l.evaluate import load_nets
    kw = dict(d=32, n_head=1, n_layer=1, n_types=3, type_embed=16)
    a = {"width": 32, "heads": 1, "layers": 1, "n_types": 3, "type_embed": 16}
    ck = {"pack": PackNet(ptr_norm=0.0, **kw).state_dict(),
          "mixer": PermNet(ptr_norm=0.0, **kw).state_dict(), "args": a}
    torch.save(ck, tmp_path / "old.pt")
    nets = load_nets(tmp_path / "old.pt", "cpu", ("pack", "mixer"))
    assert nets["pack"].ptr.norm == 0.0 and nets["mixer"].ptr.norm == 0.0
    torch.save(dict(ck, args=dict(a, ptr_norm=3.0)), tmp_path / "new.pt")
    assert load_nets(tmp_path / "new.pt", "cpu", ("pack",))["pack"].ptr.norm == 3.0


def test_pointer_center_leaves_the_policy_of_a_healthy_pointer_alone():
    """Far from saturation the tanh is nearly linear, so taking out the score
    every candidate shares barely moves the softmax -- and never which
    candidate is most likely.  Masked candidates stay out of the mean."""
    torch.manual_seed(0)
    from ar2l.model import Pointer
    xbar, nodes = torch.randn(4, 16) * 0.1, torch.randn(4, 7, 16) * 0.1
    mask = torch.ones(4, 7, dtype=torch.bool)
    mask[:, 5:] = False
    plain = Pointer(d=16, c_temp=10.0, norm=0.0, center=0)
    cent = Pointer(d=16, c_temp=10.0, norm=0.0, center=1)
    cent.load_state_dict(plain.state_dict())
    p0, p1 = plain(xbar, nodes, mask).softmax(-1), cent(xbar, nodes, mask).softmax(-1)
    assert (p0.argmax(-1) == p1.argmax(-1)).all()
    assert (p0 - p1).abs().max() < 0.05
    assert p1[:, 5:].max() == 0.0
    # the masked slots' scores do not reach the mean: changing them is a no-op
    nodes2 = nodes.clone(); nodes2[:, 5:] = 100.0
    assert torch.allclose(cent(xbar, nodes2, mask)[:, :5], cent(xbar, nodes, mask)[:, :5])


def test_pointer_center_revives_the_state_eq28_dies_in():
    """The dead selector's state: a shared offset puts every candidate far past
    tanh's range.  Centred, the shared part is gone, the policy is not uniform
    and the gradient flows -- without capping the score as ptr_norm does."""
    torch.manual_seed(0)
    from ar2l.model import Pointer
    xbar, nodes = torch.randn(4, 16), torch.randn(4, 6, 16)
    mask = torch.ones(4, 6, dtype=torch.bool)
    p = Pointer(d=16, c_temp=10.0, norm=0.0, center=1)
    with torch.no_grad():
        for lin in (p.q, p.k):
            lin.bias.fill_(3.0)
    lg = p(xbar, nodes, mask)
    lg.log_softmax(-1)[:, 0].sum().backward()
    assert p.sat.item() < 0.9
    assert p.q.weight.grad.abs().max().item() > 1e-3
    assert lg.std(-1).min() > 0.1                 # not one logit for all


def test_pointer_center_max_breaks_the_tie_a_spread_makes():
    """The w128/l1 selector's state: two groups of candidates far apart.
    Centred on the mean, every candidate of the upper group lands past tanh's
    range and they tie at +c; centred on the max, the best of them sits at 0
    and its rivals keep their order and a gradient.  Masked slots never set
    the max."""
    from ar2l.model import Pointer
    p = Pointer(d=1, c_temp=10.0, norm=0.0, center=1)
    with torch.no_grad():
        p.q.weight.fill_(1.0); p.q.bias.zero_()
        p.k.weight.fill_(1.0); p.k.bias.zero_()
    xbar = torch.ones(1, 1)
    nodes = torch.tensor([[-37.0, -38.0, -39.0, -270.0, 500.0]])[..., None]
    mask = torch.tensor([[True, True, True, True, False]])
    tied = p(xbar, nodes, mask)[0, :3]
    assert torch.allclose(tied, tied[0].expand(3), atol=1e-4)
    p.center = 2
    lg = p(xbar, nodes, mask)
    assert lg[0, 0] == 0.0 and lg[0, 0] > lg[0, 1] + 5 > lg[0, 2] + 5
    assert lg[0, 3] < -9.9
    lg[0, 1].backward()
    assert p.k.weight.grad.abs().item() > 1e-3


def test_pointer_center_per_call_overrides_without_touching_the_net():
    """A viewer plays a shared, cached net under another centering: the call's
    `center` must give what a net built with it gives, and leave the net's own."""
    torch.manual_seed(0)
    from ar2l.model import Pointer
    xbar, nodes = torch.randn(3, 16), torch.randn(3, 6, 16)
    mask = torch.ones(3, 6, dtype=torch.bool)
    mask[:, 4:] = False
    p = Pointer(d=16, c_temp=10.0, norm=0.0, center=1)
    for c in (0, 1, 2):
        ref = Pointer(d=16, c_temp=10.0, norm=0.0, center=c)
        ref.load_state_dict(p.state_dict())
        assert torch.equal(p(xbar, nodes, mask, center=c), ref(xbar, nodes, mask))
    assert p.center == 1


def test_checkpoints_without_ptr_center_load_uncentred(tmp_path):
    from ar2l.evaluate import load_nets
    kw = dict(d=32, n_head=1, n_layer=1, n_types=3, type_embed=16)
    a = {"width": 32, "heads": 1, "layers": 1, "n_types": 3, "type_embed": 16,
         "ptr_norm": 0.0}
    ck = {"pack": PackNet(**kw).state_dict(), "mixer": PermNet(**kw).state_dict(),
          "args": a}
    torch.save(ck, tmp_path / "old.pt")
    nets = load_nets(tmp_path / "old.pt", "cpu", ("pack", "mixer"))
    assert not nets["pack"].ptr.center and not nets["mixer"].ptr.center
    torch.save(dict(ck, args=dict(a, ptr_center=1)), tmp_path / "new.pt")
    assert load_nets(tmp_path / "new.pt", "cpu", ("pack",))["pack"].ptr.center
