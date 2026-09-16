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
    m = torch.ones(2, 5, dtype=torch.bool)
    idx = torch.tensor([3, 0])
    nb, nm = move_to_front(b, m, idx)
    assert torch.equal(nb[0, 0], b[0, 3]) and torch.equal(nb[1, 0], b[1, 0])
    assert torch.equal(nb[0, 1:], torch.cat([b[0, :3], b[0, 4:]]))
    assert nm.all()


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
         "b": torch.randn(n, nb, 3), "b_mask": torch.ones(n, nb, dtype=torch.bool),
         "l": torch.randn(n, nl, 6),
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
