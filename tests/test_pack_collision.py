"""The arm/pack capsule check, against its own brute-force oracle."""
import math

import numpy as np
import pytest

import ar2l.pack_collision as P


def random_capsule(rng, n=40):
    a = rng.uniform([-5, -5, 0], [n + 5, n + 5, 30])
    b = rng.uniform([-5, -5, 0], [n + 5, n + 5, 30])
    return P.Capsule(a=a, b=b, ra=rng.uniform(0.5, 6), rb=rng.uniform(0.5, 6))


def random_hm(rng, n=40):
    hm = np.zeros((n, n), int)
    for _ in range(12):
        x, y = rng.integers(0, n, 2)
        w, l = rng.integers(2, 12, 2)
        hm[x:x + w, y:y + l] = hm[x:x + w, y:y + l].max() + rng.integers(1, 8)
    return hm


def test_vectorised_probe_matches_scalar():
    rng = np.random.default_rng(0)
    for _ in range(50):
        c = random_capsule(rng)
        px, py = rng.uniform(-5, 45, (2, 200))
        h = rng.uniform(0, 30, 200)
        cov, hit, t, zl, zc = P.probe_columns(c, px, py, h)
        for i in range(200):
            s = P.probe_column(c, px[i], py[i], h[i])
            assert s.covered == cov[i] and s.hit == hit[i]
            if s.covered:
                assert zl[i] == pytest.approx(s.z_low, abs=1e-9)
                assert zc[i] == pytest.approx(s.z_center, abs=1e-9)


def test_direct_pyramid_and_oracle_agree():
    rng = np.random.default_rng(1)
    n_hits = 0
    for _ in range(150):
        hm = random_hm(rng)
        c = random_capsule(rng)
        direct = P.capsule_hits_pack(c, hm, pad=0.0)
        assert P.HeightPyramid(hm).hits(c, pad=0.0) == direct
        # The padded test must be conservative: never miss an oracle hit.
        oracle = P.capsule_hits_pack_bruteforce(c, hm, pad=0.0, z_step=0.1)
        if oracle:
            assert P.capsule_hits_pack(c, hm)
        n_hits += direct
    assert 10 < n_hits < 140   # the sample exercises both outcomes


def test_pyramid_refresh_matches_rebuild():
    rng = np.random.default_rng(2)
    hm = random_hm(rng, 37)
    pyr = P.HeightPyramid(hm)
    hm[5:14, 20:31] += 9
    pyr.refresh(hm, 5, 20, 9, 11)
    ref = P.HeightPyramid(hm)
    for a, b in zip(pyr.lv, ref.lv):
        np.testing.assert_array_equal(a, b)


def test_lazy_ik_round_trips_through_fk():
    rng = np.random.default_rng(3)
    n = 0
    while n < 200:
        q = rng.uniform(-math.pi, math.pi, 6)
        goal = P.get_forward_kinematics(q, (6,))[0]
        if goal[2, 2] >= 0:        # lazy IK is for a tool pointing down
            continue
        n += 1
        sols = P.calc_lazy_inverse_kinematics(goal)
        for i in range(2):
            s = sols[:, i]
            if np.isnan(s).any():
                continue
            np.testing.assert_allclose(P.get_forward_kinematics(s, (6,))[0], goal,
                                       atol=1e-6)
        assert sols[1, 0] >= 0 or np.isnan(sols[1, 0])
        assert sols[1, 1] <= 0 or np.isnan(sols[1, 1])


def test_unreachable_pose_is_empty():
    far = P.transform(np.diag([1.0, -1.0, -1.0]), [5.0, 0.0, 0.0])
    assert P.get_inverse_kinematics(far) == []


def test_arm_over_empty_and_tall_pack():
    # Box 1.2 x 1.0 m at 100 cells/m, its corner at (-1.2, -0.5) m in the robot base frame.
    base_from_box = P.transform(t=[-1.2, -0.5, 0.0])
    chk = P.ArmPackChecker(tool_length_int=30)
    size, pos = (20, 20, 20), (50, 40, 0)

    chk.set_heightmap(np.zeros((120, 100), int))
    hit, caps = chk.is_arm_collid_with_pack(size, pos, base_from_box)
    assert len(caps) == 3 and not hit

    chk.set_heightmap(np.full((120, 100), 200))   # 2 m of boxes swallows the arm
    hit, _ = chk.is_arm_collid_with_pack(size, pos, base_from_box)
    assert hit


def test_render_and_mask_agree():
    rng = np.random.default_rng(4)
    hm = random_hm(rng)
    caps = [random_capsule(rng) for _ in range(3)]
    img = np.zeros(hm.shape, np.uint8)
    P.render_capsules(caps, hm, img)
    mask = P.collision_mask(caps, hm)
    np.testing.assert_array_equal(img == 255, mask == 255)
    assert (P.max_penetration_depth(caps, hm) >= 0) == bool(mask.any())
