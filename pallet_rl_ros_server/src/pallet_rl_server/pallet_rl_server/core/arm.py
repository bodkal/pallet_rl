"""Robot-arm reach check: can a UR20 place a box without a link hitting the pack?

The batched path of the cell's `pack_collision.hpp` port: for many placements
at once, solve the lazy IK (tool pointing down, solution `ik_solution`), run
the forward kinematics, model the upper arm, forearm and wrist as tapered
capsules, and test them against the height map as a field of solid columns.
An unreachable pose counts as a collision, as on the cell.

All geometry is in grid cells; a column (x, y) is probed at its centre.
Several float32 roundings are kept on purpose: they make the capsules
bit-identical to the C++.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_f32 = np.float32
# half a cell's diagonal: padding every radius by it keeps the column test
# conservative (a hit may come up to 0.707 cells early, never late)
K_CELL_PAD = float(np.float32(0.70710678))
_TINY = 1e-9

# UR20 standard DH, metres
_HALF_PI = math.pi / 2
UR20_D = (0.2363, 0.0, 0.0, 0.2010, 0.1593, 0.1543)
UR20_A = (0.0, -0.8620, -0.7287, 0.0, 0.0, 0.0)
UR20_ALPHA = (_HALF_PI, 0.0, 0.0, _HALF_PI, -_HALF_PI, 0.0)
# the tool points down: 180 deg about x
_TOOL_DOWN = np.diag([1.0, -1.0, -1.0])


# ----------------------------------------------------------------- geometry
@dataclass
class LinkParams:
    radius_cells: float = 5.0          # tube radius at the start
    radius_cells_end: float = -1.0     # at the end; < 0 = same
    margin_cells: float = 0.0          # clearance on top of the radius
    extend_front: float = 0.0          # lengthen past the end
    extend_back: float = 0.0           # lengthen past the start
    centerline_offset_m: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def effective_radius(self):
        return float(_f32(self.radius_cells) + _f32(self.margin_cells))

    def effective_radius_end(self):
        r = (self.radius_cells if self.radius_cells_end < 0.0
             else self.radius_cells_end)
        return float(_f32(r) + _f32(self.margin_cells))


def ur20_links(cells_per_meter):
    """(upper arm, forearm, wrist) capsules.  Joint diameters as the cell's
    `UrKin::get_joints_diameter`; extensions are cm, kept the same length in
    metres on any grid."""
    dia = (0.45, 0.16, 0.13, 0.10, 0.10, 0.10)
    dia_end = (0.45, 0.12, 0.09, 0.10, 0.10, 0.10)
    s = cells_per_meter
    ext = s / 100.0

    def link(j, front, back):
        return LinkParams(
            radius_cells=float(_f32(0.5) * _f32(dia[j]) * _f32(s)),
            radius_cells_end=float(_f32(0.5) * _f32(dia_end[j]) * _f32(s)),
            extend_front=front * ext, extend_back=back * ext)

    return link(1, 3.0, 0.0), link(2, 4.0, 4.0), link(4, 3.0, 0.0)


def transform(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def quat_tf(v):
    """[x, y, z, qx, qy, qz, qw] -> 4x4, as tf2::Transform."""
    x, y, z, qx, qy, qz, qw = (float(t) for t in v)
    s = 2.0 / (qx * qx + qy * qy + qz * qz + qw * qw)
    R = np.array([
        [1 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw), s * (qx * qz + qy * qw)],
        [s * (qx * qy + qz * qw), 1 - s * (qx * qx + qz * qz), s * (qy * qz - qx * qw)],
        [s * (qx * qz - qy * qw), s * (qy * qz + qx * qw), 1 - s * (qx * qx + qy * qy)]])
    return transform(R, [x, y, z])


# --------------------------------------------------------------- kinematics
def _ah_b(theta, a, d, alpha):
    """DH transform Rz(theta) Tz(d) Tx(a) Rx(alpha) over (N,) theta."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    T = np.zeros(theta.shape + (4, 4))
    T[:, 0, 0], T[:, 0, 1], T[:, 0, 2], T[:, 0, 3] = ct, -st * ca, st * sa, a * ct
    T[:, 1, 0], T[:, 1, 1], T[:, 1, 2], T[:, 1, 3] = st, ct * ca, -ct * sa, a * st
    T[:, 2, 1], T[:, 2, 2], T[:, 2, 3] = sa, ca, d
    T[:, 3, 3] = 1.0
    return T


def _ah_inverse_b(theta, a, d, alpha):
    """Closed-form inverse of `_ah_b`."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    T = np.zeros(theta.shape + (4, 4))
    T[:, 0, 0], T[:, 0, 1], T[:, 0, 3] = ct, st, -a
    T[:, 1, 0], T[:, 1, 1], T[:, 1, 2], T[:, 1, 3] = -st * ca, ct * ca, sa, -sa * d
    T[:, 2, 0], T[:, 2, 1], T[:, 2, 2], T[:, 2, 3] = st * sa, -ct * sa, ca, -ca * d
    T[:, 3, 3] = 1.0
    return T


def lazy_ik_batch(goals, i):
    """Column `i` (0 or 1) of the lazy two-solution IK for (N, 4, 4) goals.
    -> (joints (N, 6), ok (N,)); unreachable rows are NaN."""
    d, a, alpha = UR20_D, UR20_A, UR20_ALPHA
    d3, d5, a1, a2 = d[3], d[5], a[1], a[2]
    gol = np.asarray(goals, float)
    th = np.full((len(gol), 6), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        P_05 = gol @ np.array([0.0, 0.0, -d5, 1.0])
        half_pi_psi = _HALF_PI + np.arctan2(P_05[:, 1], P_05[:, 0])
        phi = np.arccos(d3 / np.hypot(P_05[:, 0], P_05[:, 1]))
        th[:, 0] = half_pi_psi + (phi if i == 0 else -phi)
        T_16 = _ah_inverse_b(th[:, 0], a[0], d[0], alpha[0]) @ gol
        th[:, 4] = (-1 if i == 0 else 1) * np.arccos((T_16[:, 2, 3] - d3) / d5)

        ok = ~np.isnan(th[:, [0, 4]]).any(1)
        T_16[~ok] = np.eye(4)
        T_inv = np.linalg.inv(T_16)
        s5 = np.sin(th[:, 4])
        th[:, 5] = np.arctan2(-T_inv[:, 1, 2] / s5, T_inv[:, 0, 2] / s5)

        T_56 = _ah_inverse_b(th[:, 5], a[5], d5, alpha[5])
        T_45 = _ah_inverse_b(th[:, 4], a[4], d[4], alpha[4])
        T_14 = T_16 @ T_56 @ T_45
        P_13 = T_14 @ np.array([0.0, -d3, 0.0, 1.0])
        P_13_norm = np.sqrt((P_13[:, :3] * P_13[:, :3]).sum(1))

        th[:, 2] = (1 if i == 0 else -1) * np.arccos(
            (P_13_norm * P_13_norm - (a1 * a1 + a2 * a2)) / (2.0 * a1 * a2))
        th[:, 1] = (-np.arctan2(P_13[:, 1], -P_13[:, 0])
                    + np.arcsin(a2 * np.sin(th[:, 2]) / P_13_norm))
        sgn = -1.0 if i == 1 else 1.0
        th[:, 1] = np.where(sgn * th[:, 1] < 0, th[:, 1] + sgn * 2.0 * math.pi,
                            th[:, 1])

        T_32 = _ah_inverse_b(th[:, 2], a2, d[2], alpha[2])
        T_21 = _ah_inverse_b(th[:, 1], a1, d[1], alpha[1])
        T_34 = T_32 @ T_21 @ T_14
        th[:, 3] = np.arctan2(T_34[:, 1, 0], T_34[:, 0, 0])
    ok &= ~np.isnan(th).any(1)
    th[~ok] = np.nan
    return th, ok


def fk_batch(joints):
    """Joint frames 1..6 for (N, 6) joints -> (6, N, 4, 4)."""
    q = np.asarray(joints, float)
    T = np.broadcast_to(np.eye(4), (len(q), 4, 4))
    out = []
    for i in range(6):
        T = T @ _ah_b(q[:, i], UR20_A[i], UR20_D[i], UR20_ALPHA[i])
        out.append(T)
    return np.stack(out)


# ----------------------------------------------------------------- capsules
def _to_cell(cpm, v):
    """World metres -> cell coordinates, as the cell's ceil(world*s - eps)
    shifted onto cells that cover [k, k+1).  z uses the lateral scale."""
    v = np.asarray(v, float).astype(_f32)
    s, eps = _f32(cpm), _f32(1e-3)
    return np.stack([(v[:, 0] * s - eps).astype(float) + 1.0,
                     (v[:, 1] * s - eps).astype(float) + 1.0,
                     (v[:, 2] * s).astype(float)], 1)


def _capsule(world_a, world_b, lp, box_from_world, cpm):
    """One link's capsule over (N, 4, 4) frames -> end points a, b."""
    A = box_from_world @ world_a
    B = box_from_world @ world_b
    o_box = A[:, :3, :3] @ np.asarray(lp.centerline_offset_m, float)
    a = _to_cell(cpm, A[:, :3, 3] + o_box)
    b = _to_cell(cpm, B[:, :3, 3] + o_box)
    dv = b - a
    L = np.sqrt((dv * dv).sum(1)).astype(_f32)
    live = L >= 1e-6
    with np.errstate(divide="ignore"):
        inv = (_f32(1.0) / np.where(live, L, _f32(1.0))).astype(float)
    u = dv * inv[:, None]
    b = np.where(live[:, None], b + u * lp.extend_front, b)
    a = np.where(live[:, None], a - u * lp.extend_back, a)
    return a, b


def _column_hits(a, b, ra, rb, px, py, h):
    """Does the capsule's underside reach down to column height `h`?
    Over matching arrays of (capsule, column) pairs; radii carry the pad."""
    dr = rb - ra
    ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
    dx, dy, dz = b[:, 0] - ax, b[:, 1] - ay, b[:, 2] - az
    ex, ey = ax - px, ay - py
    dd = dx * dx + dy * dy
    ed = ex * dx + ey * dy
    ee = ex * ex + ey * ey
    A = dd - dr * dr
    B = ed - ra * dr
    C = ee - ra * ra
    nan = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        cand = [np.zeros_like(px), np.ones_like(px),
                np.where(dd > _TINY, -ed / dd, nan)]
        bigA = np.abs(A) > _TINY
        disc = B * B - A * C
        s = np.sqrt(np.where(bigA & (disc >= 0.0), disc, nan))
        cand += [(-B - s) / A, (-B + s) / A,
                 np.where(~bigA & (np.abs(B) > _TINY), -C / (2.0 * B), nan)]
        bigz = np.abs(dz) > _TINY
        P = A * (A + dz * dz)
        Q = 2.0 * B * (A + dz * dz)
        R = B * B + dz * dz * C
        bigP = np.abs(P) > _TINY
        disc = Q * Q - 4.0 * P * R
        s = np.sqrt(np.where(bigz & bigP & (disc >= 0.0), disc, nan))
        cand += [(-Q - s) / (2.0 * P), (-Q + s) / (2.0 * P),
                 np.where(bigz & ~bigP & (np.abs(Q) > _TINY), -R / Q, nan)]

        T = np.clip(np.stack(cand), 0.0, 1.0)
        qx = ex + T * dx
        qy = ey + T * dy
        d2 = qx * qx + qy * qy
        r = ra + T * dr
        r2 = r * r
        ok = ~np.isnan(T) & (d2 <= r2)
        Z = np.where(ok, az + T * dz - np.sqrt(np.where(ok, r2 - d2, 0.0)),
                     np.inf)
    return Z.min(0) <= h


def capsules_hit_packs(caps, hms, which, pad=K_CELL_PAD):
    """(N,) does any capsule of each placement hit its own height map?

    `caps`: [(a (N, 3), b (N, 3), ra, rb)] per link; `hms` (M, nx, ny);
    `which` (N,) the map each placement is over.  Only columns inside the
    capsule's padded bounding box and tall enough to reach its lowest point
    are probed; a whole row is skipped by a sparse-table range max.
    """
    hms = np.asarray(hms)
    _, nx, ny = hms.shape
    N = len(which)
    hit = np.zeros(N, bool)
    if N == 0:
        return hit
    which = np.asarray(which)
    cx = np.arange(nx) + 0.5
    cy = np.arange(ny) + 0.5
    # tab[j, m, x, y] = max of hms[m, x, y : y + 2^j]
    lg = np.zeros(ny + 1, np.int64)
    for i in range(2, ny + 1):
        lg[i] = lg[i // 2] + 1
    tab = np.zeros((lg[ny] + 1,) + hms.shape, hms.dtype)
    tab[0] = hms
    for j in range(1, lg[ny] + 1):
        s = 1 << (j - 1)
        np.maximum(tab[j - 1][..., : ny - s], tab[j - 1][..., s:],
                   out=tab[j][..., : ny - s])
    for a, b, ra, rb in caps:
        live = ~hit
        if not live.any():
            break
        rap, rbp = ra + pad, rb + pad
        rmax = max(rap, rbp)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        zf = np.minimum(a[:, 2] - ra, b[:, 2] - rb) - pad
        inx = ((cx[None, :] >= lo[:, None, 0] - rmax)
               & (cx[None, :] <= hi[:, None, 0] + rmax))
        iny = ((cy[None, :] >= lo[:, None, 1] - rmax)
               & (cy[None, :] <= hi[:, None, 1] + rmax))
        kx, ky = inx.sum(1), iny.sum(1)
        x0, y0 = inx.argmax(1), iny.argmax(1)
        # one entry per (placement, x row), kept if any column reaches
        nr = np.where(live & (ky > 0), kx, 0)
        k = np.repeat(np.arange(N), nr)
        if not len(k):
            continue
        x = x0[k] + np.arange(len(k)) - np.repeat(np.cumsum(nr) - nr, nr)
        j, m, ya, w = lg[ky[k]], which[k], y0[k], ky[k]
        top = np.maximum(tab[j, m, x, ya], tab[j, m, x, ya + w - (1 << j)])
        row = top.astype(float) >= zf[k] - 1e-9
        k, x = k[row], x[row]
        if not len(k):
            continue
        # each kept row expanded into its cells
        w = ky[k]
        r = np.repeat(np.arange(len(k)), w)
        y = y0[k][r] + np.arange(len(r)) - np.repeat(np.cumsum(w) - w, w)
        k, x = k[r], x[r]
        h = hms[which[k], x, y].astype(float)
        reach = h >= zf[k] - 1e-9
        k, x, y, h = k[reach], x[reach], y[reach], h[reach]
        if not len(k):
            continue
        h = _column_hits(a[k], b[k], rap, rbp, cx[x], cy[y], h)
        hit[k[h]] = True
    return hit


# ------------------------------------------------------------------ checker
class ArmChecker:
    """`collides(sizes, poses, base, hms, which)` for many placements.

    `bases` are the robot base poses to try, in order: the configured
    `base_from_box_tf` slid along its x axis by each of `base_x_moves`
    (fractions of half the pallet's y side).

    The IK/FK/capsule geometry depends only on (base, box size, pose), never
    on the pack, so it is memoised across requests; that is a pure-function
    cache, not pallet state.
    """

    GEO_MEMO_MAX = 400_000

    def __init__(self, robot, cell_m, pallet_y_cells):
        cpm = 1.0 / float(cell_m)
        self.cpm = cpm
        self.links = ur20_links(cpm)
        self.radii = [(lp.effective_radius(), lp.effective_radius_end())
                      for lp in self.links]
        self.tool_length = int(round(float(robot["tool_length_m"]) * cpm))
        self.ik_solution = int(robot["ik_solution"])
        if self.ik_solution not in (0, 1):
            raise ValueError("robot.ik_solution is 0 or 1")
        self.upper_off_a = transform(t=[0.0, 0.0, 0.25])
        self.upper_off_b = transform(t=[0.0, 0.0, 0.23])
        moves = [float(v) for v in (robot.get("base_x_moves") or [0.0])]
        if any(not -1.0 <= v <= 1.0 for v in moves):
            raise ValueError(f"robot.base_x_moves must lie in -1..1: {moves}")
        base = quat_tf(robot["base_from_box_tf"])
        pallet_y_m = pallet_y_cells * float(cell_m)
        self.bases = [transform(t=[-v * pallet_y_m / 2.0, 0.0, 0.0]) @ base
                      for v in moves]
        self._memo = {}

    def _geometry_b(self, sizes, poses, B):
        """(ok (N,), seg (N, 3, 2, 3)): reachable, and each link's capsule
        end points in cells."""
        n = len(sizes)
        tool = poses + np.stack([sizes[:, 0] // 2, sizes[:, 1] // 2,
                                 sizes[:, 2] + self.tool_length], 1)
        s = np.array([self.cpm] * 3, np.float32)
        box_from_tool = np.zeros((n, 4, 4))
        box_from_tool[:, :3, :3] = _TOOL_DOWN
        box_from_tool[:, :3, 3] = (tool.astype(np.float32) / s).astype(float)
        box_from_tool[:, 3, 3] = 1.0
        q, ok = lazy_ik_batch(B @ box_from_tool, self.ik_solution)
        seg = np.full((n, 3, 2, 3), np.nan)
        idx = np.flatnonzero(ok)
        if len(idx):
            jt = fk_batch(q[idx])
            inv = np.linalg.inv(B)
            up, fore, wrist = self.links
            ends = [
                _capsule(jt[0] @ self.upper_off_a, jt[1] @ self.upper_off_b,
                         up, inv, self.cpm),
                _capsule(jt[1], jt[2], fore, inv, self.cpm),
                _capsule(jt[3], jt[4], wrist, inv, self.cpm),
            ]
            for li, (a, b) in enumerate(ends):
                seg[idx, li, 0], seg[idx, li, 1] = a, b
        return ok, seg

    def _geometry(self, sizes, poses, B):
        """`_geometry_b`, memoised per (base, size, pose)."""
        if (sizes.min() < 0 or poses.min() < 0 or sizes.max() >= 1024
                or poses.max() >= 1024):
            return self._geometry_b(sizes, poses, B)
        if sum(len(m[0]) for m in self._memo.values()) > self.GEO_MEMO_MAX:
            self._memo.clear()
        index, oks, segs, used = self._memo.setdefault(
            B.tobytes(), ({}, np.zeros(4096, bool),
                          np.zeros((4096, 3, 2, 3)), 0))
        key = (((((sizes[:, 0] << 10) | sizes[:, 1]) << 10 | sizes[:, 2]) << 10
                | poses[:, 0]) << 10 | poses[:, 1]) << 10 | poses[:, 2]
        keys = key.tolist()
        row = np.fromiter((index.get(k, -1) for k in keys), np.int64, len(keys))
        miss = np.flatnonzero(row < 0)
        if len(miss):
            new, first = np.unique(key[miss], return_index=True)
            ok, seg = self._geometry_b(sizes[miss[first]], poses[miss[first]], B)
            end = used + len(new)
            if end > len(oks):
                cap = max(end, 2 * len(oks))
                oks = np.concatenate([oks[:used], np.zeros(cap - used, bool)])
                segs = np.concatenate([segs[:used],
                                       np.zeros((cap - used, 3, 2, 3))])
            oks[used:end], segs[used:end] = ok, seg
            for j, k in enumerate(new.tolist()):
                index[k] = used + j
            self._memo[B.tobytes()] = (index, oks, segs, end)
            row = np.fromiter((index[k] for k in keys), np.int64, len(keys))
        return oks[row], segs[row]

    def collides(self, sizes, poses, base, hms, which):
        """(N,) bool for placements of `sizes` at corner `poses` (cells) from
        robot base `base` over height maps `hms[which]`."""
        sizes = np.asarray(sizes, int).reshape(-1, 3)
        poses = np.asarray(poses, int).reshape(-1, 3)
        if not len(sizes):
            return np.zeros(0, bool)
        ok, seg = self._geometry(sizes, poses, np.asarray(base, float))
        out = ~ok
        idx = np.flatnonzero(ok)
        if len(idx):
            caps = [(seg[idx, li, 0], seg[idx, li, 1], ra, rb)
                    for li, (ra, rb) in enumerate(self.radii)]
            out[idx] = capsules_hit_packs(caps, hms, np.asarray(which)[idx])
        return out
