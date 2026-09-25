"""Robot-arm reach check: can a UR arm place an item without its links hitting the pack?

Python port of `pack_collision.hpp` plus the `is_arm_collid_with_pack` /
`get_arm_joint_tf_for_collusion` glue around it.  Each arm link is modelled as
a (possibly tapered) capsule and tested against the height map as a field of
solid columns `[0, h]`.

UNITS: all geometry is in height-map cells.  `hm[x, y]` -- rows indexed by x,
columns by y, the same layout as `BPPBatch.hmap[b]`.  Cell (x, y) covers
`[x, x+1) x [y, y+1)`; a column is probed at its centre `(x + .5, y + .5)`.
z is in the same units as the stored heights, times `height_scale` (see
`CellFrame`).

Transforms are 4x4 homogeneous numpy arrays in place of `tf2::Transform`.

Layout, mirroring the header:

    Capsule, probe_column / column_hits        the per-column predicate
    probe_columns                              the same predicate, vectorised
    capsule_hits_pack, any_capsule_hits_pack   direct scan
    HeightPyramid, any_capsule_hits_pyramid    max-pyramid accelerated scan
    LinkParams, ArmParams, arm_params_ur20     arm geometry
    CellFrame, build_arm_capsules              world -> cell capsules
    capsule_hits_pack_bruteforce               reference oracle (slow)
    render_capsule(s), collision_mask, ...     debug rendering, numpy only

and the parts of `UrKin` it needs, UR20 only:

    get_forward_kinematics, calc_lazy_inverse_kinematics,
    get_inverse_kinematics, get_link_1_DH_offset_tf
    ArmPackChecker                             is_arm_collid_with_pack

The tool always points down, so only the lazy two-solution IK is ported and
the cell runs solution 1 (`IK_SOLUTION_NUMBER`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Half the diagonal of a unit cell.  Padding every radius by this keeps column
# tests conservative: a hit may be reported up to 0.707 cells early, never a
# miss that's a hit.
K_CELL_PAD = float(np.float32(0.70710678))   # the C++ constant is a float

_TINY = 1e-9

# The C++ builds capsules partly in float.  Reproducing those roundings with
# float32 makes the endpoints and radii bit-identical to it; the hot-path
# column test stays in float64, which is at least as accurate.
_f32 = np.float32


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# =============================================================================
#  Capsule and the column predicate
# =============================================================================
@dataclass
class Capsule:
    a: np.ndarray = field(default_factory=lambda: np.zeros(3))   # axis endpoint, cells
    b: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ra: float = 1.0   # radius at a, cells
    rb: float = 1.0   # radius at b, cells

    def __post_init__(self):
        self.a = np.asarray(self.a, float)
        self.b = np.asarray(self.b, float)

    def axis(self):
        return self.b - self.a

    def rmax(self):
        return max(self.ra, self.rb)

    def extend(self, front_cells, back_cells):
        """Lengthen along the capsule's own 3D axis.

        `ra` stays attached to the *new* a, so a strongly tapered capsule
        shifts its taper slightly -- irrelevant for near constant-radius links.
        """
        d = self.axis()
        L = _f32(np.linalg.norm(d))          # const float L = d.length();
        if L < 1e-6:
            return
        u = d * float(_f32(1.0) / L)         # d * (1.0f / L)
        self.b = self.b + u * front_cells
        self.a = self.a - u * back_cells

    def z_floor(self):
        """Lowest z the capsule can reach anywhere.  Used for coarse rejection."""
        return min(self.a[2] - self.ra, self.b[2] - self.rb)

    def __repr__(self):
        a, b = self.a, self.b
        return (f"Capsule{{ a=({a[0]:g}, {a[1]:g}, {a[2]:g}), "
                f"b=({b[0]:g}, {b[1]:g}, {b[2]:g}), ra={self.ra:g}, rb={self.rb:g} }}")


@dataclass
class ColumnProbe:
    covered: bool = False    # column is laterally inside the capsule cross-section
    hit: bool = False        # ...and the pack column reaches the capsule underside
    t: float = 0.0           # axis parameter of the deciding point
    z_low: float = 0.0       # capsule underside above this column
    z_center: float = 0.0    # capsule axis height at t


def probe_column(c, px, py, h, pad=K_CELL_PAD):
    """Does the solid column [0, h] at (px, py) intersect the capsule?

    Let e0 = a_xy - P, d = b_xy - a_xy, dr = rb - ra.  The column meets the
    capsule iff there is t in [0, 1] with

        | e0 + t*d |^2  <=  ( ra + t*dr )^2                  (lateral)
        a.z + t*dz - sqrt( r(t)^2 - dist(t)^2 )  <=  h       (underside)

    The lateral condition is a quadratic f(t) <= 0.  The underside minimum over
    the valid set sits at an interval endpoint, the XY-nearest parameter, a
    lateral-boundary root, or an interior critical point, so evaluating those
    candidates is exact up to floating point -- including steep links.

    This is the single source of truth: `column_hits`, the pyramid leaves and
    the renderer all call it (or its vectorised twin `probe_columns`).
    """
    out = ColumnProbe()

    ra = c.ra + pad
    rb = c.rb + pad
    dr = rb - ra

    ax, ay, az = float(c.a[0]), float(c.a[1]), float(c.a[2])
    dx = float(c.b[0]) - ax
    dy = float(c.b[1]) - ay
    dz = float(c.b[2]) - az

    ex = ax - px
    ey = ay - py

    dd = dx * dx + dy * dy   # |d_xy|^2
    ed = ex * dx + ey * dy   # e0 . d_xy
    ee = ex * ex + ey * ey   # |e0|^2

    # f(t) = A t^2 + 2 B t + C,  f(t) <= 0  <=>  laterally inside
    A = dd - dr * dr
    B = ed - ra * dr
    C = ee - ra * ra

    # Any valid t gives a real point on the capsule above the column, so extra
    # candidates only tighten the result -- never below the true minimum.
    cand = [0.0, 1.0]                                  # 1. segment endpoints

    if dd > _TINY:                                     # 2. XY-nearest point
        cand.append(-ed / dd)

    if abs(A) > _TINY:                                 # 3. lateral boundary
        disc = B * B - A * C
        if disc >= 0.0:
            s = math.sqrt(disc)
            cand += [(-B - s) / A, (-B + s) / A]
    elif abs(B) > _TINY:
        cand.append(-C / (2.0 * B))

    # 4. Interior critical points.  With w(t) = -(A t^2 + 2B t + C),
    #    g'(t) = 0  iff  2 dz sqrt(w) = w'(t); squaring gives
    #    A(A + dz^2) t^2 + 2B(A + dz^2) t + (B^2 + dz^2 C) = 0.
    #    Spurious roots from squaring are harmless -- each is re-evaluated.
    if abs(dz) > _TINY:
        P = A * (A + dz * dz)
        Q = 2.0 * B * (A + dz * dz)
        R = B * B + dz * dz * C
        if abs(P) > _TINY:
            disc = Q * Q - 4.0 * P * R
            if disc >= 0.0:
                s = math.sqrt(disc)
                cand += [(-Q - s) / (2.0 * P), (-Q + s) / (2.0 * P)]
        elif abs(Q) > _TINY:
            cand.append(-R / Q)

    z_low = math.inf
    best_t = 0.0
    for t in cand[:8]:
        t = _clamp(t, 0.0, 1.0)
        qx = ex + t * dx
        qy = ey + t * dy
        d2 = qx * qx + qy * qy
        r = ra + t * dr
        r2 = r * r
        if d2 > r2:
            continue   # this t is not laterally inside
        z = az + t * dz - math.sqrt(r2 - d2)
        if z < z_low:
            z_low = z
            best_t = t

    if z_low == math.inf:
        return out   # outside the capsule footprint entirely

    out.covered = True
    out.t = best_t
    out.z_low = z_low
    out.z_center = az + best_t * dz
    out.hit = z_low <= h
    return out


def column_hits(c, px, py, h, pad=K_CELL_PAD):
    return probe_column(c, px, py, h, pad).hit


def probe_columns(c, px, py, h, pad=K_CELL_PAD):
    """`probe_column` over arrays of columns at once.

    `px, py, h` broadcast together.  Returns `(covered, hit, t, z_low,
    z_center)` arrays; where `covered` is False the other fields are 0.  Same
    candidate set and the same arithmetic as the scalar version, so the two
    agree to floating point -- `tests/test_pack_collision.py` checks it.
    """
    px, py, h = np.broadcast_arrays(np.asarray(px, float), np.asarray(py, float),
                                    np.asarray(h, float))
    ra = c.ra + pad
    rb = c.rb + pad
    dr = rb - ra

    ax, ay, az = (float(v) for v in c.a)
    dx, dy, dz = (float(v) for v in c.b - c.a)

    ex = ax - px
    ey = ay - py
    dd = dx * dx + dy * dy
    ed = ex * dx + ey * dy
    ee = ex * ex + ey * ey

    A = dd - dr * dr        # scalar: depends only on the capsule
    B = ed - ra * dr
    C = ee - ra * ra

    nan = np.full(px.shape, np.nan)
    cand = [np.zeros(px.shape), np.ones(px.shape)]

    with np.errstate(invalid="ignore", divide="ignore"):
        if dd > _TINY:
            cand.append(-ed / dd)

        if abs(A) > _TINY:
            disc = B * B - A * C
            s = np.sqrt(np.where(disc >= 0.0, disc, np.nan))
            cand += [(-B - s) / A, (-B + s) / A]
        else:
            cand.append(np.where(np.abs(B) > _TINY, -C / (2.0 * B), nan))

        if abs(dz) > _TINY:
            P = A * (A + dz * dz)
            Q = 2.0 * B * (A + dz * dz)
            R = B * B + dz * dz * C
            if abs(P) > _TINY:
                disc = Q * Q - 4.0 * P * R
                s = np.sqrt(np.where(disc >= 0.0, disc, np.nan))
                cand += [(-Q - s) / (2.0 * P), (-Q + s) / (2.0 * P)]
            else:
                cand.append(np.where(np.abs(Q) > _TINY, -R / Q, nan))

        T = np.clip(np.stack(cand), 0.0, 1.0)          # (k, ...) NaN = absent
        qx = ex + T * dx
        qy = ey + T * dy
        d2 = qx * qx + qy * qy
        r = ra + T * dr
        r2 = r * r
        ok = ~np.isnan(T) & (d2 <= r2)
        Z = np.where(ok, az + T * dz - np.sqrt(np.where(ok, r2 - d2, 0.0)), np.inf)

    k = np.argmin(Z, axis=0)
    z_low = np.take_along_axis(Z, k[None], 0)[0]
    t = np.take_along_axis(T, k[None], 0)[0]
    covered = np.isfinite(z_low)
    t = np.where(covered, t, 0.0)
    z_low = np.where(covered, z_low, 0.0)
    z_center = np.where(covered, az + t * dz, 0.0)
    hit = covered & (z_low <= h)
    return covered, hit, t, z_low, z_center


# =============================================================================
#  Direct scan
# =============================================================================
def _footprint(c, nx, ny, pad):
    """Capsule AABB clipped to the field, inclusive; None if it misses it."""
    if nx <= 0 or ny <= 0:
        return None
    rmax = c.rmax() + pad
    x0 = max(0, int(math.floor(min(c.a[0], c.b[0]) - rmax)))
    x1 = min(nx - 1, int(math.ceil(max(c.a[0], c.b[0]) + rmax)))
    y0 = max(0, int(math.floor(min(c.a[1], c.b[1]) - rmax)))
    y1 = min(ny - 1, int(math.ceil(max(c.a[1], c.b[1]) + rmax)))
    if x0 > x1 or y0 > y1:
        return None
    return x0, y0, x1, y1


def _probe_footprint(c, hm, pad, height_scale):
    """`probe_columns` over the capsule's footprint; None if it misses the field."""
    hm = np.asarray(hm)
    fp = _footprint(c, hm.shape[0], hm.shape[1], pad)
    if fp is None:
        return None
    x0, y0, x1, y1 = fp
    xs = np.arange(x0, x1 + 1, dtype=float)[:, None] + 0.5
    ys = np.arange(y0, y1 + 1, dtype=float)[None, :] + 0.5
    h = hm[x0:x1 + 1, y0:y1 + 1].astype(float) * height_scale
    return (x0, y0, x1, y1), h, probe_columns(c, xs, ys, h, pad)


def capsule_hits_pack(c, hm, pad=K_CELL_PAD, height_scale=1.0):
    """Direct scan, no acceleration structure.

    Fine for one or two queries per height-map state; for many, build a
    `HeightPyramid`.
    """
    hm = np.asarray(hm)
    fp = _footprint(c, hm.shape[0], hm.shape[1], pad)
    if fp is None:
        return False
    x0, y0, x1, y1 = fp
    # Coarse reject: the tallest column in the footprint cannot reach the
    # lowest point of the capsule.
    if float(hm[x0:x1 + 1, y0:y1 + 1].max()) * height_scale < c.z_floor() - pad:
        return False
    _, _, (_, hit, _, _, _) = _probe_footprint(c, hm, pad, height_scale)
    return bool(hit.any())


def any_capsule_hits_pack(caps, hm, pad=K_CELL_PAD, height_scale=1.0):
    """Whole-arm check: True as soon as any capsule hits."""
    return any(capsule_hits_pack(c, hm, pad, height_scale) for c in caps)


# =============================================================================
#  Pyramid pruning
# =============================================================================
def _seg_rect_overlap(ax, ay, bx, by, rx0, ry0, rx1, ry1):
    """Separating-axis test: segment vs axis-aligned rectangle."""
    if max(ax, bx) < rx0 or min(ax, bx) > rx1:
        return False
    if max(ay, by) < ry0 or min(ay, by) > ry1:
        return False
    nx, ny = -(by - ay), bx - ax   # segment normal
    if abs(nx) < _TINY and abs(ny) < _TINY:
        return True
    s = ax * nx + ay * ny
    p = [cx * nx + cy * ny for cx in (rx0, rx1) for cy in (ry0, ry1)]
    return min(p) <= s <= max(p)


def _z_low_bound_over_rect(c, pad, rx0, ry0, rx1, ry1):
    """Lower bound on the capsule underside over a rectangle of cells.

    z_low(t) >= a.z + t*dz - r(t), affine in t, so its minimum over an interval
    is at an endpoint.  The interval is the nearest-point parameters of the
    rect corners widened by r/|d_xy|, because the deciding t can be a lateral
    root that far away -- without it the pyramid could prune real hits near
    the end caps.
    """
    ra = c.ra + pad
    rb = c.rb + pad
    dr = rb - ra
    dx, dy, dz = (float(v) for v in c.b - c.a)
    dd = dx * dx + dy * dy

    t0, t1 = 0.0, 1.0
    if dd > _TINY:
        ts = [((cx - c.a[0]) * dx + (cy - c.a[1]) * dy) / dd
              for cx in (rx0, rx1) for cy in (ry0, ry1)]
        slack = max(ra, rb) / math.sqrt(dd)
        t0 = _clamp(min(ts) - slack, 0.0, 1.0)
        t1 = _clamp(max(ts) + slack, 0.0, 1.0)

    az = float(c.a[2])
    return min(az + t0 * dz - (ra + t0 * dr), az + t1 * dz - (ra + t1 * dr))


def _max_pool2(m):
    """2x2 max-pool, odd edges folded onto themselves (as the C++ min(i0+1, n-1))."""
    r, c = m.shape
    if r % 2:
        m = np.vstack([m, m[-1:]])
    if c % 2:
        m = np.hstack([m, m[:, -1:]])
    return m.reshape(m.shape[0] // 2, 2, m.shape[1] // 2, 2).max(axis=(1, 3))


class HeightPyramid:
    """Max-pyramid over the height field (~1.33x memory).

    Queries descend quadtree-style and prune whole subtrees when the tile,
    inflated by the capsule radius, misses the axis laterally, or when the
    tile's tallest column cannot reach the capsule underside.  Placements only
    ever raise heights, so `refresh` updates just the written patch.
    """

    def __init__(self, hm=None, height_scale=1.0):
        self.lv = []
        self.nx = self.ny = 0
        self.hs = height_scale
        if hm is not None:
            self.build(hm, height_scale)

    def build(self, hm, height_scale=1.0):
        """`height_scale` converts stored heights into lateral cell units;
        pass `CellFrame.height_scale()`."""
        hm = np.asarray(hm)
        self.nx, self.ny = hm.shape
        self.hs = height_scale
        self.lv = []
        if self.nx <= 0 or self.ny <= 0:
            return
        self.lv.append(hm.astype(float) * height_scale)
        while self.lv[-1].shape[0] > 1 or self.lv[-1].shape[1] > 1:
            self.lv.append(_max_pool2(self.lv[-1]))

    def refresh(self, hm, x0, y0, nx, ny):
        """Re-sync after writing an object into the height field.
        (x0, y0) is the patch origin in cells, (nx, ny) its extent."""
        if not self.lv:
            self.build(hm, self.hs)
            return
        ax0, ay0 = max(0, x0), max(0, y0)
        ax1, ay1 = min(self.nx - 1, x0 + nx - 1), min(self.ny - 1, y0 + ny - 1)
        if ax0 > ax1 or ay0 > ay1:
            return
        hm = np.asarray(hm)
        self.lv[0][ax0:ax1 + 1, ay0:ay1 + 1] = hm[ax0:ax1 + 1, ay0:ay1 + 1] * self.hs
        for L in range(1, len(self.lv)):
            prev, cur = self.lv[L - 1], self.lv[L]
            bx0, bx1, by0, by1 = ax0 // 2, ax1 // 2, ay0 // 2, ay1 // 2
            for i in range(bx0, bx1 + 1):
                i0, i1 = 2 * i, min(2 * i + 1, prev.shape[0] - 1)
                for j in range(by0, by1 + 1):
                    j0, j1 = 2 * j, min(2 * j + 1, prev.shape[1] - 1)
                    cur[i, j] = max(prev[i0, j0], prev[i0, j1], prev[i1, j0], prev[i1, j1])
            ax0, ax1, ay0, ay1 = bx0, bx1, by0, by1

    def hits(self, c, pad=K_CELL_PAD):
        if not self.lv:
            return False
        return self._node(c, pad, len(self.lv) - 1, 0, 0)

    def levels(self):
        return len(self.lv)

    def _node(self, c, pad, L, tx, ty):
        m = self.lv[L]
        if tx >= m.shape[0] or ty >= m.shape[1]:
            return False
        span = 1 << L
        cx0, cy0 = tx * span, ty * span
        cx1, cy1 = min(self.nx, cx0 + span), min(self.ny, cy0 + span)   # exclusive
        if cx0 >= cx1 or cy0 >= cy1:
            return False

        # Lateral prune: tile inflated by rmax as a box is a superset of the
        # Minkowski sum with a disc, so it never prunes a tile really reached.
        rmax = c.rmax() + pad
        if not _seg_rect_overlap(c.a[0], c.a[1], c.b[0], c.b[1],
                                 cx0 - rmax, cy0 - rmax, cx1 + rmax, cy1 + rmax):
            return False
        if m[tx, ty] < _z_low_bound_over_rect(c, pad, cx0, cy0, cx1, cy1):
            return False
        if L == 0:
            return column_hits(c, tx + 0.5, ty + 0.5, m[tx, ty], pad)
        return any(self._node(c, pad, L - 1, 2 * tx + i, 2 * ty + j)
                   for i in (0, 1) for j in (0, 1))


def any_capsule_hits_pyramid(caps, pyramid, pad=K_CELL_PAD):
    """Whole-arm check against a prebuilt `HeightPyramid`."""
    return any(pyramid.hits(c, pad) for c in caps)


# =============================================================================
#  Arm parameters
# =============================================================================
@dataclass
class LinkParams:
    radius_cells: float = 5.0        # physical tube radius at a (start)
    radius_cells_end: float = -1.0   # radius at b; < 0 means same as radius_cells
    margin_cells: float = 2.0        # safety clearance ON TOP of the radius
    extend_front: float = 0.0        # lengthen past b
    extend_back: float = 0.0         # lengthen past a
    # Lateral offset from the DH line to the tube centreline, METRES, in the
    # link's own frame -- rotates with the arm.
    centerline_offset_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # If the offset is unknown, leave it zero and bound its magnitude here, in
    # cells: a capsule of radius r + u contains the true link whichever way
    # the offset points.  Inflating is strictly safer than guessing.
    offset_uncertainty_cells: float = 0.0

    def effective_radius(self):
        return float(_f32(self.radius_cells) + _f32(self.margin_cells)
                     + _f32(self.offset_uncertainty_cells))

    def effective_radius_end(self):
        r = self.radius_cells if self.radius_cells_end < 0.0 else self.radius_cells_end
        return float(_f32(r) + _f32(self.margin_cells) + _f32(self.offset_uncertainty_cells))


@dataclass
class ArmParams:
    upper_arm: LinkParams = field(default_factory=LinkParams)   # link 2
    forearm: LinkParams = field(default_factory=LinkParams)     # link 3
    wrist: LinkParams = field(default_factory=LinkParams)       # link 4


def arm_params_ur20(cells_per_meter=100.0, margin_cells=0.0):
    """UR20 link capsules.  Joint diameters (m) as in `UrKin::get_joints_diameter`,
    which are uncalibrated guesses -- `offset_uncertainty_cells` can absorb that."""
    dia = (0.45, 0.16, 0.13, 0.10, 0.10, 0.10)
    dia_end = (0.45, 0.12, 0.09, 0.10, 0.10, 0.10)
    s = cells_per_meter
    return ArmParams(
        upper_arm=LinkParams(radius_cells=float(_f32(0.5) * _f32(dia[1]) * _f32(s)),
                             radius_cells_end=float(_f32(0.5) * _f32(dia_end[1]) * _f32(s)),
                             margin_cells=margin_cells,
                             extend_front=3.0, extend_back=0.0),
        forearm=LinkParams(radius_cells=float(_f32(0.5) * _f32(dia[2]) * _f32(s)),
                           radius_cells_end=float(_f32(0.5) * _f32(dia_end[2]) * _f32(s)),
                           margin_cells=margin_cells,
                           extend_front=4.0, extend_back=4.0),
        wrist=LinkParams(radius_cells=float(_f32(0.5) * _f32(dia[4]) * _f32(s)),
                         radius_cells_end=float(_f32(0.5) * _f32(dia_end[4]) * _f32(s)),
                         margin_cells=margin_cells,
                         extend_front=3.0, extend_back=0.0),
    )


@dataclass
class CellFrame:
    """Float version of `world_to_int_point`:  ceil(world * scale - EPS).

    ceil puts u into index k for u in (k-1, k]; this module's cells cover
    [k, k+1), so the float coordinate is shifted by +1 on x, y -- by (1 - eps)
    exactly, since ceil(u - eps) == floor(u - eps + 1).  Do not "fix" the
    ceil convention: the pack is rasterised with the same function, so the
    offset cancels.

    The lateral grid must be square (scale_x == scale_y) because a capsule
    has one radius.  z is emitted in *lateral* cell units (scaled by scale_x),
    so stored heights must be multiplied by `height_scale()`.
    """
    scale_x: float = 100.0   # cells per metre
    scale_y: float = 100.0
    scale_z: float = 100.0
    shift: tuple = (1.0, 1.0, 0.0)
    eps: float = 1e-3        # must match the EPS in world_to_int_point

    def lateral_scale(self):
        return self.scale_x

    def height_scale(self):
        return self.scale_x / self.scale_z

    def is_square(self):
        return abs(self.scale_x - self.scale_y) < 1e-6 * abs(self.scale_x)

    def __call__(self, v):
        v = np.asarray(v, float)
        # float(v.x()) * scale_x - eps is float arithmetic; + shift is double.
        sx, sy, eps = _f32(self.scale_x), _f32(self.scale_y), _f32(self.eps)
        return np.array([float(_f32(v[0]) * sx - eps) + self.shift[0],
                         float(_f32(v[1]) * sy - eps) + self.shift[1],
                         float(_f32(v[2]) * sx) + self.shift[2]])   # scale_x on purpose

    def world_to_int_point(self, v):
        v = np.asarray(v, float)
        s = np.array([self.scale_x, self.scale_y, self.scale_z])
        return np.ceil(v * s - self.eps).astype(int)

    def int_point_to_world(self, p):
        # int / float(dividing_box_size) per axis, in float32 like the C++.
        s = np.array([self.scale_x, self.scale_y, self.scale_z], np.float32)
        return (np.asarray(p, np.float32) / s).astype(float)


def build_arm_capsules(joint_tf, box_from_world, upper_arm_off_a, upper_arm_off_b,
                       p, to_cell):
    """Link capsules in box cells from joint transforms.

    `joint_tf` is forward kinematics for joints 1..6 (`joint_tf[0]` is joint 1),
    as returned by `get_forward_kinematics(q, (1, 2, 3, 4, 5, 6))`.  All
    transforms are 4x4.  The centreline offset is expressed in the link's first
    joint frame and applied to both endpoints.
    """
    if len(joint_tf) < 5:
        return []

    def make(world_a, world_b, lp):
        A = box_from_world @ world_a
        B = box_from_world @ world_b
        o_box = A[:3, :3] @ np.asarray(lp.centerline_offset_m, float)
        c = Capsule(a=to_cell(A[:3, 3] + o_box), b=to_cell(B[:3, 3] + o_box),
                    ra=lp.effective_radius(), rb=lp.effective_radius_end())
        c.extend(lp.extend_front, lp.extend_back)
        return c

    return [
        make(joint_tf[0] @ upper_arm_off_a, joint_tf[1] @ upper_arm_off_b, p.upper_arm),
        make(joint_tf[1], joint_tf[2], p.forearm),
        make(joint_tf[3], joint_tf[4], p.wrist),
    ]


# =============================================================================
#  Reference oracle
# =============================================================================
def capsule_hits_pack_bruteforce(c, hm, z_step=0.25, pad=0.0, t_steps=256,
                                 height_scale=1.0):
    """Dense 3D sampling of each column against the capsule.  Orders of
    magnitude slower -- tests only.  Should agree with `capsule_hits_pack`
    except within about K_CELL_PAD of the boundary."""
    hm = np.asarray(hm)
    nx, ny = hm.shape
    if nx <= 0 or ny <= 0:
        return False
    rmax = c.rmax() + pad
    x0 = max(0, int(math.floor(min(c.a[0], c.b[0]) - rmax)) - 1)
    x1 = min(nx - 1, int(math.ceil(max(c.a[0], c.b[0]) + rmax)) + 1)
    y0 = max(0, int(math.floor(min(c.a[1], c.b[1]) - rmax)) - 1)
    y1 = min(ny - 1, int(math.ceil(max(c.a[1], c.b[1]) + rmax)) + 1)

    t = np.linspace(0.0, 1.0, t_steps + 1)
    q = c.a[None] + t[:, None] * c.axis()[None]           # (T, 3)
    r2 = (c.ra + t * (c.rb - c.ra) + pad) ** 2
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            h = float(hm[x, y]) * height_scale
            zs = np.minimum(np.arange(0.0, h + 0.5 * z_step, z_step), h)
            pts = np.stack([np.full_like(zs, x + 0.5), np.full_like(zs, y + 0.5), zs], 1)
            d2 = ((pts[:, None, :] - q[None]) ** 2).sum(-1)  # (Z, T)
            if (d2 <= r2[None]).any():
                return True
    return False


# =============================================================================
#  Debug rendering (numpy; images are rows = x, cols = y like the C++ ROI)
# =============================================================================
@dataclass
class RenderStyle:
    collision_value: int = 255
    floor_value: int = 0
    depth: str = "centerline"          # or "underside" -- the value actually compared
    reserve_collision_value: bool = True
    preserve_collisions: bool = True


def render_capsule(c, hm, img, st=None, pad=K_CELL_PAD, height_scale=1.0):
    """Paint one capsule into `img` in place:
    covered & clear -> depth, covered & hit -> collision_value, else untouched."""
    st = st or RenderStyle()
    res = _probe_footprint(c, hm, pad, height_scale)
    if res is None:
        return
    (x0, y0, x1, y1), _, (covered, hit, _, z_low, z_center) = res
    view = img[x0:x1 + 1, y0:y1 + 1]
    zf = z_low if st.depth == "underside" else z_center
    # int(std::lround(float)): halves round away from zero, not to even.
    zf = zf.astype(np.float32).astype(float)
    z = (np.sign(zf) * np.floor(np.abs(zf) + 0.5)).astype(np.int64)
    if st.reserve_collision_value:
        z = np.minimum(z, st.collision_value - 1)
    value = np.where(hit, st.collision_value, np.maximum(z, st.floor_value))
    paint = covered.copy()
    if st.preserve_collisions:
        paint &= view != st.collision_value
    view[paint] = value[paint].astype(img.dtype, copy=False)


def render_capsules(caps, hm, img, st=None, pad=K_CELL_PAD, height_scale=1.0):
    """Two passes, so collisions always win over depths whatever the link order."""
    st = st or RenderStyle()
    depth_only = RenderStyle(**{**st.__dict__, "preserve_collisions": True})
    for c in caps:
        render_capsule(c, hm, img, depth_only, pad, height_scale)
    img[collision_mask(caps, hm, pad, height_scale) > 0] = st.collision_value


def collision_mask(caps, hm, pad=K_CELL_PAD, height_scale=1.0):
    """uint8 mask, 0 or 255, sized to the height field."""
    hm = np.asarray(hm)
    m = np.zeros(hm.shape, np.uint8)
    for c in caps:
        res = _probe_footprint(c, hm, pad, height_scale)
        if res is None:
            continue
        (x0, y0, x1, y1), _, (_, hit, _, _, _) = res
        m[x0:x1 + 1, y0:y1 + 1][hit] = 255
    return m


def collision_extent_y(mask):
    """(found, y0, y1, width) -- the column span of a collision mask."""
    cols = np.flatnonzero(np.asarray(mask).any(axis=0))
    if cols.size == 0:
        return False, 0, 0, 0
    y0, y1 = int(cols[0]), int(cols[-1])
    return True, y0, y1, y1 - y0 + 1


def max_penetration_depth(caps, hm, pad=K_CELL_PAD, height_scale=1.0):
    """Max (column height - capsule underside) over hit cells; -1 if none."""
    best = -1.0
    for c in caps:
        res = _probe_footprint(c, hm, pad, height_scale)
        if res is None:
            continue
        _, h, (_, hit, _, z_low, _) = res
        if hit.any():
            best = max(best, float((h - z_low)[hit].max()))
    return best


def escape_direction(caps, hm, pad=K_CELL_PAD, height_scale=1.0):
    """Depth-weighted unit (x, y) pointing from colliding columns toward the
    capsule axis points that hit them -- moving the arm base this way adds
    clearance.  Local gradient only.  None if nothing collides or the pushes
    cancel out."""
    sx = sy = w_sum = 0.0
    for c in caps:
        res = _probe_footprint(c, hm, pad, height_scale)
        if res is None:
            continue
        (x0, y0, x1, y1), h, (_, hit, t, z_low, _) = res
        if not hit.any():
            continue
        px = np.arange(x0, x1 + 1, dtype=float)[:, None] + 0.5
        py = np.arange(y0, y1 + 1, dtype=float)[None, :] + 0.5
        d = c.b - c.a
        ax = c.a[0] + t * d[0] - px
        ay = c.a[1] + t * d[1] - py
        w = np.where(hit, np.maximum(h - z_low, 0.0), 0.0)
        sx += float((ax * w).sum())
        sy += float((ay * w).sum())
        w_sum += float(w.sum())
    if w_sum <= 1e-6:
        return None
    n = math.hypot(sx, sy)
    if n <= 1e-6:
        return None
    return sx / n, sy / n


def upscale(img, factor=6):
    """Nearest-neighbour zoom."""
    img = np.asarray(img)
    if factor <= 1:
        return img.copy()
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


def side_by_side(left, right, factor_l=6, factor_r=6, gap=4):
    """Two images, zoomed, as one RGB uint8 image with a grey divider."""
    def rgb(m):
        m = np.asarray(m)
        return np.repeat(m[..., None], 3, axis=2) if m.ndim == 2 else m
    l, r = upscale(rgb(left), factor_l), upscale(rgb(right), factor_r)
    rows = max(l.shape[0], r.shape[0])
    out = np.zeros((rows, l.shape[1] + gap + r.shape[1], 3), np.uint8)
    out[:l.shape[0], :l.shape[1]] = l
    out[:, l.shape[1]:l.shape[1] + gap] = 40
    out[:r.shape[0], l.shape[1] + gap:] = r
    return out


# =============================================================================
#  UR20 kinematics (port of the parts of UrKin this check uses)
# =============================================================================
# Standard DH, metres, UR20 only:
# https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
_HALF_PI = math.pi / 2
UR20_D = (0.2363, 0.0, 0.0, 0.2010, 0.1593, 0.1543)
UR20_A = (0.0, -0.8620, -0.7287, 0.0, 0.0, 0.0)
UR20_ALPHA = (_HALF_PI, 0.0, 0.0, _HALF_PI, -_HALF_PI, 0.0)

# The IK branch the cell runs with -- column 1 of the lazy solution.
IK_SOLUTION_NUMBER = 1


def _ah(theta, a, d, alpha):
    """Single-joint DH transform, Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


def _ah_inverse(theta, a, d, alpha):
    """Closed-form inverse of `_ah`: [R^T  -R^T t]."""
    ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
    return np.array([[ct, st, 0.0, -a],
                     [-st * ca, ct * ca, sa, -sa * d],
                     [st * sa, -ct * sa, ca, -ca * d],
                     [0.0, 0.0, 0.0, 1.0]])


# NaN-propagating acos/asin/division, as the C++ gets from out-of-range
# arguments and x/0.  math.acos would raise instead, and NaN is how an
# unreachable pose is reported.
def _acos(x):
    return math.acos(x) if -1.0 <= x <= 1.0 else math.nan


def _asin(x):
    return math.asin(x) if -1.0 <= x <= 1.0 else math.nan


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(a) / np.float64(b))


def transform(R=None, t=None):
    """4x4 from a 3x3 rotation and a translation."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def get_forward_kinematics(joints, joint_number_pos=(1, 2, 3, 4, 5, 6)):
    """Base->frame transforms for the requested indices (ascending, 1-based):

        1 shoulder_link   top of the base pedestal
        2 upper_arm_link  shoulder pivot
        3 forearm_link    elbow pivot
        4 wrist_1_link
        5 wrist_2_link
        6 wrist_3_link    TOOL FLANGE -- the TCP needs a further transform
    """
    out = []
    T = np.eye(4)
    k = 0
    for i in range(joint_number_pos[-1]):
        T = T @ _ah(joints[i], UR20_A[i], UR20_D[i], UR20_ALPHA[i])
        if joint_number_pos[k] - 1 == i:
            out.append(T)
            k += 1
    return out


def get_link_1_DH_offset_tf(joint_number_pos):
    """Offset from the DH line to the physical upper-arm tube, for joint 2 or 3."""
    if joint_number_pos not in (2, 3):
        raise ValueError(f"joint_number_pos: have to be 2 or 3 we get: {joint_number_pos}")
    return transform(t=[0.0, 0.0, 0.25 if joint_number_pos == 2 else 0.23])


def calc_lazy_inverse_kinematics(goal):
    """The two IK solutions for a downward-pointing tool, as (6, 2).

    Column 0: theta1 = pi/2 + psi + phi, theta5 < 0, theta3 > 0, theta2 >= 0.
    Column 1: theta1 = pi/2 + psi - phi, theta5 > 0, theta3 < 0, theta2 <= 0.
    theta2 is shifted by 2 pi into that sign.  Unreachable -> NaN.
    """
    d, a, alpha = UR20_D, UR20_A, UR20_ALPHA
    gol = np.asarray(goal, float)
    th = np.zeros((6, 2))

    d3, d5 = d[3], d[5]
    a1, a2 = a[1], a[2]
    joint_offset = np.array([0.0, -d3, 0.0, 1.0])

    P_05 = gol @ np.array([0.0, 0.0, -d5, 1.0])
    half_pi_psi = _HALF_PI + math.atan2(P_05[1], P_05[0])
    phi = _acos(_div(d3, math.hypot(P_05[0], P_05[1])))

    for i in range(2):
        th[0, i] = half_pi_psi + (phi if i == 0 else -phi)
        T_16 = _ah_inverse(th[0, i], a[0], d[0], alpha[0]) @ gol
        th[4, i] = (-1 if i == 0 else 1) * _acos((T_16[2, 3] - d3) / d5)
        if np.isnan(th[:, i]).any():
            th[:, i] = np.nan
            continue

        T_inv = np.linalg.inv(T_16)
        s5 = math.sin(th[4, i])
        th[5, i] = math.atan2(_div(-T_inv[1, 2], s5), _div(T_inv[0, 2], s5))

        T_56 = _ah_inverse(th[5, i], a[5], d5, alpha[5])
        T_45 = _ah_inverse(th[4, i], a[4], d[4], alpha[4])
        T_14 = T_16 @ T_56 @ T_45
        P_13 = T_14 @ joint_offset
        P_13_norm = float(np.linalg.norm(P_13[:3]))

        th[2, i] = (1 if i == 0 else -1) * _acos(
            (P_13_norm * P_13_norm - (a1 * a1 + a2 * a2)) / (2.0 * a1 * a2))
        th[1, i] = -math.atan2(P_13[1], -P_13[0]) + _asin(_div(a2 * math.sin(th[2, i]), P_13_norm))
        if np.isnan(th[:, i]).any():
            th[:, i] = np.nan
            continue
        if (-1 if i == 1 else 1) * th[1, i] < 0:
            th[1, i] += (-1.0 if i == 1 else 1.0) * 2.0 * math.pi

        T_32 = _ah_inverse(th[2, i], a2, d[2], alpha[2])
        T_21 = _ah_inverse(th[1, i], a1, d[1], alpha[1])
        T_34 = T_32 @ T_21 @ T_14
        th[3, i] = math.atan2(T_34[1, 0], T_34[0, 0])
    return th


def get_inverse_kinematics(goal, solution_number=IK_SOLUTION_NUMBER):
    """One lazy IK solution as a list, or [] if the pose is unreachable."""
    if solution_number not in (0, 1):
        raise ValueError("pass solution_number > 1 to calc_lazy_inverse_kinematics ")
    q = calc_lazy_inverse_kinematics(goal)[:, solution_number]
    return [] if np.isnan(q).any() else q.tolist()


# =============================================================================
#  is_arm_collid_with_pack
# =============================================================================
# box_from_tool rotation, tf2::Quaternion(1, 0, 0, 0): 180 deg about x, tool z down.
_TOOL_DOWN = np.diag([1.0, -1.0, -1.0])


class ArmPackChecker:
    """Ports `is_arm_collid_with_pack` and `get_arm_joint_tf_for_collusion`
    for a UR20 on the lazy IK, solution 1.

    Keep one per pack: `set_heightmap` builds the pyramid, `refresh` updates it
    after a placement, `is_arm_collid_with_pack` answers per candidate.

    tool_length_int      KinematicData::tool_length_as_int(), in z cells
    base_from_box        4x4, the C++ tmp_box_tf / tmp_base_from_box
    """

    def __init__(self, tool_length_int=0, cell_frame=None, arm_params=None,
                 ik_solution_number=IK_SOLUTION_NUMBER, pad=K_CELL_PAD):
        self.tool_length_int = tool_length_int
        self.cell_frame = cell_frame or CellFrame()
        self.arm_params = arm_params or arm_params_ur20(self.cell_frame.lateral_scale())
        self.ik_solution_number = ik_solution_number
        self.upper_arm_off_a = get_link_1_DH_offset_tf(2)
        self.upper_arm_off_b = get_link_1_DH_offset_tf(3)
        self.pad = pad
        self.pyramid = HeightPyramid()
        if not self.cell_frame.is_square():
            raise ValueError("CellFrame lateral grid must be square (scale_x == scale_y)")

    def set_heightmap(self, hm):
        self.pyramid.build(hm, self.cell_frame.height_scale())

    def refresh(self, hm, x0, y0, nx, ny):
        self.pyramid.refresh(hm, x0, y0, nx, ny)

    def get_arm_joints(self, object_size, object_pos, base_from_box):
        """IK joint angles (rad) with the tool on top of the item's centre,
        or [] when there is no solution."""
        sx, sy, sz = (int(v) for v in object_size)
        tool_int_point = np.asarray(object_pos, int) + np.array(
            [sx // 2, sy // 2, sz + self.tool_length_int])
        box_from_tool = transform(_TOOL_DOWN, self.cell_frame.int_point_to_world(tool_int_point))
        base_from_tool = np.asarray(base_from_box, float) @ box_from_tool
        return get_inverse_kinematics(base_from_tool, self.ik_solution_number)

    def get_arm_joint_tf_for_collision(self, object_size, object_pos, base_from_box):
        """Joint frames 1..6 in the robot base frame, or [] when IK has no solution."""
        joints = self.get_arm_joints(object_size, object_pos, base_from_box)
        if not joints:
            return []
        return get_forward_kinematics(joints, (1, 2, 3, 4, 5, 6))

    def arm_capsules(self, object_size, object_pos, base_from_box):
        """Link capsules for placing an item; None when IK fails."""
        joint_tf = self.get_arm_joint_tf_for_collision(object_size, object_pos, base_from_box)
        if not joint_tf:
            return None
        return build_arm_capsules(joint_tf, np.linalg.inv(base_from_box),
                                  self.upper_arm_off_a, self.upper_arm_off_b,
                                  self.arm_params, self.cell_frame)

    def is_arm_collid_with_pack(self, object_size, object_pos, base_from_box):
        """(collides, capsules).  Unreachable poses count as a collision and
        return no capsules, as the C++ does."""
        caps = self.arm_capsules(object_size, object_pos, base_from_box)
        if caps is None:
            return True, []
        return any_capsule_hits_pyramid(caps, self.pyramid, self.pad), caps
