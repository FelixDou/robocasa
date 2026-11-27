"""2-D path planning for the mobile base among kitchen fixtures (simulator-free).

The R2 executor's navigation drove a straight line to RoboCasa's base placement pose and
pinned the base against the corners of islands and dining counters (diagnostic self-replay,
27 November 2026). A first planner treated the base as a point 10 cm from every footprint;
the second self-replay still pinned it on open doors (27 November 2026). The base's collision
box is 0.70 x 0.50 m and sits 0.20 m behind the pose the controller steers, so its rear corners
reach 0.6 m back and swing into doors as it turns. This module plans the box itself:

- obstacles are the fixtures' footprints (convex polygons in the floor plane); thin ones (doors)
  are rasterised conservatively so a box cannot slip through them between grid cells;
- for each of ``HEADINGS`` headings, a grid holds the box's clearance with its turning axis
  (the yaw joint, ``BaseShape.pivot``) at each cell, a minimum filter of the distance field
  over the box's cells;
- A* searches (cell, heading) states: 8-connected moves at a fixed heading and turns in
  place, each needing ``HARD_GAP`` of box clearance, with an added cost below ``SOFT_GAP``;
- the box's clearance is exact to within half a cell diagonal (3.5 cm), so a planned box never
  comes within ``HARD_GAP`` - 3.5 cm of a footprint (doors: ``HARD_GAP``);
- a start or goal box already closer than ``HARD_GAP`` (a counter's overhang, the robot parked
  against it) first moves straight out at its own heading, never closer to a door;
- ``PathFollower`` gives a look-ahead pose on the path, as the controller's (x, y, yaw).
"""

from __future__ import annotations

import heapq
import math

import numpy as np

GRID_RES = 0.05          # m per cell
HEADINGS = 16            # heading bins (22.5 degrees)
HARD_GAP = 0.03          # m: the box never plans closer than this to a footprint
SOFT_GAP = 0.15          # m: below this a state costs extra
SOFT_WEIGHT = 10.0       # extra cost per step at HARD_GAP, falling to 0 at SOFT_GAP
TURN_COST = 0.05         # m-equivalent cost of one heading bin, before the clearance factor
REGION_MARGIN = 2.5      # m around the start and goal that the grid covers
OBSTACLE_PAD = 0.5       # m: ... but no farther than this beyond the obstacles
ESCAPE_SLACK = 0.01      # m: a start/goal escape never comes closer than this to what it touches
ESCAPE_MAX = 12          # cells: the farthest an escape moves
LOOKAHEAD = 0.15         # m-equivalent: the follower aims this far along the path
TURN_SCALE = 0.3         # m per radian when comparing poses (about the box's half-diagonal)
OBSTACLE_MAX_Z = 0.60    # m: a fixture whose bottom is above this (wall cabinets) does not block the base
OBSTACLE_MIN_Z = 0.05    # m: ... and one whose top is below this (floors) neither

# robosuite's Omron mobile base (omron_mobile_base.xml): the collision box ``pedestal_feet_col``
# has half sizes 0.35 x 0.25 m at x = -0.20 in the base body frame; the yaw joint is at x = -0.21.
BASE_HALF = (0.35, 0.25)
BASE_CENTRE = (-0.20, 0.0)
BASE_PIVOT = (-0.21, 0.0)


def wrap(a):
    return (float(a) + math.pi) % (2 * math.pi) - math.pi


def convex_hull(points):
    """Counter-clockwise convex hull of 2-D points (monotone chain), without repeated end point."""
    pts = sorted({(float(x), float(y)) for x, y in np.asarray(points, dtype=float)[:, :2]})
    if len(pts) <= 2:
        return np.array(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def polygon_distance(points, poly):
    """Distance from each 2-D point to a convex CCW polygon (0 inside)."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    poly = np.asarray(poly, dtype=float)
    if len(poly) == 1:
        return np.linalg.norm(points - poly[0], axis=1)
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a                                                     # (M, 2)
    ap = points[:, None, :] - a[None, :, :]                        # (N, M, 2)
    denom = np.maximum((ab ** 2).sum(-1), 1e-12)
    t = np.clip((ap * ab[None]).sum(-1) / denom[None], 0.0, 1.0)
    closest = a[None] + t[..., None] * ab[None]
    d = np.linalg.norm(points[:, None, :] - closest, axis=-1).min(axis=1)
    if len(poly) >= 3:
        inside = ((ab[None, :, 0] * ap[..., 1] - ab[None, :, 1] * ap[..., 0]) >= 0).all(axis=1)
        d[inside] = 0.0
    return d


def inside_depth(point, poly):
    """How far a 2-D point is inside a convex CCW polygon (negative outside)."""
    p = np.asarray(point, dtype=float)[:2]
    poly = np.asarray(poly, dtype=float)
    depth = math.inf
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        e = b - a
        n = float(np.linalg.norm(e))
        if n > 1e-12:
            depth = min(depth, float(e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])) / n)
    return depth


def point_in_polygon(point, poly):
    return bool(polygon_distance(np.asarray(point, dtype=float)[None, :2], poly)[0] == 0.0)


def footprint(corners, max_z=OBSTACLE_MAX_Z, min_z=OBSTACLE_MIN_Z):
    """Floor-plane hull of a fixture's 3-D box corners, or None if it cannot block the base."""
    corners = np.asarray(corners, dtype=float).reshape(-1, 3)
    if corners[:, 2].min() > max_z or corners[:, 2].max() < min_z:
        return None
    hull = convex_hull(corners[:, :2])
    return hull if len(hull) >= 3 else None


GEOM_SPHERE, GEOM_CAPSULE, GEOM_ELLIPSOID, GEOM_CYLINDER, GEOM_BOX, GEOM_MESH = 2, 3, 4, 5, 6, 7  # mjtGeom


def geom_corners(geom_type, size, xpos, xmat, rbound=0.0, aabb=None):
    """World corners of a MuJoCo geom's local bounding box (8 x 3).

    Boxes, ellipsoids, spheres, cylinders and capsules from ``size``; meshes from ``aabb``
    (centre and half sizes in the geom frame, MuJoCo's ``geom_aabb``), else a cube of ``rbound``.
    """
    size = np.asarray(size, dtype=float).reshape(-1)
    centre = np.zeros(3)
    if geom_type in (GEOM_BOX, GEOM_ELLIPSOID):
        half = size[:3]
    elif geom_type == GEOM_SPHERE:
        half = np.full(3, size[0])
    elif geom_type == GEOM_CYLINDER:
        half = np.array([size[0], size[0], size[1]])
    elif geom_type == GEOM_CAPSULE:
        half = np.array([size[0], size[0], size[1] + size[0]])
    elif geom_type == GEOM_MESH and aabb is not None:
        aabb = np.asarray(aabb, dtype=float).reshape(-1)
        centre, half = aabb[:3], aabb[3:6]
    else:
        half = np.full(3, float(rbound))
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    R = np.asarray(xmat, dtype=float).reshape(3, 3)
    return np.asarray(xpos, dtype=float).reshape(1, 3) + (centre + signs * half) @ R.T


def polygons_overlap(a, b):
    """Convex polygons ``a`` and ``b`` intersect (separating-axis test; touching counts)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    for poly in (a, b):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            axis = np.array([-e[1], e[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


class BaseShape:
    """The base's collision box in its own frame and the axis it turns about.

    Poses are the controller's (x, y, yaw) of the base body; ``to_pivot`` and ``to_origin``
    convert to and from the turning axis, about which a turn in place keeps the box in place.
    """

    def __init__(self, half=BASE_HALF, centre=BASE_CENTRE, pivot=BASE_PIVOT):
        self.half = np.asarray(half, dtype=float).reshape(-1)[:2]
        self.centre = np.asarray(centre, dtype=float).reshape(-1)[:2]
        self.pivot = np.asarray(pivot, dtype=float).reshape(-1)[:2]

    def _offset(self, yaw, v):
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

    def to_pivot(self, pose):
        x, y, yaw = (float(v) for v in pose[:3])
        o = self._offset(yaw, self.pivot)
        return (x + o[0], y + o[1], yaw)

    def to_origin(self, pose):
        x, y, yaw = (float(v) for v in pose[:3])
        o = self._offset(yaw, self.pivot)
        return (x - o[0], y - o[1], yaw)

    def corners(self, pose):
        """World corners (4 x 2, CCW) of the box at the controller pose ``pose``."""
        x, y, yaw = (float(v) for v in pose[:3])
        hx, hy = self.half
        local = self.centre + np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
        return np.array([x, y]) + np.array([self._offset(yaw, p) for p in local])

    def summary(self):
        return {"half": [round(float(v), 3) for v in self.half], "centre": [round(float(v), 3) for v in self.centre],
                "pivot": [round(float(v), 3) for v in self.pivot]}


class NavGrid:
    """Box clearance over (cell, heading); ``plan`` returns turning-axis poses or None.

    ``polygons``: obstacle footprints; ``thin``: per polygon, rasterise it conservatively (doors).
    ``extra_points``: the start and goal positions the grid must cover.
    """

    def __init__(self, polygons, extra_points=(), *, thin=None, shape=None, res=GRID_RES, headings=HEADINGS):
        self.base = shape if shape is not None else BaseShape()
        self.polygons = [np.asarray(p, dtype=float) for p in polygons]
        thin = list(thin) if thin is not None else [False] * len(self.polygons)
        self.res, self.headings = float(res), int(headings)
        ends = np.asarray(extra_points, dtype=float).reshape(-1, 2)
        lo, hi = ends.min(axis=0) - REGION_MARGIN, ends.max(axis=0) + REGION_MARGIN
        if self.polygons:
            allp = np.concatenate(self.polygons + [ends], axis=0)
            lo = np.maximum(lo, allp.min(axis=0) - OBSTACLE_PAD)
            hi = np.minimum(hi, allp.max(axis=0) + OBSTACLE_PAD)
        self.lo = lo
        self.dims = tuple(int(math.ceil(v)) + 1 for v in (hi - lo) / self.res)
        xs = lo[0] + self.res * np.arange(self.dims[0])
        ys = lo[1] + self.res * np.arange(self.dims[1])
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        centres = np.stack([gx.ravel(), gy.ravel()], axis=1)
        solid = 0.5 * math.sqrt(2) * self.res
        fields = {False: np.full(len(centres), np.inf), True: np.full(len(centres), np.inf)}
        for poly, is_thin in zip(self.polygons, thin):
            pl, ph = poly.min(axis=0) - SOFT_GAP - solid, poly.max(axis=0) + SOFT_GAP + solid
            near = np.all((centres >= pl) & (centres <= ph), axis=1)
            if near.any():
                d = polygon_distance(centres[near], poly)
                if is_thin:
                    d = np.maximum(d - solid, 0.0)   # every cell the door touches reads 0
                f = fields[bool(is_thin)]
                f[near] = np.minimum(f[near], d)
        self.clear_thin = self._box_filter(fields[True].reshape(self.dims))
        self.clear = [np.minimum(a, b) for a, b in zip(self._box_filter(fields[False].reshape(self.dims)),
                                                       self.clear_thin)]
        self.cost = []
        for c in self.clear:
            k = np.where(c >= SOFT_GAP, 1.0,
                         1.0 + SOFT_WEIGHT * ((SOFT_GAP - np.clip(c, HARD_GAP, SOFT_GAP)) / (SOFT_GAP - HARD_GAP)) ** 2)
            self.cost.append(np.where(c < HARD_GAP, np.inf, k))

    def _box_filter(self, dist):
        """Per heading: the minimum of ``dist`` over the cells under the box, turning axis at each cell.

        The box is widened by half a cell diagonal, so every point of it has a counted cell within
        that distance: with the axis on a cell centre (every planned state), the clearance is the
        exact gap to within half a cell diagonal (3.5 cm) either way.
        """
        b = self.base.centre - self.base.pivot
        h = self.base.half + 0.5 * math.sqrt(2) * self.res
        R = int(math.ceil((np.linalg.norm(b) + np.linalg.norm(h)) / self.res)) + 1
        pad = np.pad(dist, R, constant_values=np.inf)
        nx, ny = self.dims
        out = []
        for k in range(self.headings):
            th = 2 * math.pi * k / self.headings
            c, s = math.cos(th), math.sin(th)
            best = np.full(self.dims, np.inf)
            for di in range(-R, R + 1):
                for dj in range(-R, R + 1):
                    ox, oy = di * self.res, dj * self.res
                    bx, by = c * ox + s * oy - b[0], -s * ox + c * oy - b[1]
                    if abs(bx) <= h[0] and abs(by) <= h[1]:
                        np.minimum(best, pad[R + di:R + di + nx, R + dj:R + dj + ny], out=best)
            out.append(best)
        return out

    def cell(self, xy):
        i, j = np.round((np.asarray(xy, dtype=float)[:2] - self.lo) / self.res).astype(int)
        return int(np.clip(i, 0, self.dims[0] - 1)), int(np.clip(j, 0, self.dims[1] - 1))

    def centre(self, cell):
        return self.lo + self.res * np.asarray(cell, dtype=float)

    def heading(self, yaw):
        return int(round((float(yaw) % (2 * math.pi)) / (2 * math.pi / self.headings))) % self.headings

    def yaw(self, k):
        return 2 * math.pi * k / self.headings

    def clearance(self, pose):
        """Box clearance at the controller pose ``pose`` (heading binned)."""
        px, py, yaw = self.base.to_pivot(pose)
        return float(self.clear[self.heading(yaw)][self.cell((px, py))])

    def _escape(self, cell, k):
        """Nearest cell with ``HARD_GAP`` at heading ``k``, moving straight out (4-neighbours).

        Never through a cell where the box is closer to anything than it starts (less
        ``ESCAPE_SLACK``), nor closer to a door, so it leaves on its own side of thin obstacles.
        """
        clear, thin = self.clear[k], self.clear_thin[k]
        if clear[cell] >= HARD_GAP:
            return cell
        floor = float(clear[cell]) - ESCAPE_SLACK
        thin_floor = min(float(thin[cell]), HARD_GAP) - ESCAPE_SLACK
        seen, frontier = {cell}, [cell]
        for _ in range(ESCAPE_MAX):
            nxt = []
            for c in frontier:
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (c[0] + di, c[1] + dj)
                    if n in seen or not (0 <= n[0] < self.dims[0] and 0 <= n[1] < self.dims[1]):
                        continue
                    seen.add(n)
                    if thin[n] < thin_floor or clear[n] < floor:
                        continue
                    if clear[n] >= HARD_GAP:
                        return n
                    nxt.append(n)
            frontier = nxt
        return None

    def plan(self, start, goal):
        """Turning-axis poses (x, y, yaw) from controller pose ``start`` to ``goal``, or None.

        The exact start and goal axis poses are at the ends; between them, cell centres at
        binned headings.
        """
        sp, gp = self.base.to_pivot(start), self.base.to_pivot(goal)
        sk, gk = self.heading(sp[2]), self.heading(gp[2])
        s, g = self._escape(self.cell(sp), sk), self._escape(self.cell(gp), gk)
        if s is None or g is None:
            return None
        s, g = (*s, sk), (*g, gk)
        H = self.headings
        moves = [(di, dj, self.res * math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
        h = lambda c: self.res * math.hypot(c[0] - g[0], c[1] - g[1])  # noqa: E731
        best, parent = {s: 0.0}, {s: None}
        heap = [(h(s), 0.0, s)]
        nx, ny = self.dims
        while heap:
            _, cost, c = heapq.heappop(heap)
            if c == g:
                break
            if cost > best.get(c, math.inf):
                continue
            i, j, k = c
            steps = [((i + di, j + dj, k), length) for di, dj, length in moves]
            steps += [((i, j, (k + 1) % H), TURN_COST), ((i, j, (k - 1) % H), TURN_COST)]
            for n, length in steps:
                if not (0 <= n[0] < nx and 0 <= n[1] < ny):
                    continue
                step_cost = self.cost[n[2]][n[0], n[1]]
                if math.isinf(step_cost):
                    continue
                new = cost + length * step_cost
                if new < best.get(n, math.inf):
                    best[n] = new
                    parent[n] = c
                    heapq.heappush(heap, (new + h(n), new, n))
        if g not in parent:
            return None
        states = []
        c = g
        while c is not None:
            states.append(c)
            c = parent[c]
        states.reverse()
        poses = [tuple(float(v) for v in sp)]
        poses += [(*(float(v) for v in self.centre(c[:2])), self.yaw(c[2])) for c in states]
        poses.append(tuple(float(v) for v in gp))
        return poses


def path_length(points):
    """Length of the path's (x, y) part."""
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])))


def path_turn(points):
    """Total heading change along a path of (x, y, yaw) poses, radians."""
    return float(sum(abs(wrap(b[2] - a[2])) for a, b in zip(points, points[1:])))


class PathFollower:
    """Look-ahead pose on a path of turning-axis poses, as a controller pose; never goes back."""

    def __init__(self, points, shape=None, lookahead=LOOKAHEAD, turn_scale=TURN_SCALE):
        self.points = [tuple(float(v) for v in p[:3]) for p in points]
        self.base = shape if shape is not None else BaseShape()
        self.lookahead, self.turn_scale = float(lookahead), float(turn_scale)
        self.index = 0

    def _d(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1]) + self.turn_scale * abs(wrap(a[2] - b[2]))

    def target(self, mobile):
        here = self.base.to_pivot(mobile)
        window = self.points[self.index:self.index + 40]
        self.index += int(np.argmin([self._d(p, here) for p in window]))
        travelled, i = 0.0, self.index
        while i + 1 < len(self.points) and travelled < self.lookahead:
            travelled += self._d(self.points[i + 1], self.points[i])
            i += 1
        return self.base.to_origin(self.points[i])

    def at_end(self):
        return self.index >= len(self.points) - 2
