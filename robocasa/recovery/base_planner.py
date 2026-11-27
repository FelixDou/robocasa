"""2-D path planning for the mobile base among kitchen fixtures (simulator-free).

The R2 executor's navigation drove a straight line to RoboCasa's base placement pose and
pinned the base against the corners of islands and dining counters (diagnostic self-replay,
27 November 2026). This module plans around fixture footprints instead:

- obstacles are the fixtures' footprints (convex polygons in the floor plane);
- a grid holds each cell's distance to the nearest footprint;
- A* searches 8-connected cells at least ``HARD_CLEARANCE`` from every footprint, with an
  added cost below ``SOFT_CLEARANCE``, so the path keeps away from furniture but can still
  leave a counter the robot starts against;
- ``PathFollower`` gives a look-ahead point on the path for the base controller.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

GRID_RES = 0.05          # m per cell
HARD_CLEARANCE = 0.10    # m: the base centre never plans closer than this to a footprint
SOFT_CLEARANCE = 0.45    # m: below this a cell costs extra (about the base's half-diagonal + margin)
SOFT_WEIGHT = 20.0       # extra cost per cell at HARD_CLEARANCE, falling to 0 at SOFT_CLEARANCE
GRID_MARGIN = 1.0        # m around the obstacles, start and goal
LOOKAHEAD = 0.30         # m: the follower aims this far along the path
OBSTACLE_MAX_Z = 0.60    # m: a fixture whose bottom is above this (wall cabinets) does not block the base
OBSTACLE_MIN_Z = 0.05    # m: ... and one whose top is below this (floors) neither


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


def point_in_polygon(point, poly):
    return bool(polygon_distance(np.asarray(point, dtype=float)[None, :2], poly)[0] == 0.0)


def footprint(corners, max_z=OBSTACLE_MAX_Z, min_z=OBSTACLE_MIN_Z):
    """Floor-plane hull of a fixture's 3-D box corners, or None if it cannot block the base."""
    corners = np.asarray(corners, dtype=float).reshape(-1, 3)
    if corners[:, 2].min() > max_z or corners[:, 2].max() < min_z:
        return None
    hull = convex_hull(corners[:, :2])
    return hull if len(hull) >= 3 else None


class NavGrid:
    """Clearance grid over obstacle polygons; ``plan`` returns a list of (x, y) points or None."""

    def __init__(self, polygons, extra_points=(), res=GRID_RES, margin=GRID_MARGIN):
        self.polygons = [np.asarray(p, dtype=float) for p in polygons]
        pts = [p for p in self.polygons] + [np.asarray(extra_points, dtype=float).reshape(-1, 2)]
        pts = np.concatenate([p for p in pts if len(p)], axis=0)
        self.res = float(res)
        self.lo = pts.min(axis=0) - margin
        hi = pts.max(axis=0) + margin
        self.shape = tuple(int(math.ceil(v)) + 1 for v in (hi - self.lo) / self.res)
        xs = self.lo[0] + self.res * np.arange(self.shape[0])
        ys = self.lo[1] + self.res * np.arange(self.shape[1])
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        centres = np.stack([gx.ravel(), gy.ravel()], axis=1)
        dist = np.full(len(centres), np.inf)
        for poly in self.polygons:
            pl, ph = poly.min(axis=0) - SOFT_CLEARANCE, poly.max(axis=0) + SOFT_CLEARANCE
            near = np.all((centres >= pl) & (centres <= ph), axis=1)
            if near.any():
                dist[near] = np.minimum(dist[near], polygon_distance(centres[near], poly))
        self.dist = dist.reshape(self.shape)

    def cell(self, xy):
        i, j = np.round((np.asarray(xy, dtype=float)[:2] - self.lo) / self.res).astype(int)
        return int(np.clip(i, 0, self.shape[0] - 1)), int(np.clip(j, 0, self.shape[1] - 1))

    def centre(self, cell):
        return self.lo + self.res * np.asarray(cell, dtype=float)

    def clearance(self, xy):
        return float(self.dist[self.cell(xy)])

    def cost(self, cell):
        d = self.dist[cell]
        if d < HARD_CLEARANCE:
            return math.inf
        if d >= SOFT_CLEARANCE:
            return 1.0
        return 1.0 + SOFT_WEIGHT * ((SOFT_CLEARANCE - d) / (SOFT_CLEARANCE - HARD_CLEARANCE)) ** 2

    def _escape(self, cell):
        """Nearest cell with ``HARD_CLEARANCE`` (breadth-first), for a start or goal inside it."""
        if self.dist[cell] >= HARD_CLEARANCE:
            return cell
        seen, frontier = {cell}, [cell]
        while frontier:
            nxt = []
            for c in frontier:
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        n = (c[0] + di, c[1] + dj)
                        if n in seen or not (0 <= n[0] < self.shape[0] and 0 <= n[1] < self.shape[1]):
                            continue
                        if self.dist[n] >= HARD_CLEARANCE:
                            return n
                        seen.add(n)
                        nxt.append(n)
            frontier = nxt
        return None

    def plan(self, start, goal):
        """Cells from ``start`` to ``goal`` as points (the exact start and goal at the ends), or None."""
        s0, g0 = self.cell(start), self.cell(goal)
        s, g = self._escape(s0), self._escape(g0)
        if s is None or g is None:
            return None
        steps = [(di, dj, math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
        h = lambda c: self.res * math.hypot(c[0] - g[0], c[1] - g[1])  # noqa: E731
        best = {s: 0.0}
        parent = {s: None}
        heap = [(h(s), 0.0, s)]
        while heap:
            _, cost, c = heapq.heappop(heap)
            if c == g:
                break
            if cost > best.get(c, math.inf):
                continue
            for di, dj, length in steps:
                n = (c[0] + di, c[1] + dj)
                if not (0 <= n[0] < self.shape[0] and 0 <= n[1] < self.shape[1]):
                    continue
                step_cost = self.cost(n)
                if math.isinf(step_cost):
                    continue
                new = cost + self.res * length * step_cost
                if new < best.get(n, math.inf):
                    best[n] = new
                    parent[n] = c
                    heapq.heappush(heap, (new + h(n), new, n))
        if g not in parent:
            return None
        cells = []
        c = g
        while c is not None:
            cells.append(c)
            c = parent[c]
        cells.reverse()
        points = [np.asarray(start, dtype=float)[:2]] + [self.centre(c) for c in cells]
        points.append(np.asarray(goal, dtype=float)[:2])
        return points


def path_length(points):
    return float(sum(np.linalg.norm(np.asarray(b) - np.asarray(a)) for a, b in zip(points, points[1:])))


class PathFollower:
    """Look-ahead point on a path: the point ``lookahead`` beyond the nearest one, never going back."""

    def __init__(self, points, lookahead=LOOKAHEAD):
        self.points = [np.asarray(p, dtype=float)[:2] for p in points]
        self.lookahead = float(lookahead)
        self.index = 0

    def target(self, xy):
        xy = np.asarray(xy, dtype=float)[:2]
        window = self.points[self.index:self.index + 40]
        d = [float(np.linalg.norm(p - xy)) for p in window]
        self.index += int(np.argmin(d))
        travelled, i = 0.0, self.index
        while i + 1 < len(self.points) and travelled < self.lookahead:
            travelled += float(np.linalg.norm(self.points[i + 1] - self.points[i]))
            i += 1
        return self.points[i]

    def at_end(self):
        return self.index >= len(self.points) - 2
