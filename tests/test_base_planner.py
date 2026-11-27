"""Base path planner: footprints, clearance grid, A*, path following (no simulator)."""

import math
import unittest

import numpy as np

from robocasa.recovery import base_planner as bp


def box(x0, x1, y0, y1, z0=0.0, z1=0.9):
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])


ISLAND = bp.footprint(box(1.5, 2.5, -1.5, 1.5))


class GeometryTest(unittest.TestCase):
    def test_hull_and_distance(self):
        hull = bp.convex_hull([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
        self.assertEqual(len(hull), 4)
        d = bp.polygon_distance([[0.5, 0.5], [2, 0.5], [2, 2], [-1, 0.5]], hull)
        np.testing.assert_allclose(d, [0.0, 1.0, math.sqrt(2), 1.0])
        self.assertTrue(bp.point_in_polygon([0.2, 0.9], hull))
        self.assertFalse(bp.point_in_polygon([1.2, 0.9], hull))

    def test_footprints_keep_only_what_blocks_the_base(self):
        self.assertEqual(len(bp.footprint(box(0, 1, 0, 0.6))), 4)                 # counter
        self.assertIsNone(bp.footprint(box(0, 1, 0, 0.4, z0=1.4, z1=2.1)))        # wall cabinet
        self.assertIsNone(bp.footprint(box(-5, 5, -5, 5, z0=-0.1, z1=0.0)))       # floor
        rotated = bp.footprint(np.array([[math.cos(a), math.sin(a), z] for a in np.arange(4) * math.pi / 2
                                         for z in (0, 1)]))
        self.assertTrue(bp.point_in_polygon([0, 0], rotated))


class GeomCornersTest(unittest.TestCase):
    def test_box_turned_a_quarter_about_z(self):
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        c = bp.geom_corners(bp.GEOM_BOX, [0.4, 0.02, 0.3], [1.0, 2.0, 0.5], R.ravel())
        np.testing.assert_allclose(c.min(axis=0), [0.98, 1.6, 0.2])
        np.testing.assert_allclose(c.max(axis=0), [1.02, 2.4, 0.8])

    def test_capsule_cylinder_mesh_and_fallback(self):
        I = np.eye(3).ravel()
        np.testing.assert_allclose(bp.geom_corners(bp.GEOM_CAPSULE, [0.1, 0.5, 0], [0, 0, 0], I).max(axis=0),
                                   [0.1, 0.1, 0.6])
        np.testing.assert_allclose(bp.geom_corners(bp.GEOM_CYLINDER, [0.1, 0.5, 0], [0, 0, 0], I).max(axis=0),
                                   [0.1, 0.1, 0.5])
        mesh = bp.geom_corners(bp.GEOM_MESH, [0, 0, 0], [1, 0, 0], I, aabb=[0.2, 0, 0, 0.1, 0.3, 0.4])
        np.testing.assert_allclose(mesh.min(axis=0), [1.1, -0.3, -0.4])
        np.testing.assert_allclose(bp.geom_corners(bp.GEOM_MESH, [0, 0, 0], [0, 0, 0], I, rbound=0.2).max(axis=0),
                                   [0.2, 0.2, 0.2])


SHAPE = bp.BaseShape()


def exact_gap(pose, poly, shape=SHAPE):
    """Distance between the box at ``pose`` and a convex polygon (0 if they overlap)."""
    box = shape.corners(pose)
    if bp.polygons_overlap(box, poly):
        return 0.0
    return float(min(bp.polygon_distance(box, poly).min(), bp.polygon_distance(poly, box).min()))


def swept_poses(points, n=6, shape=SHAPE):
    """Controller poses along a path of turning-axis poses, interpolated between its points."""
    for a, b in zip(points, points[1:]):
        for t in np.linspace(0, 1, n):
            yaw = a[2] + t * bp.wrap(b[2] - a[2])
            yield shape.to_origin(((1 - t) * a[0] + t * b[0], (1 - t) * a[1] + t * b[1], yaw))


class BaseShapeTest(unittest.TestCase):
    def test_pose_conversions_and_corners(self):
        pose = (1.0, 2.0, 0.7)
        np.testing.assert_allclose(SHAPE.to_origin(SHAPE.to_pivot(pose)), pose)
        c = SHAPE.corners((0.0, 0.0, 0.0))
        np.testing.assert_allclose(c.min(axis=0), [-0.55, -0.25])
        np.testing.assert_allclose(c.max(axis=0), [0.15, 0.25])
        # Turning in place about the yaw axis keeps the box centre (0.01 m from the axis) in place.
        pivot = SHAPE.to_pivot((0.0, 0.0, 0.0))
        turned = SHAPE.to_origin((pivot[0], pivot[1], math.pi / 2))
        np.testing.assert_allclose(SHAPE.corners(turned).mean(axis=0), c.mean(axis=0), atol=0.015)

    def test_overlap(self):
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        self.assertTrue(bp.polygons_overlap(sq, sq + 0.5))
        self.assertFalse(bp.polygons_overlap(sq, sq + [1.1, 0.0]))
        diamond = np.array([[1.6, 0.5], [2.1, 1.0], [1.6, 1.5], [1.1, 1.0]])
        self.assertFalse(bp.polygons_overlap(sq, diamond))   # only a diagonal axis separates them


class ClearanceTest(unittest.TestCase):
    def test_box_clearance_matches_the_exact_gap(self):
        counter = bp.footprint(box(-1.0, 1.0, 0.0, 0.6))
        grid = bp.NavGrid([counter], extra_points=[(0, -1), (0, -1)])
        for k in range(bp.HEADINGS):
            yaw = 2 * math.pi * k / bp.HEADINGS
            for y in (-0.5, -0.7, -0.9):
                pivot = (0.0, y, yaw)                 # on a cell centre: the grid is anchored at the points
                pose = SHAPE.to_origin(pivot)
                gap, read = exact_gap(pose, counter), grid.clearance(pose)
                if gap < bp.SOFT_GAP:
                    self.assertAlmostEqual(read, gap, delta=0.5 * math.sqrt(2) * bp.GRID_RES + 1e-9)
                else:
                    self.assertGreaterEqual(read, bp.SOFT_GAP - 0.5 * math.sqrt(2) * bp.GRID_RES)

    def test_a_thin_door_through_the_box_is_never_missed(self):
        door = bp.footprint(box(0.012, 0.03, -1.0, 1.0))
        grid = bp.NavGrid([door], extra_points=[(-1, 0), (1, 0)], thin=[True])
        for x in np.linspace(-0.3, 0.3, 13):
            for yaw in (0.0, math.pi / 2, math.pi / 8):
                pose = SHAPE.to_origin((x, 0.0, yaw))
                if exact_gap(pose, door) == 0.0:
                    self.assertLess(grid.clearance(pose), bp.HARD_GAP, (x, yaw))


class PlanTest(unittest.TestCase):
    def assert_clear(self, points, polygons, skip_ends=False):
        segs = points[1:-1] if skip_ends else points
        for pose in swept_poses(segs):
            for poly in polygons:
                self.assertFalse(bp.polygons_overlap(SHAPE.corners(pose), poly), pose)

    def test_goes_around_an_island(self):
        island = bp.footprint(box(1.5, 2.5, -1.5, 1.5))
        start, goal = (0.0, 0.0, 0.0), (4.5, 0.0, 0.0)
        grid = bp.NavGrid([island], extra_points=[start[:2], goal[:2]])
        points = grid.plan(start, goal)
        self.assertIsNotNone(points)
        np.testing.assert_allclose(points[0], SHAPE.to_pivot(start))
        np.testing.assert_allclose(points[-1], SHAPE.to_pivot(goal))
        self.assert_clear(points, [island])
        self.assertLess(bp.path_length(points), 8.0)

    def test_turns_lengthwise_before_a_narrow_corridor(self):
        # A 0.62 m corridor: the 0.50 m wide box passes only lengthwise. It starts and ends across it.
        walls = [bp.footprint(box(1.0, 3.0, 0.31, 3.0)), bp.footprint(box(1.0, 3.0, -3.0, -0.31))]
        start, goal = (0.0, 0.0, math.pi / 2), (4.2, 0.0, math.pi / 2)
        grid = bp.NavGrid(walls, extra_points=[start[:2], goal[:2]])
        points = grid.plan(start, goal)
        self.assertIsNotNone(points)
        self.assert_clear(points, walls)
        inside = [p for p in points if 1.0 < p[0] < 3.0]
        self.assertTrue(inside)
        self.assertTrue(all(abs(math.sin(p[2])) < 0.2 for p in inside))
        self.assertGreater(bp.path_turn(points), math.pi - 0.1)

    def test_leaves_a_counter_the_robot_is_parked_against(self):
        counter = bp.footprint(box(-1.0, 1.0, 0.0, 0.6))
        start = (0.0, -0.17, math.pi / 2)          # facing the counter, the box front 2 cm from it
        goal = (2.5, -1.5, 0.0)
        self.assertAlmostEqual(exact_gap(start, counter), 0.02)
        grid = bp.NavGrid([counter], extra_points=[start[:2], goal[:2]])
        points = grid.plan(start, goal)
        self.assertIsNotNone(points)
        self.assert_clear(points, [counter], skip_ends=True)

    def test_a_start_beside_an_open_door_never_crosses_it(self):
        # Parked at a cabinet with its door open along x = 0.29..0.31, the box 1.5 cm left of it.
        cabinet = bp.footprint(box(-1.5, 1.5, 0.0, 0.6))
        door = bp.footprint(box(0.29, 0.31, -0.6, 0.0))
        start = (0.025, -0.2, math.pi / 2)        # box x in [-0.225, 0.275], front 5 cm from the cabinet
        goal = (2.0, -1.8, 0.0)
        grid = bp.NavGrid([cabinet, door], extra_points=[start[:2], goal[:2]], thin=[False, True])
        self.assertLess(grid.clearance(start), bp.HARD_GAP)
        points = grid.plan(start, goal)
        self.assertIsNotNone(points)
        for pose in swept_poses(points):
            self.assertFalse(bp.polygons_overlap(SHAPE.corners(pose), door), pose)

    def test_no_path_out_of_a_closed_room(self):
        walls = [bp.footprint(box(-2, 2, 2, 2.2)), bp.footprint(box(-2, 2, -2.2, -2)),
                 bp.footprint(box(-2.2, -2, -2.2, 2.2)), bp.footprint(box(2, 2.2, -2.2, 2.2))]
        grid = bp.NavGrid(walls, extra_points=[(0, 0), (4, 0)])
        self.assertIsNone(grid.plan((0.0, 0.0, 0.0), (4.0, 0.0, 0.0)))

    def test_follower_aims_ahead_as_a_controller_pose(self):
        points = [(x, 0.0, 0.0) for x in np.linspace(0, 2, 41)]
        f = bp.PathFollower(points, SHAPE, lookahead=0.3)
        aim = f.target(SHAPE.to_origin((0.0, 0.1, 0.0)))
        np.testing.assert_allclose(aim, SHAPE.to_origin((0.3, 0.0, 0.0)), atol=1e-9)
        f.target(SHAPE.to_origin((1.0, 0.0, 0.0)))
        np.testing.assert_allclose(f.target(SHAPE.to_origin((0.0, 0.0, 0.0))), SHAPE.to_origin((1.3, 0.0, 0.0)),
                                   atol=1e-9)   # never goes back
        f.target(SHAPE.to_origin((1.98, 0.0, 0.0)))
        self.assertTrue(f.at_end())

    def test_follower_turns_in_place_where_the_path_does(self):
        points = [(0.0, 0.0, k * math.pi / 8) for k in range(5)] + [(x, 0.0, math.pi / 2) for x in (0.05, 0.1)]
        f = bp.PathFollower(points, SHAPE, lookahead=0.1)
        aim = f.target(SHAPE.to_origin(points[0]))
        self.assertEqual(SHAPE.to_pivot(aim)[:2], (0.0, 0.0))
        self.assertGreater(aim[2], 0.0)


if __name__ == "__main__":
    unittest.main()
