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

    def test_an_open_door_blocks_the_straight_way_out(self):
        counter = bp.footprint(box(-1.0, 1.0, 0.0, 0.6))
        door = bp.footprint(bp.geom_corners(bp.GEOM_BOX, [0.02, 0.4, 0.4], [1.0, -0.4, 0.45], np.eye(3).ravel()))
        start, goal = (0.6, -0.45), (3.0, -0.45)
        grid = bp.NavGrid([counter, door], extra_points=[start, goal])
        points = grid.plan(start, goal)
        self.assertIsNotNone(points)
        self.assertLess(min(p[1] for p in points), -0.8 - bp.HARD_CLEARANCE + 1e-9)   # around the door's end


class EscapeTest(unittest.TestCase):
    def test_a_start_beside_a_thin_door_leaves_on_its_own_side(self):
        # The robot boxed in a 17 cm pocket between a neighbouring fixture's side (x <= -0.05) and a
        # door open along x = 0.12..0.16: no clear cell inside the pocket, the nearest clear cell is
        # just across the door, and the way out is down the pocket.
        cabinet = bp.footprint(box(-1.5, 1.5, 0.0, 0.6))
        side = bp.footprint(box(-0.6, -0.05, -0.6, 0.0))
        door = bp.footprint(box(0.12, 0.16, -0.6, 0.0))
        start, goal = (0.035, -0.05), (2.0, -1.5)
        grid = bp.NavGrid([cabinet, side, door], extra_points=[start, goal])
        points = grid.plan(start, goal)
        self.assertIsNotNone(points)
        for a, b in zip(points, points[1:]):
            for s in np.linspace(0, 1, 11):
                p = (1 - s) * np.asarray(a) + s * np.asarray(b)
                self.assertFalse(bp.point_in_polygon(p, door), (a, b))
        self.assertLess(points[1][1], -0.6)   # first clear cell: below the pocket, not across the door


class PlanTest(unittest.TestCase):
    def check_path(self, grid, points, start, goal):
        np.testing.assert_allclose(points[0], start[:2])
        np.testing.assert_allclose(points[-1], goal[:2])
        for p in points[1:-1]:
            self.assertGreaterEqual(grid.clearance(p), bp.HARD_CLEARANCE - 1e-9)

    def test_goes_around_an_island(self):
        start, goal = (0.0, 0.0), (4.0, 0.0)
        grid = bp.NavGrid([ISLAND], extra_points=[start, goal])
        points = grid.plan(start, goal)
        self.check_path(grid, points, start, goal)
        self.assertGreater(max(abs(p[1]) for p in points), 1.5 + bp.HARD_CLEARANCE)   # around, not through
        self.assertLess(bp.path_length(points), 7.0)

    def test_leaves_a_counter_the_robot_starts_against(self):
        counter = bp.footprint(box(-1.0, 1.0, 0.0, 0.6))
        start, goal = (0.0, -0.05), (3.0, -1.0)     # base centre 5 cm from the counter front
        grid = bp.NavGrid([counter], extra_points=[start, goal])
        self.assertLess(grid.clearance(start), bp.HARD_CLEARANCE)
        points = grid.plan(start, goal)
        self.assertIsNotNone(points)
        self.check_path(grid, points, start, goal)

    def test_no_path_through_a_closed_room(self):
        walls = [bp.footprint(box(-2, 2, 2, 2.2)), bp.footprint(box(-2, 2, -2.2, -2)),
                 bp.footprint(box(-2.2, -2, -2.2, 2.2)), bp.footprint(box(2, 2.2, -2.2, 2.2))]
        grid = bp.NavGrid(walls, extra_points=[(0, 0), (4, 0)])
        self.assertIsNone(grid.plan((0, 0), (4, 0)))

    def test_follower_moves_forward_along_the_path(self):
        points = [np.array([x, 0.0]) for x in np.linspace(0, 2, 41)]
        f = bp.PathFollower(points, lookahead=0.3)
        np.testing.assert_allclose(f.target([0.0, 0.1]), [0.3, 0.0], atol=1e-9)
        np.testing.assert_allclose(f.target([1.0, 0.0]), [1.3, 0.0], atol=1e-9)
        np.testing.assert_allclose(f.target([0.0, 0.0]), [1.3, 0.0], atol=1e-9)   # never goes back
        f.target([1.98, 0.0])
        self.assertTrue(f.at_end())


if __name__ == "__main__":
    unittest.main()
