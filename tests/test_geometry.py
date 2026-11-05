import unittest
import numpy as np

from tesseract.geometry import (corner_angles, corner_pairs, hypercube,
    point_segment_distance, project, rotate, samples, segment_distance)


class GeometryTests(unittest.TestCase):
    def test_topology_and_bit_identity(self):
        v, e = hypercube()
        self.assertEqual(v.shape, (16, 4))
        self.assertEqual(e.shape, (32, 2))
        np.testing.assert_array_equal(np.bincount(e.ravel()), np.full(16, 4))
        np.testing.assert_array_equal(v[0], [-1]*4)
        np.testing.assert_array_equal(v[8], [-1, -1, -1, 1])
        np.testing.assert_allclose(np.linalg.norm(v[e[:, 1]]-v[e[:, 0]], axis=1), 2)
        cv, ce = hypercube(3)
        self.assertEqual(cv.shape, (8, 3)); self.assertEqual(ce.shape, (12, 2))

    def test_all_rotation_planes_preserve_distances_and_period(self):
        v, _ = hypercube()
        distance = np.linalg.norm(v[:, None]-v[None, :], axis=-1)
        for plane in ["xy", "xz", "xw", "yz", "yw", "zw"]:
            for angle in [0, 17, 90, 137, 360]:
                r = rotate(v, angle, plane)
                np.testing.assert_allclose(np.linalg.norm(r[:, None]-r[None, :], axis=-1), distance, atol=1e-12)
            np.testing.assert_allclose(rotate(v, 360, plane), v, atol=1e-12)

    def test_known_perspective_pose(self):
        v, _ = hypercube()
        p = project(v)
        np.testing.assert_allclose(p[0], [-.8]*3)
        np.testing.assert_allclose(p[8], [-4/3]*3)
        np.testing.assert_allclose(project(rotate(v, 90))[0], [.8, -.8, -.8], atol=1e-12)

    def test_eye_plane_and_zero_edge(self):
        v, e = hypercube()
        self.assertFalse(np.isfinite(project(v, distance=1)).all())
        p = project(v, "orthographic")
        lengths = np.linalg.norm(p[e[:, 0]]-p[e[:, 1]], axis=1)
        self.assertEqual(np.count_nonzero(lengths == 0), 8)
        self.assertTrue(np.isnan(corner_angles(p, corner_pairs(e))).any())

    def test_endpoint_sampling(self):
        np.testing.assert_allclose(samples(-.3, .3, .1), [-.3,-.2,-.1,0,.1,.2,.3], atol=1e-12)
        np.testing.assert_allclose(samples(0, 1, .3), [0,.3,.6,.9,1])

    def test_segment_distances(self):
        cases = [
            ((0,0,0),(1,0,0),(.5,-1,0),(.5,1,0),0),
            ((0,0,0),(1,0,0),(0,2,0),(1,2,0),2),
            ((0,0,0),(1,0,0),(.5,0,0),(2,0,0),0),
            ((0,0,0),(1,0,0),(2,0,0),(3,0,0),1),
            ((0,0,0),(1,0,0),(.5,-1,3),(.5,1,3),3),
            ((0,0,0),(0,0,0),(1,-1,0),(1,1,0),1),
            ((0,0,0),(0,0,0),(0,0,2),(0,0,2),2),
            ((0,0,0),(100,0,0),(0,-1e-6,0),(100,1e-6,0),0),
        ]
        for a,b,c,d,expected in cases:
            with self.subTest(case=(a,b,c,d)):
                self.assertAlmostEqual(float(segment_distance(a,b,c,d)), expected, places=9)
                self.assertAlmostEqual(float(segment_distance(c,d,a,b)), expected, places=9)

    def test_vectorized_distance_matches_scalar(self):
        rng=np.random.default_rng(121)
        a,b,c,d=rng.normal(size=(4,50,3))
        result=segment_distance(a,b,c,d)
        for i in range(50):
            self.assertAlmostEqual(float(result[i]), float(segment_distance(a[i],b[i],c[i],d[i])),places=12)
        self.assertAlmostEqual(float(point_segment_distance((2,1,0),(0,0,0),(1,0,0))),np.sqrt(2))


if __name__ == "__main__":
    unittest.main()
