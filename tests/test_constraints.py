from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tesseract.analysis import analyze
from tesseract.config import Config
from tesseract.constraints import (SIGNS, constraint_study, edge_groups,
    export_constraints, independent_angles, intrinsic_rotation_dof, jacobians,
    numerical_jacobian, rank, rigid_basis, square_constraints,
    square_coordinates, square_vertices)
from tesseract.geometry import corner_angles, corner_pairs, hypercube, project, rotate


class ConstraintGeometryTests(unittest.TestCase):
    def setUp(self):
        self.v, self.e = hypercube()
        self.corners = corner_pairs(self.e)

    def test_four_squares_exactly_reconstruct_projected_vertices(self):
        for distance in (2, 4, 10):
            for angle in (-180, -15, 0, 45, 90, 179, 270, 360):
                with self.subTest(distance=distance, angle=angle):
                    actual = square_vertices(square_coordinates(angle, distance, 87))
                    expected = 87*project(rotate(self.v, angle), distance=distance)
                    np.testing.assert_allclose(actual, expected, atol=1e-11)

    def test_conic_and_phase_equations_hold_through_phase_wrap(self):
        for angle in (0, 45, 179.999, 180, 270, 359.999):
            q = square_coordinates(angle, scale=60)
            residual, branch = square_constraints(q, scale=60)
            np.testing.assert_allclose(residual, 0, atol=1e-12)
            np.testing.assert_allclose(branch, 1, atol=1e-12)

    def test_opposite_phase_requires_branch_inequality(self):
        q = square_coordinates(17)
        q[1] = square_coordinates(197)[1]
        residual, branch = square_constraints(q)
        np.testing.assert_allclose(residual, 0, atol=1e-12)
        self.assertLess(branch[0], 0)

    def test_direct_ellipse_equation(self):
        distance, scale = 4, 80
        for angle in (0, 13, 57, 129, 270):
            x, b = square_coordinates(angle, distance, scale).T
            np.testing.assert_allclose(x*x + distance**2*(b-scale)**2, 2*b*b, atol=1e-9)

    def test_eight_symbolic_length_families_cover_all_edges(self):
        groups = edge_groups(self.v, self.e)
        self.assertEqual(len(groups), 8)
        self.assertEqual(sorted(i for ids in groups.values() for i in ids), list(range(32)))
        for angle in (-15, 0, 17, 45, 131):
            p = project(rotate(self.v, angle))
            lengths = np.linalg.norm(p[self.e[:, 1]]-p[self.e[:, 0]], axis=1)
            for ids in groups.values():
                self.assertEqual(len(ids), 4)
                np.testing.assert_allclose(lengths[ids], lengths[ids[0]], atol=1e-12)

    def test_connector_revolute_axes_are_exact_on_fixed_xw_path(self):
        for angle in (0, 7, 45, 123):
            p = square_vertices(square_coordinates(angle))
            for a, b in self.e:
                if (int(a) ^ int(b)) not in (1, 8):
                    continue
                # Hinge normal to the common X / yz-diagonal plane.
                normal = np.array((0, self.v[a, 2], -self.v[a, 1]))
                self.assertAlmostEqual(float(normal @ (p[b]-p[a])), 0, places=12)

    def test_analytic_length_and_cosine_jacobians(self):
        p = project(rotate(self.v, 7)) + np.random.default_rng(42).normal(size=(16, 3))*.01
        jl, jc = jacobians(p, self.e, self.corners)
        numeric_l = numerical_jacobian(lambda q: np.linalg.norm(
            q.reshape(16, 3)[self.e[:, 1]]-q.reshape(16, 3)[self.e[:, 0]], axis=1), p)
        numeric_c = numerical_jacobian(lambda q: np.cos(np.deg2rad(
            corner_angles(q.reshape(16, 3), self.corners))), p)
        np.testing.assert_allclose(jl, numeric_l, atol=1e-8)
        np.testing.assert_allclose(jc, numeric_c, atol=1e-8)

    def test_jacobians_ignore_global_rigid_motion(self):
        p = project(rotate(self.v, 13))
        jl, jc = jacobians(p, self.e, self.corners)
        np.testing.assert_allclose(np.vstack((jl, jc)) @ rigid_basis(p), 0, atol=1e-12)

    def test_singular_bar_framework_is_not_generic_bar_count(self):
        p = project(rotate(self.v, 7))
        self.assertEqual(rank(jacobians(p, self.e, self.corners)[0]), 31)
        perturbed = p+np.random.default_rng(42).normal(size=p.shape)*.02
        self.assertEqual(rank(jacobians(perturbed, self.e, self.corners)[0]), 32)

    def test_local_angle_complement_has_eleven_independent_rows(self):
        jl, jc = jacobians(project(self.v), self.e, self.corners)
        selected = independent_angles(jl, jc)
        self.assertEqual(len(selected), 11)
        for angle in (-15, 0, 7, 45, 91, 180):
            jl, jc = jacobians(project(rotate(self.v, angle)), self.e, self.corners)
            self.assertEqual(rank(np.vstack((jl, jc[selected]))), 42)

    def test_four_square_constraints_leave_exactly_one_tangent(self):
        for angle in (-15, 0, 45, 179, 270):
            q = square_coordinates(angle)
            j = numerical_jacobian(lambda q: square_constraints(q)[0], q)
            tangent = (square_coordinates(angle+1e-4)-square_coordinates(angle-1e-4)).ravel()/np.deg2rad(2e-4)
            self.assertEqual(rank(j), 7)
            np.testing.assert_allclose(j @ tangent, 0, atol=1e-8)

    def test_general_rotation_has_three_intrinsic_shape_directions(self):
        self.assertEqual(intrinsic_rotation_dof(self.v, 4), 3)

    def test_invalid_reduced_projection_is_rejected(self):
        for distance, scale in ((1, 1), (4, 0)):
            with self.assertRaises(ValueError):
                square_coordinates(0, distance, scale)


class ConstraintStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c = Config()
        cls.study = analyze(replace(c, model=replace(c.model, coarse_step_deg=10, fine_step_deg=1)))
        cls.result = constraint_study(cls.study)

    def test_local_mobility_models_and_rank_stability(self):
        for row in self.result['mobility_samples']:
            self.assertEqual(row['equal_family_internal_infinitesimal_dof'], 18)
            self.assertEqual(row['equal_family_and_yz_internal_infinitesimal_dof'], 15)
            self.assertEqual(row['locked_lengths_internal_infinitesimal_dof'], 11)
            self.assertEqual(row['scheduled_internal_infinitesimal_dof'], 1)
            self.assertEqual(row['four_square_infinitesimal_dof'], 1)

    def test_range_only_counterexample_is_inside_limits_but_off_path(self):
        witness = self.result['range_counterexample']
        self.assertTrue(witness['available'])
        self.assertTrue(witness['all_length_limits_satisfied'])
        self.assertLess(witness['max_corner_angle_change_deg'], 1e-10)
        self.assertGreater(witness['fixed_scale_path_constraint_error'], .001)

    def test_input_reversals_are_distinguished_from_square_widths(self):
        groups = {g['name']: g for g in self.result['length_families']}
        self.assertEqual(groups['x_link_w-1']['slope_nearest_zero_mm_per_rad'], 0)
        self.assertFalse(groups['x_link_w-1']['strictly_monotone_in_sampled_candidate'])
        self.assertFalse(groups['w_link_x+1']['strictly_monotone_in_sampled_candidate'])
        for name, group in groups.items():
            if name.startswith('square'):
                self.assertTrue(group['strictly_monotone_in_sampled_candidate'])

    def test_cam_tables_use_the_same_fixed_scale_as_original_study(self):
        for row in self.result['cam_profiles']:
            scan = self.study.scans[row['size']+'_candidate']
            i = int(np.argmin(abs(scan['angles_deg']-row['angle_deg'])))
            v = next(j for j, p in enumerate(self.study.vertices)
                     if p[0] == row['source_x'] and p[3] == row['source_w'])
            self.assertAlmostEqual(scan['points_mm'][i, v, 0], row['center_x_mm'])
            self.assertAlmostEqual(abs(scan['points_mm'][i, v, 1]), row['half_width_mm'])

    def test_export_has_traceable_metadata_tables_and_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            export_constraints(self.study, out)
            metadata = json.loads((out/'constraints.json').read_text())
            self.assertEqual(metadata['config_id'], self.study.config.digest())
            self.assertEqual(metadata['mobility_summary']['prototype_candidate']['four_square_infinitesimal_dof'], {'min': 1, 'max': 1})
            for name in ('mobility.csv', 'cam_profiles.csv', 'constraints.png', 'frame-layout.png', 'report.md'):
                self.assertGreater((out/name).stat().st_size, 50)

    def test_unsupported_plane_is_not_silently_treated_as_xw(self):
        c = self.study.config
        other = analyze(replace(c, model=replace(c.model, plane='yw')))
        with self.assertRaisesRegex(ValueError, 'xw perspective'):
            constraint_study(other)
