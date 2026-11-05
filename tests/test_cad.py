from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest

from tesseract.analysis import analyze
from tesseract.config import Config


@unittest.skipUnless(importlib.util.find_spec('build123d'), 'optional CAD dependency not installed')
class CadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tesseract.cad import make_parts
        c=Config(); c=replace(c,model=replace(c.model,coarse_step_deg=5,fine_step_deg=1))
        cls.study=analyze(c)
        cls.parts,cls.sample,cls.length,cls.gauge=make_parts(cls.study)

    def test_positive_single_solids(self):
        for name,shape in self.parts.items():
            with self.subTest(name=name):
                self.assertTrue(shape.is_valid); self.assertGreater(shape.volume,0)
                self.assertEqual(len(shape.solids()),1)

    def test_sample_edge_and_gauge_traceability(self):
        self.assertTrue(self.sample['two_stage_fits'])
        self.assertEqual(len(self.gauge['directions']),4)
        self.assertEqual(set(r['edge_id'] for r in self.gauge['directions']),{0,1,2,3})

    def test_fit_parameter_changes_bore_volume(self):
        from tesseract.cad import fit_coupon
        h=self.study.config.hardware
        tight,loose=fit_coupon(h,.1),fit_coupon(h,.5)
        self.assertLess(loose.volume,tight.volume)

    def test_step_round_trip_and_stl_mesh(self):
        import build123d as b
        import struct
        for name in ['telescopic_outer','ball_stem','socket_left','hub_layout_gauge']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                shape=self.parts[name]; path=Path(tmp)/'part.step'
                b.export_step(shape,path)
                restored=b.import_step(path)
                self.assertTrue(restored.is_valid)
                self.assertAlmostEqual(restored.volume,shape.volume,places=4)
                mesh=Path(tmp)/'part.stl'; self.assertTrue(b.export_stl(shape,mesh))
                data=mesh.read_bytes(); triangles=struct.unpack('<I',data[80:84])[0]
                self.assertGreater(triangles,0); self.assertEqual(len(data),84+triangles*50)

    def test_telescope_can_slide_at_both_stops_without_solid_overlap(self):
        import build123d as b
        h=self.study.config.hardware
        outer=self.parts['telescopic_outer']; inner=self.parts['telescopic_inner']
        for extension in [0,self.length-h.end_wall_mm-h.min_overlap_mm]:
            placed=b.Pos(0,0,h.end_wall_mm+extension)*inner
            common=outer & placed
            self.assertLess(common.volume if common is not None else 0,1e-5)

    def test_socket_clearance_and_tilt_limit(self):
        import build123d as b
        housing=self.parts['socket_left']+self.parts['socket_right']
        ball=self.parts['ball_stem']
        common=ball & housing
        self.assertLess(common.volume if common is not None else 0,1e-5)
        self.assertGreater(((b.Rot(0,30,0)*ball)&housing).volume,1)


if __name__ == '__main__':
    unittest.main()
