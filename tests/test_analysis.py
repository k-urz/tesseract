from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from tesseract.analysis import Envelope, analyze, export_analysis, telescope_requirements
from tesseract.config import Config, load_config
from tesseract.geometry import hypercube


class ConfigTests(unittest.TestCase):
    def test_file_and_builtin_defaults_match(self):
        self.assertEqual(load_config(Path("config/default.toml")).digest(), Config().digest())

    def test_invalid_config(self):
        base=Config()
        for name,value in [("fine_step_deg",0),("size_mm",float('nan')),("plane","xx"),("size_mm","large"),("candidate_end_deg",-30),("coarse_step_deg",1e-8)]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                replace(base,model=replace(base.model,**{name:value})).validate()
        with self.assertRaises(ValueError):
            replace(base,hardware=replace(base.hardware,tube_wall_mm=10)).validate()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"bad.toml"; p.write_text('[model]\nunknown=3\n')
            with self.assertRaisesRegex(ValueError,'unknown'):
                load_config(p)


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c=Config(); cls.config=replace(c,model=replace(c.model,coarse_step_deg=5,fine_step_deg=1))
        cls.study=analyze(cls.config)

    def test_fixed_scale_and_size_budget(self):
        for prefix,size in [("prototype",300),("target",150)]:
            scans=[self.study.scans[f'{prefix}_{n}'] for n in ('full','candidate','candidate_coarse')]
            self.assertEqual(len({s['summary']['scale_mm_per_unit'] for s in scans}),1)
            for s in scans:
                self.assertLessEqual(s['summary']['max_sampled_envelope_mm'],size)

    def test_smaller_model_keeps_real_hardware(self):
        a=self.study.scans['prototype_candidate']; b=self.study.scans['target_candidate']
        self.assertLess(b['summary']['min_clearance_mm'],a['summary']['min_clearance_mm'])
        self.assertGreater(sum(r['two_stage_fits'] for r in telescope_requirements(self.study,'prototype_candidate')),
                           sum(r['two_stage_fits'] for r in telescope_requirements(self.study,'target_candidate')))

    def test_counts_and_per_edge_requirements(self):
        scan=self.study.scans['prototype_candidate']
        self.assertEqual(len(scan['edges']),32); self.assertEqual(len(scan['corners']),96)
        self.assertEqual(scan['summary']['samples'],31)
        rows=telescope_requirements(self.study,'prototype_candidate')
        for row in rows:
            self.assertAlmostEqual(row['stroke_mm'],row['max_mm']-row['min_mm'])
            if row['two_stage_fits']:
                self.assertLessEqual(row['required_stroke_mm'],row['capacity_mm']+1e-8)

    def test_shared_vertex_does_not_skip_outside_hub_collision(self):
        v,e=hypercube(); points=v[:,:3]*100
        points[0]=[0,0,0]; points[1]=[100,0,0]; points[2]=[100,.1,0]
        result=Envelope(e).check(points,5,2)
        incident_edges=[i for i,ends in enumerate(e) if set(ends)=={0,1} or set(ends)=={0,2}]
        self.assertTrue(all(i in result['colliding_edges'] for i in incident_edges))

    def test_singular_pose_is_reported(self):
        _,e=hypercube(); p=np.zeros((16,3)); p[0]=np.nan
        result=Envelope(e).check(p,18,7)
        self.assertTrue(result['projection_invalid']); self.assertTrue(result['blocked'])
        self.assertIsNone(result['minimum_clearance_mm'])

    def test_orthographic_scan_reports_degeneracy_in_strict_json(self):
        c=replace(self.config,model=replace(self.config.model,projection='orthographic'))
        study=analyze(c)
        self.assertGreater(study.scans['prototype_full']['summary']['zero_edge_samples'],0)
        with tempfile.TemporaryDirectory() as tmp:
            export_analysis(study,Path(tmp))
            text=(Path(tmp)/'analysis.json').read_text()
            self.assertNotIn('NaN',text); self.assertNotIn('Infinity',text)
            result=json.loads(text)
            self.assertEqual(result['config_id'],c.digest())
            self.assertTrue((Path(tmp)/'prototype_candidate_edges.csv').exists())


if __name__ == "__main__":
    unittest.main()
