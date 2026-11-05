"""A manufacturing pack must never mix CAD with stale visualizations."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.package_example import validate_outputs


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        (self.output / 'cad').mkdir()
        for name in ('analysis.json', 'cad/manifest.json'):
            (self.output / name).write_text(json.dumps({'config_id': 'current'}))
        hashes = {}
        for name in ('viewer.html', 'requirements.png', 'motion.png'):
            content = ('test artifact: ' + name).encode()
            (self.output / name).write_bytes(content)
            hashes[name] = hashlib.sha256(content).hexdigest()
        (self.output / 'render_manifest.json').write_text(
            json.dumps({'config_id': 'current', 'sha256': hashes}))

    def test_consistent_pack_is_accepted(self):
        validate_outputs(self.output, 'current')

    def test_different_analysis_cad_or_render_config_is_rejected(self):
        for name in ('analysis.json', 'cad/manifest.json', 'render_manifest.json'):
            with self.subTest(name=name):
                path = self.output / name
                original = path.read_text()
                data = json.loads(original)
                data['config_id'] = 'old'
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
                    validate_outputs(self.output, 'current')
                path.write_text(original)

    def test_stale_or_missing_render_is_rejected(self):
        for name in ('viewer.html', 'requirements.png', 'motion.png'):
            with self.subTest(name=name):
                path = self.output / name
                original = path.read_bytes()
                path.write_bytes(b'stale')
                with self.assertRaisesRegex(ValueError, 'stale or changed'):
                    validate_outputs(self.output, 'current')
                path.unlink()
                with self.assertRaisesRegex(ValueError, 'stale or changed'):
                    validate_outputs(self.output, 'current')
                path.write_bytes(original)

    def test_missing_render_manifest_is_rejected(self):
        (self.output / 'render_manifest.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing output'):
            validate_outputs(self.output, 'current')
