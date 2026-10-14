"""Output migration preserves files and old command paths without aliases."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from common import output_layout as layout
from data_tools import organize_outputs as migration


class OutputLayoutTests(unittest.TestCase):
    def test_categories_and_traversal(self):
        for name,expected in [('scripted_touch_abc','sweeps'),('remote_loop_abc','simulator_runs/cloud'),
                              ('openvla_loop_abc','simulator_runs/local'),('gpu-benchmarks','benchmarks/training'),
                              ('phase3_training','training'),('hf-sweep-archives','archives/hf-sweeps')]:
            self.assertEqual(layout.output_path(name),layout.OUTPUTS_ROOT/expected/name)
        for name in ('../outside','/outside',''):
            with self.assertRaises(ValueError):layout.output_path(name)

    def test_rename_preserves_data_and_legacy_command_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'scripted_touch_abc';source.mkdir();(source/'results.json').write_text('original')
            before=(source/'results.json').stat().st_ino
            with patch.object(layout,'OUTPUTS_ROOT',root),patch.object(migration,'OUTPUTS_ROOT',root):
                migration.organize(apply=True)
                destination=root/'sweeps/scripted_touch_abc'
                self.assertEqual((destination/'results.json').read_text(),'original')
                self.assertEqual((destination/'results.json').stat().st_ino,before)
                self.assertEqual(layout.resolve_output_path(source/'results.json'),destination/'results.json')
                self.assertEqual(migration.migration_plan(),[])

    def test_collision_rejected_before_any_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'scripted_a').mkdir();(root/'sweeps/scripted_a').mkdir(parents=True)
            with patch.object(layout,'OUTPUTS_ROOT',root),patch.object(migration,'OUTPUTS_ROOT',root):
                with self.assertRaisesRegex(ValueError,'Destination already exists'):migration.organize(apply=True)
            self.assertTrue((root/'scripted_a').is_dir())
