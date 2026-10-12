"""Check inference timing/cost aggregation independently of a cloud GPU."""
import json
from pathlib import Path
import tempfile
import unittest

from analysis.summarize_gpu_inference import comparison, measurement


class InferenceComparisonTests(unittest.TestCase):
    def make_run(self, root):
        folder=root/'trial'; folder.mkdir()
        (folder/'startup_info.json').write_text(json.dumps(dict(
            gpu=dict(name='NVIDIA A40',total_vram_bytes=48*2**30),
            model_load_seconds=60,model_preparation_seconds=10)))
        for i in range(7):
            step=folder/f'step_{i+1:03d}'; step.mkdir()
            # Warmup is deliberately very slow and must not inflate steady-state cost.
            request=100 if i<5 else 1
            (step/'execution.json').write_text(json.dumps(dict(
                executed=i!=6, request_timings_seconds=dict(request_round_trip=request),
                server_timings_seconds=dict(server_processing=.6,model_inference=.3),
                timings_seconds=dict(cycle=request+.6,capture_and_state_save=.05,ik=.01,
                                     robot_execution_wall=.4,next_capture_and_state_save=.04))))
        return folder

    def test_warmup_exclusion_and_no_double_counting(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=self.make_run(Path(tmp)); m=measurement(folder)
            self.assertEqual(m['measured'],2)
            self.assertEqual(m['count'],7)
            self.assertEqual(m['request'],1)
            self.assertAlmostEqual(m['network'],.4)
            self.assertAlmostEqual(m['local_total'],.5)
            self.assertAlmostEqual(m['overhead'],.1)
            self.assertAlmostEqual(m['cycle'],1.6)
            self.assertEqual(m['p95'],1)

    def test_report_costs_and_missing_gpu_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_run(root)
            output=root/'report.md'
            self.assertEqual(comparison(root,output),1)
            text=output.read_text()
            self.assertIn('160.0 | 0.0262 | 0.0377',text)
            self.assertIn('NVIDIA RTX A5000 | Not measured',text)
            self.assertIn('Not recorded',text)
            self.assertLess(text.index('## Full cycle'),text.index('## Remote request breakdown'))

    def test_short_run_does_not_invent_steady_state_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=self.make_run(Path(tmp))
            m=measurement(folder,warmup=7)
            self.assertEqual(m['measured'],0)
            self.assertIsNone(m['cycle'])
            self.assertIsNone(m['p95'])
            with self.assertRaises(ValueError): comparison(folder,folder/'report.md',warmup=-1)


if __name__=='__main__':
    unittest.main()
