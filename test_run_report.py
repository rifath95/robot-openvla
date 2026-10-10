"""Verify timing aggregation and honest reporting of missing startup data."""
import json
from pathlib import Path
import tempfile
import unittest
from run_report import write_run_report

class ReportTests(unittest.TestCase):
    def test_finalized_cycles_include_rejected_hold_and_no_double_count(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for index, (request, move) in enumerate([(1., .4), (2., 0.)], 1):
                step = folder / f'step_{index:03d}'; step.mkdir()
                row = {'executed': move > 0, 'action_scale': 1 if move else None,
                       'timings_seconds': {'capture_and_state_save': .1, 'ik': .01,
                                           'robot_execution_wall': move, 'next_capture_and_state_save': .1,
                                           'cycle': request + move + .26},
                       'request_timings_seconds': {'request_round_trip': request},
                       'server_timings_seconds': {'server_processing': .5}}
                (step/'execution.json').write_text(json.dumps(row))
            (folder/'startup_info.json').write_text(json.dumps({'model_load_seconds': 64.26,
                'volume_mount': {'filesystems': [{'fstype':'nfs4','source':'host:/networkvolumes/test'}]},
                'gpu': {'name':'NVIDIA A40', 'total_vram_bytes':48*1024**3}}))
            text = write_run_report(folder).read_text()
            self.assertIn('| Complete cycle | 1.9600 s |', text)
            self.assertIn('| Local simulation and recording | 0.4100 s |', text)
            self.assertIn('| Remaining cycle overhead | 0.0500 s |', text)
            self.assertIn('| Remaining network/tunnel/request overhead | 1.0000 s |', text)
            self.assertIn('2 finalized cycle records', text)
            self.assertIn('48.00 GiB', text)
            self.assertIn('Network volume', text)
            self.assertLess(text.index('## Average full cycle'), text.index('## Remote request breakdown'))

    def test_empty_failed_run_does_not_invent_timing_or_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'status.json').write_text('{"phase":"Failed"}')
            text = write_run_report(folder).read_text()
            self.assertIn('| Model loading | Not recorded |', text)
            self.assertIn('| Storage | Not recorded |', text)
            self.assertIn('| Complete cycle | Not recorded |', text)
            self.assertIn('| Result | Failed |', text)

if __name__ == '__main__':
    unittest.main()
