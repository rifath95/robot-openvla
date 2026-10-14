"""Local replay links and safe restoration of downloaded archives."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from common.paths import PROJECT_ROOT
from analysis.replay_launcher import Launcher,configure
from analysis.multi_color_report import checkpoint_summary
from cloud.sweep_plan import trial_specs
from data_tools.restore_sweep_archive import restore
from simulation.multi_color_scene import positions


class ReplayButtonTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=PROJECT_ROOT/'outputs');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'source';self.dest=self.root/'restored'
        (self.source/'reports').mkdir(parents=True);(self.source/'archives').mkdir()
        plan=dict(task='multi_color_touch',sampling=False,steps=1,training_session='panda-train-test',
                  checkpoints=[50],scene_seeds=[2000],cube_counts=[4],training_colors=['red','blue'],
                  color_sets=[dict(name='test',colors=['red','blue','green','yellow'])],
                  multi_cube_positions={'4':[positions(2000,4)]})
        self.rows=[dict(s,outcome='no_contact',phase='Completed') for s in trial_specs(plan)]
        (self.source/'reports/results.json').write_text(json.dumps(dict(plan=plan,trials=self.rows)))
        self.archive=self.source/'archives/part.tar.gz'
        self.pack()

    def pack(self,bad=None):
        with tarfile.open(self.archive,'w:gz') as archive:
            for row in self.rows:
                info=tarfile.TarInfo(row['directory']+'/summary.json');info.size=2
                archive.addfile(info,io.BytesIO(b'{}'))
            if bad:
                info=tarfile.TarInfo(bad);info.size=2;archive.addfile(info,io.BytesIO(b'{}'))
        entries=[]
        for path in (self.archive,self.source/'reports/results.json'):
            entries.append(dict(path=str(path.relative_to(self.source)),bytes=path.stat().st_size,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (self.source/'manifest.json').write_text(json.dumps(dict(sweep='scripted_test',trials=4,files=entries)))

    def test_restore_and_buttons(self):
        restore(self.source,self.dest)
        restore(self.source,self.dest)
        allowed=configure(self.dest,'http://127.0.0.1:8765','test-token')
        self.assertEqual(len(allowed),4)
        report=(self.dest/'REPORT.md').read_text()
        self.assertEqual(report.count('[▶ Replay]'),4)
        self.assertIn('| Details | Replay |',report)
        self.assertIn('token=test-token',report)

    def test_checksum_rejects_corrupted_download(self):
        data=self.archive.read_bytes();self.archive.write_bytes(bytes([data[0]^1])+data[1:])
        with self.assertRaisesRegex(ValueError,'Checksum mismatch'):restore(self.source,self.dest)

    def test_archive_traversal_rejected(self):
        self.pack('../escape.json')
        with self.assertRaisesRegex(ValueError,'Unsafe archive'):restore(self.source,self.dest)
        self.assertFalse((self.root/'escape.json').exists())

    def test_report_symlink_rejected(self):
        self.dest.mkdir();(self.dest/'results.json').symlink_to(self.root/'outside.json')
        with self.assertRaisesRegex(ValueError,'Unsafe report'):restore(self.source,self.dest)
        self.assertFalse((self.root/'outside.json').exists())

    def test_token_known_trial_and_only_one_window(self):
        trial=self.rows[0]['directory'];launcher=Launcher(self.dest,{trial},'secret')
        with patch('analysis.replay_launcher.subprocess.Popen') as popen:
            self.assertEqual(launcher.launch(trial,'wrong')[0],403)
            self.assertEqual(launcher.launch('../escape','secret')[0],404)
            popen.assert_not_called()
            self.assertEqual(launcher.launch(trial,'secret')[0],200)
            self.assertEqual(popen.call_args.args[0][:2],[str(PROJECT_ROOT/'.venv/bin/python'),str(PROJECT_ROOT/'.venv/bin/mjpython')])
            self.assertIn('--start-immediately',popen.call_args.args[0])
            popen.return_value.poll.return_value=None
            self.assertEqual(launcher.launch(trial,'secret')[0],409)
            self.assertEqual(popen.call_count,1)

    def test_summary_denominators_and_unstarted_checkpoints(self):
        plan=dict(checkpoints=[0,50,100],include_base_model=True,training_colors=['red'])
        rows=[dict(checkpoint=50,target_color=color,outcome=outcome) for color,outcome in
              [('red','correct_cube'),('blue','correct_cube'),('blue','wrong_cube'),
               ('blue','no_contact'),('blue','multiple_cubes'),('blue','unmeasured')]]
        text='\n'.join(checkpoint_summary(plan,rows))
        self.assertIn('| 50 | 33.3% (2/6) | 100.0% (1/1) | 20.0% (1/5) | 50.0% (1/2) |',text)
        self.assertIn('| Pretrained base | — | — | — | — |',text)
        self.assertIn('| 100 | — | — | — | — |',text)
