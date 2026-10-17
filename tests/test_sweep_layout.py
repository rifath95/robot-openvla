"""Recording moves preserve data, reports, resume identities and old commands."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit
from common import output_layout
from common.sweep_layout import trial_directory, validate_trial_directory
from cloud.sweep_plan import trial_specs, validate_saved
from data_tools.organize_sweep_trials import organize, write_json


def fixture(root):
    folder=root/'sweeps/scripted_test';folder.mkdir(parents=True)
    plan=dict(task='touch',sampling=False,steps=50,training_session='panda-train-test',
              checkpoints=[50],scene_seeds=[1000,1001])
    rows=[dict(spec,phase='Completed',completed_steps=1,contact=True) for spec in trial_specs(plan)]
    for row in rows:
        trial=folder/row['directory'];(trial/'step_001').mkdir(parents=True)
        (trial/'summary.json').write_text('{"phase":"Completed","completed_steps":1}')
        (trial/'step_001/scene_state.npz').write_bytes(b'unchanged recording')
        (trial/'RUN_REPORT.md').write_text(f'[▶ Replay](http://127.0.0.1:8765/replay?run=outputs%2Fsweeps%2Fscripted_test%2F{trial.name}&mode=poses&token=test)\n')
    payload=dict(plan=plan,trials=rows,launcher_error=None)
    write_json(folder/'results.json',payload)
    (folder/'REPORT.md').write_text('Custom analysis stays.\n'+''.join(
        f'[Run]({r["directory"]}/RUN_REPORT.md)\n' for r in rows))
    return folder,payload


class SweepLayoutTests(unittest.TestCase):
    def test_move_preserves_inodes_results_and_custom_report_and_old_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,original=fixture(root)
            old=folder/original['trials'][0]['directory']/'step_001/scene_state.npz'
            inode=old.stat().st_ino
            self.assertEqual(organize(folder),2)
            self.assertTrue(old.exists())
            self.assertEqual(organize(folder,apply=True),2)
            saved=json.loads((folder/'results.json').read_text());validate_saved(saved)
            self.assertEqual(saved['trials'][0]['contact'],original['trials'][0]['contact'])
            self.assertEqual(len([s for s in trial_specs(saved['plan']) if s['directory'] not in {r['directory'] for r in saved['trials']}]),0)
            new=folder/'trials'/original['trials'][0]['directory']/'step_001/scene_state.npz'
            self.assertEqual(new.stat().st_ino,inode)
            self.assertEqual(new.read_bytes(),b'unchanged recording')
            with patch.object(output_layout,'OUTPUTS_ROOT',root):
                self.assertEqual(output_layout.resolve_output_path(old),new)
                self.assertEqual(output_layout.resolve_output_path(root/'scripted_test'/old.relative_to(folder)),new)
            report=(folder/'REPORT.md').read_text()
            self.assertTrue(report.startswith('Custom analysis stays.'))
            self.assertIn('[Run](trials/',report)
            run=(new.parent.parent/'RUN_REPORT.md').read_text().split('](')[1].rstrip(')\n')
            self.assertIn('/trials/',parse_qs(urlsplit(run).query)['run'][0])
            self.assertEqual(organize(folder,apply=True),0)
            self.assertEqual((folder/'REPORT.md').read_text(),report)
            backup=json.loads((folder/'.trial_layout_migration.json').read_text())
            self.assertEqual(backup['original_results'],original)

    def test_collision_rejected_before_any_moves(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,payload=fixture(Path(tmp));name=payload['trials'][1]['directory']
            (folder/'trials'/name).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'already exists'):organize(folder,apply=True)
            self.assertTrue((folder/payload['trials'][0]['directory']).is_dir())

    def test_interrupted_move_can_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,payload=fixture(Path(tmp));names=[r['directory'] for r in payload['trials']]
            write_json(folder/'.trial_layout_migration.json',dict(original_results=payload,trial_names=names,complete=False))
            (folder/'trials').mkdir();(folder/names[0]).rename(folder/'trials'/names[0])
            self.assertEqual(organize(folder,apply=True),1)
            validate_saved(json.loads((folder/'results.json').read_text()))
            self.assertTrue(json.loads((folder/'.trial_layout_migration.json').read_text())['complete'])

    def test_unsafe_or_unknown_layout_rejected(self):
        for name in ('../bad','trials/../bad','/absolute','elsewhere/trial','trials','trials/a/b'):
            with self.assertRaises(ValueError):validate_trial_directory(name)
        with self.assertRaises(ValueError):trial_directory({'trial_directory':'../bad'},'checkpoint_000050')
        self.assertEqual(trial_directory({},'checkpoint_000050'),'checkpoint_000050')
        self.assertEqual(trial_directory({'trial_directory':'trials'},'checkpoint_000050'),'trials/checkpoint_000050')


if __name__=='__main__':unittest.main()
