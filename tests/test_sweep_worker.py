"""Cloud worker preserves independent trial state/output and skips measured trials."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch,MagicMock
from cloud import sweep_worker as worker
from cloud.sweep_plan import trial_specs,timeout_budget

class WorkerTests(unittest.TestCase):
    def test_resume_dispatch_and_persistent_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'sweep';directory.mkdir()
            plan=dict(task='color_transfer',checkpoints=[100,300],scene_seeds=[1000],color_pairs=[['black','purple']],
                steps=50,sampling=False,cube_positions=[{'red':[.4,.1],'blue':[.6,-.1]}],training_session='panda-train-test',timeout_budget=timeout_budget(100))
            specs=list(trial_specs(plan))
            saved=[dict(specs[0],phase='Failed',completed_steps=10,outcome='no_contact',contact=False)]
            (directory/'results.json').write_text(json.dumps(dict(plan=plan,trials=saved)))
            calls=[]
            def run(command,**kwargs):
                if 'inference.remote_loop' in command:
                    calls.append(command)
                    trial=Path(command[command.index('--output-dir')+1]);trial.mkdir(parents=True)
                    target=command[command.index('--target-color')+1]
                    (trial/'summary.json').write_text(json.dumps(dict(phase='Completed',completed_steps=12,
                        task_result=dict(correct_cube_contact=True,wrong_cube_contact=False,outcome='correct_cube',first_contact=target))))
                    (trial/'before.png').write_bytes(b'pixels')
                return SimpleNamespace(returncode=0)
            server=MagicMock();server.poll.return_value=0
            with patch.object(worker,'PROJECT_ROOT',root),patch.object(worker,'stage_weights',return_value=root/'staged') as stage, \
                patch.object(worker.subprocess,'run',side_effect=run),patch.object(worker.subprocess,'Popen',return_value=server), \
                patch.object(worker.subprocess,'check_output',return_value='commit'), \
                patch.object(worker,'wait_server',return_value=dict(startup_info={})), \
                patch.object(worker,'write_reports',side_effect=lambda folder,plan,rows,error=None:(folder/'results.json').write_text(json.dumps(dict(plan=plan,trials=rows)))):
                worker.run(directory,'/sim/python')
            self.assertEqual(len(calls),3)
            self.assertEqual(calls[0][calls[0].index('--target-color')+1],'purple')
            self.assertTrue(all('--no-view' in c and '--start-immediately' in c for c in calls))
            for command in calls:
                coordinates=Path(command[command.index('--scene-positions')+1])
                self.assertEqual(json.loads(coordinates.read_text()),plan['cube_positions'][0])
            result=json.loads((directory/'results.json').read_text())
            self.assertEqual(len(result['trials']),4);self.assertEqual(result['trials'][0],saved[0])
            self.assertTrue(all(r['execution_mode']=='cloud_sequential' for r in result['trials'][1:]))
            for row in result['trials'][1:]:self.assertEqual((directory/row['directory']/'before.png').read_bytes(),b'pixels')
            self.assertEqual(stage.call_count,1)

if __name__=='__main__':unittest.main()
