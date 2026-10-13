"""Multi-cube evaluation: stable plans, collision contacts and replay fidelity."""
import copy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import mujoco
from common.paths import PROJECT_ROOT
from cloud.sweep_plan import trial_specs,validate_saved,validate_multi_scene_positions
from simulation import multi_color_scene as scene
from analysis.multi_color_report import metrics,write_report


def payload():
    p=json.loads((PROJECT_ROOT/'inference/sweep_profiles/multi_color_all_models.json').read_text())
    p.update(sampling=False,multi_cube_positions={str(n):[scene.positions(s,n) for s in p['scene_seeds']] for n in p['cube_counts']})
    return dict(plan=p,trials=[])


class MultiColorTests(unittest.TestCase):
    def test_all_targets_models_counts_and_resume_identity(self):
        p=payload();specs=validate_saved(p)
        self.assertEqual(len(specs),1080)
        for cp in p['plan']['checkpoints']:
            self.assertEqual(sum(s['checkpoint']==cp for s in specs.values()),180)
        s=next(iter(specs.values()));p['trials']=[dict(s,outcome='no_contact',phase='Completed')]
        validate_saved(p)
        p['trials'][0]['cube_count']=5
        with self.assertRaises(ValueError):validate_saved(p)

    def test_layouts_nested_separated_and_rounding_only_tolerance(self):
        for seed in range(2000,2100):
            six=scene.positions(seed,6)
            scene.validate_positions(six,6)
            self.assertEqual(scene.positions(seed,4),{s:xy for s,xy in list(six.items())[:4]})
        saved=copy.deepcopy(six);saved['cube_0'][0]+=1e-16
        validate_multi_scene_positions(six,saved)
        saved['cube_0'][0]+=.001
        with self.assertRaises(ValueError):validate_multi_scene_positions(six,saved)

    def test_both_correct_and_wrong_contacts_are_not_success(self):
        names={'cube_0':'red','cube_1':'blue','cube_2':'green','cube_3':'yellow'}
        env=SimpleNamespace()
        with patch.object(scene,'cube_position',return_value=np.array([.5,0,.27])):
            tracker=scene.ContactOutcome(env,'red',names)
            with patch.object(scene,'contacts',side_effect=[[],[{'force':1}],[],[]]):tracker.observe(env)
            with patch.object(scene,'contacts',side_effect=[[{'force':1}],[],[],[]]):tracker.observe(env)
            result=tracker.result(env)
        self.assertEqual(result['outcome'],'multiple_cubes');self.assertFalse(result['success'])
        self.assertEqual(result['first_contact'],'blue');self.assertEqual(result['contacted_colors'],['red','blue'])

    def test_renderer_replay_and_all_target_inputs_match(self):
        from simulation.panda_env import PandaEnv
        from simulation.replay_run import replay_model
        from inference.remote_loop import run_loop
        for n in (4,5,6):
            colors=['black','white','purple','orange','cyan','pink'][:n]
            with tempfile.TemporaryDirectory() as tmp:
                captured=[]
                # Capture initial image through real loop initialization, before predictions.
                for color in colors[:2]:
                    folder=Path(tmp)/color;stop=threading.Event();stop.set()
                    with patch('analysis.summarize_gpu_inference.comparison'):
                        run_loop(task='multi_color_touch',cube_colors=colors,target_color=color,
                            scene_seed=2000,steps=1,no_view=True,stop_event=stop,output_dir=folder)
                    meta=json.loads((folder/'episode.json').read_text());model,camera=replay_model(folder)
                    self.assertEqual(model.nq,9+7*n);self.assertEqual(camera,'touch_camera')
                    with PandaEnv(scene.scene_path(n),camera=scene.CAMERA_NAME) as env:
                        scene.initialize_scene(env,meta['scenario']['scene']['cube_positions'],meta['cube_colors'])
                        captured.append(env.observe())
                        self.assertEqual(scene.ContactOutcome(env,color,meta['cube_colors']).result(env)['outcome'],'no_contact')
                        for i in range(n):
                            g=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f'cube_{i}_geom')
                            np.testing.assert_allclose(model.geom_rgba[g],scene.PALETTE[colors[i]])
                np.testing.assert_array_equal(captured[0],captured[1])

    def test_missing_trial_diagnostics_are_reportable(self):
        from cloud.sweep_worker import summarize
        p=payload();spec=next(trial_specs(p['plan']))
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);row=summarize(spec,folder,1)
            self.assertEqual(row['outcome'],'unmeasured')
            self.assertEqual(row['target_color'],spec['target_color'])
            validate_saved(dict(plan=p['plan'],trials=[row]))
            write_report(folder,p['plan'],[row],'Missing summary')
            self.assertIn('Missing summary',(folder/'REPORT.md').read_text())

    def test_worker_dispatches_all_visible_targets_and_resumes(self):
        from cloud import sweep_worker as worker
        from cloud.sweep_plan import timeout_budget
        from unittest.mock import MagicMock
        p=payload();plan=p['plan'];plan.update(checkpoints=[50],include_base_model=False,
            cube_counts=[4],scene_seeds=[2000],color_sets=plan['color_sets'][:1],minimum_results_free_gib=0,
            timeout_budget=timeout_budget(60))
        plan['multi_cube_positions']={'4':[scene.positions(2000,4)]}
        specs=list(trial_specs(plan));p['trials']=[dict(specs[0],phase='Completed',outcome='no_contact',contact=False)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'sweep';directory.mkdir();(directory/'results.json').write_text(json.dumps(p));calls=[]
            def run(command,**kwargs):
                if 'inference.remote_loop' in command:
                    calls.append(command);trial=Path(command[command.index('--output-dir')+1]);trial.mkdir(parents=True)
                    colors=command[command.index('--cube-colors')+1:]
                    self.assertEqual(len(colors),4)
                    coords=json.loads(Path(command[command.index('--scene-positions')+1]).read_text())
                    self.assertEqual(coords,plan['multi_cube_positions']['4'][0])
                    target=command[command.index('--target-color')+1]
                    (trial/'summary.json').write_text(json.dumps(dict(phase='Completed',task_result=dict(
                        correct_cube_contact=True,wrong_cube_contact=False,outcome='correct_cube',first_contact=target,contacted_colors=[target]))))
                return SimpleNamespace(returncode=0)
            server=MagicMock();server.poll.return_value=0
            with patch.object(worker,'PROJECT_ROOT',root),patch.object(worker,'stage_weights',return_value=root/'staged'), \
                 patch.object(worker.subprocess,'run',side_effect=run),patch.object(worker.subprocess,'Popen',return_value=server), \
                 patch.object(worker.subprocess,'check_output',return_value='commit'), \
                 patch.object(worker,'wait_server',return_value=dict(startup_info={},server_port=12345)):
                worker.run(directory,'/sim/python')
            self.assertEqual(len(calls),3)
            saved=json.loads((directory/'results.json').read_text());validate_saved(saved)
            self.assertEqual(len(saved['trials']),4)
            self.assertTrue(all(r['cube_count']==4 for r in saved['trials']))

    def test_prediction_instruction_execution_and_contact_stop(self):
        from inference.remote_loop import run_loop
        predicted=dict(request_id='multi-fixture',unnorm_key='panda_grasp_v1',
            action=[0,0,-.005,0,0,0,0],client_timings_seconds={})
        colors=['red','blue','green','yellow']
        initial=dict(correct_cube_contact=False,wrong_cube_contact=False,outcome='no_contact',success=False)
        contact=dict(correct_cube_contact=True,wrong_cube_contact=False,outcome='correct_cube',success=True)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'run'
            with patch('inference.remote_loop.request_prediction',return_value=predicted) as request, \
                 patch.object(scene.ContactOutcome,'result',side_effect=[initial,contact]), \
                 patch('analysis.summarize_gpu_inference.comparison'):
                run_loop(task='multi_color_touch',cube_colors=colors,target_color='green',scene_seed=2000,
                    steps=5,no_view=True,output_dir=folder)
            self.assertEqual(request.call_count,1);self.assertEqual(request.call_args.args[1],'touch the green cube.')
            saved=json.loads((folder/'summary.json').read_text());self.assertEqual(saved['phase'],'Completed')
            self.assertTrue(saved['task_result']['success']);self.assertEqual(saved['completed_steps'],1)
            with np.load(folder/'step_001/motion_frames.npz') as motion:
                self.assertEqual(motion['qpos'].shape[1],37)

    def test_report_separates_failure_and_familiarity(self):
        p=payload();s=list(trial_specs(p['plan']))[:4]
        rows=[dict(t,outcome=o,phase='Completed',first_contact=None) for t,o in zip(s,['correct_cube','wrong_cube','no_contact','multiple_cubes'])]
        m=metrics(rows);self.assertEqual(m['success_rate'],.25);self.assertEqual(m['choice_accuracy'],.5)
        with tempfile.TemporaryDirectory() as tmp:
            write_report(Path(tmp),p['plan'],rows)
            text=(Path(tmp)/'REPORT.md').read_text()
            self.assertIn('25%',text);self.assertIn('seen targets',text)
            self.assertEqual(json.loads((Path(tmp)/'statistics.json').read_text())[0]['multiple'],1)

if __name__=='__main__':unittest.main()
