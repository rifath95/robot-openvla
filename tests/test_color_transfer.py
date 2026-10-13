"""Recoloring, target identity, paired sweep and scene-level statistical checks."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch,MagicMock
import mujoco
import numpy as np
from simulation.color_palette import configure_colors,parse_pairs,select_instruction,PALETTE
from simulation.color_touch_scene import SCENE_PATH
from simulation.replay_run import replay_model
from inference.color_touch_task import ContactOutcome
from analysis.color_transfer_report import metrics
from cloud import scripted_touch_session as session


class TransferTests(unittest.TestCase):
    def test_recoloring_changes_only_color_and_replay_preserves_it(self):
        model=mujoco.MjModel.from_xml_path(str(SCENE_PATH));size=model.geom_size.copy();mass=model.body_mass.copy()
        names={'red':'purple','blue':'black'};configure_colors(model,names)
        np.testing.assert_array_equal(size,model.geom_size);np.testing.assert_array_equal(mass,model.body_mass)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'episode.json').write_text(json.dumps(dict(
                scene_profile='panda_color_transfer_v1',cube_colors=names)))
            replay,_=replay_model(folder)
        for geom_name,color in [('cube_geom','purple'),('blue_cube_geom','black')]:
            g=mujoco.mj_name2id(replay,mujoco.mjtObj.mjOBJ_GEOM,geom_name)
            np.testing.assert_allclose(replay.geom_rgba[g],PALETTE[color])

    def test_instruction_targets_actual_color_after_slot_exchange(self):
        instruction,slot,names=select_instruction('touch the black cube.','black',['purple','black'])
        self.assertEqual(slot,'blue');self.assertEqual(names['blue'],'black')
        with self.assertRaises(ValueError):select_instruction('touch the red cube.',None,['black','purple'])
        with self.assertRaises(ValueError):parse_pairs(['black:purple','purple:black'])
        env=SimpleNamespace(model=None,data=None)
        with patch('inference.color_touch_task.cube_position',return_value=np.zeros(3)):
            result=ContactOutcome(env,slot,names)
            with patch('inference.color_touch_task.finger_cube_contacts',side_effect=[[],[{'color':'blue'}]]):
                result.observe(env)
            out=result.result(env)
        self.assertTrue(out['success']);self.assertEqual(out['target_color'],'black');self.assertEqual(out['first_contact'],'black')
        self.assertEqual(set(out['final_cubes']),{'purple','black'})

    def test_more_than_half_is_not_automatically_significant(self):
        # Nine scenes always choose one cube: one correct and one wrong. One
        # scene succeeds for both prompts: 11/20 correct, but only one block
        # supplies instruction-selection evidence, giving p=0.5.
        rows=[]
        for seed in range(10):
            outcomes=['correct_cube','wrong_cube'] if seed<9 else ['correct_cube','correct_cube']
            rows += [dict(seed=seed,color_pair='black:purple',outcome=o) for o in outcomes]
        m=metrics(rows,16)
        self.assertEqual(m['choice_accuracy'],.55);self.assertEqual(m['one_sided_label_swap_p'],.5)
        self.assertEqual(m['bonferroni_p'],1)
        for r in rows:r['outcome']='correct_cube'
        m=metrics(rows,16)
        self.assertEqual(m['one_sided_label_swap_p'],1/1024)
        self.assertEqual(m['bonferroni_p'],16/1024)

    def test_no_contact_is_not_a_choice_or_chance_level_failure(self):
        rows=[dict(seed=i,color_pair='black:purple',outcome='no_contact') for i in range(10)]
        m=metrics(rows)
        self.assertIsNone(m['choice_accuracy']);self.assertIsNone(m['one_sided_label_swap_p'])
        self.assertEqual(m['overall_success'],0)

    def test_transfer_sweep_uses_both_prompts_and_balances_color_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'datasets/panda_color_touch/manifest.json';manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({'episodes':[{'cube_positions':{'red':[0,0],'blue':[0,0]}}]}))
            out=root/'outputs/test';out.mkdir(parents=True);calls=[]
            args=SimpleNamespace(task='color_transfer',color_pairs=['black:purple'],scene_seeds=[1000,1001],
                checkpoints=[50,100],steps=50,training_session='panda-train-test',startup_timeout_seconds=10,trial_timeout_seconds=10)
            def run(command,**kwargs):
                if 'inference.remote_loop' in command:
                    calls.append(command);p=Path(command[command.index('--output-dir')+1]);p.mkdir()
                    (p/'summary.json').write_text(json.dumps(dict(phase='Completed',completed_steps=18,
                        task_result=dict(correct_cube_contact=True,wrong_cube_contact=False,outcome='correct_cube'))))
                return SimpleNamespace(returncode=0)
            with patch.object(session,'ROOT',root),patch.object(session.subprocess,'run',side_effect=run), \
                patch.object(session.subprocess,'Popen',return_value=MagicMock()), \
                patch.object(session,'wait_for_server',return_value={'startup_info':{}}):
                session.run_trials(args,out,['ssh','root@host'])
            self.assertEqual(len(calls),8)
            assignments=[c[c.index('--cube-colors')+1:c.index('--cube-colors')+3] for c in calls]
            self.assertEqual(assignments,[['black','purple']]*2+[['purple','black']]*2+
                                        [['black','purple']]*2+[['purple','black']]*2)
            self.assertTrue(all('--sample-actions' not in c for c in calls))
            self.assertIn('Choice accuracy',(out/'REPORT.md').read_text())
            self.assertTrue((out/'statistics.json').is_file())


if __name__=='__main__':unittest.main()
