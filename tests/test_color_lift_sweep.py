"""Selection requires a held lift of the requested cube, not touch or a short bounce."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from simulation.color_lift_scene import ColorLiftOutcome,select_instruction
from cloud.sweep_plan import validate_saved,trial_specs
ROOT=Path(__file__).resolve().parents[1]

class ColorLiftTests(unittest.TestCase):
    def run_case(self,lifted_slot,advance=True,lose_contact=False):
        positions={'red':np.array([.45,0.,.27]),'blue':np.array([.65,0.,.27])}
        env=SimpleNamespace(data=SimpleNamespace(time=0.),controller=SimpleNamespace(end_effector_pose=lambda:(positions[lifted_slot]+np.array([0,0,.003]),None)))
        fingers={'left_finger','right_finger'}
        with patch('simulation.color_lift_scene.cube_position',side_effect=lambda env,slot:positions[slot].copy()), \
             patch('simulation.color_lift_scene.fingers_touching',side_effect=lambda env,slot:fingers if slot==lifted_slot else set()):
            tracker=ColorLiftOutcome(env,'blue',{'red':'blue','blue':'yellow'})
            positions[lifted_slot][2]=.34
            for t in (0.,.4,.8,1.2):
                env.data.time=t if advance else 0
                if lose_contact and t>=.8:fingers={'left_finger'}
                tracker.observe(env)
            return tracker.result(env)

    def test_requested_and_wrong_cube_are_distinct(self):
        self.assertTrue(self.run_case('blue')['correct_cube_lift'])
        wrong=self.run_case('red');self.assertTrue(wrong['wrong_cube_lift']);self.assertFalse(wrong['success'])
        self.assertEqual(wrong['outcome'],'wrong_lift')
        self.assertFalse(self.run_case('blue',advance=False)['success'])
        self.assertFalse(self.run_case('blue',lose_contact=True)['success'])

    def test_real_physics_grasp_of_second_cube_counts_the_requested_color(self):
        import mujoco
        from simulation.panda_actions import make_controller,to_controller_action,PHYSICS_STEPS
        from simulation.ik_retries import apply_with_ik_retries
        from simulation.color_lift_scene import SCENE_PATH,initialize_scene,cube_position
        model=mujoco.MjModel.from_xml_path(str(SCENE_PATH));data=mujoco.MjData(model)
        home=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'scene_home');mujoco.mj_resetDataKeyframe(model,data,home)
        env=SimpleNamespace(model=model,data=data,controller=make_controller(model,data))
        initialize_scene(env,{'red':[.45,-.1],'blue':[.60,.1]},{'red':'purple','blue':'orange'})
        initial=cube_position(env,'blue');tracker=ColorLiftOutcome(env,'blue',{'red':'purple','blue':'orange'})
        def act(action):
            accepted,ik,attempts=apply_with_ik_retries(env.controller,to_controller_action(action))
            self.assertIsNotNone(accepted)
            for count in range(0,PHYSICS_STEPS,20):env.controller.step(min(20,PHYSICS_STEPS-count));tracker.observe(env)
        for target in (initial+np.array([0,0,.10]),initial+np.array([0,0,.003])):
            for _ in range(80):
                delta=target-env.controller.end_effector_pose()[0]
                if np.linalg.norm(delta)<.0015:break
                act([*delta,0,0,0,1])
            else:self.fail('Expert could not reach cube')
        act([0,0,0,0,0,0,0])
        for _ in range(9):act([0,0,.008,0,0,0,0])
        result=tracker.result(env)
        self.assertTrue(result['correct_cube_lift']);self.assertFalse(result['wrong_cube_lift'])
        self.assertGreaterEqual(result['cubes']['orange']['maximum_grasp_seconds'],1)

    def test_instruction_matches_visible_nonred_color(self):
        instruction,slot,names=select_instruction(None,'yellow',['blue','yellow'])
        self.assertEqual(instruction,'pick up the yellow cube.');self.assertEqual(slot,'blue')
        with self.assertRaises(ValueError):select_instruction('pick up the blue cube.','yellow',['blue','yellow'])

    def test_all_160_trials_rotate_colors_and_half_have_no_red(self):
        config=json.loads((ROOT/'inference/sweep_profiles/color_lift_500_1000.json').read_text())
        config['sampling']=False
        specs=list(trial_specs(config));self.assertEqual(len(specs),160)
        self.assertEqual(sum('red' not in s['cube_colors'] for s in specs),80)
        validate_saved(dict(plan=config,trials=[]))
        for cp in (500,1000):
            for pair in config['color_pairs']:
                subset=[s for s in specs if s['checkpoint']==cp and s['color_pair']==':'.join(pair)]
                self.assertEqual(len(subset),20)
                self.assertEqual(sum(s['cube_colors'][0]==pair[0] for s in subset),10)

if __name__=='__main__':unittest.main()
