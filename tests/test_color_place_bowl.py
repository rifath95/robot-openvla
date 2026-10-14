"""Real physics and target attribution for the new colored placement scenes."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np
from simulation.color_place_bowl_scene import scene_path,positions,initialize_scene,PlacementOutcome,slots,cube_position
from simulation.panda_actions import make_controller,to_controller_action,PHYSICS_STEPS
from simulation.ik_retries import apply_with_ik_retries

class PlacementTests(unittest.TestCase):
    def environment(self,count,seed=1000):
        model=mujoco.MjModel.from_xml_path(str(scene_path(count)));data=mujoco.MjData(model)
        env=SimpleNamespace(model=model,data=data,controller=make_controller(model,data))
        palette=['black','purple','green','orange'][:count];names=dict(zip(slots(count),palette));layout=positions(seed,count)
        initialize_scene(env,layout,names)
        return env,names,layout

    def test_all_declared_layouts_settle_and_preserve_grip_camera(self):
        from cloud.bowl_chain_plan import make_plan
        for g in make_plan('panda-train-test')['groups']:
            m=mujoco.MjModel.from_xml_path(str(scene_path(len(g['colors']))));d=mujoco.MjData(m)
            env=SimpleNamespace(model=m,data=d,controller=make_controller(m,d));names=dict(zip(slots(len(g['colors'])),g['colors']))
            for layout in g['layouts']:
                start=initialize_scene(env,layout,names)['starting_pose']
                self.assertEqual(start['requested_gripper_open_fraction'],1)
                self.assertLess(np.max(np.abs(d.qvel)),.02)
                for s,xy in layout['cubes'].items():np.testing.assert_allclose(cube_position(env,s)[:2],xy,atol=.002)
                self.assertGreaterEqual(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,'bowl_camera'),0)

    def test_expert_places_second_cube_and_wrong_target_does_not_pass(self):
        for count in (2,4):
            with self.subTest(count=count):self.expert_place(count)

    def expert_place(self,count):
        env,names,layout=self.environment(count);target='cube_1';initial=cube_position(env,target)
        correct=PlacementOutcome(env,names[target],names);wrong=PlacementOutcome(env,names['cube'],names)
        def act(action):
            accepted,ik,attempts=apply_with_ik_retries(env.controller,to_controller_action(action));self.assertIsNotNone(accepted)
            for i in range(0,PHYSICS_STEPS,20):
                env.controller.step(min(20,PHYSICS_STEPS-i));correct.observe(env);wrong.observe(env)
        def move(position,opening):
            for _ in range(100):
                delta=np.asarray(position)-env.controller.end_effector_pose()[0]
                if np.linalg.norm(delta)<.0015:return
                act([*delta,0,0,0,opening])
            self.fail('Expert did not reach waypoint')
        move(initial+[0,0,.14],1);move(initial+[0,0,.003],1);act([0,0,0,0,0,0,0]);move(initial+[0,0,.15],0)
        self.assertGreater(cube_position(env,target)[2]-initial[2],.08)
        move([*layout['bowl'],.424],0);move([*layout['bowl'],.354],0);act([0,0,0,0,0,0,1]);move([*layout['bowl'],.424],1)
        a=correct.result(env);b=wrong.result(env)
        self.assertTrue(a['success'],a);self.assertFalse(b['success']);self.assertTrue(b['wrong_placement']);self.assertEqual(b['outcome'],'wrong_placed')

    def test_replay_preserves_four_colors_and_bowl_position(self):
        from simulation.replay_run import replay_model
        env,names,layout=self.environment(4)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'episode.json').write_text(json.dumps(dict(scene_profile='panda_color_place_bowl_v1',cube_colors=names,scenario=dict(scene=dict(bowl_position=[*layout['bowl'],.254])))))
            m,camera=replay_model(p);self.assertEqual(camera,'bowl_camera')
            for slot,color in names.items():
                from simulation.color_palette import PALETTE
                geom=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,slot+'_geom');np.testing.assert_allclose(m.geom_rgba[geom],PALETTE[color])
            bowl=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'bowl');np.testing.assert_allclose(m.body_pos[bowl],[*layout['bowl'],.254])

if __name__=='__main__':unittest.main()
