"""Physical-state detection, sampling state machine, limits and request routing."""
from concurrent.futures import Future
import tempfile,unittest,json
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
from inference.playground_recovery import Detector,Recovery,settings
from inference.interactive_bowl import Engine


def state(x=0,cube=0,grip=0):
    return dict(joints=[x]*7,hand=[x,0,0],gripper=grip,cubes=[[0,0,cube]])

class RecoveryTests(unittest.TestCase):
    def test_fixed_pose_stalls_but_useful_translation_and_cube_lift_do_not(self):
        d=Detector(8);d.reset(state())
        for _ in range(8):d.add(state())
        self.assertEqual(d.detect()['kind'],'stall')
        for movement in ('arm','gripper'):
            d.reset(state())
            for n in range(1,21):
                d.add(state(x=n*.002 if movement=='arm' else 0,cube=n*.002 if movement=='cube' else 0,grip=n*.001 if movement=='gripper' else 0))
                self.assertIsNone(d.detect(),movement)

    def test_detector_ignores_simulator_cube_state(self):
        d=Detector(4);d.reset(state())
        for n in range(4):d.add(state(cube=n*.1))
        self.assertEqual(d.detect()['kind'],'stall')
        d.add(dict(joints=[0]*7,hand=[0,0,0],gripper=0))

    def test_nudge_is_separate_and_returns_to_greedy(self):
        r=Recovery(dict(enabled=True,method='nudge',window=4,max_attempts=1),state())
        for _ in range(4):events=r.observe(state())
        self.assertEqual(events[0]['event'],'nudge_started')
        self.assertFalse(r.sampling)
        self.assertEqual(r.observe(state()),[])
        self.assertEqual(r.finish_nudge(state(x=.1))[0]['event'],'nudge_finished')
        for _ in range(12):events=r.observe(state(x=.1))
        self.assertEqual(events[0]['event'],'recovery_exhausted')

    def test_two_three_four_step_cycles_require_three_returns(self):
        for period in (2,3,4):
            d=Detector(8);d.reset(state())
            for n in range(1,3*period):d.add(state(x=(n%period)*.01))
            self.assertIsNone(d.detect())
            d.add(state())
            self.assertEqual(d.detect()['kind'],'cycle');self.assertEqual(d.detect()['period'],period)
        d=Detector(8);d.reset(state())
        for n in range(1,20):
            d.add(state(x=(n%2)*.01+n*.005))
            self.assertIsNone(d.detect(),'Robot progress must prevent a cycle flag')

    def test_sampling_returns_to_greedy_retries_and_stops_at_bound(self):
        r=Recovery(dict(enabled=True,sampling_steps=5,window=4,max_attempts=2),state())
        for _ in range(4):events=r.observe(state())
        self.assertEqual(events[0]['event'],'sampling_started');self.assertTrue(r.sampling)
        for _ in range(5):events=r.observe(state())
        self.assertEqual(events[0]['event'],'sampling_finished');self.assertFalse(r.sampling)
        for _ in range(12):events=r.observe(state())
        self.assertEqual(events[0]['event'],'sampling_started');self.assertEqual(r.attempts,2)
        for _ in range(5):r.observe(state())
        for _ in range(12):events=r.observe(state())
        self.assertEqual(events[0]['event'],'recovery_exhausted');self.assertEqual(r.mode,'exhausted')

    def test_progress_after_sampling_returns_to_normal_greedy(self):
        r=Recovery(dict(enabled=True,sampling_steps=5,window=4),state())
        for _ in range(4):r.observe(state())
        for _ in range(5):r.observe(state())
        for n in range(1,13):events=r.observe(state(x=n*.005))
        self.assertEqual(events[0]['event'],'progress_resumed');self.assertEqual(r.mode,'greedy')
        self.assertEqual(r.attempts,1)

    def test_off_by_default_and_invalid_controls_rejected(self):
        r=Recovery(settings(),state())
        for _ in range(30):self.assertEqual(r.observe(state()),[])
        for bad in [dict(enabled='yes'),dict(method='random'),dict(sampling_steps=0),dict(window=3),dict(max_attempts=0)]:
            with self.assertRaises(ValueError):settings(bad)
        with tempfile.TemporaryDirectory() as tmp:
            e=Engine('http://unused',tmp)
            for steps in (0,10001,True,1.5):
                with self.assertRaisesRegex(ValueError,'Action limit'):e.begin('test',steps)
            with self.assertRaisesRegex(ValueError,'turn off manual sampling'):e.begin('test',1000,True,dict(enabled=True))
            with patch.object(e,'begin') as begin:
                e.command(dict(command='play',instruction='typed',steps=1000,recovery=dict(enabled=True)))
                self.assertEqual(begin.call_args.args[1],1000)
                e.command(dict(command='act',instruction='typed',steps=1000))
                self.assertEqual(begin.call_args.args[1],1)

    def test_request_routing_turns_sampling_off_after_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);e=Engine('http://unused',root);e.folder=root/'trial';e.folder.mkdir()
            e.sample=False;e.instruction='typed';e.save_state=Mock();e.env=Mock();e.view['checkpoint']=6000
            e.recovery=Recovery(dict(enabled=True,window=4,sampling_steps=5),state())
            for _ in range(4):e.recovery.observe(state())
            with patch('simulation.capture_camera.save_rgb_png'),patch('inference.interactive_bowl.background',return_value=Future()) as request:
                e.request();self.assertTrue(request.call_args.kwargs['generation']['do_sample'])
                self.assertEqual(json.loads((e.stepdir/'generation_request.json').read_text())['source'],'recovery')
                for _ in range(5):e.recovery.observe(state())
                e.step=1;e.request();self.assertEqual(request.call_args.kwargs['generation'],dict(do_sample=False))
            self.assertEqual(e.recovery_sampled_requests,1)

class RealRecoveryIntegrationTests(unittest.TestCase):
    def test_manual_button_discards_prediction_moves_and_holds_without_model(self):
        from common.paths import PROJECT_ROOT
        from inference.playground_recovery import physical_state
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/'outputs') as tmp:
            e=Engine('http://unused',tmp);e.health_future=Future()
            try:
                e.reset(e.scene);e.view['model']=dict(model_loaded=True,checkpoint=6000)
                e.begin('typed instruction',120,recovery=dict(enabled=True,method='sampling'))
                old_folder=e.folder
                old_response=Future();e.pending=old_response
                before=physical_state(e.env)
                e.command(dict(command='manual_nudge'))
                manual_folder=e.folder
                self.assertNotEqual(manual_folder,old_folder)
                self.assertIsNone(e.pending)
                self.assertTrue(e.manual_nudge)
                with self.assertRaisesRegex(ValueError,'already in progress'):
                    e.command(dict(command='manual_nudge'))
                old_response.set_result(dict(action=[1]*7))
                with patch.object(e,'publish_frame'),patch('inference.interactive_bowl.time.sleep'),patch('cloud.connection_client.request_prediction') as predict:
                    while e.active:
                        e.request();pending=e.pending;e.pending=None;e.execute(pending.result())
                    predict.assert_not_called()
                after=physical_state(e.env)
                self.assertAlmostEqual(after['hand'][2]-before['hand'][2],.1,delta=.003)
                np.testing.assert_allclose(after['hand'][:2],before['hand'][:2],atol=.003)
                self.assertIn('press Play',e.view['phase'])
                self.assertFalse(e.view['manual_nudge_active'])
                summary=json.loads((manual_folder/'summary.json').read_text())
                self.assertTrue(summary['manual_nudge'])
                self.assertEqual(summary['prediction_attempts'],0)
                self.assertFalse(summary['recovery_settings']['enabled'])
                self.assertEqual(json.loads((old_folder/'summary.json').read_text())['completed_steps'],0)
                events=[json.loads(l)['event'] for l in (manual_folder/'recovery_events.jsonl').read_text().splitlines()]
                self.assertEqual(events,['manual_nudge_started','nudge_finished'])
                # Manual movement also works with no loaded model, and Stop
                # cancels an already prepared increment without moving the arm.
                e.reset(e.scene);e.view['model']=dict(model_loaded=False,checkpoint=6000)
                e.command(dict(command='manual_nudge'));e.request()
                initial=e.env.data.qpos.copy()
                e.command(dict(command='stop'));e.tick()
                self.assertFalse(e.active)
                np.testing.assert_array_equal(e.env.data.qpos,initial)
            finally:
                e.closed.set()
                if e.env:e.env.close()

    def test_upward_nudge_uses_physics_without_querying_model(self):
        from common.paths import PROJECT_ROOT
        from inference.playground_recovery import physical_state,nudge_policy
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/'outputs') as tmp:
            e=Engine('http://unused',tmp);e.health_future=Future()
            try:
                e.reset(e.scene);e.view['model']=dict(model_loaded=True,checkpoint=6000)
                e.begin('test',100,recovery=dict(enabled=True,method='nudge'))
                folder=e.folder
                initial=physical_state(e.env)
                quaternion=e.env.controller.end_effector_pose()[1].copy()
                gripper=e.env.data.ctrl[e.env.controller.gripper_actuator_id]
                self.assertAlmostEqual(nudge_policy()['distance_metres'],.1)
                e.recovery.mode='nudge';e.recovery.attempts=1
                with patch.object(e,'publish_frame'),patch('inference.interactive_bowl.time.sleep'),patch('cloud.connection_client.request_prediction') as predict:
                    while e.active and e.recovery.mode=='nudge':
                        e.request()
                        pending=e.pending;e.pending=None
                        e.execute(pending.result())
                    predict.assert_not_called()
                self.assertTrue(e.active,e.view['phase'])
                final=physical_state(e.env)
                self.assertAlmostEqual(final['hand'][2]-initial['hand'][2],.1,delta=.003)
                np.testing.assert_allclose(final['hand'][:2],initial['hand'][:2],atol=.003)
                self.assertGreater(abs(float(np.dot(quaternion,e.env.controller.end_effector_pose()[1]))),.999)
                self.assertAlmostEqual(e.env.data.ctrl[e.env.controller.gripper_actuator_id],gripper)
                self.assertEqual(e.recovery.mode,'assessing')
                self.assertNotIn('cubes',final)
                e.finish('test finished')
                summary=json.loads((folder/'summary.json').read_text())
                self.assertEqual(summary['prediction_attempts'],0)
                self.assertGreater(summary['nudge_actions'],0)
                actions=[json.loads(l)['action'] for l in (folder/'transitions.jsonl').read_text().splitlines()]
                self.assertTrue(all(np.linalg.norm(a[:3])<=.010001 for a in actions))
                self.assertTrue(all(a[2]>=0 and a[3:6]==[0,0,0] for a in actions))
                # Replaying the same robot feedback at the height ceiling must
                # stop instead of allowing repeated recoveries to climb forever.
                e.begin('test',100,recovery=dict(enabled=True,method='nudge'))
                e.recovery.mode='nudge';e.request()
                self.assertFalse(e.active)
                self.assertIn('height limit',e.view['phase'])
                # The global action cap can interrupt a multi-increment nudge.
                e.reset(e.scene);e.begin('test',1,recovery=dict(enabled=True,method='nudge'))
                e.recovery.mode='nudge'
                with patch.object(e,'publish_frame'),patch('inference.interactive_bowl.time.sleep'):
                    e.request();pending=e.pending;e.pending=None;e.execute(pending.result())
                self.assertFalse(e.active)
                self.assertEqual(e.nudge_actions,1)
                self.assertIn('Completed requested actions',e.view['phase'])
            finally:
                e.closed.set()
                if e.env:e.env.close()

    def test_real_stationary_arm_triggers_sample_burst_then_greedy_and_stops(self):
        import time
        from common.paths import PROJECT_ROOT
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/'outputs') as tmp:
            e=Engine('http://unused',tmp);e.health_future=Future()
            try:
                e.reset(e.scene);e.view['model']=dict(model_loaded=True,checkpoint=6000)
                e.begin('pick up the red cube and place it in the blue bowl.',1000,recovery=dict(enabled=True,window=4,sampling_steps=5,max_attempts=1))
                folder=e.folder
                response=dict(unnorm_key='panda_grasp_v1',action=[0,0,0,0,0,0,1],checkpoint=0)
                with patch.object(e,'publish_frame'),patch('inference.interactive_bowl.time.sleep'),patch('cloud.connection_client.request_prediction',return_value=response) as predict:
                    e.request()
                    for _ in range(40):
                        e.pending.result(timeout=10);e.tick()
                        if not e.active:break
                self.assertFalse(e.active);self.assertEqual(e.step,21)
                modes=[c.kwargs['generation']['do_sample'] for c in predict.call_args_list]
                self.assertEqual(modes,[False]*4+[True]*5+[False]*12)
                summary=json.loads((folder/'summary.json').read_text())
                self.assertEqual(summary['action_limit'],1000);self.assertEqual(summary['recovery_sampled_requests'],5)
                self.assertIn('exhausted',summary['phase'])
                events=[json.loads(l) for l in (folder/'recovery_events.jsonl').read_text().splitlines()]
                self.assertEqual([x['event'] for x in events],['sampling_started','sampling_finished','recovery_exhausted'])
                self.assertIn('physical_state_after',json.loads((folder/'step_004/execution.json').read_text()))
            finally:
                e.closed.set()
                if e.env:e.env.close()

if __name__=='__main__':unittest.main()
