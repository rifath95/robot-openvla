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
        for movement in ('arm','cube','gripper'):
            d.reset(state())
            for n in range(1,21):
                d.add(state(x=n*.002 if movement=='arm' else 0,cube=n*.002 if movement=='cube' else 0,grip=n*.001 if movement=='gripper' else 0))
                self.assertIsNone(d.detect(),movement)

    def test_two_three_four_step_cycles_require_three_returns(self):
        for period in (2,3,4):
            d=Detector(8);d.reset(state())
            for n in range(1,3*period):d.add(state(x=(n%period)*.01))
            self.assertIsNone(d.detect())
            d.add(state())
            self.assertEqual(d.detect()['kind'],'cycle');self.assertEqual(d.detect()['period'],period)
        d=Detector(8);d.reset(state())
        for n in range(1,20):
            d.add(state(x=(n%2)*.01,cube=n*.002))
            self.assertIsNone(d.detect(),'Cube progress must prevent a cycle flag')

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
        for bad in [dict(enabled='yes'),dict(sampling_steps=0),dict(window=3),dict(max_attempts=0)]:
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
    def test_real_stationary_arm_triggers_sample_burst_then_greedy_and_stops(self):
        import time
        from common.paths import PROJECT_ROOT
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/'outputs') as tmp:
            e=Engine('http://unused',tmp);e.health_future=Future()
            try:
                e.reset(e.scene);e.view['model']=dict(model_loaded=True,checkpoint=0)
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
