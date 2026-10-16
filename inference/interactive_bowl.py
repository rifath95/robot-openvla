"""Local browser controls and live MuJoCo view for manual cloud checkpoint trials."""
import argparse
from concurrent.futures import Future
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import secrets
import threading
import time
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen
import uuid
import webbrowser

from common.paths import PROJECT_ROOT


def background(function, *args, **kwargs):
    future = Future()
    def work():
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
    threading.Thread(target=work, daemon=True).start()
    return future


def remote_json(origin, path, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(origin+path, data=body, headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=10) as response:
        return json.load(response)


def validate_scene(value):
    from simulation.color_palette import PALETTE
    count = value.get('count')
    colors = value.get('colors')
    seed = value.get('seed')
    if type(count) is not int or count not in (1, 2, 3, 4):
        raise ValueError('Choose 1–4 cubes')
    if (not isinstance(colors, list) or len(colors) != count or
        any(not isinstance(c, str) for c in colors) or
        len(set(colors)) != count or not set(colors) <= set(PALETTE)):
        raise ValueError('Choose a different available color for each cube')
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError('Use a nonnegative scene seed below 2147483648')
    target = value.get('target')
    if target not in colors:
        raise ValueError('Scored target must be a visible cube')
    return dict(count=count, colors=list(colors), seed=seed, target=target)


class Engine:
    """Only the owning main thread reads/writes MuJoCo or renders images."""
    def __init__(self, origin, output):
        self.origin = origin
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.commands = queue.Queue()
        self.stop = threading.Event()
        self.closed = threading.Event()
        self.env = None
        self.pending = None
        self.epoch = 0
        self.active = False
        self.folder = None
        self.step = 0
        self.moves = 0
        self.rejections = 0
        self.frame = b''
        self.view = dict(phase='Starting', checkpoint=0, model=dict(status='connecting'),
                         scene=None, steps=0, instruction='', output=str(output))
        self.health_future = None
        self.checkpoint_selected = False
        self.last_health = 0
        self.scene = dict(count=1, colors=['red'], seed=1000, target='red')
        self.instruction = 'pick up the red cube and place it in the blue bowl.'
        self.recovery = None
        self.recovery_events = []
        self.sampled_requests = 0
        self.recovery_sampled_requests = 0
        self.nudge = None
        self.nudge_actions = 0

    def publish_frame(self):
        from simulation.capture_camera import save_rgb_png
        save_rgb_png(self.output/'live.png', self.env.observe())
        self.frame = (self.output/'live.png').read_bytes()

    def save(self, name, value):
        (self.folder/name).write_text(json.dumps(value, indent=2)+'\n')

    def save_state(self, path):
        from data_tools.panda_demonstrations import save_state
        save_state(path, self.env)

    def finish(self, phase):
        self.active = False
        self.view['phase'] = phase
        if self.recovery:self.view['recovery']=dict(self.recovery.status(),active=False)
        if self.folder:
            result = self.outcome.result(self.env)
            self.save('summary.json', dict(phase=phase, completed_steps=self.moves, prediction_attempts=self.step-self.nudge_actions,total_action_attempts=self.step,
                checkpoint=self.view['checkpoint'], instruction=self.instruction,
                task_result=result, rejected_steps=self.rejections,action_limit=self.limit,
                recovery_settings=self.recovery.config,recovery_events=self.recovery_events,
                sampled_prediction_requests=self.sampled_requests,recovery_sampled_requests=self.recovery_sampled_requests,
                nudge_actions=self.nudge_actions))
            from common.replay_links import replay_buttons
            (self.folder/'RUN_REPORT.md').write_text(
                '# Interactive bowl trial\n\n'+self.instruction+'\n\n'+
                f'Checkpoint: {self.view["checkpoint"]}. Scene seed: {self.scene["seed"]}. '
                f'Status: {phase}. Outcome: {result["outcome"]}.\n\n'
                f'Action limit: {self.limit}. Recovery: {self.recovery.config["method"] if self.recovery.config["enabled"] else "off"}. '
                f'Recovery attempts: {self.recovery.attempts}. Sampled prediction requests: {self.sampled_requests}. '
                f'Upward recovery actions: {self.nudge_actions}.\n\n'
                '[Recovery events](recovery_events.jsonl) · [Trial summary](summary.json)\n\n'+replay_buttons(self.folder)+'\n')
            self.folder = None

    def reset(self, scene):
        from simulation.panda_env import PandaEnv
        from simulation.color_place_bowl_scene import (scene_path, CAMERA_NAME,
            initialize_scene, positions, slots, PlacementOutcome)
        self.finish('Stopped for scene reset')
        self.epoch += 1  # Any response from the former scene is stale.
        self.stop.clear()
        self.scene = validate_scene(scene)
        self.recovery=None
        self.view['recovery']=dict(mode='disabled',attempts=0)
        if self.env:
            self.env.close()
            self.env = None
        self.env = PandaEnv(scene_path(scene['count']), camera=CAMERA_NAME)
        self.names = dict(zip(slots(scene['count']), scene['colors']))
        self.scene_metadata = initialize_scene(self.env, positions(scene['seed'], scene['count']), self.names)
        self.outcome = PlacementOutcome(self.env, scene['target'], self.names)
        self.view.update(scene=self.scene, phase='Ready — enter instruction and press Play or Act once', steps=0)
        self.publish_frame()

    def begin(self, instruction, steps, sample=False, recovery=None):
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 1000:
            raise ValueError('Enter an instruction (up to 1,000 characters)')
        from inference.playground_recovery import integer,settings
        integer(steps,1,10000,'Action limit')
        config=settings(recovery)
        if type(sample) is not bool:raise ValueError('Sampling must be boolean')
        if sample and config['enabled']:raise ValueError('Automatic recovery starts from greedy; turn off manual sampling')
        if self.active or self.pending is not None:
            raise ValueError('A prediction is still in progress; wait for it to finish')
        model = self.view['model']
        if (not model.get('model_loaded') or model.get('checkpoint') != self.view['checkpoint']):
            raise ValueError('Wait for the selected checkpoint to finish loading')
        from simulation.color_place_bowl_scene import PlacementOutcome, CAMERA_NAME, scene_fingerprint
        from data_tools.bowl_demonstrations import controller_fingerprint
        from simulation.panda_actions import action_contract
        from simulation.replay_run import write_launcher
        self.outcome = PlacementOutcome(self.env, self.scene['target'], self.names)
        self.instruction = instruction  # Preserve exactly what the user entered.
        self.limit = steps
        self.sample = sample
        from inference.playground_recovery import Recovery,physical_state,DETECTION_POLICY,nudge_policy
        self.recovery=Recovery(config,physical_state(self.env,self.names) if config['enabled'] else None)
        self.recovery_events=[]
        self.sampled_requests=0
        self.recovery_sampled_requests=0
        self.nudge=None
        self.nudge_actions=0
        self.stop.clear()
        self.epoch += 1
        self.step = 0
        self.moves = 0
        self.rejections = 0
        self.folder = self.output/('trial_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        self.folder.mkdir()
        write_launcher(self.folder)
        self.save_state(self.folder/'initial_state.npz')
        self.save('episode.json', dict(scene_profile='panda_color_place_bowl_v1',
            instruction=instruction, target_color=self.scene['target'], cube_colors=self.names,
            scene_seed=self.scene['seed'], scenario=dict(scene=self.scene_metadata),
            action_contract=action_contract(camera=CAMERA_NAME), controller_sha256=controller_fingerprint(),
            scene_xml_sha256=scene_fingerprint(self.scene['count']), training_eligible=False,
            checkpoint=self.view['checkpoint'], model=self.view['model'], sample_actions=sample,
            action_limit=steps,recovery_settings=self.recovery.config,recovery_detection_policy=DETECTION_POLICY,
            nudge_policy=nudge_policy(),
            scoring_note='Scored target is the selected target dropdown; instruction text is sent unchanged.'))
        (self.folder/'recovery_events.jsonl').touch()
        self.view.update(instruction=instruction, phase='Running', steps=0, action_limit=steps,
            recovery=dict(self.recovery.status(),active=True),run=str(self.folder))
        self.active = True

    def request(self):
        from cloud.connection_client import request_prediction
        from simulation.capture_camera import save_rgb_png
        self.stepdir = self.folder/f'step_{self.step+1:03d}'
        self.stepdir.mkdir()
        self.save_state(self.stepdir/'scene_state.npz')
        save_rgb_png(self.stepdir/'before.png', self.env.observe())
        if self.recovery.mode=='nudge':
            import numpy as np
            from inference.playground_recovery import nudge_policy
            policy=nudge_policy()
            hand=self.env.controller.end_effector_pose()[0]
            if self.nudge is None:
                goal=hand.copy()
                goal[2]=min(hand[2]+policy['distance_metres'],policy['ceiling_z_metres'])
                if goal[2]-hand[2]<.002:
                    self.finish('Stopped — upward recovery height limit reached')
                    return
                self.nudge=dict(goal=goal.tolist(),actions=0,
                    gripper=float(np.clip(self.env.data.ctrl[self.env.controller.gripper_actuator_id]/255.,0.,1.)))
                self.save('nudge_attempt_'+str(self.recovery.attempts)+'.json',dict(self.nudge,policy=policy))
            delta=np.asarray(self.nudge['goal'])-hand
            delta[2]=max(0.,delta[2])
            self.nudge['actions']+=1
            self.prediction_sampled=False
            self.prediction_source='nudge'
            self.save(str(self.stepdir.relative_to(self.folder)/'generation_request.json'),
                dict(source='nudge',model_queried=False,sampled=False))
            self.pending_epoch=self.epoch
            self.pending=Future()
            self.pending.set_result(dict(unnorm_key='panda_grasp_v1',
                action=[*delta.tolist(),0.,0.,0.,self.nudge['gripper']],source='nudge',model_queried=False))
            self.view['phase']='Recovery — moving upward'
            return
        options = dict(server_url=self.origin, checkpoint=self.view['checkpoint'])
        automatic=self.recovery.sampling
        self.prediction_sampled=self.sample or automatic
        self.prediction_source='recovery' if automatic else 'manual' if self.sample else 'greedy'
        if self.prediction_sampled:
            options['generation'] = dict(do_sample=True, temperature=1., top_p=1.)
            self.sampled_requests+=1
            if automatic:self.recovery_sampled_requests+=1
        elif self.recovery.config['enabled']:
            options['generation']=dict(do_sample=False)
        (self.stepdir/'generation_request.json').write_text(json.dumps(dict(
            sampled=self.prediction_sampled,source=self.prediction_source,
            generation=options.get('generation',dict(do_sample=False))),indent=2)+'\n')
        self.pending_epoch = self.epoch
        self.pending = background(request_prediction, self.stepdir/'before.png', self.instruction, **options)
        self.view['phase'] = 'Waiting for prediction'+(' — recovery sampling' if automatic else '')

    def execute(self, prediction):
        import numpy as np
        from simulation.ik_retries import apply_with_ik_retries
        from simulation.panda_actions import to_controller_action, from_controller_action, PHYSICS_STEPS
        if prediction['unnorm_key'] != 'panda_grasp_v1':
            raise ValueError('Unexpected action normalization; no action executed')
        bounded = to_controller_action(prediction['action'])
        action, ik, attempts = apply_with_ik_retries(self.env.controller, bounded)
        (self.stepdir/'prediction.json').write_text(json.dumps(prediction, indent=2)+'\n')
        self.view['phase'] = 'Moving'
        poses = [self.env.data.qpos.copy()]
        times = [float(self.env.data.time)]
        executed = 0
        started = time.monotonic()
        if ik.converged:
            while executed < PHYSICS_STEPS and not self.stop.is_set() and not self.closed.is_set():
                count = min(20, PHYSICS_STEPS-executed)
                self.env.controller.step(count)
                executed += count
                self.outcome.observe(self.env)
                poses.append(self.env.data.qpos.copy())
                times.append(float(self.env.data.time))
                self.publish_frame()
                # Show the same controller motion at real simulation speed.
                time.sleep(max(0, executed*float(self.env.model.opt.timestep)-(time.monotonic()-started)))
        else:
            self.rejections += 1
        np.savez_compressed(self.stepdir/'motion_frames.npz', qpos=np.asarray(poses), time=np.asarray(times))
        self.save_state(self.stepdir/'after_state.npz')
        from simulation.capture_camera import save_rgb_png
        save_rgb_png(self.stepdir/'after.png', self.env.observe())
        tracking = float(np.linalg.norm(self.env.controller.target_position-self.env.controller.end_effector_pose()[0]))
        row = dict(step=self.step+1, raw_action=prediction['action'],
            controller_action=action.tolist() if action is not None else None,
            bounded_action=bounded.tolist(), ik_attempts=attempts, executed=bool(ik.converged),
            executed_physics_steps=executed, position_tracking_error_metres=tracking,
            task_result=self.outcome.result(self.env),sampling_source=self.prediction_source,
            sampled=self.prediction_sampled)
        if self.recovery.config['enabled']:
            from inference.playground_recovery import physical_state
            row['physical_state_after']=physical_state(self.env,self.names)
        (self.stepdir/'execution.json').write_text(json.dumps(row, indent=2)+'\n')
        if executed == PHYSICS_STEPS:
            with (self.folder/'transitions.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(index=self.moves+1, action=from_controller_action(action).tolist(),
                    source_step_directory=self.stepdir.name,source=self.prediction_source,
                    physics_steps=executed, instruction=self.instruction, executed=True))+'\n')
            self.moves += 1
            self.rejections = 0
        self.step += 1
        if self.prediction_source=='nudge':self.nudge_actions+=1
        self.view.update(steps=self.step, result=row['task_result'], last_action=prediction['action'])
        self.publish_frame()
        if self.stop.is_set():
            self.finish('Stopped — arm held at its current position')
        elif executed and (tracking > .005 or not np.all(np.isfinite(self.env.data.qpos))):
            self.finish(f'Controller tracking failed ({tracking*1000:.2f} mm)')
        elif self.rejections >= 3:
            self.finish('Stopped after three rejected actions')
        elif row['task_result']['correct_placement'] or row['task_result']['wrong_placement']:
            self.finish('Placement reached: '+row['task_result']['outcome'])
        elif self.step >= self.limit:
            self.finish('Completed requested actions — Play or Act once to continue')
        elif self.recovery.config['enabled']:
            state=row['physical_state_after']
            if self.prediction_source=='nudge':
                from inference.playground_recovery import nudge_policy
                import numpy as np
                reached=np.linalg.norm(np.asarray(state['hand'])-self.nudge['goal'])<=.002
                if reached:
                    events=self.recovery.finish_nudge(state)
                    self.nudge=None
                elif self.nudge['actions']>=nudge_policy()['max_actions']:
                    self.finish('Stopped — upward recovery did not reach its target')
                    return
                else:events=[]
            else:events=self.recovery.observe(state)
            for event in events:
                event.update(step=self.step,simulation_time=float(self.env.data.time),physical_state=state)
                self.recovery_events.append(event)
                with (self.folder/'recovery_events.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
            self.view['recovery']=dict(self.recovery.status(),active=True)
            if self.recovery.mode=='exhausted':self.finish('Stopped — automatic recovery attempts exhausted')

    def command(self, value):
        kind = value.get('command')
        if kind == 'stop':
            self.epoch += 1
            self.finish('Stopped — arm held at its current position')
        elif kind == 'scene':
            self.reset(validate_scene(value['scene']))
        elif kind == 'checkpoint':
            cp = value.get('checkpoint')
            from cloud.interactive_bowl_host import CHECKPOINTS
            if type(cp) is not int or cp not in CHECKPOINTS:
                raise ValueError('Unknown checkpoint')
            if self.pending is not None:
                raise ValueError('Wait for the cancelled prediction before changing checkpoint')
            self.finish('Stopped for checkpoint change')
            remote_json(self.origin, '/select', dict(checkpoint=cp))
            self.checkpoint_selected = True
            self.view.update(checkpoint=cp, model=dict(status='loading'), phase='Loading checkpoint')
            self.reset(self.scene)
        elif kind in ('play', 'act'):
            self.begin(value.get('instruction'), value.get('steps',120) if kind == 'play' else 1,
                value.get('sample') is True,value.get('recovery'))
        elif kind == 'quit':
            self.closed.set()
        else:
            raise ValueError('Unknown command')

    def tick(self):
        if self.health_future is None and time.monotonic()-self.last_health > 1:
            self.last_health = time.monotonic()
            self.health_future = background(remote_json, self.origin, '/health')
        if self.health_future is not None and self.health_future.done():
            try:
                self.view['model'] = self.health_future.result()
                if not self.checkpoint_selected and self.view['model'].get('checkpoint') is not None:
                    self.view['checkpoint'] = self.view['model']['checkpoint']
            except Exception as exc:
                self.view['model'] = dict(status='disconnected', error=str(exc))
            self.health_future = None
        while not self.commands.empty():
            try:
                self.command(self.commands.get_nowait())
            except Exception as exc:
                self.view['phase'] = str(exc)
        if self.stop.is_set() and self.active:
            self.epoch += 1
            self.finish('Stopped — any pending prediction is discarded')
        if self.pending is not None and self.pending.done():
            pending, self.pending = self.pending, None
            if self.pending_epoch == self.epoch and self.active and not self.stop.is_set():
                try:
                    self.execute(pending.result())
                except Exception as exc:
                    self.finish('Error: '+str(exc))
        if self.active and self.pending is None and not self.stop.is_set():
            self.request()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, status, body, content_type='application/json'):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlsplit(self.path)
        if parse_qs(url.query).get('token', [''])[0] != self.server.token:
            return self.reply(403, dict(error='Open the URL printed in your terminal'))
        if url.path == '/':
            self.reply(200, (PROJECT_ROOT/'inference/interactive_bowl.html').read_bytes(), 'text/html; charset=utf-8')
        elif url.path == '/state':
            self.reply(200, dict(self.server.engine.view))
        elif url.path == '/frame':
            self.reply(200, self.server.engine.frame, 'image/png')
        else:
            self.reply(404, dict(error='Not found'))

    def do_POST(self):
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 8192:
                raise ValueError('Invalid request size')
            value = json.loads(self.rfile.read(size))
            if value.pop('token', '') != self.server.token:
                return self.reply(403, dict(error='Invalid session token'))
            if value.get('command') in ('stop', 'scene', 'checkpoint', 'quit'):
                self.server.engine.stop.set()  # Immediate cancellation, even during movement.
            self.server.engine.commands.put(value)
            self.reply(202, dict(queued=True))
        except Exception as exc:
            self.reply(400, dict(error=str(exc)))


def serve(origin, output, *, port=0, open_browser=True):
    engine = Engine(origin, output)
    token = secrets.token_urlsafe(32)
    with ThreadingHTTPServer(('127.0.0.1', port), Handler) as server:
        server.engine = engine
        server.token = token
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            engine.reset(engine.scene)
            url = f'http://127.0.0.1:{server.server_port}/?token={token}'
            print('Simulator controls:', url, flush=True)
            print('Ctrl+C ends local controls. Cloud model stays loaded; pod stays running.', flush=True)
            if open_browser:
                webbrowser.open(url)
            while not engine.closed.is_set():
                engine.tick()
                time.sleep(.02)
        except KeyboardInterrupt:
            pass
        finally:
            engine.stop.set()
            engine.finish('Local controls closed')
            if engine.env:
                engine.env.close()
            server.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-url', default='http://127.0.0.1:8010')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    serve(args.server_url, args.output_dir, open_browser=not args.no_browser)


if __name__ == '__main__':
    main()
