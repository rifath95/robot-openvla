"""Physical-state stall/cycle detection and bounded temporary sampling recovery."""
from collections import deque
import numpy as np

DETECTION_POLICY=dict(version=1,joint_tolerance_radians=.015,position_tolerance_metres=.003,
    gripper_tolerance_metres=.002,cycle_periods=[2,3,4],cycle_repeats=3,minimum_greedy_assessment=12)


def integer(value,low,high,label):
    if type(value) is not int or not low<=value<=high:raise ValueError(f'{label} must be an integer from {low} to {high}')
    return value


def settings(value=None):
    value={} if value is None else value
    if not isinstance(value,dict):raise ValueError('Invalid recovery settings')
    enabled=value.get('enabled',False)
    if type(enabled) is not bool:raise ValueError('Recovery enabled must be boolean')
    return dict(enabled=enabled,sampling_steps=integer(value.get('sampling_steps',8),5,10,'Recovery sampling actions'),
        window=integer(value.get('window',8),4,50,'Stall/assessment window'),
        max_attempts=integer(value.get('max_attempts',3),1,10,'Maximum recovery attempts'))


def physical_state(env,names):
    import mujoco
    from simulation.color_place_bowl_scene import cube_position
    fingers=[mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in ('finger_joint1','finger_joint2')]
    if min(fingers)<0:raise ValueError('Missing gripper joints')
    return dict(joints=env.data.qpos[env.controller.qpos_indices].tolist(),
        hand=env.controller.end_effector_pose()[0].tolist(),
        gripper=float(sum(env.data.qpos[env.model.jnt_qposadr[j]] for j in fingers)),
        cubes=[cube_position(env,slot).tolist() for slot in names])


class Detector:
    def __init__(self,window=8):
        self.window=window;self.history=deque(maxlen=max(window+1,13))

    def reset(self,state):
        self.history.clear();self.add(state)

    def add(self,state):
        joints=np.asarray(state['joints'],dtype=float);hand=np.asarray(state['hand'],dtype=float)
        cubes=np.asarray(state['cubes'],dtype=float).reshape(-1)
        if joints.shape!=(7,) or hand.shape!=(3,) or cubes.size%3:raise ValueError('Invalid physical-state dimensions')
        vector=np.r_[joints/.015,hand/.003,float(state['gripper'])/.002,cubes/.003]
        if not np.all(np.isfinite(vector)):raise ValueError('Nonfinite physical state')
        if self.history and vector.shape!=self.history[-1].shape:raise ValueError('Scene changed during recovery detection')
        self.history.append(vector)

    def detect(self):
        points=np.asarray(self.history)
        if len(points)>=self.window+1:
            tail=points[-self.window-1:]
            if np.max(np.ptp(tail,axis=0))<=1:return dict(kind='stall',window=self.window)
        for period in (2,3,4):
            length=3*period+1
            if len(points)<length:continue
            tail=points[-length:]
            # Require a real excursion and three returns, with no cumulative drift.
            if np.max(np.ptp(tail,axis=0))<=1:continue
            if all(np.max(np.abs(row-tail[i%period]))<=1 for i,row in enumerate(tail)):
                return dict(kind='cycle',period=period,repeats=3)
        return None


class Recovery:
    def __init__(self,config,initial=None):
        self.config=settings(config);self.detector=Detector(self.config['window'])
        if initial is not None:self.detector.reset(initial)
        self.mode='greedy' if self.config['enabled'] else 'disabled'
        self.attempts=0;self.remaining=0;self.assessed=0;self.last_detection=None

    @property
    def sampling(self):return self.mode=='sampling'

    def status(self):
        return dict(mode=self.mode,attempts=self.attempts,max_attempts=self.config['max_attempts'],
            sampling_remaining=self.remaining,assessment_steps=self.assessed,last_detection=self.last_detection)

    def observe(self,state):
        if self.mode in ('disabled','exhausted'):return []
        self.detector.add(state)
        if self.mode=='sampling':
            self.remaining-=1
            if self.remaining==0:
                self.mode='assessing';self.assessed=0;self.detector.reset(state)
                return [dict(event='sampling_finished',next_mode='greedy',attempt=self.attempts)]
            return []
        if self.mode=='assessing':
            self.assessed+=1
            if self.assessed<max(self.config['window'],12):return []
        detection=self.detector.detect()
        if detection:
            self.last_detection=detection
            if self.attempts>=self.config['max_attempts']:
                self.mode='exhausted'
                return [dict(event='recovery_exhausted',detection=detection,attempt=self.attempts)]
            self.attempts+=1;self.remaining=self.config['sampling_steps'];self.mode='sampling'
            return [dict(event='sampling_started',detection=detection,attempt=self.attempts,actions=self.remaining)]
        if self.mode=='assessing':
            self.mode='greedy'
            return [dict(event='progress_resumed',attempt=self.attempts,note='No stall/cycle in greedy assessment window; task success not implied')]
        return []
