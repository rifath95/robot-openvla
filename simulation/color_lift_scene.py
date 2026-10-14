"""Two-cube lift evaluation using existing geometry, camera and contact grip."""
import mujoco
import numpy as np
from simulation.color_touch_scene import SCENE_PATH,CAMERA_NAME,BODIES,cube_position
from simulation.color_touch_scene import initialize_scene as initialize_two_cubes
from simulation.bowl_scene import configure_grip
from simulation.color_palette import color_mapping,configure_colors


def select_instruction(instruction,target,colors):
    names=color_mapping(colors)
    if target not in colors:raise ValueError('Requested color must be visible')
    expected=f'pick up the {target} cube.'
    if instruction is not None and instruction.strip().lower().rstrip('.')!=expected.rstrip('.'):
        raise ValueError('Instruction and target disagree')
    return instruction or expected,next(slot for slot,color in names.items() if color==target),names


def initialize_scene(env,positions,names):
    configure_grip(env.model)
    scene=initialize_two_cubes(env,positions,gripper=1)
    configure_colors(env.model,names)
    return scene


def fingers_touching(env,slot):
    cube=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_BODY,BODIES[slot])
    fingers={mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_BODY,n):n for n in ('left_finger','right_finger')}
    touching=set()
    for index,contact in enumerate(env.data.contact):
        bodies={int(env.model.geom_bodyid[g]) for g in (contact.geom1,contact.geom2)}
        if cube not in bodies or contact.dist>1e-5:continue
        for finger in bodies.intersection(fingers):
            force=np.empty(6);mujoco.mj_contactForce(env.model,env.data,index,force)
            if force[0]>.01:touching.add(fingers[finger])
    return touching


class ColorLiftOutcome:
    def __init__(self,env,target_slot,names):
        self.target=target_slot;self.names=names
        self.initial={slot:cube_position(env,slot) for slot in BODIES}
        self.states={slot:dict(since=None,maximum_lift_metres=0.,maximum_grasp_seconds=0.,lifted=False,touched=False) for slot in BODIES}

    def observe(self,env):
        now=float(env.data.time);hand=env.controller.end_effector_pose()[0]
        for slot,state in self.states.items():
            cube=cube_position(env,slot);lift=float(cube[2]-self.initial[slot][2]);contacts=fingers_touching(env,slot)
            state['touched']|=bool(contacts)
            qualified=(contacts=={'left_finger','right_finger'} and np.linalg.norm(cube-hand)<=.045 and lift>=.05)
            state['since']=(now if state['since'] is None else state['since']) if qualified else None
            duration=0 if state['since'] is None else now-state['since']
            state['maximum_lift_metres']=max(state['maximum_lift_metres'],lift)
            state['maximum_grasp_seconds']=max(state['maximum_grasp_seconds'],duration)
            state['lifted']|=bool(qualified and duration>=1-1e-9)

    def result(self,env):
        other='blue' if self.target=='red' else 'red'
        correct=self.states[self.target]['lifted'];wrong=self.states[other]['lifted']
        return dict(success=correct and not wrong,target_color=self.names[self.target],correct_cube_lift=correct,
            wrong_cube_lift=wrong,outcome='both_lifts' if correct and wrong else 'correct_lift' if correct else 'wrong_lift' if wrong else 'no_lift',
            cubes={self.names[slot]:{k:v for k,v in state.items() if k!='since'} for slot,state in self.states.items()},
            initial_cubes={self.names[slot]:p.tolist() for slot,p in self.initial.items()},
            final_cubes={self.names[slot]:cube_position(env,slot).tolist() for slot in BODIES})
