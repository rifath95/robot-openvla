"""Separate 4–6-cube evaluation scenes; keep the trained camera and controller."""
import itertools
import math
from common.paths import PROJECT_ROOT
from simulation.color_palette import PALETTE
CAMERA_NAME="touch_camera"
START_POSITION=(.50,0.,.45)
CUBE_X_BOUNDS=(.40,.68)
CUBE_Y_BOUNDS=(-.16,.16)


def scene_path(count):
    if type(count) is not int or count not in (4,5,6):raise ValueError('Require 4, 5 or 6 cubes')
    return PROJECT_ROOT/f'simulation/scenes/scene_multi_color_{count}.xml'


def positions(seed,count):
    """Nested jittered lattice: separated positions without easier-scene resampling."""
    scene_path(count)
    if type(seed) is not int or seed<0:raise ValueError('Nonnegative integer scene seed required')
    import numpy as np
    rng=np.random.default_rng(seed)
    grid=np.array([[x,y] for x in (.415,.535,.655) for y in (-.11,.11)])
    grid+=rng.uniform(-.01,.01,grid.shape)
    grid=grid[rng.permutation(6)]
    return {f'cube_{i}':xy.tolist() for i,xy in enumerate(grid[:count])}


def validate_positions(value,count):
    if set(value)!={f'cube_{i}' for i in range(count)}:raise ValueError('Cube positions must cover every slot')
    for xy in value.values():
        if (len(xy)!=2 or not all(math.isfinite(x) for x in xy)
            or not CUBE_X_BOUNDS[0]<=xy[0]<=CUBE_X_BOUNDS[1]
            or not CUBE_Y_BOUNDS[0]<=xy[1]<=CUBE_Y_BOUNDS[1]):raise ValueError('Cube outside evaluation bounds')
    if any(math.dist(a,b)<.095 for a,b in itertools.combinations(value.values(),2)):
        raise ValueError('Cube positions too close')


def validate_colors(colors):
    scene_path(len(colors))
    if len(set(colors))!=len(colors) or any(c not in PALETTE for c in colors):raise ValueError('Distinct palette colors required')


def configure_colors(model,names):
    import mujoco
    validate_colors(list(names.values()))
    if set(names)!={f'cube_{i}' for i in range(len(names))}:raise ValueError('Invalid physical slots')
    for slot,color in names.items():
        g=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,slot+'_geom')
        if g<0:raise ValueError('Missing cube geometry')
        model.geom_rgba[g]=PALETTE[color]


def initialize_scene(env,value,names):
    import mujoco
    from simulation.panda_start_poses import initialize_start_pose
    validate_positions(value,len(names));configure_colors(env.model,names)
    reference=env.controller.end_effector_pose()[1].copy()
    start=initialize_start_pose(env,START_POSITION,0,reference,gripper=0,environment_bodies=tuple(names))
    for slot,xy in value.items():
        j=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_JOINT,slot+'_joint')
        q=int(env.model.jnt_qposadr[j]);v=int(env.model.jnt_dofadr[j])
        env.data.qpos[q:q+7]=[*xy,.26989224457978134,1,0,0,0]
        env.data.qvel[v:v+6]=0
    mujoco.mj_forward(env.model,env.data)
    return dict(starting_pose=start,cube_positions=value,cube_colors=names,camera=CAMERA_NAME,
                cube_orientation_wxyz=[1,0,0,0])


def cube_position(env,slot):
    import mujoco
    return env.data.xpos[mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_BODY,slot)].copy()


def contacts(env,slot):
    import mujoco
    import numpy as np
    model,data=env.model,env.data
    body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,slot)
    fingers={mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in ('left_finger','right_finger')}
    result=[]
    for i,c in enumerate(data.contact):
        bodies={int(model.geom_bodyid[g]) for g in (c.geom1,c.geom2)}
        if body in bodies and bodies.intersection(fingers) and c.dist<=1e-5:
            force=np.empty(6);mujoco.mj_contactForce(model,data,i,force)
            if force[0]>.01:result.append(dict(normal_force_newtons=float(force[0]),distance_metres=float(c.dist)))
    return result


class ContactOutcome:
    def __init__(self,env,target_color,names):
        if target_color not in names.values():raise ValueError('Target must be visible')
        self.names=names;self.target=target_color;self.initial={s:cube_position(env,s) for s in names}
        self.contacts={s:[] for s in names};self.first=None
    def observe(self,env):
        current={s:contacts(env,s) for s in self.names};touching=[self.names[s] for s,c in current.items() if c]
        if touching and self.first is None:self.first=touching[0] if len(touching)==1 else 'multiple'
        for s,c in current.items():
            if c:self.contacts[s]=c
    def result(self,env):
        touched=[self.names[s] for s,c in self.contacts.items() if c]
        correct=self.target in touched;wrong=any(c!=self.target for c in touched)
        return dict(target_color=self.target,correct_cube_contact=correct,wrong_cube_contact=wrong,
            success=correct and not wrong,first_contact=self.first,contacted_colors=touched,
            outcome='multiple_cubes' if len(touched)>1 else 'wrong_cube' if wrong else 'correct_cube' if correct else 'no_contact',
            finger_cube_contacts={self.names[s]:c for s,c in self.contacts.items()},
            initial_cubes={self.names[s]:p.tolist() for s,p in self.initial.items()},
            final_cubes={self.names[s]:cube_position(env,s).tolist() for s in self.names})
