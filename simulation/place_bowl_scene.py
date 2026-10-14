"""Phase 4 bowl placement: fixed successful lift start, blue interior, variable destinations."""
from types import SimpleNamespace
import mujoco
import numpy as np
from common.paths import PROJECT_ROOT
from simulation.touch_scene import START_POSITION
from simulation.bowl_scene import CAMERA_NAME, configure_grip
from simulation.panda_actions import make_controller
from simulation.panda_start_poses import initialize_start_pose
SCENE_PATH=PROJECT_ROOT/'simulation/scenes/scene_place_bowl.xml'
INSTRUCTION='pick up the red cube and place it in the blue bowl.'


def scene_fingerprint():
    from data_tools.bowl_demonstrations import scene_fingerprint as fingerprint
    return fingerprint(SCENE_PATH)


def positions(slot):
    rng=np.random.default_rng([20261014,slot,5])
    for _ in range(10000):
        cube=rng.uniform([.40,-.16],[.68,.16])
        bowl=rng.uniform([.40,-.16],[.68,.16])
        if .18<=np.linalg.norm(cube-bowl)<=.32:
            return dict(cube_xy=cube.tolist(),bowl_xy=bowl.tolist())
    raise RuntimeError('Unable to predeclare separated positions')


def initialize_scene(model,data,rng,**kwargs):
    # Collector seeds carry the immutable slot; this initializer never substitutes positions.
    slot=kwargs.pop('slot',None)
    if slot is None:raise ValueError('Placement initializer requires its fixed slot')
    spec=positions(slot)
    home=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'scene_home')
    mujoco.mj_resetDataKeyframe(model,data,home);configure_grip(model)
    bowl=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'bowl')
    model.body_pos[bowl]=[*spec['bowl_xy'],.254]
    env=SimpleNamespace(model=model,data=data,controller=make_controller(model,data))
    reference=env.controller.end_effector_pose()[1].copy()
    start=initialize_start_pose(env,START_POSITION,0,reference,gripper=1)
    joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'cube_joint');q=int(model.jnt_qposadr[joint]);v=int(model.jnt_dofadr[joint])
    data.qpos[q:q+3]=[*spec['cube_xy'],.271];data.qpos[q+3:q+7]=[1,0,0,0];data.qvel[v:v+6]=0
    mujoco.mj_forward(model,data)
    return env.controller,dict(starting_pose=start,bowl_position=model.body_pos[bowl].tolist(),
        requested_cube_yaw_radians=0,grasp_quaternion_wxyz=reference.tolist(),**spec)
