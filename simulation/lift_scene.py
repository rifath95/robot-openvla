"""Single-cube lifting using the existing camera, physics grip and current controller."""
import hashlib
import mujoco
import numpy as np
from common.paths import PROJECT_ROOT
from simulation.touch_scene import SCENE_PATH,CAMERA_NAME,START_POSITION,CUBE_X_BOUNDS,CUBE_Y_BOUNDS,finger_cube_contacts
from simulation.bowl_scene import configure_grip
from simulation.panda_start_poses import initialize_start_pose

INSTRUCTION='pick up the red cube.'
MINIMUM_LIFT_METRES=.05
REQUIRED_GRASP_SECONDS=1.0


def scene_fingerprint():
    return hashlib.sha256(b''.join((PROJECT_ROOT/name).read_bytes() for name in (
        'simulation/scenes/scene_touch.xml','scene.xml',
        'mujoco_menagerie/franka_emika_panda/panda.xml',
        'simulation/lift_scene.py','simulation/bowl_scene.py'))).hexdigest()


def initialize_scene(env,cube_xy):
    xy=np.asarray(cube_xy,dtype=float)
    if xy.shape!=(2,) or not np.isfinite(xy).all() or not (CUBE_X_BOUNDS[0]<=xy[0]<=CUBE_X_BOUNDS[1] and CUBE_Y_BOUNDS[0]<=xy[1]<=CUBE_Y_BOUNDS[1]):
        raise ValueError('Cube outside the declared workspace')
    grip=configure_grip(env.model)
    reference=env.controller.end_effector_pose()[1].copy()
    start=initialize_start_pose(env,START_POSITION,0,reference,gripper=1)
    joint=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_JOINT,'cube_joint')
    q=int(env.model.jnt_qposadr[joint]);v=int(env.model.jnt_dofadr[joint])
    env.data.qpos[q:q+2]=xy;env.data.qpos[q+3:q+7]=(1,0,0,0)
    env.data.qvel[v:v+6]=0;mujoco.mj_forward(env.model,env.data)
    return dict(starting_pose=start,cube_xy=xy.tolist(),cube_orientation_wxyz=[1,0,0,0],
                camera=CAMERA_NAME,gripper_physics=grip)


class LiftSuccess:
    """Require continuous bilateral grip above threshold in simulator time, while moving or still."""
    def __init__(self,initial_cube):
        self.initial=np.asarray(initial_cube,dtype=float);self.since=None;self.maximum_lift=0.;self.maximum_duration=0.
        self.last=dict(success=False,continuous_grasp_seconds=0.,cube_lift_metres=0.)

    def observe(self,env):
        from data_tools.panda_demonstrations import cube_position
        cube=cube_position(env);lift=float(cube[2]-self.initial[2]);now=float(env.data.time)
        contacts=finger_cube_contacts(env.model,env.data)
        bilateral={'left_finger','right_finger'}<={c['finger'] for c in contacts}
        distance=float(np.linalg.norm(cube-env.controller.end_effector_pose()[0]))
        held=bilateral and distance<=.045
        qualified=held and lift>=MINIMUM_LIFT_METRES
        self.since=(now if self.since is None else self.since) if qualified else None
        duration=0. if self.since is None else now-self.since
        self.maximum_lift=max(self.maximum_lift,lift);self.maximum_duration=max(self.maximum_duration,duration)
        self.last=dict(success=bool(qualified and duration>=REQUIRED_GRASP_SECONDS-1e-9),
                       cube_lift_metres=lift,maximum_cube_lift_metres=self.maximum_lift,
                       continuous_grasp_seconds=duration,maximum_continuous_grasp_seconds=self.maximum_duration,
                       bilateral_finger_contact=bilateral,held_near_grasp_point=held,
                       grasp_point_cube_distance_metres=distance,finger_cube_contacts=contacts,
                       initial_cube=self.initial.tolist(),final_cube=cube.tolist())
        return self.last
