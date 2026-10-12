"""Checked random starts and contact-based grip for the cube-to-bowl task."""

from common.paths import PROJECT_ROOT
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from simulation.panda_actions import make_controller
from simulation.panda_start_poses import initialize_start_pose
from simulation.panda_controller import _euler_xyz_to_quat, _quat_multiply

SCENE_PATH = (PROJECT_ROOT / 'simulation/scenes/scene_bowl.xml')
CAMERA_NAME = 'bowl_camera'
INSTRUCTION = 'pick up the red cube and drop it in the blue bowl.'
BOWL_RADIUS = .085


def configure_grip(model):
    """Increase closing pressure and pad friction only in the new task scene.

    Preserve the 0..255 control mapping and force limit. Contacts, not a weld
    or attachment, must support the cube. Call before stepping this model.
    """
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'actuator8')
    if actuator < 0:
        raise ValueError('Panda gripper actuator missing')
    model.actuator_gainprm[actuator, 0] = .04 * 400 / 255
    model.actuator_biasprm[actuator, :3] = [0, -400, -20]
    fingers = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
               for name in ('left_finger', 'right_finger')}
    pads = [g for g in range(model.ngeom) if model.geom_bodyid[g] in fingers
            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    if len(pads) != 10:
        raise ValueError('Expected ten Panda fingertip collision pads')
    for geom in pads:
        model.geom_friction[geom] = [2, .02, .002]
        model.geom_condim[geom] = 4
        model.geom_priority[geom] = 1
        # Firmer pad contacts resist slow settling/sliding under the closing load.
        model.geom_solref[geom, :2] = [.005, 1]
        model.geom_solimp[geom, :3] = [.95, .99, .001]
    return dict(position_gain=400, velocity_gain=20,
                control_range=model.actuator_ctrlrange[actuator].tolist(),
                force_range=model.actuator_forcerange[actuator].tolist(),
                pad_friction=[2, .02, .002], pad_contact_dimension=4,
                pad_solref=[.005, 1], pad_solimp=[.95, .99, .001],
                simulated_contact_grip=True)


def randomize_scene(model, data, rng, *, max_attempts=100, full_pose=False,
                    initial_gripper=None, random_cube_yaw=False, coverage_target=None):
    """Reset objects and sample a collision-checked arm position and yaw.

    The fixed bowl's model body position is part of this scene's reset state.
    Returns the controller bound to the settled state and its specification.
    Sampling is restricted to the table's reachable interior, not every joint
    configuration or every point on the physical table.
    """
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'scene_home')
    bowl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'bowl')
    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_joint')
    if min(home, bowl, cube) < 0:
        raise ValueError('The cube-to-bowl scene requires scene_home, bowl, and cube_joint')
    grip = configure_grip(model)
    address = int(model.jnt_qposadr[cube])
    for attempt in range(max_attempts):
        mujoco.mj_resetDataKeyframe(model, data, home)
        if coverage_target:
            from data_tools.bowl_coverage import sample
            sampled=sample(rng,coverage_target)
            cube_xy,bowl_xy=sampled['cube_xy'],sampled['bowl_xy']
            position=sampled['position']
            if int(np.linalg.norm(position-np.r_[cube_xy,.27])>=.30)!=coverage_target['hand_distance']:
                continue
        else:
            sampled=None
            cube_xy = rng.uniform([.38, -.22], [.72, .22])
            bowl_xy = rng.uniform([.40, -.21], [.70, .21])
        if not .16 <= np.linalg.norm(cube_xy-bowl_xy) <= .36:
            continue
        model.body_pos[bowl] = [*bowl_xy, .254]
        data.qpos[address:address+3] = [*cube_xy, .271]
        cube_yaw = sampled['cube_yaw'] if sampled else float(rng.uniform(-np.pi,np.pi)) if random_cube_yaw else 0.0
        data.qpos[address+3:address+7] = _euler_xyz_to_quat(0,0,cube_yaw)
        mujoco.mj_forward(model, data)
        env = SimpleNamespace(model=model, data=data, controller=make_controller(model, data))
        reference = env.controller.end_effector_pose()[1].copy()
        # A square cube has four equivalent edge-aligned grasps. Choose the one
        # nearest the home yaw, rather than unnecessarily twisting the wrist.
        grasp_yaw = float((cube_yaw+np.pi/4) % (np.pi/2)-np.pi/4)
        grasp_quaternion = _quat_multiply(_euler_xyz_to_quat(0,0,grasp_yaw),reference)
        # Both manipulation locations must have reachable downward-facing poses.
        if not all(env.controller._solve_ik(np.array([*xy, z]), grasp_quaternion,
                                           max_iterations=400).converged
                   for xy, z in ((cube_xy, .274), (bowl_xy, .286))):
            continue
        position = sampled['position'] if sampled else rng.uniform([.36, -.24, .38], [.70, .24, .60])
        yaw = sampled['yaw'] if sampled else float(rng.uniform(-2.2 if full_pose else -1.15,
                                                              2.2 if full_pose else 1.15))
        roll, pitch = (sampled['roll'],sampled['pitch']) if sampled else rng.uniform(-.85, .85, size=2) if full_pose else (0, 0)
        gripper = sampled['gripper'] if sampled else (float(rng.uniform(0, 1)) if full_pose else 1) if initial_gripper is None else initial_gripper
        try:
            start = initialize_start_pose(env, position, yaw, reference, roll=float(roll),
                                          pitch=float(pitch), gripper=gripper,
                                          yaw_limit=2.2 if full_pose else 1.3)
        except ValueError:
            continue
        cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
        camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        spec = dict(instruction=INSTRUCTION, sampling_attempt=attempt+1,
                    camera=dict(name=CAMERA_NAME, position=model.cam_pos[camera].tolist(),
                                quaternion_wxyz=model.cam_quat[camera].tolist(),
                                fovy_degrees=float(model.cam_fovy[camera])),
                    cube_position=data.xpos[cube_body].tolist(),
                    requested_cube_yaw_radians=cube_yaw,
                    cube_quaternion_wxyz=data.xquat[cube_body].tolist(),
                    grasp_yaw_radians=grasp_yaw,
                    grasp_quaternion_wxyz=grasp_quaternion.tolist(),
                    bowl_position=data.xpos[bowl].tolist(), bowl_radius_metres=BOWL_RADIUS,
                    bowl_floor_height_metres=.259, bowl_rim_height_metres=.314,
                    gripper_physics=grip, starting_pose=start,
                    reference_quaternion_wxyz=reference.tolist(),
                    randomized_components=['x','y','z','roll','pitch','yaw','gripper'] if full_pose
                    else ['x','y','z','yaw'])
        if coverage_target:
            from data_tools.bowl_coverage import classify
            if classify(spec)!=coverage_target:continue
            spec['coverage_target']=coverage_target
        return env.controller, spec
    raise RuntimeError('No valid random scene found; try another seed')
