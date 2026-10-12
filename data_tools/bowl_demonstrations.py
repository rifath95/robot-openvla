"""Contact-based scripted bowl demonstrations and deterministic physics replay."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from common.paths import PROJECT_ROOT
from data_tools.panda_demonstrations import (write_json, save_state, restore_state,
                                            cube_position)
from data_tools.prepare_panda_dataset import fingerprint
from simulation.bowl_scene import SCENE_PATH, CAMERA_NAME, INSTRUCTION, configure_grip, randomize_scene
from simulation.capture_camera import save_rgb_png
from simulation.ik_retries import apply_with_ik_retries
from simulation.panda_actions import PHYSICS_STEPS, action_contract, to_controller_action, from_controller_action
from simulation.panda_env import PandaEnv
from simulation.panda_start_poses import unsafe_motion_contacts
from simulation.replay_run import write_launcher


def scene_fingerprint():
    return hashlib.sha256(b''.join(path.read_bytes() for path in (
        SCENE_PATH, PROJECT_ROOT/'scene.xml',
        PROJECT_ROOT/'mujoco_menagerie/franka_emika_panda/panda.xml'))).hexdigest()


def configure_recorded_scene(model, metadata):
    configure_grip(model)
    position = np.asarray(metadata['scenario']['scene']['bowl_position'], dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('Invalid saved bowl position')
    bowl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'bowl')
    model.body_pos[bowl] = position


def orientation_delta(current, target):
    current_matrix, target_matrix = np.empty(9), np.empty(9)
    mujoco.mju_quat2Mat(current_matrix, current)
    mujoco.mju_quat2Mat(target_matrix, target)
    relative = target_matrix.reshape(3, 3) @ current_matrix.reshape(3, 3).T
    return np.array([np.arctan2(relative[2, 1], relative[2, 2]),
                     np.arcsin(np.clip(-relative[2, 0], -1, 1)),
                     np.arctan2(relative[1, 0], relative[0, 0])])


def task_result(initial, bowl, positions, env):
    final = positions[-1]
    cube = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
    velocity = np.empty(6)
    mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, cube, velocity, 0)
    lift = float(max(point[2] for point in positions) - initial[2])
    distance = float(np.linalg.norm(final[:2] - bowl[:2]))
    height_error = float(abs(final[2] - .279))
    separation = float(np.linalg.norm(final - env.controller.end_effector_pose()[0]))
    settled = bool(np.linalg.norm(velocity[:3]) < .1 and np.linalg.norm(velocity[3:]) < .02)
    return dict(success=bool(lift >= .08 and distance <= .045 and height_error <= .004
                             and settled and separation >= .08),
                maximum_cube_lift_metres=lift, bowl_center_distance_metres=distance,
                bowl_floor_height_error_metres=height_error, cube_settled=settled,
                cube_velocity=velocity.tolist(), empty_hand_cube_distance_metres=separation,
                initial_cube=initial.tolist(), final_cube=final.tolist(),
                destination_cube=[*bowl[:2].tolist(), .279])


def record_episode(folder, scenario):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    write_launcher(folder)
    rows = []
    metadata = dict(schema_version=1, scene_profile='panda_bowl_v1',
                    scene_xml_sha256=scene_fingerprint(), mujoco_version=mujoco.__version__,
                    action_contract=action_contract(camera=CAMERA_NAME), camera=CAMERA_NAME,
                    instruction=INSTRUCTION, image_size=[640, 480],
                    source='scripted expert with privileged object coordinates; physical contact grasp',
                    scenario=scenario, success=False, training_eligible=False,
                    physics_seconds_per_action=.4)
    write_json(folder/'episode.json', metadata)
    try:
        with PandaEnv(SCENE_PATH, camera=CAMERA_NAME) as env:
            rng = np.random.default_rng(scenario['random_seed'])
            env.controller, scene = randomize_scene(env.model, env.data, rng, full_pose=True,
                                                    initial_gripper=scenario['initial_gripper'])
            metadata['scenario'] = dict(scenario, scene=scene)
            metadata['starting_pose'] = scene['starting_pose']
            metadata['physics_seconds_per_action'] = PHYSICS_STEPS * float(env.model.opt.timestep)
            write_json(folder/'episode.json', metadata)
            initial = cube_position(env)
            bowl = np.asarray(scene['bowl_position'])
            reference = np.asarray(scene['reference_quaternion_wxyz'])
            positions = [initial.copy()]
            save_state(folder/'initial_state.npz', env)

            def act(proposal, stage):
                directory = folder/f'step_{len(rows)+1:04d}'
                directory.mkdir()
                before, quaternion_before = env.controller.end_effector_pose()
                save_rgb_png(directory/'before.png', env.observe())
                save_state(directory/'before_state.npz', env)
                bounded = to_controller_action(proposal)
                accepted, ik, attempts = apply_with_ik_retries(env.controller, bounded)
                if accepted is None:
                    raise RuntimeError(f'IK rejected expert action during {stage}')
                motion_poses, motion_times = [env.data.qpos.copy()], [float(env.data.time)]
                for count in range(0, PHYSICS_STEPS, 20):
                    env.controller.step(min(20, PHYSICS_STEPS-count))
                    if unsafe_motion_contacts(env.model, env.data):
                        raise RuntimeError(f'Unsafe robot contact during {stage}')
                    motion_poses.append(env.data.qpos.copy())
                    motion_times.append(float(env.data.time))
                after, quaternion_after = env.controller.end_effector_pose()
                tracking = float(np.linalg.norm(after-env.controller.target_position))
                if not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all() or tracking > .005:
                    raise RuntimeError(f'Expert tracking failed during {stage}: {tracking:g} m')
                np.savez_compressed(directory/'motion_frames.npz', qpos=motion_poses, time=motion_times)
                save_rgb_png(directory/'after.png', env.observe())
                save_state(directory/'after_state.npz', env)
                cube = cube_position(env)
                positions.append(cube)
                row = dict(index=len(rows)+1, stage=stage, instruction=INSTRUCTION,
                           observation=f'{directory.name}/before.png', next_observation=f'{directory.name}/after.png',
                           proposed_action=np.asarray(proposal).tolist(),
                           bounded_action=from_controller_action(bounded).tolist(),
                           action=from_controller_action(accepted).tolist(), executed=True,
                           action_scale=attempts[-1]['scale'], ik_attempts=attempts,
                           physics_steps=PHYSICS_STEPS, physics_seconds=metadata['physics_seconds_per_action'],
                           control_position_before=before.tolist(), control_position_after=after.tolist(),
                           quaternion_before_wxyz=quaternion_before.tolist(), quaternion_after_wxyz=quaternion_after.tolist(),
                           measured_translation=(after-before).tolist(), tracking_error_metres=tracking,
                           cube_position=cube.tolist())
                write_json(directory/'transition.json', row)
                rows.append(row)
                with (folder/'transitions.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')

            def move(target, opening, stage, orientation=reference):
                print(f'{scenario["id"]}: {stage}', flush=True)
                for _ in range(160):
                    position, quaternion = env.controller.end_effector_pose()
                    delta = np.asarray(target)-position
                    rotation = orientation_delta(quaternion, orientation)
                    if np.linalg.norm(delta) < .0015 and np.linalg.norm(rotation) < .006:
                        return
                    act([*delta, *rotation, opening], stage)
                raise RuntimeError(f'Waypoint did not settle during {stage}')

            # Open even if the initial gripper is closed; reorient above table/rim.
            act([0, 0, 0, 0, 0, 0, 1], 'open_for_pickup')
            position, orientation = env.controller.end_effector_pose()
            move([*position[:2], max(position[2], .50)], 1, 'clear_start', orientation)
            move([*position[:2], max(position[2], .50)], 1, 'orient_for_grasp')
            move(initial+[0, 0, .14], 1, 'approach')
            move(initial+[0, 0, .003], 1, 'lower_to_grasp')
            act([0, 0, 0, 0, 0, 0, 0], 'close')
            move(initial+[0, 0, .15], 0, 'lift')
            if cube_position(env)[2] < initial[2]+.08:
                raise RuntimeError('Cube did not lift through contact grasp')
            move([*bowl[:2], .424], 0, 'transfer_to_bowl')
            # Release above the rim rather than putting open fingers inside walls.
            move([*bowl[:2], .354], 0, 'lower_above_bowl')
            act([0, 0, 0, 0, 0, 0, 1], 'drop')
            move([*bowl[:2], .424], 1, 'retreat')
            act([0, 0, 0, 0, 0, 0, 1], 'settle_in_bowl')
            result = task_result(initial, bowl, positions, env)
            metadata.update(result, transition_count=len(rows))
            write_json(folder/'episode.json', metadata)
            if not result['success']:
                raise RuntimeError('Cube failed settled bowl-containment checks')
    except Exception as exc:
        metadata.update(success=False, training_eligible=False, error=str(exc), transition_count=len(rows))
        write_json(folder/'episode.json', metadata)
        raise
    return metadata


def replay_episode(folder):
    folder = Path(folder)
    metadata = json.loads((folder/'episode.json').read_text())
    if (metadata['scene_xml_sha256'] != scene_fingerprint() or metadata['mujoco_version'] != mujoco.__version__
        or metadata['action_contract'] != action_contract(camera=CAMERA_NAME) or not metadata['success']):
        raise ValueError('Episode scene, action convention, version, or success mismatch')
    rows = [json.loads(line) for line in (folder/'transitions.jsonl').read_text().splitlines()]
    errors = []
    with PandaEnv(SCENE_PATH, camera=CAMERA_NAME) as env:
        configure_recorded_scene(env.model, metadata)
        restore_state(folder/'initial_state.npz', env)
        initial = cube_position(env)
        positions = [initial]
        for row in rows:
            result = env.controller.apply_action(to_controller_action(row['action']))
            if not result.converged:
                raise RuntimeError(f'Replay IK failed at step {row["index"]}')
            env.controller.step(PHYSICS_STEPS)
            positions.append(cube_position(env))
            with np.load(folder/f'step_{row["index"]:04d}/after_state.npz') as saved:
                errors.append(float(np.max(np.abs(env.data.qpos-saved['qpos']))))
        result = task_result(initial, np.asarray(metadata['scenario']['scene']['bowl_position']), positions, env)
        save_rgb_png(folder/'replay_final.png', env.observe())
    report = dict(result, replayed_transitions=len(rows), maximum_qpos_difference=max(errors),
                  episode_sha256=fingerprint(folder), passed=bool(result['success'] and max(errors) < 1e-6))
    write_json(folder/'replay.json', report)
    metadata['training_eligible'] = report['passed']
    write_json(folder/'episode.json', metadata)
    if not report['passed']:
        raise RuntimeError('Successful trajectory failed deterministic physics replay')
    return report
