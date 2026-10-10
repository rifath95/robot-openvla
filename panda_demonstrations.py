"""Record and replay scripted expert Panda commands, without model inference."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from capture_camera import save_rgb_png
from panda_actions import (PHYSICS_STEPS, action_contract, from_controller_action,
                           to_controller_action)
from dataset_paths import PANDA_DATASET_ROOT
from panda_env import PandaEnv
from remote_loop import apply_with_ik_retries

ROOT = Path(__file__).resolve().parent
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def scene_fingerprint():
    files = [ROOT / 'scene.xml', ROOT / 'mujoco_menagerie/franka_emika_panda/panda.xml']
    return hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()


def save_state(path, env):
    controller = env.controller
    state = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
    mujoco.mj_getState(env.model, env.data, state, STATE_SPEC)
    np.savez(path, integration=state, target_position=controller.target_position,
             target_quaternion=controller.target_quaternion,
             correction_counter=controller._steps_since_ik,
             qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), time=env.data.time)


def restore_state(path, env):
    with np.load(path) as state:
        mujoco.mj_setState(env.model, env.data, state['integration'], STATE_SPEC)
        env.controller.target_position = state['target_position'].copy()
        env.controller.target_quaternion = state['target_quaternion'].copy()
        env.controller._steps_since_ik = int(state['correction_counter'])
    mujoco.mj_forward(env.model, env.data)


def cube_position(env):
    mujoco.mj_forward(env.model, env.data)
    return env.data.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'cube')].copy()


def task_result(initial, destination, positions):
    final = positions[-1]
    lift = max(p[2] for p in positions) - initial[2]
    error = float(np.linalg.norm(final[:2] - destination[:2]))
    height_error = float(abs(final[2] - initial[2]))
    return dict(success=bool(lift >= 0.08 and error <= 0.015 and height_error <= 0.01),
                maximum_cube_lift_metres=float(lift), placement_xy_error_metres=error,
                final_height_error_metres=height_error, initial_cube=initial.tolist(),
                destination_cube=destination.tolist(), final_cube=final.tolist())


def record_episode(folder, *, cube_xy=None, destination_offset=(0, 0.12), instruction=None, scenario=None):
    offset = np.asarray(destination_offset, dtype=float)
    if offset.shape != (2,) or not np.isfinite(offset).all() or not 0.04 <= np.linalg.norm(offset) <= 0.18:
        raise ValueError('Destination offset must be a finite XY vector of length 4–18 cm')
    if cube_xy is not None:
        cube_xy = np.asarray(cube_xy, dtype=float)
        if cube_xy.shape != (2,) or not np.isfinite(cube_xy).all():
            raise ValueError('Cube XY must contain two finite coordinates')
        for point in (cube_xy, cube_xy + offset):
            if not (.45 <= point[0] <= .65 and -.20 <= point[1] <= .20):
                raise ValueError('Cube/destination is outside the pilot collection area')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    from replay_run import write_launcher
    write_launcher(folder)
    instruction = instruction or ('pick up the red cube and place it 12 cm in the positive world Y direction'
                                  if np.array_equal(offset, [0, .12]) else
                                  f'pick up the red cube and move it {offset[0]*100:g} cm along world X and {offset[1]*100:g} cm along world Y')
    metadata = dict(schema_version=1, action_contract=action_contract(),
                    instruction=instruction, source='scripted expert with privileged cube coordinates',
                    camera='workspace_camera', image_size=[640, 480],
                    mujoco_version=mujoco.__version__, scene_xml_sha256=scene_fingerprint(),
                    scenario=scenario, requested_cube_xy=None if cube_xy is None else cube_xy.tolist(),
                    requested_destination_offset_xy=offset.tolist(),
                    success=False, training_eligible=False,
                    note='Task success and deterministic replay required before dataset preparation.')
    write_json(folder / 'episode.json', metadata)
    rows = []
    try:
        with PandaEnv() as env:
            if cube_xy is not None:
                joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_joint')
                address = int(env.model.jnt_qposadr[joint])
                env.data.qpos[address:address + 2] = cube_xy
                mujoco.mj_forward(env.model, env.data)
            env.controller.step(500)
            initial = cube_position(env)
            destination = initial + [*offset, 0]
            positions = [initial]
            save_state(folder / 'initial_state.npz', env)

            def act(proposal, stage):
                directory = folder / f'step_{len(rows) + 1:04}'
                directory.mkdir()
                before, quaternion_before = env.controller.end_effector_pose()
                save_rgb_png(directory / 'before.png', env.observe())
                save_state(directory / 'before_state.npz', env)
                bounded = to_controller_action(proposal)
                accepted, ik, attempts = apply_with_ik_retries(env.controller, bounded)
                if accepted is None:
                    write_json(directory / 'rejection.json', dict(proposed_action=list(proposal),
                               attempts=attempts, executed=False, training_eligible=False))
                    raise RuntimeError('Expert action rejected; episode is ineligible for training')
                env.controller.step(PHYSICS_STEPS)
                after, quaternion_after = env.controller.end_effector_pose()
                tracking = float(np.linalg.norm(after - env.controller.target_position))
                finite = np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()
                if not finite or tracking > 0.005:
                    raise RuntimeError('Expert action tracking failed')
                save_rgb_png(directory / 'after.png', env.observe())
                save_state(directory / 'after_state.npz', env)
                cube = cube_position(env)
                positions.append(cube)
                row = dict(index=len(rows) + 1, stage=stage, instruction=instruction,
                           observation=f'{directory.name}/before.png',
                           next_observation=f'{directory.name}/after.png',
                           proposed_action=np.asarray(proposal).tolist(),
                           bounded_action=from_controller_action(bounded).tolist(),
                           action=from_controller_action(accepted).tolist(),
                           executed=True, action_scale=attempts[-1]['scale'], ik_attempts=attempts,
                           physics_steps=PHYSICS_STEPS,
                           physics_seconds=PHYSICS_STEPS * float(env.model.opt.timestep),
                           control_position_before=before.tolist(), control_position_after=after.tolist(),
                           quaternion_before_wxyz=quaternion_before.tolist(),
                           quaternion_after_wxyz=quaternion_after.tolist(),
                           measured_translation=(after - before).tolist(),
                           tracking_error_metres=tracking, cube_position=cube.tolist())
                write_json(directory / 'transition.json', row)
                rows.append(row)
                with (folder / 'transitions.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')

            def move(target, gripper, stage):
                print(stage, flush=True)
                for _ in range(100):
                    position, _ = env.controller.end_effector_pose()
                    delta = target - position
                    if np.linalg.norm(delta) < 0.0015:
                        return
                    # The shared adapter bounds it to 1 cm; record its executed value.
                    act([*delta, 0, 0, 0, gripper], stage)
                raise RuntimeError('Expert waypoint did not settle')

            move(initial + [0, 0, 0.14], 1, 'approach')
            move(initial + [0, 0, 0.003], 1, 'lower_to_grasp')
            act([0, 0, 0, 0, 0, 0, 0], 'close')
            move(initial + [0, 0, 0.14], 0, 'lift')
            if cube_position(env)[2] < initial[2] + 0.08:
                raise RuntimeError('Expert grasp failed')
            move(destination + [0, 0, 0.14], 0, 'transfer')
            move(destination + [0, 0, 0.006], 0, 'lower_to_place')
            act([0, 0, 0, 0, 0, 0, 1], 'release')
            move(destination + [0, 0, 0.14], 1, 'retreat')
            result = task_result(initial, destination, positions)
            metadata.update(result)
            metadata.update(transition_count=len(rows),
                            physics_seconds_per_action=PHYSICS_STEPS * float(env.model.opt.timestep))
            write_json(folder / 'episode.json', metadata)
            if not result['success']:
                raise RuntimeError('Expert placement failed')
            print(f'Recorded {len(rows)} transitions: {folder}', flush=True)
    except Exception as exc:
        metadata.update(success=False, training_eligible=False, error=str(exc), transition_count=len(rows))
        write_json(folder / 'episode.json', metadata)
        raise
    return metadata


def replay_episode(folder):
    folder = Path(folder)
    metadata = json.loads((folder / 'episode.json').read_text())
    metadata['training_eligible'] = False
    write_json(folder / 'episode.json', metadata)
    if metadata['action_contract'] != action_contract():
        raise ValueError('Action contract mismatch')
    if metadata['scene_xml_sha256'] != scene_fingerprint() or metadata['mujoco_version'] != mujoco.__version__:
        raise ValueError('Scene or MuJoCo version differs from recorded episode')
    if not metadata['success']:
        raise ValueError('Episode did not pass expert task checks')
    rows = [json.loads(line) for line in (folder / 'transitions.jsonl').read_text().splitlines()]
    errors = []
    with PandaEnv() as env:
        restore_state(folder / 'initial_state.npz', env)
        initial = cube_position(env)
        positions = [initial]
        for row in rows:
            if not row['executed'] or row['physics_steps'] != PHYSICS_STEPS:
                raise ValueError('Rejected or incompatible transition cannot be replayed')
            result = env.controller.apply_action(to_controller_action(row['action']))
            if not result.converged:
                raise RuntimeError(f"Replay IK failed at {row['index']}")
            env.controller.step(PHYSICS_STEPS)
            positions.append(cube_position(env))
            with np.load(folder / f"step_{row['index']:04}/after_state.npz") as saved:
                errors.append(float(np.max(np.abs(env.data.qpos - saved['qpos']))))
        save_rgb_png(folder / 'replay_final.png', env.observe())
        result = task_result(initial, np.asarray(metadata['destination_cube']), positions)
        from prepare_panda_dataset import fingerprint
        report = dict(**result, replayed_transitions=len(rows), episode_sha256=fingerprint(folder),
                      maximum_qpos_difference=max(errors),
                      passed=bool(result['success'] and max(errors) < 1e-6))
        write_json(folder / 'replay.json', report)
        metadata['training_eligible'] = report['passed']
        write_json(folder / 'episode.json', metadata)
        print(json.dumps(report, indent=2), flush=True)
        if not report['passed']:
            raise RuntimeError('Replay differs from recorded successful demonstration')
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, help='Replay an existing episode instead of recording')
    parser.add_argument('--output-dir', type=Path, default=PANDA_DATASET_ROOT / 'episodes' /
                        datetime.now().strftime('panda_demo_%Y%m%d_%H%M%S_%f'))
    args = parser.parse_args()
    if args.replay:
        replay_episode(args.replay)
    else:
        record_episode(args.output_dir)
        replay_episode(args.output_dir)


if __name__ == '__main__':
    main()
