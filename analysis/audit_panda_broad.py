"""Audit measured diversity, replay eligibility and saved path contacts."""

from common.paths import PROJECT_ROOT
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from simulation.panda_actions import make_controller
from simulation.panda_controller import _orientation_error
from simulation.panda_start_poses import unsafe_motion_contacts
from data_tools.prepare_panda_dataset import fingerprint


def audit(folder):
    folder = Path(folder)
    report = json.loads((folder / 'collection_report.json').read_text())
    rows = report['results']
    model = mujoco.MjModel.from_xml_path(str((PROJECT_ROOT / 'scene.xml')))
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'scene_home')
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)
    controller = make_controller(model, data)
    home_quaternion = controller.end_effector_pose()[1]
    measurements, frame_count = [], 0
    combinations = {s: set() for s in ('train', 'validation', 'test')}
    for row in rows:
        episode = Path(row['episode'])
        replay = json.loads((episode / 'replay.json').read_text())
        if not replay['passed'] or replay['episode_sha256'] != fingerprint(episode):
            raise ValueError(f'Episode changed or failed replay: {episode}')
        with np.load(episode / 'initial_state.npz') as saved:
            data.qpos[:] = saved['qpos']
            if not np.isfinite(saved['qvel']).all() or np.max(np.abs(saved['qvel'])) > .02:
                raise ValueError('Recorded initial state was not settled')
        mujoco.mj_forward(model, data)
        position, quaternion = controller.end_effector_pose()
        actual_yaw = float(_orientation_error(quaternion, home_quaternion)[2])
        np.testing.assert_allclose(position, row['starting_pose']['actual_position'], atol=1e-10)
        if abs(actual_yaw - row['starting_pose']['requested_yaw_radians']) > .03:
            raise ValueError('Actual starting yaw differs from declared pose')
        if unsafe_motion_contacts(model, data):
            raise ValueError(f'Unsafe initial contact: {episode}')
        measurements.append(dict(id=row['id'], split=row['split'], position=position.tolist(),
                                 actual_yaw_degrees=float(np.rad2deg(actual_yaw)),
                                 hand_cube_distance_metres=float(np.linalg.norm(position-row['initial_cube']))))
        combinations[row['split']].add(tuple(row['combination']))
        for path in sorted(episode.glob('step_*/motion_frames.npz')):
            with np.load(path) as saved:
                for pose in saved['qpos']:
                    data.qpos[:] = pose
                    mujoco.mj_forward(model, data)
                    joints = data.qpos[controller.qpos_indices]
                    if np.any(joints < controller.joint_limits[:, 0]) or np.any(joints > controller.joint_limits[:, 1]):
                        raise ValueError(f'Joint limit violated: {path}')
                    contacts = unsafe_motion_contacts(model, data)
                    if contacts:
                        raise ValueError(f'Unsafe saved contact: {path}: {contacts}')
                    frame_count += 1
    for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if combinations[left] & combinations[right]:
            raise ValueError('Held-out combinations leaked between splits')
    poses = np.array([m['position'] for m in measurements])
    yaws = [m['actual_yaw_degrees'] for m in measurements]
    distances = [m['hand_cube_distance_metres'] for m in measurements]
    summary = dict(passed=True, verified_episodes=len(rows), saved_frames_checked=frame_count,
                   split_counts={s: sum(r['split']==s for r in rows) for s in combinations},
                   actual_hand_span_metres=np.ptp(poses, axis=0).tolist(),
                   actual_start_yaw_range_degrees=[min(yaws), max(yaws)],
                   initial_hand_cube_distance_range_metres=[min(distances), max(distances)],
                   failed_candidate_attempts=len(report['failures']), measurements=measurements,
                   note='Contacts checked at recorded 20-physics-step intervals, not continuous path planning. '
                        'Task success and full deterministic replay are checked separately per episode.')
    (folder / 'coverage_audit.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# Broad demonstration collection audit', '',
             f"Verified trajectories: **{len(rows)}**. Splits: **{summary['split_counts']}**.", '',
             '| Measured starting-pose variation | Value |', '|---|---|',
             f"| Hand X / Y / Z spans | {' / '.join(f'{x*100:.1f} cm' for x in np.ptp(poses, axis=0))} |",
             f'| Starting yaw range | {min(yaws):.1f}° to {max(yaws):.1f}° |',
             f'| Initial hand-to-cube distance | {min(distances)*100:.1f}–{max(distances)*100:.1f} cm |', '',
             f'All trajectories passed actual grasp, ≥8 cm lift, placement within 1.5 cm, '
             f'and deterministic physics replay within 1e-6 qpos. {frame_count:,} saved motion poses '
             'passed joint-limit and sampled contact checks. Failed candidates stay excluded.', '',
             'New hand-quadrant / cube-quadrant / initial-yaw-sign combinations are disjoint '
             'between train, validation and test. The legacy 50-trajectory corpus is retained separately.', '',
             'The expert uses known cube coordinates. The training inputs remain RGB image, instruction '
             'and the executed seven-component action; initial setup teleports are not training actions.', '',
             'Contact checks inspect recorded intervals, not every point of a continuous planned path. '
             'Coverage is broad within this table and downward-gripper/yaw family; it does not include '
             'arbitrary roll/pitch or every possible joint configuration.', '',
             '[Choose a trajectory to replay](WATCH_TRAJECTORIES.md). Each episode has replay.py.', '']
    (folder / 'COLLECTION_SUMMARY.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in summary.items() if k!='measurements'}, indent=2))
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    audit(parser.parse_args().folder)
