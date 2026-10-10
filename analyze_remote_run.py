"""Offline analysis of saved remote trials; no cloud, rendering, or model loading."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def analyze_run(folder):
    import mujoco
    import numpy as np
    from control_diagnostics import pose_diagnostics
    from openvla_single_action import adapt_bridge_action
    from panda_controller import PandaController, _euler_xyz_to_quat, _quat_multiply

    folder = Path(folder)
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'scene.xml'))
    records = []
    failures = []

    def restore(path):
        data = mujoco.MjData(model)
        with np.load(path) as state:
            for key in ('qpos', 'qvel', 'act', 'ctrl'):
                getattr(data, key)[:] = state[key]
            data.time = float(state['time'])
        return PandaController(model, data)

    for directory in sorted(folder.glob('step_*')):
        if not (directory / 'scene_state.npz').exists() or not (directory / 'prediction.json').exists():
            continue
        controller = restore(directory / 'scene_state.npz')
        prediction = json.loads((directory / 'prediction.json').read_text())
        raw = prediction['action']
        row = {'step': int(directory.name.split('_')[1]), 'raw_action': raw,
               'before': pose_diagnostics(controller)}
        if (directory / 'after_state.npz').exists():
            row['after'] = pose_diagnostics(restore(directory / 'after_state.npz'))
        if (directory / 'ik_attempts.json').exists():
            attempts = json.loads((directory / 'ik_attempts.json').read_text())
            row['selected_scale'] = attempts['selected_scale']
            if attempts['selected_scale'] is None:
                action = adapt_bridge_action(raw)
                position, quaternion = controller.end_effector_pose()
                probes = []
                # These tests solve on scratch data only; no actuator/physics changes.
                for scale, iterations, damping in ((1, 150, .03), (1, 1000, .03),
                                                   (1, 1000, .01), (.5, 150, .03),
                                                   (.25, 150, .03), (.125, 150, .03)):
                    controller.damping = damping
                    target_quaternion = _quat_multiply(_euler_xyz_to_quat(*(action[3:6] * scale)), quaternion)
                    result = controller._solve_ik(position + action[:3] * scale, target_quaternion,
                                                  max_iterations=iterations)
                    probes.append({'scale': scale, 'iteration_budget': iterations, 'damping': damping,
                                   'converged': bool(result.converged), 'iterations': int(result.iterations),
                                   'position_error_metres': float(result.position_error),
                                   'orientation_error_radians': float(result.orientation_error)})
                failures.append({'step': row['step'], 'probes': probes})
        records.append(row)
    if not records:
        raise ValueError('No saved observations/predictions found')
    initial = records[0]['before']
    final = records[-1].get('after', records[-1]['before'])
    grippers = [row['raw_action'][6] for row in records]
    return {
        'run': str(folder.resolve()), 'prediction_count': len(records),
        'initial_pose': initial, 'final_pose': final,
        'negative_world_y_prediction_count': sum(row['raw_action'][1] < 0 for row in records),
        'raw_gripper_min_max': [min(grippers), max(grippers)],
        'reduced_action_steps': [{'step': row['step'], 'scale': row['selected_scale']} for row in records
                                 if row.get('selected_scale') is not None and row['selected_scale'] < 1],
        'rejected_pose_solver_probes': failures, 'trajectory': records,
        'limitations': [
            'Hand-origin distance to cube is a drift diagnostic, not a grasp success metric.',
            'Jacobian conditioning mixes position/rotation units and is only compared within this model.',
            'Poor conditioning suggests harder IK; it does not prove a target is physically unreachable.',
            'Bridge world XYZ/Euler to Panda mapping and camera/domain compatibility remain unvalidated.',
            'Joint limits are respected by IK; collision avoidance and task-specific workspace bounds are not implemented.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = analyze_run(args.run_dir)
    output = args.output or args.run_dir / 'trajectory_analysis.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Analysis saved: {output}')
    print(f"Hand/cube distance: {report['initial_pose']['hand_to_cube_distance_metres']:.3f} → {report['final_pose']['hand_to_cube_distance_metres']:.3f} m")
    print(f"Negative world-Y requests: {report['negative_world_y_prediction_count']}/{report['prediction_count']}")
    print(f"Raw gripper range (0 closed, 1 open): {report['raw_gripper_min_max']}")


if __name__ == '__main__':
    main()
