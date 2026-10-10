"""Play saved simulator poses offline; never call inference, IK, or physics."""
import argparse
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def write_launcher(folder):
    """Keep each run playable from its output folder within the repository."""
    launcher = '''"""Offline playback. Run with the repository simulation Python environment."""
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from replay_run import main
if __name__ == '__main__':
    main([str(Path(__file__).resolve().parent), *sys.argv[1:]])
'''
    launcher = launcher.replace('sys.path.insert(0, str(root))',
                                f'if not (root / "replay_run.py").exists():\n    root = Path({str(ROOT)!r})\nsys.path.insert(0, str(root))')
    (Path(folder) / 'replay.py').write_text(launcher)


def load_frames(folder, model):
    """Return saved poses, or approximate intermediate poses for older runs."""
    import mujoco
    import numpy as np
    frames = []
    approximate = False
    for step in sorted(Path(folder).glob('step_*')):
        trajectory = step / 'motion_frames.npz'
        if trajectory.exists():
            with np.load(trajectory, allow_pickle=False) as saved:
                poses, times = saved['qpos'].copy(), saved['time'].copy()
            if poses.ndim != 2 or poses.shape[1] != model.nq or len(times) != len(poses):
                raise ValueError(f'Invalid trajectory: {trajectory}')
        else:
            before, after = step / 'scene_state.npz', step / 'after_state.npz'
            if not before.exists():
                continue
            with np.load(before, allow_pickle=False) as saved:
                start, start_time = saved['qpos'].copy(), float(saved['time'])
            if after.exists():
                with np.load(after, allow_pickle=False) as saved:
                    end, end_time = saved['qpos'].copy(), float(saved['time'])
                duration = max(0, end_time - start_time)
                count = max(1, int(round(duration * 25)))
                velocity = np.empty(model.nv)
                mujoco.mj_differentiatePos(model, velocity, 1, start, end)
                poses = []
                for fraction in np.linspace(0, 1, count + 1):
                    pose = start.copy()
                    mujoco.mj_integratePos(model, pose, velocity, float(fraction))
                    poses.append(pose)
                poses[0], poses[-1] = start, end
                times = np.linspace(start_time, end_time, count + 1)
                approximate |= duration > 0
            else:
                poses, times = [start], [start_time]
        for pose, timestamp in zip(poses, times):
            if np.shape(pose) != (model.nq,) or not np.all(np.isfinite(pose)) or not np.isfinite(timestamp):
                raise ValueError(f'Invalid saved pose in {step}')
            frames.append((step.name, float(timestamp), np.asarray(pose).copy()))
    if not frames:
        raise ValueError('No saved simulator states found in this run')
    return frames, approximate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--speed', type=float, default=1, help='Movement playback multiplier; e.g. 0.5 or 2')
    parser.add_argument('--check', action='store_true', help='Validate saved states without opening a viewer')
    args = parser.parse_args(argv)
    import math
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be finite and positive')
    if not args.check:
        from end_effector_demo import _restart_with_mjpython
        _restart_with_mjpython()
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'scene.xml'))
    frames, approximate = load_frames(args.run_dir, model)
    print(f'Loaded {len(frames)} poses from {args.run_dir}')
    if approximate:
        print('Older run: intermediate motion is interpolated between saved before/after poses.')
    else:
        print('Playing recorded motion frames; no actions are recomputed.')
    print('Cloud prediction waits and original pauses are omitted. Uses current repository scene/assets.')
    if args.check:
        return
    import mujoco.viewer
    import glfw
    data = mujoco.MjData(model)
    paused, stopped = True, False
    def key_callback(key):
        nonlocal paused, stopped
        if key == glfw.KEY_SPACE:
            paused = not paused
        elif key == glfw.KEY_ESCAPE:
            stopped = True
    data.qpos[:] = frames[0][2]
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback,
                                     show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'workspace_camera')
        index, remaining = 0, 0.0
        previous = time.perf_counter()
        while viewer.is_running() and not stopped:
            now = time.perf_counter()
            if not paused:
                remaining -= now - previous
                if remaining <= 0 and index < len(frames) - 1:
                    index += 1
                    with viewer.lock():
                        data.qpos[:] = frames[index][2]
                        data.time = frames[index][1]
                        mujoco.mj_forward(model, data)
                    if index < len(frames) - 1:
                        remaining = max(0.001, frames[index + 1][1] - frames[index][1]) / args.speed
                    else:
                        paused = True
            previous = now
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                             f'Offline replay | {frames[index][0]} | pose {index + 1}/{len(frames)}',
                             'Space: start/pause/resume | Esc / close: exit' + (' | Finished' if index == len(frames)-1 else '')))
            viewer.sync()
            time.sleep(0.005)


if __name__ == '__main__':
    main()
