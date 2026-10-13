"""Execute demonstration actions through IK and physics from one initial state."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

import mujoco
import numpy as np

from common.paths import PROJECT_ROOT
from data_tools.panda_demonstrations import restore_state, save_state, cube_position
from simulation.panda_actions import action_contract, make_controller, to_controller_action
from simulation.replay_run import replay_model, write_launcher


def replay_actions(folder,*,no_view=False,speed=1,output_dir=None,max_steps=None,controller_mode='current'):
    folder=Path(folder).resolve()
    if not math.isfinite(speed) or speed<=0:raise ValueError('Speed must be finite and positive')
    if max_steps is not None and max_steps<1:raise ValueError('max_steps must be positive')
    if controller_mode not in ('current','collection'):raise ValueError('Unknown controller mode')
    metadata=json.loads((folder/'episode.json').read_text())
    rows=[json.loads(line) for line in (folder/'transitions.jsonl').read_text().splitlines()]
    if not rows or any(row.get('executed') is not True for row in rows):
        raise ValueError('Expected executed demonstration transitions')
    if [r['index'] for r in rows]!=list(range(1,len(rows)+1)):
        raise ValueError('Demonstration action sequence is not contiguous')
    model,camera=replay_model(folder)
    if metadata.get('scene_profile')=='panda_bowl_v1':
        from data_tools.bowl_demonstrations import scene_fingerprint,controller_fingerprint
        if metadata.get('scene_xml_sha256')!=scene_fingerprint():
            raise ValueError('Scene assets differ from the recorded demonstration')
        if metadata.get('controller_sha256') and (
                controller_mode!='current' or metadata['controller_sha256']!=controller_fingerprint()):
            raise ValueError('Controller differs from the version recorded for this demonstration')
    if metadata['action_contract']!=action_contract(camera=camera):
        raise ValueError('Demonstration action convention differs from the current controller')
    data=mujoco.MjData(model)
    env=SimpleNamespace(model=model,data=data,controller=make_controller(model,data))
    restore_state(folder/'initial_state.npz',env)
    # Restore once only. Subsequent states must result from executing the actions.
    initial=cube_position(env);positions=[initial.copy()]
    output=Path(output_dir) if output_dir else PROJECT_ROOT/'outputs'/f'action_replay_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}'
    output.mkdir(parents=True,exist_ok=False)
    (output/'episode.json').write_text(json.dumps(dict(metadata,training_eligible=False,
        replay_source=str(folder)),indent=2)+'\n')
    write_launcher(output)
    save_state(output/'initial_state.npz',env)
    print(f'Action replay results: {output}',flush=True)
    print('Executing saved action labels with current IK/controller/physics; no model and no per-step pose resets.',flush=True)
    print(f'Controller target-setting behavior: {controller_mode}',flush=True)
    paused=not no_view
    stopped=False
    results=[]
    reason=None

    def key_callback(key):
        nonlocal paused,stopped
        import glfw
        if key==glfw.KEY_SPACE:paused=not paused
        elif key==glfw.KEY_ESCAPE:stopped=True

    context=nullcontext(None)
    if not no_view:
        from mujoco import viewer as mujoco_viewer
        context=mujoco_viewer.launch_passive(model,data,key_callback=key_callback,
                                            show_left_ui=False,show_right_ui=False)
    def ready(viewer,label):
        nonlocal stopped
        while viewer is not None:
            if not viewer.is_running() or stopped:
                stopped=True
                return False
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                             f'Action replay | {label}',
                             'Space: start/pause/resume | Esc / close: stop'+(' | Paused' if paused else '')))
            viewer.sync()
            if not paused:return True
            time.sleep(.01)
        return not stopped

    try:
        with context as viewer:
            if viewer is not None:
                viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,camera)
            for row in rows[:max_steps]:
                label=f"step {row['index']}/{len(rows)} | {row.get('stage','')}"
                if not ready(viewer,label):break
                directory=output/f"step_{row['index']:04d}";directory.mkdir()
                save_state(directory/'before_state.npz',env)
                action=to_controller_action(row['action'])
                with viewer.lock() if viewer is not None else nullcontext():
                    ik=env.controller.apply_action(action)
                    if ik.converged and controller_mode=='collection':
                        # Collection predates the holding-offset preservation fix.
                        data.ctrl[env.controller.actuator_ids]=ik.joint_targets
                if not ik.converged:
                    reason=f"IK rejected recorded action at step {row['index']}"
                    break
                poses=[data.qpos.copy()];times=[float(data.time)]
                executed=0
                physics_steps=row['physics_steps']
                if type(physics_steps) is not int or physics_steps<1:
                    raise ValueError('Invalid recorded physics step count')
                while executed<physics_steps and ready(viewer,label):
                    started=time.perf_counter();count=min(20,physics_steps-executed)
                    with viewer.lock() if viewer is not None else nullcontext():
                        env.controller.step(count)
                        poses.append(data.qpos.copy());times.append(float(data.time))
                    executed+=count
                    if viewer is not None:
                        viewer.sync()
                        time.sleep(max(0,count*model.opt.timestep/speed-(time.perf_counter()-started)))
                save_state(directory/'after_state.npz',env)
                np.savez_compressed(directory/'motion_frames.npz',qpos=poses,time=times)
                positions.append(cube_position(env))
                with np.load(folder/f"step_{row['index']:04d}"/'after_state.npz') as saved:
                    difference=float(np.max(np.abs(data.qpos-saved['qpos'])))
                result=dict(index=row['index'],stage=row.get('stage'),action=row['action'],
                            executed_physics_steps=executed,
                            recorded_qpos_max_difference=difference,
                            tracking_error_metres=float(np.linalg.norm(env.controller.end_effector_pose()[0]-env.controller.target_position)),
                            cube_position=positions[-1].tolist())
                results.append(result)
                (directory/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
                if not np.isfinite(data.qpos).all():raise RuntimeError('Nonfinite simulator state')
                if stopped:break
            if viewer is not None and not stopped and reason is None:
                print('Action sequence finished. Close window or press Esc to exit.',flush=True)
                paused=True
                while ready(viewer,'Finished'):
                    time.sleep(.01)
    except KeyboardInterrupt:
        stopped=True
    except Exception as exc:
        reason=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        completed=sum(r['executed_physics_steps']==rows[r['index']-1]['physics_steps'] for r in results)
        report=dict(source=str(folder),replay_type='actions_through_current_physics',
                    controller_mode=controller_mode,mujoco_version=mujoco.__version__,
                    completed_actions=completed,total_actions=len(rows),steps=results,
                    outcome='failed' if reason else 'completed' if completed==len(rows) else 'partial',
                    reason=reason,initial_cube=initial.tolist(),final_cube=cube_position(env).tolist(),
                    maximum_cube_lift_metres=float(max(p[2] for p in positions)-initial[2]),
                    maximum_recorded_qpos_difference=max((r['recorded_qpos_max_difference'] for r in results),default=0),
                    note='Recorded-pose differences are measured, not corrected. Collection mode reproduces the pre-holding-fix actuator target assignment, not an arbitrary historical code version.')
        if metadata.get('scene_profile')=='panda_bowl_v1':
            from data_tools.bowl_demonstrations import task_result
            report['task_result']=task_result(initial,np.asarray(metadata['scenario']['scene']['bowl_position']),positions,env)
        (output/'action_replay_report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f"Action replay {report['outcome']}: {completed}/{len(rows)} actions; report {output/'action_replay_report.json'}",flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path)
    parser.add_argument('--no-view',action='store_true')
    parser.add_argument('--speed',type=float,default=1)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--controller',choices=['current','collection'],default='current',
                        help='Current fixed controller, or pre-holding-fix target assignment used during collection')
    args=parser.parse_args(argv)
    if not args.no_view:
        from simulation.viewer import _restart_with_mjpython
        _restart_with_mjpython()
    replay_actions(args.episode,no_view=args.no_view,speed=args.speed,
                   output_dir=args.output_dir,max_steps=args.max_steps,controller_mode=args.controller)


if __name__=='__main__':main()
