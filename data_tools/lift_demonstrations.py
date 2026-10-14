"""Grasp/lift demonstrations with moving verification and continuous physics replay."""
import json
from pathlib import Path
import mujoco
import numpy as np
from data_tools.bowl_demonstrations import controller_fingerprint
from data_tools.panda_demonstrations import write_json, save_state, restore_state, cube_position
from data_tools.prepare_panda_dataset import fingerprint
from simulation.capture_camera import save_rgb_png
from simulation.ik_retries import apply_with_ik_retries
from simulation.panda_actions import PHYSICS_STEPS, action_contract, to_controller_action, from_controller_action
from simulation.panda_env import PandaEnv
from simulation.panda_start_poses import unsafe_motion_contacts
from simulation.replay_run import write_launcher
from simulation.lift_scene import SCENE_PATH, CAMERA_NAME, INSTRUCTION, initialize_scene, scene_fingerprint, finger_cube_contacts, LiftSuccess
from simulation.bowl_scene import configure_grip


def record_episode(folder, scenario):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    write_launcher(folder, demonstration=True)
    metadata = dict(schema_version=1, scene_profile='panda_lift_v1',
        scene_xml_sha256=scene_fingerprint(), controller_sha256=controller_fingerprint(),
        mujoco_version=mujoco.__version__, camera=CAMERA_NAME, image_size=[640, 480],
        action_contract=action_contract(camera=CAMERA_NAME), instruction=INSTRUCTION,
        scenario=scenario, source='scripted contact grasp and moving lift; continuous bilateral grip above 5 cm for 1 simulator second',
        controller_policy='holding_correction_v1',
        success=False, training_eligible=False)
    rows=[]
    try:
        with PandaEnv(SCENE_PATH, camera=CAMERA_NAME) as env:
            scene=initialize_scene(env, scenario['cube_xy'])
            metadata['scenario']=dict(scenario, scene=scene)
            metadata['starting_pose']=scene['starting_pose']
            metadata['physics_seconds_per_action']=PHYSICS_STEPS*float(env.model.opt.timestep)
            write_json(folder/'episode.json', metadata)
            initial=cube_position(env)
            tracker=LiftSuccess(initial)
            stage='approach_above_cube'
            target=initial+np.array([0,0,.10])
            save_state(folder/'initial_state.npz', env)
            if finger_cube_contacts(env.model, env.data):
                raise RuntimeError('Starting hand already touches the cube')
            for index in range(1, 121):
                directory=folder/f'step_{index:04d}';directory.mkdir()
                before, qbefore=env.controller.end_effector_pose()
                delta=target-before
                if stage in ('approach_above_cube','lower_to_grasp') and np.linalg.norm(delta)<.0015:
                    if stage=='approach_above_cube':
                        stage='lower_to_grasp';target=initial+np.array([0,0,.003]);delta=target-before
                    else:stage='close_gripper'
                if stage=='close_gripper':proposal=np.array([0,0,0,0,0,0,0.])
                elif stage=='lift':proposal=np.array([0,0,.008,0,0,0,0.])
                else:proposal=np.array([*delta,0,0,0,1.])
                if stage!='close_gripper' and np.linalg.norm(proposal[:3])<.0001:
                    raise RuntimeError('Refusing idle demonstration label')
                save_rgb_png(directory/'before.png', env.observe())
                save_state(directory/'before_state.npz', env)
                bounded=to_controller_action(proposal)
                accepted,ik,attempts=apply_with_ik_retries(env.controller,bounded)
                if accepted is None:raise RuntimeError('Lift expert command rejected by IK')
                poses=[env.data.qpos.copy()];times=[float(env.data.time)]
                for count in range(0,PHYSICS_STEPS,20):
                    env.controller.step(min(20,PHYSICS_STEPS-count))
                    if unsafe_motion_contacts(env.model,env.data):raise RuntimeError('Unsafe robot contact')
                    tracker.observe(env)
                    poses.append(env.data.qpos.copy());times.append(float(env.data.time))
                after,qafter=env.controller.end_effector_pose()
                tracking=float(np.linalg.norm(after-env.controller.target_position))
                if not np.isfinite(env.data.qpos).all() or tracking>.005:
                    raise RuntimeError(f'Lift action did not track within tolerance: {tracking:g} m')
                np.savez_compressed(directory/'motion_frames.npz',qpos=poses,time=times)
                save_rgb_png(directory/'after.png',env.observe());save_state(directory/'after_state.npz',env)
                row=dict(index=index,stage=stage,instruction=INSTRUCTION,
                    observation=f'{directory.name}/before.png',next_observation=f'{directory.name}/after.png',
                    proposed_action=proposal.tolist(),bounded_action=from_controller_action(bounded).tolist(),
                    action=from_controller_action(accepted).tolist(),executed=True,
                    action_scale=attempts[-1]['scale'],ik_attempts=attempts,
                    physics_steps=PHYSICS_STEPS,physics_seconds=metadata['physics_seconds_per_action'],
                    control_position_before=before.tolist(),control_position_after=after.tolist(),
                    quaternion_before_wxyz=qbefore.tolist(),quaternion_after_wxyz=qafter.tolist(),
                    measured_translation=(after-before).tolist(),tracking_error_metres=tracking,
                    cube_position=cube_position(env).tolist(),finger_cube_contacts=finger_cube_contacts(env.model,env.data),lift_result=tracker.last)
                write_json(directory/'transition.json',row);rows.append(row)
                with (folder/'transitions.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                if stage=='close_gripper':
                    if not tracker.last['bilateral_finger_contact']:
                        raise RuntimeError('Closing failed to establish bilateral finger contact')
                    stage='lift'
                if tracker.last['success']:break
                if tracker.maximum_lift>.16:raise RuntimeError('Lift height cap reached without retained grip')
            result=tracker.last
            metadata.update(result,transition_count=len(rows))
            if not result['success']:raise RuntimeError('No retained lift within step limit')
            write_json(folder/'episode.json',metadata)
    except Exception as exc:
        metadata.update(success=False,training_eligible=False,error=str(exc),transition_count=len(rows))
        write_json(folder/'episode.json',metadata)
        raise
    return metadata


def replay_episode(folder):
    folder=Path(folder);metadata=json.loads((folder/'episode.json').read_text())
    if (metadata['scene_xml_sha256']!=scene_fingerprint() or
        metadata['controller_sha256']!=controller_fingerprint() or
        metadata['mujoco_version']!=mujoco.__version__ or not metadata['success'] or
        metadata['action_contract']!=action_contract(camera=CAMERA_NAME)):
        raise ValueError('Lift replay scene/controller/contract/version mismatch')
    rows=[json.loads(line) for line in (folder/'transitions.jsonl').read_text().splitlines()]
    errors=[]
    with PandaEnv(SCENE_PATH,camera=CAMERA_NAME) as env:
        configure_grip(env.model)
        restore_state(folder/'initial_state.npz',env);initial=cube_position(env);tracker=LiftSuccess(initial)
        for row in rows:
            ik=env.controller.apply_action(to_controller_action(row['action']))
            if not ik.converged:raise RuntimeError('Recorded lift action rejected')
            for count in range(0,row['physics_steps'],20):
                env.controller.step(min(20,row['physics_steps']-count));tracker.observe(env)
            with np.load(folder/f"step_{row['index']:04d}/after_state.npz") as saved:
                errors.append(float(np.max(np.abs(env.data.qpos-saved['qpos']))))
        result=tracker.last;save_rgb_png(folder/'replay_final.png',env.observe())
    report=dict(result,replayed_transitions=len(rows),maximum_qpos_difference=max(errors),
                episode_sha256=fingerprint(folder),passed=bool(result['success'] and max(errors)<1e-6))
    write_json(folder/'replay.json',report)
    metadata['training_eligible']=report['passed'];write_json(folder/'episode.json',metadata)
    if not report['passed']:raise RuntimeError('Lift demonstration failed continuous physics replay')
    return report
