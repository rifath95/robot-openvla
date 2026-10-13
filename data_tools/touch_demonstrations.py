"""Translation-only contact demonstrations, with independent action replay."""
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
from simulation.touch_scene import SCENE_PATH, CAMERA_NAME, INSTRUCTION, initialize_scene, scene_fingerprint, finger_cube_contacts


def task_result(env, initial):
    final = cube_position(env)
    contacts = finger_cube_contacts(env.model, env.data)
    return dict(success=bool(contacts), finger_cube_contacts=contacts,
                initial_cube=np.asarray(initial).tolist(), final_cube=final.tolist(),
                cube_displacement_metres=float(np.linalg.norm(final-initial)))


def record_episode(folder, scenario):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    write_launcher(folder, demonstration=True)
    metadata = dict(schema_version=1, scene_profile='panda_touch_v1',
        scene_xml_sha256=scene_fingerprint(), controller_sha256=controller_fingerprint(),
        mujoco_version=mujoco.__version__, camera=CAMERA_NAME, image_size=[640, 480],
        action_contract=action_contract(camera=CAMERA_NAME), instruction=INSTRUCTION,
        scenario=scenario, source='scripted translation toward cube; verified physical finger contact',
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
            target=initial+np.array([0, 0, .023])
            save_state(folder/'initial_state.npz', env)
            if finger_cube_contacts(env.model, env.data):
                raise RuntimeError('Starting hand already touches the cube')
            for index in range(1, 101):
                directory=folder/f'step_{index:04d}';directory.mkdir()
                before, qbefore=env.controller.end_effector_pose()
                proposal=np.array([*(target-before), 0, 0, 0, 0])
                if np.linalg.norm(proposal[:3]) < .0001:
                    raise RuntimeError('Reached target without finger contact; refusing idle labels')
                save_rgb_png(directory/'before.png', env.observe())
                save_state(directory/'before_state.npz', env)
                bounded=to_controller_action(proposal)
                accepted,ik,attempts=apply_with_ik_retries(env.controller,bounded)
                if accepted is None:raise RuntimeError('Touch expert command rejected by IK')
                poses=[env.data.qpos.copy()];times=[float(env.data.time)]
                for count in range(0,PHYSICS_STEPS,20):
                    env.controller.step(min(20,PHYSICS_STEPS-count))
                    if unsafe_motion_contacts(env.model,env.data):raise RuntimeError('Unsafe robot contact')
                    poses.append(env.data.qpos.copy());times.append(float(env.data.time))
                after,qafter=env.controller.end_effector_pose()
                tracking=float(np.linalg.norm(after-env.controller.target_position))
                if not np.isfinite(env.data.qpos).all() or tracking>.005:
                    raise RuntimeError('Touch action did not track within tolerance')
                np.savez_compressed(directory/'motion_frames.npz',qpos=poses,time=times)
                save_rgb_png(directory/'after.png',env.observe());save_state(directory/'after_state.npz',env)
                row=dict(index=index,stage='move_to_touch',instruction=INSTRUCTION,
                    observation=f'{directory.name}/before.png',next_observation=f'{directory.name}/after.png',
                    proposed_action=proposal.tolist(),bounded_action=from_controller_action(bounded).tolist(),
                    action=from_controller_action(accepted).tolist(),executed=True,
                    action_scale=attempts[-1]['scale'],ik_attempts=attempts,
                    physics_steps=PHYSICS_STEPS,physics_seconds=metadata['physics_seconds_per_action'],
                    control_position_before=before.tolist(),control_position_after=after.tolist(),
                    quaternion_before_wxyz=qbefore.tolist(),quaternion_after_wxyz=qafter.tolist(),
                    measured_translation=(after-before).tolist(),tracking_error_metres=tracking,
                    cube_position=cube_position(env).tolist(),finger_cube_contacts=finger_cube_contacts(env.model,env.data))
                write_json(directory/'transition.json',row);rows.append(row)
                with (folder/'transitions.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                if row['finger_cube_contacts']:break
            result=task_result(env,initial)
            metadata.update(result,transition_count=len(rows))
            if not result['success']:raise RuntimeError('No finger contact within step limit')
            if any(row['finger_cube_contacts'] for row in rows[:-1]):
                raise RuntimeError('Expert continued after first verified contact')
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
        raise ValueError('Touch replay scene/controller/contract/version mismatch')
    rows=[json.loads(line) for line in (folder/'transitions.jsonl').read_text().splitlines()]
    errors=[]
    with PandaEnv(SCENE_PATH,camera=CAMERA_NAME) as env:
        restore_state(folder/'initial_state.npz',env);initial=cube_position(env)
        for row in rows:
            ik=env.controller.apply_action(to_controller_action(row['action']))
            if not ik.converged:raise RuntimeError('Recorded touch action rejected')
            env.controller.step(row['physics_steps'])
            with np.load(folder/f"step_{row['index']:04d}/after_state.npz") as saved:
                errors.append(float(np.max(np.abs(env.data.qpos-saved['qpos']))))
        result=task_result(env,initial);save_rgb_png(folder/'replay_final.png',env.observe())
    report=dict(result,replayed_transitions=len(rows),maximum_qpos_difference=max(errors),
                episode_sha256=fingerprint(folder),passed=bool(result['success'] and max(errors)<1e-6))
    write_json(folder/'replay.json',report)
    metadata['training_eligible']=report['passed'];write_json(folder/'episode.json',metadata)
    if not report['passed']:raise RuntimeError('Touch demonstration failed continuous physics replay')
    return report
