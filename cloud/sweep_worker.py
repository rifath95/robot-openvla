"""Sequential headless MuJoCo trials entirely on the pod; fixed source checkout."""

from common.output_layout import output_path
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen
from common.paths import PROJECT_ROOT
from cloud.sweep_plan import trial_specs,validate_saved
from cloud.sweep_storage import stage_weights
from cloud.scripted_touch_session import write_reports


def wait_server(process,port,seconds=900,ready_file=None):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if process.poll() is not None:raise RuntimeError('Cloud inference server exited; inspect server log')
        try:
            selected=port
            if ready_file is not None:
                selected=json.loads(Path(ready_file).read_text())['port']
            if type(selected) is not int or not 1<=selected<=65535:raise ValueError('Invalid server port')
            with urlopen(f'http://127.0.0.1:{selected}/health',timeout=2) as response:health=json.load(response)
            if health.get('model_loaded') is True:
                health['server_port']=selected
                return health
        except (OSError,ValueError):pass
        time.sleep(1)
    raise TimeoutError('Cloud model readiness timeout')


def wait_checkpoint(folder,seconds=0,status_path=None):
    """Wait only for a complete save; never load a checkpoint being written."""
    from cloud.checkpoint_ready import check_ready
    deadline=time.monotonic()+seconds;announced=False
    while True:
        try:return check_ready(folder)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            if not seconds or time.monotonic()>=deadline:
                raise RuntimeError(f'Checkpoint did not finish saving within {seconds}s: {folder}') from exc
        if status_path is not None and Path(status_path).is_file():
            try:status=json.loads(Path(status_path).read_text())
            except (OSError,ValueError):status={}
            if status.get('phase') in ('completed','incomplete','training_failed','wrapper_failed','wall_limit_reached','awaiting_sync','failed','inner_timeout'):
                raise RuntimeError(f'Training ended without required complete checkpoint: {folder}')
        if not announced:
            print(f'Waiting for fully saved checkpoint: {folder} (up to {seconds}s)',flush=True);announced=True
        time.sleep(min(5,max(0,deadline-time.monotonic())))


def summarize(spec,folder,returncode):
    summary=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else {}
    status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
    outcome=summary.get('task_result') or {}
    row=dict(checkpoint=spec['checkpoint'],seed=spec['seed'],directory=spec['directory'],
        phase=summary.get('phase','Missing summary'),completed_steps=summary.get('completed_steps',0),
        contact=outcome.get('correct_cube_contact',outcome.get('success',False)),error=status.get('error'),
        returncode=returncode,execution_mode='cloud_sequential',finished_at_utc=datetime.now(timezone.utc).isoformat())
    if 'correct_cube_lift' in outcome:
        row.update(correct_lift=outcome['correct_cube_lift'],wrong_lift=outcome['wrong_cube_lift'],cubes=outcome['cubes'])
    if 'cube_settled' in outcome:
        row.update(placement_success=outcome.get('success',False),
            maximum_lift_metres=outcome['maximum_cube_lift_metres'],
            bowl_center_distance_metres=outcome['bowl_center_distance_metres'],
            empty_hand_cube_distance_metres=outcome['empty_hand_cube_distance_metres'])
    elif 'maximum_cube_lift_metres' in outcome:
        row.update(lift_success=outcome.get('success',False),
            maximum_lift_metres=outcome['maximum_cube_lift_metres'],
            maximum_grasp_seconds=outcome['maximum_continuous_grasp_seconds'])
    if 'outcome' in outcome:
        row.update(target_color=spec['target_color'],wrong_contact=outcome.get('wrong_cube_contact',False),
            outcome=outcome['outcome'],first_contact=outcome.get('first_contact'))
    if spec['cube_colors'] is not None:row.update(color_pair=spec['color_pair'],cube_colors=spec['cube_colors'])
    if spec.get('cube_count'):
        row.update(cube_count=spec['cube_count'],color_set=spec['color_set'],contacted_colors=outcome.get('contacted_colors',[]))
        row.setdefault('target_color',spec['target_color'])
        row.setdefault('outcome','unmeasured')
    return row


def run(directory,sim_python,*,port=0):
    directory=Path(directory);payload=json.loads((directory/'results.json').read_text())
    validate_saved(payload);plan=payload['plan'];rows=payload['trials']
    previous=float(plan.get('elapsed_seconds',0));started=time.monotonic()
    if plan['task']=='multi_color_touch':
        rows[:]=[r for r in rows if r.get('outcome')!='unmeasured' and r.get('phase')!='Missing summary']
    done={r['directory'] for r in rows};specs=[s for s in trial_specs(plan) if s['directory'] not in done]
    checkpoints=sorted({s['checkpoint'] for s in specs})
    minimum=plan.get('minimum_results_free_gib',0)
    if minimum and rows:
        minimum=min(minimum,2+max(0,minimum-2)*len(specs)/len(list(trial_specs(plan))))
    if minimum and shutil.disk_usage(directory).free<minimum*1024**3:
        raise RuntimeError(f'Need {minimum:.1f} GiB free on the results volume before this sweep')
    base=Path('/workspace/openvla-7b')
    training=Path('/workspace/panda-training/sessions')/plan['training_session']/'runs/training'
    pending_wait=plan.get('checkpoint_wait_seconds',0)
    incremental=bool(pending_wait)
    if incremental:
        first=plan.get('action_statistics_checkpoint',50) if 0 in checkpoints and plan.get('include_base_model') else checkpoints[0]
        wait_checkpoint(training/f'checkpoint_{first:06d}',pending_wait,training.parent.parent/'status.json')
        staged_adapters={first:training/f'checkpoint_{first:06d}'}
    else:
        adapter_indices={n for n in checkpoints if n!=0 or not plan.get('include_base_model')}
        if 0 in checkpoints and plan.get('include_base_model'):adapter_indices.add(plan.get('action_statistics_checkpoint',50))
        staged_adapters={n:training/f'checkpoint_{n:06d}' for n in sorted(adapter_indices)}
    cache=Path('/root/panda-sweep-cache')/directory.name
    if incremental and (cache/'staging.json').is_file():
        cached=json.loads((cache/'staging.json').read_text())
        allowed=set(plan['checkpoints'])|{plan.get('action_statistics_checkpoint',50)}
        for name in cached['sources']:
            if name=='base':continue
            index=int(name.removeprefix('checkpoint_'))
            if index not in allowed:raise ValueError('Staged checkpoint outside sweep plan')
            staged_adapters[index]=training/f'checkpoint_{index:06d}'
    staging=stage_weights(base,dict(staged_adapters),cache)
    local=output_path('cloud-sweeps',root=PROJECT_ROOT/'outputs')/directory.name;local.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.pop('RUNPOD_API_KEY',None);env['MUJOCO_GL']='egl';env['PYOPENGL_PLATFORM']='egl'
    # Render once before expensive model loading; a dependency/graphics failure is explicit.
    if plan['task'] in ('color_touch','color_transfer','color_lift'):
        seed_check="from data_tools.collect_panda_color_touch import pair_spec; from cloud.sweep_plan import validate_scene_positions; validate_scene_positions([pair_spec(s)['cube_positions'] for s in "+repr(plan['scene_seeds'])+"], "+repr(plan['cube_positions'])+"); "
    else:seed_check=''
    probe=seed_check+"from simulation.panda_env import PandaEnv; from simulation.color_touch_scene import SCENE_PATH,CAMERA_NAME; e=PandaEnv(SCENE_PATH,camera=CAMERA_NAME); print(e.observe().shape); e.close()"
    if plan['task']=='color_lift':
        probe=seed_check+"from simulation.panda_env import PandaEnv; from simulation.color_lift_scene import SCENE_PATH,CAMERA_NAME,initialize_scene; e=PandaEnv(SCENE_PATH,camera=CAMERA_NAME); initialize_scene(e,"+repr(plan['cube_positions'][0])+", {'red':'red','blue':'blue'}); print(e.observe().shape); e.close()"
    if plan['task']=='place_bowl':
        probe="from simulation.panda_env import PandaEnv; from simulation.place_bowl_scene import SCENE_PATH,CAMERA_NAME,initialize_scene; from inference.place_bowl_task import layout,validate_positions; import numpy as np; validate_positions("+repr(plan['cube_positions'])+",[layout(s) for s in "+repr(plan['scene_seeds'])+"]); e=PandaEnv(SCENE_PATH,camera=CAMERA_NAME)\nfor slot in "+repr(plan['scene_seeds'])+":\n    e.controller,s=initialize_scene(e.model,e.data,np.random.default_rng(0),slot=slot); e.observe()\nprint('All bowl layouts initialized and rendered'); e.close()"
    if plan['task']=='lift':
        probe="from simulation.panda_env import PandaEnv; from simulation.lift_scene import SCENE_PATH,CAMERA_NAME,initialize_scene; e=PandaEnv(SCENE_PATH,camera=CAMERA_NAME); initialize_scene(e,"+repr(plan['cube_positions'][0]['red'])+"); print(e.observe().shape); e.close()"
    if plan['task']=='multi_color_touch':
        probe="from simulation.panda_env import PandaEnv; from simulation.multi_color_scene import scene_path,CAMERA_NAME,positions; from cloud.sweep_plan import validate_multi_scene_positions; "
        for n in plan['cube_counts']:
            for i,seed in enumerate(plan['scene_seeds']):
                probe+=f"validate_multi_scene_positions(positions({seed},{n}),{plan['multi_cube_positions'][str(n)][i]!r}); "
            probe+=f"e=PandaEnv(scene_path({n}),camera=CAMERA_NAME); print(e.observe().shape); e.close(); "
    subprocess.run([str(sim_python),'-c',probe],cwd=PROJECT_ROOT,env=env,check=True,timeout=120)
    plan.setdefault('execution_segments',[]).append(dict(mode='cloud_sequential',started_at_utc=datetime.now(timezone.utc).isoformat(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,text=True).strip(),remaining_trials=len(specs),
        model_staging=str(staging),mujoco_rendering='egl',batch_size=1))
    server=None
    error=None
    try:
        for checkpoint in checkpoints:
            is_base=checkpoint==0 and plan.get('include_base_model')
            if incremental and not is_base and checkpoint not in staged_adapters:
                wait_checkpoint(training/f'checkpoint_{checkpoint:06d}',pending_wait,training.parent.parent/'status.json')
                staged_adapters[checkpoint]=training/f'checkpoint_{checkpoint:06d}'
                staging=stage_weights(base,dict(staged_adapters),cache,incremental=True)
            selected=[s for s in specs if s['checkpoint']==checkpoint]
            server_log=directory/f'cloud_server_{checkpoint:06d}.log'
            ready_file=local/f'server_ready_{checkpoint:06d}.json'
            ready_file.unlink(missing_ok=True)
            with server_log.open('a',buffering=1) as log:
                is_base=checkpoint==0 and plan.get('include_base_model')
                model_args=(['--action-statistics',str(staging/f"checkpoint_{plan.get('action_statistics_checkpoint',50):06d}"/'dataset_statistics.json')]
                    if is_base else ['--adapter',str(staging/f'checkpoint_{checkpoint:06d}')])
                server=subprocess.Popen([sys.executable,'-u','-m','cloud.connection_server','--mode','openvla',
                    '--model-dir',str(staging/'base'),*model_args,
                    '--unnorm-key','panda_grasp_v1','--device','cuda','--port',str(port),'--ready-file',str(ready_file),
                    '--max-runtime-seconds',str(plan['timeout_budget']['inner_seconds'])],
                    cwd=PROJECT_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                try:
                    health=wait_server(server,port,ready_file=ready_file)
                    selected_port=health.get('server_port',port)
                    startup=directory/f'startup_cloud_{checkpoint:06d}.json'
                    startup_info=dict(health.get('startup_info',{}),execution_mode='cloud_sequential',server_port=selected_port,
                        adapter_checkpoint=None if is_base else str(training/f'checkpoint_{checkpoint:06d}'),
                        model_label='Pretrained base' if is_base else f'Checkpoint {checkpoint}')
                    startup.write_text(json.dumps(startup_info,indent=2)+'\n')
                    for spec in selected:
                        trial=local/spec['directory']
                        if trial.exists():trial.rename(trial.with_name(trial.name+f'_interrupted_{time.time_ns()}'))
                        command=[str(sim_python),'-u','-m','inference.remote_loop','--task',plan['task'],
                            '--instruction',('pick up the red cube and place it in the blue bowl.' if plan['task']=='place_bowl' else f"pick up the {spec['target_color']} cube." if plan['task']=='color_lift' else 'pick up the red cube.' if plan['task']=='lift' else f"touch the {spec['target_color']} cube."),'--scene-seed',str(spec['seed']),
                            '--steps',str(plan['steps']),'--no-view','--start-immediately','--startup-info',str(startup),
                            '--output-dir',str(trial),'--server-url',f'http://127.0.0.1:{selected_port}',
                            '--max-runtime-seconds','180']
                        if plan['task'] in ('lift','place_bowl'):
                            positions_file=local/f"lift_positions_{spec['seed']}.json"
                            positions_file.write_text(json.dumps(plan['cube_positions'][plan['scene_seeds'].index(spec['seed'])]))
                            command+=['--scene-positions',str(positions_file)]
                        elif plan['task']!='touch':
                            positions_file=local/f"scene_positions_{spec.get('cube_count',2)}_{spec['seed']}.json"
                            layouts=plan['multi_cube_positions'][str(spec['cube_count'])] if plan['task']=='multi_color_touch' else plan['cube_positions']
                            positions_file.write_text(json.dumps(layouts[plan['scene_seeds'].index(spec['seed'])]))
                            command+=['--target-color',spec['target_color'],'--scene-positions',str(positions_file)]
                        if spec['cube_colors']:command+=['--cube-colors',*spec['cube_colors']]
                        print(f"Checkpoint {checkpoint}, seed {spec['seed']}, target {spec['target_color']}",flush=True)
                        with (directory/'trials.log').open('a',buffering=1) as log:
                            try:code=subprocess.run(command,cwd=PROJECT_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=420).returncode
                            except subprocess.TimeoutExpired:code='timeout'
                        row=summarize(spec,trial,code)
                        row['model_label']='Pretrained base' if is_base else f'Checkpoint {checkpoint}'
                        destination=directory/spec['directory']
                        if destination.exists():destination.rename(destination.with_name(destination.name+f'_interrupted_{time.time_ns()}'))
                        if trial.exists():shutil.copytree(trial,destination)
                        rows.append(row);plan['elapsed_seconds']=previous+time.monotonic()-started
                        write_reports(directory,plan,rows)
                        print(f"Saved {len(rows)} trials: {row['phase']}; contact={row['contact']}",flush=True)
                        if (row.get('outcome')=='unmeasured' or row['phase']=='Missing summary'
                            or (row.get('error') and 'Server returned HTTP' in row['error'])):
                            raise RuntimeError('Trial infrastructure failed; diagnostics saved instead of continuing')
                        wait_server(server,selected_port,30)
                finally:
                    if server.poll() is None:server.terminate()
                    try:server.wait(timeout=30)
                    except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
                    server=None
    except BaseException as exc:
        error=str(exc) or type(exc).__name__;raise
    finally:
        plan['elapsed_seconds']=previous+time.monotonic()-started
        write_reports(directory,plan,rows,error)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--sim-python',type=Path,required=True);a=p.parse_args()
    raise SystemExit(run(a.directory,a.sim_python))
