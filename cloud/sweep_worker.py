"""Sequential headless MuJoCo trials entirely on the pod; fixed source checkout."""
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


def summarize(spec,folder,returncode):
    summary=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else {}
    status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
    outcome=summary.get('task_result') or {}
    row=dict(checkpoint=spec['checkpoint'],seed=spec['seed'],directory=spec['directory'],
        phase=summary.get('phase','Missing summary'),completed_steps=summary.get('completed_steps',0),
        contact=outcome.get('correct_cube_contact',outcome.get('success',False)),error=status.get('error'),
        returncode=returncode,execution_mode='cloud_sequential',finished_at_utc=datetime.now(timezone.utc).isoformat())
    if 'outcome' in outcome:
        row.update(target_color=spec['target_color'],wrong_contact=outcome.get('wrong_cube_contact',False),
            outcome=outcome['outcome'],first_contact=outcome.get('first_contact'))
    if spec['cube_colors'] is not None:row.update(color_pair=spec['color_pair'],cube_colors=spec['cube_colors'])
    return row


def run(directory,sim_python,*,port=0):
    directory=Path(directory);payload=json.loads((directory/'results.json').read_text())
    validate_saved(payload);plan=payload['plan'];rows=payload['trials']
    previous=float(plan.get('elapsed_seconds',0));started=time.monotonic()
    done={r['directory'] for r in rows};specs=[s for s in trial_specs(plan) if s['directory'] not in done]
    checkpoints=sorted({s['checkpoint'] for s in specs})
    base=Path('/workspace/openvla-7b')
    training=Path('/workspace/panda-training/sessions')/plan['training_session']/'runs/training'
    adapter_indices={n for n in checkpoints if n!=0 or not plan.get('include_base_model')}
    if 0 in checkpoints and plan.get('include_base_model'):adapter_indices.add(plan.get('action_statistics_checkpoint',50))
    staging=stage_weights(base,{n:training/f'checkpoint_{n:06d}' for n in sorted(adapter_indices)},Path('/root/panda-sweep-cache')/directory.name)
    local=PROJECT_ROOT/'outputs/cloud-sweeps'/directory.name;local.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.pop('RUNPOD_API_KEY',None);env['MUJOCO_GL']='egl';env['PYOPENGL_PLATFORM']='egl'
    # Render once before expensive model loading; a dependency/graphics failure is explicit.
    if plan['task'] in ('color_touch','color_transfer'):
        seed_check="from data_tools.collect_panda_color_touch import pair_spec; from cloud.sweep_plan import validate_scene_positions; validate_scene_positions([pair_spec(s)['cube_positions'] for s in "+repr(plan['scene_seeds'])+"], "+repr(plan['cube_positions'])+"); "
    else:seed_check=''
    probe=seed_check+"from simulation.panda_env import PandaEnv; from simulation.color_touch_scene import SCENE_PATH,CAMERA_NAME; e=PandaEnv(SCENE_PATH,camera=CAMERA_NAME); print(e.observe().shape); e.close()"
    subprocess.run([str(sim_python),'-c',probe],cwd=PROJECT_ROOT,env=env,check=True,timeout=120)
    plan.setdefault('execution_segments',[]).append(dict(mode='cloud_sequential',started_at_utc=datetime.now(timezone.utc).isoformat(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,text=True).strip(),remaining_trials=len(specs),
        model_staging=str(staging),mujoco_rendering='egl',batch_size=1))
    server=None
    error=None
    try:
        for checkpoint in checkpoints:
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
                            '--instruction',f"touch the {spec['target_color']} cube.",'--scene-seed',str(spec['seed']),
                            '--steps',str(plan['steps']),'--no-view','--start-immediately','--startup-info',str(startup),
                            '--output-dir',str(trial),'--server-url',f'http://127.0.0.1:{selected_port}',
                            '--max-runtime-seconds','180']
                        if plan['task']!='touch':
                            positions_file=local/f"scene_positions_{spec['seed']}.json"
                            positions_file.write_text(json.dumps(plan['cube_positions'][plan['scene_seeds'].index(spec['seed'])]))
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
