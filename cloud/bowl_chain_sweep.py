"""Run the expanded sweep without a pod-stop helper or network-volume recordings."""
import json,os,subprocess,sys,time
from pathlib import Path
from common.paths import PROJECT_ROOT
from cloud.bowl_chain_plan import make_plan,trial_specs,validate_saved
from cloud.sweep_worker import wait_server
from cloud.scripted_touch_session import write_reports
from cloud.chain_timing import Timings,utc


def mirror_reports(source,destination):
    import shutil
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('results.json','REPORT.md'):
        p=source/name
        if p.exists():shutil.copyfile(p,destination/name)


def preflight(plan):
    from simulation.panda_env import PandaEnv
    from simulation.color_place_bowl_scene import scene_path,CAMERA_NAME,initialize_scene,slots,cube_position
    import numpy as np
    validate_saved(dict(plan=plan,trials=[]))
    for group in plan['groups']:
        with PandaEnv(scene_path(len(group['colors'])),camera=CAMERA_NAME) as env:
            names=dict(zip(slots(len(group['colors'])),group['colors']))
            for layout in group['layouts']:
                initialize_scene(env,layout,names);image=env.observe()
                if image.shape!=(480,640,3):raise ValueError('Wrong rendered input shape')
                if any(np.linalg.norm(cube_position(env,s)[:2]-xy)>.002 for s,xy in layout['cubes'].items()):raise ValueError('Cube moved during initialization')
    print('All 100 bowl layouts initialized and rendered.',flush=True)


def classify_failure(code,summary,status):
    error=status.get('error','')
    controller_failed=(isinstance(error,str) and error.startswith('Action executed but tracking failed:')) or summary.get('stop_reason')=='consecutive_rejection_limit'
    measured=(summary.get('task_result') or {}).get('outcome') in ('correct_placed','wrong_placed','both_placed','target_lifted_not_placed','wrong_cube_lifted','no_lift')
    cancelled=summary.get('stop_reason') in ('keyboard_interrupt','user_stop','viewer_closed')
    infrastructure_failed=not measured or cancelled or (code!=0 and not controller_failed)
    return bool(controller_failed),bool(infrastructure_failed)


def run(config):
    directory=Path(config['sweep']);persistent=Path(config['session'])/'sweep-reports';timings=Timings(Path(config['session']))
    payload=json.loads((directory/'results.json').read_text());plan=payload['plan'];rows=payload['trials'];total=len(validate_saved(payload))
    plan.pop('compute_stop_reason',None)
    done={r['directory'] for r in rows};specs=[r for r in trial_specs(plan) if r['directory'] not in done]
    env=os.environ.copy();env.pop('RUNPOD_API_KEY',None);env.pop('HF_TOKEN',None);env['MUJOCO_GL']='egl';env['PYOPENGL_PLATFORM']='egl'
    base=Path(config['base']);training=Path(config['training_run']);server=None;started=time.monotonic();error=None
    try:
        from cloud.compute_deadline import check,ComputeDeadlineReached
        for cp in plan['checkpoints']:
            check()
            selected=[s for s in specs if s['checkpoint']==cp]
            if not selected:continue
            # Base is reused from the container staging done before training.
            # Avoid making a second base copy: adapters alone are copied below.
            adapter=Path(config['local'])/f'adapters/checkpoint_{cp or 500:06d}'
            from cloud.checkpoint_ready import check_ready
            with timings.stage(f'checkpoint_wait_{cp:06d}'):
                from cloud.sweep_worker import wait_checkpoint
                source=wait_checkpoint(training/f'checkpoint_{cp or 500:06d}',
                    config.get('checkpoint_wait_seconds',0),Path(config['source_status']) if config.get('source_status') else None)
            if config.get('source_dataset_sha256'):
                saved=json.loads((source/'training_config.json').read_text())
                provenance=saved.get('dataset_provenance',{})
                if (saved.get('checkpoint_update')!=cp or provenance.get('checksums_sha256')!=config['source_dataset_sha256']
                    or provenance.get('dataset_revision')!=config['source_dataset_revision']):raise ValueError('Checkpoint belongs to a different training dataset/update')
            with timings.stage(f'checkpoint_staging_{cp:06d}'):
                if not adapter.exists():
                    import shutil
                    shutil.copytree(source,adapter,ignore=shutil.ignore_patterns('training_state'))
                check_ready(adapter)
                if config.get('verify_adapter_copy'):
                    from cloud.archive_sweep import digest
                    for f in source.rglob('*'):
                        if f.is_file() and 'training_state' not in f.relative_to(source).parts and digest(f)!=digest(adapter/f.relative_to(source)):raise ValueError('Staged adapter checksum mismatch')
            ready=Path(config['local'])/f'ready_{cp:06d}.json';ready.unlink(missing_ok=True)
            log_path=directory/f'cloud_server_{cp:06d}.log'
            with log_path.open('a',buffering=1) as log:
                model_args=['--action-statistics',str(adapter/'dataset_statistics.json')] if cp==0 else ['--adapter',str(adapter)]
                with timings.stage(f'model_loading_{cp:06d}'):
                    server=subprocess.Popen([sys.executable,'-u','-m','cloud.connection_server','--mode','openvla','--device','cuda','--model-dir',str(base),*model_args,'--unnorm-key','panda_grasp_v1','--port','0','--ready-file',str(ready),'--max-runtime-seconds',str(plan['timeout_budget']['inner_seconds'])],cwd=PROJECT_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    health=wait_server(server,0,seconds=1800,ready_file=ready)
                port=health['server_port'];startup=directory/f'startup_cloud_{cp:06d}.json'
                startup.write_text(json.dumps(dict(health.get('startup_info',{}),execution_mode='cloud_sequential',model_label='Base' if cp==0 else f'Checkpoint {cp}'),indent=2)+'\n')
                for spec in selected:
                    check()
                    trial=directory/spec['directory'];positions=Path(config['local'])/'trial_layout.json';positions.write_text(json.dumps(spec['layout']))
                    if trial.exists():raise ValueError('Unexpected unindexed trial directory; resume explicitly after review')
                    command=[config['sim_python'],'-u','-m','inference.remote_loop','--task','multi_place_bowl','--target-color',spec['target_color'],'--cube-colors',*spec['cube_colors'],'--instruction',f"pick up the {spec['target_color']} cube and place it in the blue bowl.",'--scene-seed',str(spec['seed']),'--scene-positions',str(positions),'--steps','120','--no-view','--start-immediately','--startup-info',str(startup),'--output-dir',str(trial),'--server-url',f'http://127.0.0.1:{port}','--max-runtime-seconds','900']
                    print(f"Checkpoint {cp}, {spec['color_set']}, seed {spec['seed']}, target {spec['target_color']}",flush=True)
                    at=utc();trial_start=time.monotonic()
                    with (directory/'trials.log').open('a',buffering=1) as trial_log:
                        code=subprocess.run(command,cwd=PROJECT_ROOT,env=env,stdout=trial_log,stderr=subprocess.STDOUT,timeout=1200).returncode
                    summary=json.loads((trial/'summary.json').read_text()) if (trial/'summary.json').exists() else {}
                    trial_status=json.loads((trial/'status.json').read_text()) if (trial/'status.json').exists() else {}
                    controller_failed,infrastructure_failed=classify_failure(code,summary,trial_status)
                    result=summary.get('task_result') or {}
                    row={k:v for k,v in spec.items() if k!='layout'}
                    row.update(controller_failed=controller_failed,infrastructure_failed=infrastructure_failed,error=trial_status.get('error'),stop_reason=summary.get('stop_reason'),model_label='Base' if cp==0 else f'Checkpoint {cp}',phase=summary.get('phase','Missing summary'),completed_steps=summary.get('completed_steps',0),outcome=result.get('outcome','unmeasured'),placement_success=result.get('success',False),wrong_placement=result.get('wrong_placement',False),cubes=result.get('cubes'),returncode=code,trial_seconds=time.monotonic()-trial_start,started_at_utc=at,finished_at_utc=utc())
                    rows.append(row);plan['elapsed_seconds']=time.monotonic()-started;write_reports(directory,plan,rows);mirror_reports(directory,persistent)
                    print(f'Saved {len(rows)}/{total} trials: {row["outcome"]} ({row["trial_seconds"]:.1f}s)',flush=True)
                    if infrastructure_failed:raise RuntimeError('Trial infrastructure failure; partial results retained')
                    wait_server(server,port,30)
            if server.poll() is None:server.terminate()
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
            server=None
        if len(rows)!=total:raise ValueError(f'Sweep did not finish all {total} trials')
    except ComputeDeadlineReached:
        if len(rows)<total:
            plan['compute_stop_reason']='inner_compute_deadline'
            print('Computation deadline reached; delivering completed trials as an incomplete sweep.',flush=True)
    except BaseException as exc:error=type(exc).__name__;raise
    finally:
        if server is not None:
            if server.poll() is None:server.terminate()
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
        plan['elapsed_seconds']=time.monotonic()-started;write_reports(directory,plan,rows,error);mirror_reports(directory,persistent)
    return 0


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path);p.add_argument('--preflight',type=Path);args=p.parse_args()
    if args.preflight:preflight(json.loads(args.preflight.read_text())['plan'])
    else:run(json.loads(args.config.read_text()))
