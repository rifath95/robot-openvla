"""Detached bowl training/sweep/backup chain with independent hard-stop watchdog."""
import argparse,fcntl,getpass,json,os,subprocess,sys,time,uuid
from datetime import datetime,timezone
from pathlib import Path
from common.paths import PROJECT_ROOT
from cloud.bowl_chain_plan import timeout_budget
from cloud.chain_timing import Timings,utc,read_events
from training.detached_run import preflight,api_request,write_status


def stop_request(config,reason):
    """Avoid a global os.sync(): a stalled filesystem must not block the API call."""
    started=time.monotonic()
    receipt=dict(reason=reason,requested_at_utc=utc(),stop_state='requested')
    for attempt in range(1,4):
        receipt['attempt']=attempt
        try:
            api_request(config['pod_id'],config['pod_key'],stop=True);receipt['stop_state']='accepted';break
        except RuntimeError as exc:
            receipt.update(stop_state='failed',error=str(exc))
            if attempt<3:time.sleep(5)
    receipt.update(finished_at_utc=utc(),wall_seconds=time.monotonic()-started)
    # Write local status first; storage I/O cannot delay the stop request itself.
    Path(config['local'],'pod_stop_status.json').write_text(json.dumps(receipt,indent=2)+'\n')
    try:write_status(Path(config['session'])/'pod_stop_status.json',receipt)
    except OSError:pass
    if receipt['stop_state']=='accepted':Path(config['local'],'guard_done').write_text('accepted\n')
    print('Pod stop '+receipt['stop_state']+'. Confirm stopped status in RunPod.',flush=True)
    return receipt['stop_state']=='accepted'


def kill_group(pid):
    # Torch elastic workers can create their own sessions. Kill descendant groups,
    # not just the outer worker PGID; never kill this supervisor/watchdog group.
    from types import SimpleNamespace
    from training.benchmark_supervisor import kill_benchmark_tree
    kill_benchmark_tree(SimpleNamespace(pid=pid,wait=lambda **kwargs:None))


def process_birth(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,IndexError):return None


def watchdog(config):
    local=Path(config['local']);deadline=config['outer_deadline_monotonic']-90
    while time.monotonic()<deadline:
        if (local/'guard_done').exists():return 0
        time.sleep(min(5,max(0,deadline-time.monotonic())))
    # Kill work before stopping; local PID records contain no credentials.
    try:ids=json.loads((local/'processes.json').read_text())
    except (OSError,ValueError):ids={}
    for pid in ids.get('children',[]):
        expected=ids.get('births',{}).get(str(pid))
        if expected is None or process_birth(pid)==expected:kill_group(pid)
    main=ids.get('supervisor')
    if main and (ids.get('supervisor_birth') is None or process_birth(main)==ids['supervisor_birth']):
        kill_group(main)
    return 0 if stop_request(config,'Independent outer deadline reached') else 1


def recovery(config):
    from cloud.chain_archive import recover_to_global
    from cloud.checkpoint_recovery import recover_checkpoints,record
    session=Path(config['session']);source=Path(config['sweep'])
    # Recovery must not depend on being able to write the failed network volume.
    timings=Timings(Path(config['local'])/'recovery-timings');errors=[]
    try:
        if config.get('checkpoint_recovery'):
            with timings.stage('checkpoint_global_recovery'):recover_checkpoints(config)
    except Exception as exc:
        errors.append(dict(stage='checkpoints',error=type(exc).__name__+': '+str(exc)))
    if not source.exists():
        record(config,'recovery_errors.json',errors)
        return 1 if errors else 0
    # A final small-report failure must not recopy recordings already verified on HF.
    try:
        from cloud.archive_sweep import digest
        saved=json.loads((session/'sweep_hf_receipt.json').read_text())
        if (saved.get('verified') is True and saved.get('repo_id')==config['sweep_repo']
            and saved.get('path_in_repo')==source.name
            and saved.get('source_revision')==digest(source/'results.json')):
            print('Sweep backup already verified; no global recovery copy needed.',flush=True)
            record(config,'recovery_errors.json',errors)
            return 1 if errors else 0
    except (OSError,ValueError):pass
    try:
        with timings.stage('global_recovery'):
            from cloud.chain_archive import prepare_recovery_archive
            packed=prepare_recovery_archive(source,Path(config['local']))
            receipt=recover_to_global(packed,Path(config['global_volume'])/'panda-chain-recovery'/session.name)
            record(config,'global_recovery_receipt.json',receipt)
    except Exception as exc:
        errors.append(dict(stage='sweep',error=type(exc).__name__+': '+str(exc)))
    record(config,'recovery_errors.json',errors)
    return 1 if errors else 0


def attempt_recovery(config,child_ids,pids):
    remaining=config['outer_deadline_monotonic']-time.monotonic()-120
    if remaining<=0:return 'no_time_remaining'
    safe={k:v for k,v in config.items() if k not in ('pod_key','hf_token')}
    with (Path(config['local'])/'recovery.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-u','-m','cloud.bowl_chain','--recover'],cwd=PROJECT_ROOT,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        child_ids.append(child.pid);pids();child.stdin.write(json.dumps(safe).encode());child.stdin.close()
        try:return child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            kill_group(child.pid)
            return 'timeout'


def worker_module(config):
    choices=[(flag,module) for flag,module in (
        ('continuation_chain','cloud.color_bowl_continuation'),
        ('training_only','cloud.color_bowl_training'),('sweep_only','cloud.color_bowl_sweep')) if config.get(flag)]
    if len(choices)>1:raise ValueError('Conflicting chain worker modes')
    return choices[0][1] if choices else 'cloud.bowl_chain_worker'


def interrupted_timings(session):
    path=Path(session)/'stage_timings.jsonl'
    if not path.exists():return
    rows=read_events(path);finished={r['stage'] for r in rows if r['event']=='finished'};timer=Timings(session)
    for row in rows:
        if row['event']=='started' and row['stage'] not in finished:
            elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(row['at_utc'])).total_seconds()
            timer.event(dict(stage=row['stage'],event='finished',started_at_utc=row['at_utc'],finished_at_utc=utc(),seconds=elapsed,status='interrupted',error='Worker exited or timed out'))
    timer.report()


def supervise(config):
    local=Path(config['local']);session=Path(config['session']);budget=config['timeout_budget'];worker=None;child_ids=[];outcome='failed';started=time.monotonic()
    config.update(started_at_utc=utc(),started_monotonic=started)
    config['outer_deadline_monotonic']=started+budget['outer_seconds'];inner=started+budget['inner_seconds']
    cooperative=config.get('deadline_policy')=='compute_then_delivery_v1'
    if cooperative:config['compute_deadline_monotonic']=inner
    def pids():write_status(local/'processes.json',dict(supervisor=os.getpid(),supervisor_birth=process_birth(os.getpid()),children=child_ids,births={str(pid):process_birth(pid) for pid in child_ids}))
    safe={k:v for k,v in config.items() if k not in ('pod_key','hf_token')};
    guard={k:v for k,v in config.items() if k!='hf_token'}
    # Arm the independent guard before launching work. It is not in a worker process group.
    with (local/'watchdog.log').open('a',buffering=1) as log:
        watchdog_process=subprocess.Popen([sys.executable,'-u','-m','cloud.bowl_chain','--watchdog'],cwd=PROJECT_ROOT,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        watchdog_process.stdin.write(json.dumps(guard).encode());watchdog_process.stdin.close()
    pids()
    write_status(session/'chain_config.json',safe)
    status=dict(phase='training' if config.get('training_only') else 'running',started_at_utc=utc(),timeout_budget=budget,code_commit=config['code_commit'],pod_id=config['pod_id'])
    write_status(session/'status.json',status)
    try:
        env=os.environ.copy();env.pop('RUNPOD_API_KEY',None);env.pop('HF_TOKEN',None)
        if cooperative:env['PANDA_COMPUTE_DEADLINE_MONOTONIC']=str(inner)
        with (local/'console.log').open('a',buffering=1) as log:
            payload={k:v for k,v in config.items() if k!='pod_key'}
            module=worker_module(config)
            worker=subprocess.Popen([sys.executable,'-u','-m',module],cwd=PROJECT_ROOT,env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            child_ids.append(worker.pid);pids();worker.stdin.write(json.dumps(payload).encode());worker.stdin.close()
            try:
                code=worker.wait(timeout=max(.01,(config['outer_deadline_monotonic']-120 if cooperative else inner)-time.monotonic()));outcome='completed' if code==0 else 'failed';status['exit_code']=code
                if cooperative and code==0 and (session/'worker_completion.json').is_file():
                    delivered=json.loads((session/'worker_completion.json').read_text())
                    if delivered.get('work_complete') is False and delivered.get('delivery_verified') is True:
                        outcome='incomplete';status['delivery_verified']=True
            except subprocess.TimeoutExpired:outcome='outer_timeout' if cooperative else 'inner_timeout';kill_group(worker.pid);worker.wait(timeout=10)
        try:interrupted_timings(session)
        except OSError:print('Volume timing writes failed; local recovery remains available.',flush=True)
    except BaseException as exc:
        status['supervisor_error']=type(exc).__name__
        if worker is not None and worker.poll() is None:kill_group(worker.pid)
    finally:
        if outcome!='completed' and not (outcome=='incomplete' and status.get('delivery_verified')):
            # Also run on volume/log/timing exceptions, not just worker exit codes.
            # Keep recovery independent of persistent status writes and bounded
            # by the unchanged outer watchdog deadline.
            try:status['recovery_exit_code']=attempt_recovery(config,child_ids,pids)
            except Exception as exc:status['recovery_error']=type(exc).__name__+': '+str(exc)
        # All persistent checkpoints are retained regardless of the outcome.
        status.update(phase=outcome,finished_at_utc=utc(),elapsed_seconds=time.monotonic()-started)
        try:
            if config.get('training_only'):
                # Make final diagnostics available during the synchronization
                # window, before acknowledgment permits stopping.
                import shutil
                if (local/'console.log').exists():shutil.copyfile(local/'console.log',session/'supervisor_console.log')
                completion=session/'training_completion.json'
                if completion.exists():status.update(json.loads(completion.read_text()))
                receipt=session/'checkpoint_hf_receipt.json'
                status['hf_backup_verified']=receipt.exists() and json.loads(receipt.read_text()).get('verified') is True
                if outcome=='completed' and not (status.get('training_complete') and status['hf_backup_verified']):
                    outcome='failed';status['phase']=outcome
                status['job_complete']=outcome=='completed' and bool(status.get('training_complete')) and status['hf_backup_verified']
                from training.detached_run import wait_for_download
                wait_for_download(session,status,config['outer_deadline_monotonic']-120,
                    0 if outcome in ('inner_timeout','outer_timeout') else config.get('download_grace_seconds',120))
            write_status(session/'status.json',status)
            for name in ('console.log','recovery.log','watchdog.log'):
                if (local/name).exists():
                    import shutil
                    shutil.copyfile(local/name,session/name)
        except OSError:
            write_status(local/'status.json',status)
            print('Final volume reports unavailable; status and recovery receipts retained on container/global storage.',flush=True)
        finally:
            accepted=stop_request(config,'Bowl training/sweep/backup chain: '+outcome)
    return 0 if outcome=='completed' and accepted else 1


def launch(profile,global_volume):
    from cloud.scripted_eval_control import active_training
    if active_training():raise ValueError('Use a separate pod with no active training')
    pod,key=preflight()
    config=json.loads(Path(profile).read_text());budget=timeout_budget(config['estimated_seconds'])
    if not global_volume.is_absolute():raise ValueError('Global recovery mount must be absolute')
    # Its actual read/write probe runs inside the guarded worker.
    token=getpass.getpass('Hugging Face WRITE token (hidden): ').strip()
    if not token or '{{' in token:raise ValueError('A resolved HF write token is required')
    from huggingface_hub import HfApi
    HfApi(token=token).whoami()
    name='panda-train-chain-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
    session=Path('/workspace/panda-training/sessions')/name;local=Path('/root/panda-chain-work')/name
    session.mkdir(parents=True,exist_ok=False);local.mkdir(parents=True,exist_ok=False)
    lock=Path('/root/panda-long-chain.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    code=local/'code';commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,text=True).strip()
    subprocess.run(['git','clone','--shared',str(PROJECT_ROOT),str(code)],check=True)
    subprocess.run(['git','-C',str(code),'checkout','--detach',commit],check=True)
    sweep_name='scripted_bowl_chain_'+name.removeprefix('panda-train-chain-').replace('-','_')
    config.update(session=str(session),local=str(local),base_volume='/workspace/openvla-7b',base=str(local/'base'),dataset=str(local/'training_place_bowl_v2_300'),training_run=str(session/'runs/training'),sweep=str(local/'scripted-evaluations'/sweep_name),global_volume=str(global_volume),timeout_budget=budget,pod_id=pod,pod_key=key,hf_token=token,code_commit=commit)
    env=os.environ.copy();env.pop('RUNPOD_API_KEY',None);env.pop('HF_TOKEN',None)
    with (local/'supervisor.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-u','-m','cloud.bowl_chain','--supervise'],cwd=code,env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
        child.stdin.write(json.dumps(config).encode());child.stdin.close()
    print('Detached long bowl chain started. Mac may disconnect.',flush=True)
    print('Session:',name,flush=True);print('Watch:',local/'console.log',flush=True);print('Persistent reports:',session,flush=True)
    print(f'Inner limit: {budget["inner_seconds"]/3600:g} hours; independent outer limit: {budget["outer_seconds"]/3600:g} hours. Checkpoints retained on volume.',flush=True)
    print('Global recovery mount:',global_volume,flush=True)
    print('Supervisor setup status:',local/'supervisor.log',flush=True)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--launch',action='store_true');m.add_argument('--supervise',action='store_true');m.add_argument('--watchdog',action='store_true');m.add_argument('--recover',action='store_true')
    p.add_argument('--profile',type=Path,default=PROJECT_ROOT/'inference/chain_profiles/place_bowl_300_1000_trials.json');p.add_argument('--global-volume',type=Path,default=Path('/workspace-global'))
    a=p.parse_args()
    if a.launch:return launch(a.profile,a.global_volume)
    config=json.load(sys.stdin)
    if a.watchdog:return watchdog(config)
    if a.recover:return recovery(config)
    try:return supervise(config)
    except BaseException as exc:
        print('Chain supervisor failed: '+type(exc).__name__,flush=True)
        stop_request(config,'Chain supervisor startup or finalization failure');return 1


if __name__=='__main__':raise SystemExit(main())
