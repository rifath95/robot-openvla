"""Detached cloud sweep and Mac-driven watchdog with independent stop deadlines."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common.paths import PROJECT_ROOT
from training.benchmark_stop import stop_pod
from training.benchmark_supervisor import kill_benchmark_tree
from training.detached_run import preflight,write_status
from training.pod_environment import recover_pod_environment
from cloud.scripted_eval_control import active_training
from cloud.sweep_plan import timeout_budget

SWEEP_ROOT=Path('/workspace/panda-training/scripted-evaluations')


def supervise(directory,pod_id,key,*,mode,estimated_seconds,sim_python=None,heartbeat_grace=180):
    directory=Path(directory)
    budget=timeout_budget(estimated_seconds)
    saved=json.loads((directory/'results.json').read_text()) if (directory/'results.json').exists() else {}
    if saved.get('plan',{}).get('task')=='multi_place_bowl':
        from cloud.bowl_chain_plan import timeout_budget as bowl_budget
        budget=bowl_budget(estimated_seconds)
    started=time.monotonic();process=None
    status=dict(phase='running',mode=mode,pod_id=pod_id,started_at_utc=datetime.now(timezone.utc).isoformat(),
        timeout_budget=budget,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,text=True).strip())
    def persist():
        try:write_status(directory/'supervisor_status.json',status)
        except OSError:pass
    persist()
    try:
        if mode=='cloud':
            env=os.environ.copy();env.pop('RUNPOD_API_KEY',None)
            log=(directory/'console.log').open('a',buffering=1)
            process=subprocess.Popen([sys.executable,'-u','-m','cloud.sweep_worker','--directory',str(directory),
                '--sim-python',str(sim_python)],cwd=PROJECT_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while True:
            elapsed=time.monotonic()-started
            if elapsed>=budget['outer_seconds']:status['outcome']='outer_limit';break
            if elapsed>=budget['inner_seconds']:
                status['outcome']='inner_limit';break
            if mode=='cloud' and process.poll() is not None:
                status.update(outcome='completed' if process.returncode==0 else 'failed',exit_code=process.returncode,phase='awaiting_sync')
                persist()
                # A connected watcher may fetch everything. No Mac is required to stop.
                sync_deadline=min(started+budget['outer_seconds'],time.monotonic()+1800)
                while time.monotonic()<sync_deadline:
                    if (directory/'sync_ack').exists():break
                    heartbeat=directory/'heartbeat'
                    if not heartbeat.exists() or time.time()-heartbeat.stat().st_mtime>60:break
                    time.sleep(1)
                break
            if mode=='client_watchdog':
                heartbeat=directory/'heartbeat'
                age=time.time()-heartbeat.stat().st_mtime if heartbeat.exists() else elapsed
                if (directory/'client_finished').exists():status['outcome']='client_finished';break
                if age>heartbeat_grace:status['outcome']='mac_disconnected';break
            time.sleep(1)
    except BaseException as exc:
        status.update(outcome='supervisor_failed',error=str(exc))
    finally:
        if process is not None and process.poll() is None:
            try:kill_benchmark_tree(process)
            except OSError:status['worker_cleanup_error']='Worker cleanup failed'
        status.update(phase='stopping',elapsed_seconds=time.monotonic()-started,finished_at_utc=datetime.now(timezone.utc).isoformat())
        persist()
        result=stop_pod(pod_id,key,directory/'pod_stop_status.json',reason=f"Simulator sweep: {status.get('outcome')}")
        status.update(phase='finished',stop_state='accepted' if result==0 else 'failed');persist()
    return 0 if status.get('outcome') in ('completed','client_finished') and result==0 else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--mode',choices=['cloud','client_watchdog'],required=True)
    p.add_argument('--estimated-seconds',type=float,required=True);p.add_argument('--sim-python',type=Path)
    p.add_argument('--launch',action='store_true');p.add_argument('--lock-fd',type=int)
    args=p.parse_args();directory=args.directory.resolve()
    if directory.parent!=SWEEP_ROOT:p.error('Use a direct sweep directory under the network volume')
    timeout_budget(args.estimated_seconds)
    if args.mode=='cloud' and args.sim_python is None:p.error('Cloud mode requires simulation interpreter')
    if args.launch:
        if active_training():raise RuntimeError('Training is active on this pod; refusing to arm automatic stopping')
        preflight();directory.mkdir(parents=True,exist_ok=True)
        lock=Path('/root/panda-sweep-supervisor.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        command=[sys.executable,'-u','-m','cloud.sweep_supervisor','--directory',str(directory),
            '--mode',args.mode,'--estimated-seconds',str(args.estimated_seconds),'--lock-fd',str(lock.fileno())]
        if args.sim_python:command+=['--sim-python',str(args.sim_python)]
        with (directory/'supervisor.log').open('a') as log:
            subprocess.Popen(command,cwd=PROJECT_ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=(lock.fileno(),))
        lock.close()
        print(f'Detached simulator supervisor started: {directory}',flush=True)
        return 0
    if args.lock_fd is None:p.error('Launch using --launch')
    pod_id,key=recover_pod_environment()
    return supervise(directory,pod_id,key,mode=args.mode,estimated_seconds=args.estimated_seconds,sim_python=args.sim_python)


if __name__=='__main__':raise SystemExit(main())
