"""Detached benchmark watchdog: cloud wall limit and best-effort pod stopping."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from training.benchmark_stop import stop_pod
from training.detached_run import ROOT, preflight, write_status

BENCHMARK_ROOT=Path('/workspace/panda-training/batch-benchmarks')


class BenchmarkAlreadyRunning(RuntimeError):pass



def kill_benchmark_tree(process,proc_root=Path('/proc')):
    # Candidates deliberately have separate process groups for their own timers.
    # Include those groups, not only the outer benchmark's group.
    rows=[]
    for stat in proc_root.glob('[0-9]*/stat'):
        try:
            fields=stat.read_text().rsplit(')',1)[1].split()
            rows.append((int(stat.parent.name),int(fields[1]),int(fields[2])))
        except (OSError,ValueError,IndexError):continue
    descendants={process.pid}
    while True:
        expanded=descendants|{pid for pid,parent,_ in rows if parent in descendants}
        if expanded==descendants:break
        descendants=expanded
    groups={group for pid,_,group in rows if pid in descendants}|{process.pid}
    groups.discard(os.getpgrp())
    for group in groups:
        try:os.killpg(group,signal.SIGKILL)
        except ProcessLookupError:pass
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:pass  # Proceed to the pod stop even if a worker is stuck.


def supervise(command,directory,pod_id,key,*,hard_limit=3600,sync_grace=120):
    started=time.monotonic()
    status=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),pod_id=pod_id,
                hard_limit_seconds=hard_limit,sync_grace_seconds=sync_grace,
                phase='benchmarking',stop_state='not_requested',command=command)
    path=directory/'status.json'
    process=None
    try:
        write_status(path,status)
        env=os.environ.copy();env.pop('RUNPOD_API_KEY',None)
        with (directory/'console.log').open('w',buffering=1) as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=process.wait(timeout=max(0,hard_limit-(time.monotonic()-started)))
                status.update(exit_code=code,outcome='completed' if code==0 else 'failed')
            except subprocess.TimeoutExpired:
                # Kill the benchmark process group, including the training workers.
                kill_benchmark_tree(process)
                status.update(exit_code=process.returncode,outcome='wall_limit_reached')
        status.update(phase='awaiting_sync',benchmark_finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic()-started)
        write_status(path,status)
        # Leave a bounded chance to fetch reports. The outer deadline still wins.
        deadline=min(started+hard_limit,time.monotonic()+sync_grace)
        while time.monotonic()<deadline:
            time.sleep(min(1,deadline-time.monotonic()))
    except Exception as exc:
        status.update(phase='wrapper_failed',outcome='wrapper_failed',error=str(exc))
    finally:
        if process is not None and process.poll() is None:
            try:kill_benchmark_tree(process)
            except OSError:status['worker_cleanup_error']='Could not confirm all workers exited'
        status.update(phase='stopping',finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic()-started,stop_state='requested')
        try:write_status(path,status)
        except OSError:pass
        result=stop_pod(pod_id,key,directory/'pod_stop_status.json',
                        reason=f"Benchmark supervisor cleanup: {status.get('outcome','wrapper_failed')}")
        status.update(phase='finished',stop_state='accepted' if result==0 else 'failed')
        try:write_status(path,status)
        except OSError:pass
    return 0 if status.get('outcome')=='completed' and status['stop_state']=='accepted' else 1


def watch(directory):
    """Stream progress over SSH; disconnecting this watcher leaves the worker alive."""
    offset=0
    while True:
        log=directory/'console.log'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(offset);chunk=stream.read();offset=stream.tell()
            if chunk:
                sys.stdout.buffer.write(chunk);sys.stdout.buffer.flush()
        path=directory/'status.json'
        if path.exists():
            status=json.loads(path.read_text())
            if status['phase'] in ('awaiting_sync','stopping','finished'):
                return 0 if status.get('outcome')=='completed' else 1
        time.sleep(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch',action='store_true')
    p.add_argument('--session-dir',type=Path)
    p.add_argument('--lock-fd',type=int)
    p.add_argument('--dataset',default='/workspace/panda-training/training_v1')
    p.add_argument('--max-batch',type=int,default=64)
    p.add_argument('--controlled',action='store_true')
    p.add_argument('--hard-limit-seconds',type=int,default=3600)
    p.add_argument('--sync-grace-seconds',type=int,default=120)
    args=p.parse_args()
    if args.hard_limit_seconds<1 or args.sync_grace_seconds<0 or args.max_batch<1:
        p.error('Timeout/batch limit must be positive; sync grace cannot be negative')
    try:
        if args.launch:
            pod_id,key=preflight()
            BENCHMARK_ROOT.mkdir(parents=True,exist_ok=True)
            lock=(BENCHMARK_ROOT/'supervisor.lock').open('a')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise BenchmarkAlreadyRunning('A benchmark supervisor is already running; refusing another job') from None
            stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
            directory=BENCHMARK_ROOT/'sessions'/stamp
            directory.mkdir(parents=True,exist_ok=False)
            write_status(directory/'status.json',dict(phase='starting'))
            worker=[sys.executable,'-u','-m','training.benchmark_supervisor',
                    '--session-dir',str(directory),'--lock-fd',str(lock.fileno()),
                    '--dataset',args.dataset,'--max-batch',str(args.max_batch),
                    '--hard-limit-seconds',str(args.hard_limit_seconds),
                    '--sync-grace-seconds',str(args.sync_grace_seconds)]
            if args.controlled:worker.append('--controlled')
            with (directory/'supervisor.log').open('w') as log:
                subprocess.Popen(worker,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                                 stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
            lock.close()  # The detached worker keeps the same lock open.
            print(f'Cloud benchmark supervisor: {directory}',flush=True)
            print(f'Outer limit: {args.hard_limit_seconds}s; post-benchmark download window: {args.sync_grace_seconds}s.',flush=True)
            print('Mac disconnects do not cancel cloud cleanup. Confirm stopped status in RunPod.',flush=True)
            return watch(directory)
        if args.session_dir is None or args.lock_fd is None:
            p.error('Worker requires --session-dir and --lock-fd; use --launch')
        directory=args.session_dir.resolve()
        if not directory.is_relative_to(BENCHMARK_ROOT/'sessions'):
            p.error('Session directory must be inside the benchmark sessions directory')
        from training.pod_environment import recover_pod_environment
        pod_id,key=recover_pod_environment()
        command=[sys.executable,'-u',str(ROOT/'benchmark_training_batches.py'),
                 '--dataset',args.dataset,'--max-batch',str(args.max_batch)]
        if not args.controlled:command.append('--find-capacity')
        return supervise(command,directory,pod_id,key,hard_limit=args.hard_limit_seconds,
                         sync_grace=args.sync_grace_seconds)
    except (RuntimeError,OSError,ValueError,subprocess.CalledProcessError) as exc:
        print(f'Benchmark not started: {exc}. Check and stop the pod manually if still running.',file=sys.stderr)
        return 75 if isinstance(exc,BenchmarkAlreadyRunning) else 1


if __name__=='__main__':raise SystemExit(main())
