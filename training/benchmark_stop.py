"""Check or request a stop of this benchmark pod using its existing API secret."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import time

from training.detached_run import api_request, preflight, write_status


def stop_pod(pod_id,key,status_path):
    status=dict(pod_id=pod_id,requested_at_utc=datetime.now(timezone.utc).isoformat(),
                stop_state='requested',reason='Benchmark reports synchronized to the Mac')
    write_status(status_path,status)
    os.sync()
    for attempt in range(1,4):
        status['stop_attempt']=attempt
        try:
            api_request(pod_id,key,stop=True)
            status['stop_state']='accepted'
            write_status(status_path,status)
            print('RunPod accepted the stop request. Confirm stopped status in the portal.',flush=True)
            return 0
        except RuntimeError as exc:
            status.update(stop_state='failed',stop_error=str(exc))
            write_status(status_path,status)
            if attempt<3:time.sleep(5)
    print('Automatic stop failed. Stop the pod manually in RunPod.',flush=True)
    return 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    try:
        pod_id,key=preflight()
        if args.check_only:
            print('Benchmark API and network-volume checks passed. Pod has not been stopped.',flush=True)
            return 0
        directory=Path('/workspace/panda-training/batch-benchmarks')
        directory.mkdir(parents=True,exist_ok=True)
        return stop_pod(pod_id,key,directory/'pod_stop_status.json')
    except (RuntimeError,OSError) as exc:
        print(f'Automatic stop unavailable: {exc}. Stop the pod manually in RunPod.',flush=True)
        return 1


if __name__=='__main__':raise SystemExit(main())
