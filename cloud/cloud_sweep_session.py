"""Resume scripted trials on a detached cloud MuJoCo worker, then fetch results."""
import argparse
import json
from pathlib import Path,PurePosixPath
import re
import shlex
import subprocess
import tarfile
import tempfile
import time
import uuid
from common.paths import PROJECT_ROOT
from cloud.cloud_session import parse_ssh,ssh_arguments,REPOSITORY
from cloud.sweep_plan import estimate_remaining,validate_saved

ARCHIVE_SCRIPT="""import pathlib,sys,tarfile
p=pathlib.Path(sys.argv[1])
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
    for f in sorted(p.rglob('*')):
        if f.is_symlink():raise SystemExit('Refusing a symlink in sweep results')
        if f.is_file() and (f.suffix in ('.json','.jsonl','.md','.log','.png','.npz','.txt') or f.name in ('replay.py','replay_actions.py')):
            archive.add(f,arcname=str(f.relative_to(p)),recursive=False)
"""


def extract_results(archive,destination):
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers()
        if len(members)>100000 or sum(m.size for m in members)>10*1024**3:raise ValueError('Sweep archive exceeds limits')
        for member in members:
            name=PurePosixPath(member.name)
            if (not member.isfile() or name.is_absolute() or '..' in name.parts or
                (name.suffix not in ('.json','.jsonl','.md','.log','.png','.npz','.txt') and name.name not in ('replay.py','replay_actions.py')) or
                not (destination/str(name)).resolve().is_relative_to(destination.resolve())):raise ValueError('Unsafe sweep archive')
        record=next((m for m in members if m.name=='results.json'),None)
        if record is None:raise ValueError('Sweep results missing')
        validate_saved(json.load(stream.extractfile(record)))
        if (destination/'results.json').exists():
            (destination/f'results_before_cloud_fetch_{time.time_ns()}.json').write_bytes((destination/'results.json').read_bytes())
        for member in members:
            target=destination/member.name;target.parent.mkdir(parents=True,exist_ok=True)
            with stream.extractfile(member) as src,target.open('wb') as dst:
                while chunk:=src.read(1024**2):dst.write(chunk)


def fetch(ssh,remote,destination):
    destination.mkdir(parents=True,exist_ok=True)
    command='/opt/openvla/bin/python -c '+shlex.quote(ARCHIVE_SCRIPT)+' '+shlex.quote(remote)
    print('Downloading cloud sweep images, actions, states, reports and logs...',flush=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        archive=Path(tmp)/'results.tar.gz'
        with archive.open('wb') as out:subprocess.run(ssh+[command],stdout=out,check=True,timeout=1800)
        extract_results(archive,destination)
    subprocess.run(ssh+['touch '+shlex.quote(remote+'/sync_ack')],check=True,timeout=20)
    print(f'Results saved locally: {destination}',flush=True)


def setup_script(name):
    # Pull once. A detached private checkout and exact submodule commit freeze the
    # execution source even if another local client later updates the main checkout.
    code='/root/panda-sweep-code/'+name
    return f'''set -euo pipefail
mkdir -p /root
flock -n /root/panda-sweep-supervisor.lock -c true || {{ echo 'Another sweep is active'; exit 1; }}
if [ -d /root/robot-openvla/.git ]; then
  git -C /root/robot-openvla pull --ff-only
else
  git clone {shlex.quote(REPOSITORY)} /root/robot-openvla
fi
cd /root/robot-openvla
/opt/openvla/bin/python -m cloud.scripted_eval_control --session {shlex.quote(name)} --check-only --refuse-training
mkdir -p /root/panda-sweep-code
if [ ! -d {shlex.quote(code)} ]; then
  git clone --shared /root/robot-openvla {shlex.quote(code)}
fi
cd {shlex.quote(code)}
# Fetch only the pinned Panda assets, rather than every robot in Menagerie.
menagerie_commit="$(git ls-tree HEAD mujoco_menagerie | awk '{{print $3}}')"
test -n "$menagerie_commit"
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git mujoco_menagerie
git -C mujoco_menagerie sparse-checkout init --cone
git -C mujoco_menagerie sparse-checkout set franka_emika_panda
git -C mujoco_menagerie fetch --depth 1 origin "$menagerie_commit"
git -C mujoco_menagerie checkout --detach FETCH_HEAD
if [ ! -x /root/panda-sweep-sim/bin/python ]; then
  /opt/openvla/bin/python -m venv /root/panda-sweep-sim
fi
/root/panda-sweep-sim/bin/python -m pip install -r requirements/requirements-sim-linux.txt
if ! /root/panda-sweep-sim/bin/python -c 'import ctypes; ctypes.CDLL("libEGL.so.1"); ctypes.CDLL("libGL.so.1")'; then
  apt-get update
  apt-get install -y libegl1 libgl1 libgles2 libopengl0
fi
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ssh',required=True)
    selection=p.add_mutually_exclusive_group(required=True)
    selection.add_argument('--resume',type=Path,help='Local saved sweep folder, also used for downloads')
    selection.add_argument('--profile',type=Path,help='Create a fresh sweep from a local JSON profile')
    p.add_argument('--fetch-only',action='store_true');p.add_argument('--launch-only',action='store_true')
    p.add_argument('--estimated-seconds',type=float,help='Override conservative unbatched estimate; buffers remain one hour each')
    args=p.parse_args()
    if args.profile:
        if args.fetch_only:p.error('--fetch-only requires --resume')
        from cloud.create_sweep_plan import create
        folder=create(args.profile)
    else:folder=args.resume.resolve()
    if not folder.is_relative_to((PROJECT_ROOT/'outputs').resolve()) or not (folder/'results.json').is_file():p.error('Select saved sweep under outputs/')
    name=folder.name
    if not re.fullmatch(r'scripted_[A-Za-z0-9_]+',name):p.error('Invalid sweep folder name')
    payload=json.loads((folder/'results.json').read_text());validate_saved(payload)
    host,port,key=parse_ssh(args.ssh,None);ssh=ssh_arguments(host,port,key)
    remote='/workspace/panda-training/scripted-evaluations/'+name
    if args.fetch_only:fetch(ssh,remote,folder);return 0
    budget=estimate_remaining(payload)
    if not budget['remaining_trials']:p.error('All trials are already saved; nothing to launch')
    chosen_estimate=args.estimated_seconds if args.estimated_seconds is not None else payload['plan'].get('estimated_seconds')
    if chosen_estimate is not None:
        from cloud.sweep_plan import timeout_budget
        budget.update(timeout_budget(chosen_estimate))
    payload['plan']['timeout_budget']=budget
    payload['plan']['execution_mode']='cloud_sequential'
    code_name=name+'_'+uuid.uuid4().hex[:8]
    setup=setup_script(code_name)
    subprocess.run(ssh+['bash -s'],input=setup.encode(),check=True,timeout=600)
    # Do not replace an existing cloud session's newer results; fetch them first.
    # Mac-driven sessions already have compact reports in this same directory.
    # Preserve that prior report as long as no cloud supervisor was launched there.
    publish=('import pathlib,sys,time,json; p=pathlib.Path('+repr(remote)+'); p.mkdir(parents=True,exist_ok=True); '
             "status=json.loads((p/'supervisor_status.json').read_text()) if (p/'supervisor_status.json').exists() else {}; "
             "assert not status or status.get('phase')=='finished', 'Cloud sweep still active; do not relaunch'; "
             "old=json.loads((p/'results.json').read_text()) if (p/'results.json').exists() else {}; "
             "data=json.load(sys.stdin); assert not status or old.get('trials')==data.get('trials'), 'Fetch newer cloud results before relaunch'; "
             "[(p/n).rename(p/(n+'.previous_'+str(time.time_ns()))) for n in ('supervisor_status.json','sync_ack','heartbeat') if (p/n).exists()]; "
             "f=p/'results.json'; "
             "f.rename(p/('results_before_cloud_'+str(time.time_ns())+'.json')) if f.exists() else None; "
             'f.write_text(json.dumps(data,indent=2)); (p/"heartbeat").touch()')
    subprocess.run(ssh+['/opt/openvla/bin/python -c '+shlex.quote(publish)],input=json.dumps(payload).encode(),check=True,timeout=60)
    command='cd '+shlex.quote('/root/panda-sweep-code/'+code_name)+' && /opt/openvla/bin/python -m cloud.sweep_supervisor --launch --mode cloud --directory '+shlex.quote(remote)+' --sim-python /root/panda-sweep-sim/bin/python --estimated-seconds '+str(budget['estimated_seconds'])
    subprocess.run(ssh+[command],check=True,timeout=60)
    print(f"Detached cloud sweep started. Remaining trials: {budget['remaining_trials']}. Mac may disconnect.",flush=True)
    print(f"Estimated sweep: {budget['estimated_seconds']/3600:.2f} h; inner: {budget['inner_seconds']/3600:.2f} h; outer: {budget['outer_seconds']/3600:.2f} h.",flush=True)
    if args.launch_only:return 0
    print('Watching cloud progress; Ctrl+C ends only this watcher, not the sweep.',flush=True)
    offset=0
    try:
        while True:
            script=('import pathlib,json; p=pathlib.Path('+repr(remote)+'); (p/"heartbeat").touch(); '
                's=json.loads((p/"supervisor_status.json").read_text()) if (p/"supervisor_status.json").exists() else {}; '
                'log=(p/"console.log").read_bytes() if (p/"console.log").exists() else b""; '
                'print(json.dumps(dict(status=s,log=log['+str(offset)+':].decode(errors="replace"),offset=len(log))))')
            result=subprocess.run(ssh+['/opt/openvla/bin/python -c '+shlex.quote(script)],capture_output=True,check=True,timeout=30)
            item=json.loads(result.stdout);offset=item['offset'];print(item['log'],end='',flush=True)
            if item['status'].get('phase') in ('awaiting_sync','stopping','finished'):
                # Keep heartbeat fresh during a potentially large download.
                from cloud.sweep_heartbeat import Heartbeat
                with Heartbeat(ssh,remote):fetch(ssh,remote,folder)
                failed=item['status'].get('outcome')!='completed'
                if failed:print('Cloud sweep did not complete; failure diagnostics are saved locally.',flush=True)
                print('Cloud supervisor requests pod stop. Confirm Stopped in RunPod.',flush=True)
                return 1 if failed else 0
            time.sleep(10)
    except (KeyboardInterrupt,subprocess.SubprocessError,OSError) as exc:
        print(f'Watcher disconnected ({type(exc).__name__}). Cloud sweep continues and handles pod stopping. Fetch results later.',flush=True)
        return 0


if __name__=='__main__':raise SystemExit(main())
