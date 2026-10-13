"""Separate unattended touch trials; use the existing inference loop unchanged."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import socket
import subprocess
import uuid

from cloud.cloud_session import (ROOT, REPOSITORY, bootstrap_script, parse_ssh,
                                 ssh_arguments, wait_for_server)


def write_reports(folder, plan, rows, error=None):
    payload = dict(plan=plan, trials=rows, launcher_error=error)
    (folder / 'results.json').write_text(json.dumps(payload, indent=2) + '\n')
    groups = defaultdict(list)
    for row in rows:
        groups[row['checkpoint']].append(row)
    lines = ['# Scripted touch evaluation', '',
             'Instruction: `touch the red cube.` Sampling off. Same predeclared unseen positions for every checkpoint.', '',
             'Contact and clean completion are separate: a tracking failure can coincide with cube contact.', '',
             '| Checkpoint | Trials recorded | Contact | Clean completion |',
             '|---|---:|---:|---:|']
    for checkpoint, trials in sorted(groups.items()):
        lines.append(f'| {checkpoint} | {len(trials)} | {sum(r["contact"] for r in trials)} | '
                     f'{sum(r["contact"] and r["phase"] == "Completed" for r in trials)} |')
    lines += ['', f'Planned seeds: {plan["scene_seeds"]}', '',
              '| Checkpoint | Seed | Contact | Phase | Completed actions | Details |',
              '|---|---:|---|---|---:|---|']
    for row in rows:
        lines.append(f'| {row["checkpoint"]} | {row["seed"]} | {row["contact"]} | {row["phase"]} | '
                     f'{row["completed_steps"]} | [Run]({row["directory"]}/RUN_REPORT.md) |')
    if error:
        lines += ['', f'Launcher ended early: {error}']
    (folder / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def run_trials(args, folder, ssh):
    from data_tools.collect_panda_touch import scenario
    seeds = args.scene_seeds or list(range(1000, 1020))
    specs = [scenario(seed) for seed in seeds]
    manifest = ROOT / 'datasets/panda_touch_cube/manifest.json'
    if not manifest.is_file():
        raise ValueError('Local touch manifest required to verify unseen positions')
    known = json.loads(manifest.read_text())['episodes']
    for spec in specs:
        for episode in known:
            xy = episode['cube_xy']
            if sum((a-b)**2 for a, b in zip(spec['cube_xy'], xy)) < 1e-12:
                raise ValueError('Requested seed reproduces a collected dataset position')
    plan = dict(checkpoints=args.checkpoints, scene_seeds=seeds,
                cube_positions=[s['cube_xy'] for s in specs], steps=args.steps,
                instruction='touch the red cube.', sampling=False,
                training_session=args.training_session)
    rows = []
    write_reports(folder, plan, rows)
    tunnel = None
    error = None
    try:
        for checkpoint in args.checkpoints:
            adapter = f'/workspace/panda-training/sessions/{args.training_session}/runs/training/checkpoint_{checkpoint:06d}'
            subprocess.run(ssh + ['bash -s'], input=bootstrap_script(21600, adapter).encode(),
                           check=True, timeout=args.startup_timeout_seconds)
            if tunnel is None:
                host = ssh[-1]
                tunnel = subprocess.Popen(ssh[:-1] + ['-o', 'ExitOnForwardFailure=yes', '-N',
                    '-L', '127.0.0.1:8000:127.0.0.1:8000', host])
            health = wait_for_server(tunnel, args.startup_timeout_seconds)
            startup = dict(health.get('startup_info', {}), adapter_checkpoint=adapter)
            startup_path = folder / f'startup_{checkpoint:06d}.json'
            startup_path.write_text(json.dumps(startup, indent=2) + '\n')
            for seed in seeds:
                trial = folder / f'checkpoint_{checkpoint:06d}_seed_{seed}'
                print(f'Checkpoint {checkpoint}, seed {seed}: automatic trial starts.', flush=True)
                command = [str(ROOT / '.venv/bin/python'), '-m', 'inference.remote_loop',
                    '--task', 'touch', '--instruction', 'touch the red cube.',
                    '--scene-seed', str(seed), '--steps', str(args.steps),
                    '--no-view', '--start-immediately', '--startup-info', str(startup_path),
                    '--output-dir', str(trial), '--max-runtime-seconds', str(args.trial_timeout_seconds)]
                try:
                    with (folder / 'console.log').open('a') as log:
                        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                timeout=args.trial_timeout_seconds + 240)
                    returncode = result.returncode
                except subprocess.TimeoutExpired:
                    returncode = 'timeout'
                summary_path = trial / 'summary.json'
                summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
                status_path = trial / 'status.json'
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                rows.append(dict(checkpoint=checkpoint, seed=seed, directory=trial.name,
                    phase=summary.get('phase', 'Missing summary'),
                    completed_steps=summary.get('completed_steps', 0),
                    contact=summary.get('task_result', {}).get('success', False),
                    error=status.get('error'), returncode=returncode))
                write_reports(folder, plan, rows)
                print(f'  {rows[-1]["phase"]}; contact={rows[-1]["contact"]}', flush=True)
                # An unhealthy connection aborts the sweep instead of recording 60 bogus failures.
                wait_for_server(tunnel, 5)
    except BaseException as exc:
        error = str(exc) or type(exc).__name__
        raise
    finally:
        write_reports(folder, plan, rows, error)
        if tunnel is not None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.kill()
                tunnel.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh', required=True)
    parser.add_argument('--key', type=Path)
    parser.add_argument('--training-session', required=True)
    parser.add_argument('--checkpoints', type=int, nargs='+', default=[50, 100, 300, 500])
    parser.add_argument('--scene-seeds', type=int, nargs='+')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--trial-timeout-seconds', type=int, default=180)
    parser.add_argument('--startup-timeout-seconds', type=int, default=900)
    args = parser.parse_args()
    if (any(x < 0 for x in args.checkpoints) or len(set(args.checkpoints)) != len(args.checkpoints)
            or any(x < 50 for x in (args.scene_seeds or []))
            or len(set(args.scene_seeds or [])) != len(args.scene_seeds or [])
            or min(args.steps, args.trial_timeout_seconds, args.startup_timeout_seconds) <= 0):
        parser.error('Use unique nonnegative checkpoints, unique unseen seeds >=50, and positive limits')
    if not args.training_session.startswith('panda-train-') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-' for c in args.training_session):
        parser.error('Invalid training session name')
    try:
        host, port, key = parse_ssh(args.ssh, args.key)
    except ValueError as exc:
        parser.error(str(exc))
    with socket.socket() as probe:
        try:
            probe.bind(('127.0.0.1', 8000))
        except OSError:
            parser.error('Local port 8000 is occupied. End the previous local simulator session first.')
    ssh = ssh_arguments(host, port, key)
    name = f'scripted_touch_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}'
    folder = ROOT / 'outputs' / name
    folder.mkdir(parents=True)
    control = f'cd /root/robot-openvla && /opt/openvla/bin/python -m cloud.scripted_eval_control --session {name}'
    ready = False
    code = 0
    print('Keep the Mac awake and connected. This dedicated inference pod will be stopped at the end.', flush=True)
    try:
        setup = f'set -e\nif [ -d /root/robot-openvla/.git ]; then git -C /root/robot-openvla pull --ff-only; else git clone {REPOSITORY} /root/robot-openvla; fi\n'
        subprocess.run(ssh + ['bash -s'], input=setup.encode(), check=True, timeout=120)
        ready = True
        subprocess.run(ssh + [control + ' --check-only'], check=True, timeout=90)
        run_trials(args, folder, ssh)
    except (Exception, KeyboardInterrupt) as exc:
        print(f'Scripted evaluation ended: {exc or type(exc).__name__}', flush=True)
        code = 1
    finally:
        if ready:
            # Copy compact reports before stopping; images/actions are already on the Mac.
            try:
                payload = {p.name: p.read_text() for p in folder.glob('*')
                           if p.name in ('REPORT.md', 'results.json')}
                destination = '/workspace/panda-training/scripted-evaluations/' + name
                script = ('import json,pathlib,sys; p=pathlib.Path(' + repr(destination) + '); '
                          'p.mkdir(parents=True,exist_ok=True); '
                          '[(p/k).write_text(v) for k,v in json.load(sys.stdin).items()]')
                subprocess.run(ssh + ['python3 -c ' + shlex.quote(script)],
                               input=json.dumps(payload).encode(), check=True, timeout=45)
            except Exception as exc:
                print(f'Volume report copy failed; local results retained: {exc}', flush=True)
                code = 1
            print('Requesting stop of the inference pod...', flush=True)
            try:
                if subprocess.run(ssh + [control], timeout=100).returncode:
                    code = 1
                    print('Stop was not confirmed. Check RunPod and stop manually.', flush=True)
            except Exception:
                code = 1
                print('Stop was not confirmed. Check RunPod and stop manually.', flush=True)
        print(f'Results saved locally: {folder}', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
