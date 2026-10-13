"""Separate unattended touch trials; use the existing inference loop unchanged."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import socket
import subprocess
import time
import uuid

from cloud.cloud_session import (ROOT, REPOSITORY, bootstrap_script, parse_ssh,
                                 ssh_arguments, wait_for_server)


def write_reports(folder, plan, rows, error=None):
    payload = dict(plan=plan, trials=rows, launcher_error=error)
    (folder / 'results.json').write_text(json.dumps(payload, indent=2) + '\n')
    if plan.get('task')=='color_transfer':
        from analysis.color_transfer_report import write_report
        write_report(folder,plan,rows,error)
        return
    groups = defaultdict(list)
    for row in rows:
        groups[row['checkpoint']].append(row)
    if plan.get('task')=='color_touch':
        lines=['# Scripted red/blue touch evaluation','',
            'Sampling off. Every checkpoint sees both instructions in the same predeclared unseen scenes.',
            'Correct-only contact and clean completion are reported separately; tracking failures remain visible.','',
            '| Checkpoint | Trials | Correct only | Wrong or both | No contact | Unmeasured | Clean correct | Both colors correct in scene |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for checkpoint,trials in sorted(groups.items()):
            correct=lambda r:r['contact'] and not r.get('wrong_contact',False)
            pairs=defaultdict(list)
            for row in trials:pairs[row['seed']].append(row)
            lines.append(f'| {checkpoint} | {len(trials)} | {sum(correct(r) for r in trials)} | '
                f'{sum(r.get("wrong_contact",False) for r in trials)} | '
                f'{sum(r["outcome"]=="no_contact" for r in trials)} | '
                f'{sum(r["outcome"]=="unmeasured" for r in trials)} | '
                f'{sum(correct(r) and r["phase"]=="Completed" for r in trials)} | '
                f'{sum(len(pair)==2 and all(correct(r) for r in pair) for pair in pairs.values())} |')
        lines+=['','| Checkpoint | Target color | Trials | Correct only | Wrong or both |',
                 '|---|---|---:|---:|---:|']
        for checkpoint,trials in sorted(groups.items()):
            for color in ('red','blue'):
                subset=[r for r in trials if r['target_color']==color]
                lines.append(f'| {checkpoint} | {color} | {len(subset)} | '
                    f'{sum(r["contact"] and not r["wrong_contact"] for r in subset)} | '
                    f'{sum(r["wrong_contact"] for r in subset)} |')
        lines+=['',f'Wall time for checkpoint preparation and trials: {plan.get("elapsed_seconds",0):.1f} seconds.',
            '', '| Checkpoint | Seed | Target | Outcome | Phase | Details |','|---|---:|---|---|---|---|']
        for row in rows:
            lines.append(f'| {row["checkpoint"]} | {row["seed"]} | {row["target_color"]} | '
                f'{row["outcome"]} | {row["phase"]} | [Run]({row["directory"]}/RUN_REPORT.md) |')
        if error:lines+=['',f'Launcher ended early: {error}']
        (folder/'REPORT.md').write_text('\n'.join(lines)+'\n')
        return
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


def resume_rows(folder, plan):
    """Reuse measured trials, including task failures; never select only successes."""
    saved=json.loads((folder/'results.json').read_text())
    previous=saved['plan']
    for key in ('checkpoints','scene_seeds','cube_positions','steps','sampling','task','training_session','color_pairs'):
        if previous.get(key)!=plan.get(key):
            raise ValueError(f'Resume plan differs: {key}')
    rows=saved['trials']
    if len({r['directory'] for r in rows})!=len(rows):raise ValueError('Duplicate saved trials')
    for row in rows:
        name=row['directory']
        if Path(name).name!=name:raise ValueError('Invalid saved trial directory')
        summary=folder/name/'summary.json'
        if not summary.is_file():raise ValueError('Saved trial summary missing; inspect before resuming')
        data=json.loads(summary.read_text())
        if data.get('phase')!=row['phase'] or data.get('completed_steps')!=row['completed_steps']:
            raise ValueError('Saved summary differs from recorded trial')
    backup=folder/f'results_before_resume_{time.time_ns()}.json'
    backup.write_text(json.dumps(saved,indent=2)+'\n')
    return rows,float(previous.get('elapsed_seconds',0))


def collect_server_diagnostics(ssh, folder):
    """Best-effort bounded log capture before shutdown, without reading secrets."""
    command="cd /root/robot-openvla && tail -c 262144 outputs/cloud_server.log"
    try:
        result=subprocess.run(ssh+[command],capture_output=True,timeout=20)
        if result.returncode==0:
            (folder/f'cloud_server_failure_{time.time_ns()}.log').write_bytes(result.stdout)
    except Exception as exc:
        print(f'Cloud server log could not be collected: {exc}',flush=True)


def run_trials(args, folder, ssh):
    task=getattr(args,'task','touch')
    started=time.monotonic()
    if task in ('color_touch','color_transfer'):
        from data_tools.collect_panda_color_touch import pair_spec
        seeds=args.scene_seeds or list(range(1000,1010))
        specs=[pair_spec(seed) for seed in seeds]
        manifest=ROOT/'datasets/panda_color_touch/manifest.json'
    else:
        from data_tools.collect_panda_touch import scenario
        seeds = args.scene_seeds or list(range(1000, 1020))
        specs = [scenario(seed) for seed in seeds]
        manifest = ROOT / 'datasets/panda_touch_cube/manifest.json'
    if not manifest.is_file():
        raise ValueError('Local touch manifest required to verify unseen positions')
    known = json.loads(manifest.read_text())['episodes']
    for spec in specs:
        for episode in known:
            if task in ('color_touch','color_transfer'):
                # Both individual object positions must be new, even after exchanging colors.
                for xy in spec['cube_positions'].values():
                    for old in episode['cube_positions'].values():
                        if sum((a-b)**2 for a,b in zip(xy,old))<1e-12:
                            raise ValueError('Requested seed reproduces a collected dataset position')
            elif sum((a-b)**2 for a, b in zip(spec['cube_xy'], episode['cube_xy'])) < 1e-12:
                raise ValueError('Requested seed reproduces a collected dataset position')
    plan = dict(checkpoints=args.checkpoints, scene_seeds=seeds,
                cube_positions=[s['cube_positions'] if task in ('color_touch','color_transfer') else s['cube_xy'] for s in specs], steps=args.steps,
                instruction='Both red and blue touch instructions' if task=='color_touch' else 'touch the red cube.', sampling=False,
                task=task,
                training_session=args.training_session)
    if task=='color_transfer':
        from simulation.color_palette import parse_pairs
        plan['color_pairs']=parse_pairs(getattr(args,'color_pairs',None))
        plan['instruction']='Touch each visible color in turn'
    previous_elapsed=0
    rows = []
    if getattr(args,'resume',None):
        rows,previous_elapsed=resume_rows(folder,plan)
        print(f'Resuming with {len(rows)} saved trials; their outcomes are retained.',flush=True)
    done={r['directory'] for r in rows}
    write_reports(folder, plan, rows)
    tunnel = None
    error = None
    try:
        for checkpoint in args.checkpoints:
            if task=='color_transfer':
                trial_specs=[]
                for pair_index,pair in enumerate(plan['color_pairs']):
                    for index,seed in enumerate(seeds):
                        colors=pair[::-1] if (index+pair_index)%2 else pair
                        for color in pair:trial_specs.append((seed,color,colors,':'.join(pair)))
            else:
                trial_specs=[(seed,color,None,None) for seed in seeds for color in
                             (('red','blue') if task=='color_touch' else ('red',))]
            def trial_name(spec):
                seed,color,colors,pair_name=spec
                suffix=f'_{color}' if task in ('color_touch','color_transfer') else ''
                if task=='color_transfer':suffix=f'_{pair_name.replace(":","-")}{suffix}'
                return f'checkpoint_{checkpoint:06d}_seed_{seed}{suffix}'
            pending=[spec for spec in trial_specs if trial_name(spec) not in done]
            if not pending:
                print(f'Checkpoint {checkpoint}: all trials already saved; skipping model load.',flush=True)
                continue
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
            if startup_path.exists():
                startup_path.rename(startup_path.with_name(startup_path.stem+f'_before_resume_{time.time_ns()}.json'))
            startup_path.write_text(json.dumps(startup, indent=2) + '\n')
            for seed,color,colors,pair_name in pending:
                suffix=f'_{color}' if task in ('color_touch','color_transfer') else ''
                if task=='color_transfer':suffix=f'_{pair_name.replace(":","-")}{suffix}'
                trial = folder / f'checkpoint_{checkpoint:06d}_seed_{seed}{suffix}'
                if trial.exists():
                    trial.rename(trial.with_name(trial.name+f'_interrupted_{time.time_ns()}'))
                instruction=f'touch the {color} cube.'
                print(f'Checkpoint {checkpoint}, seed {seed}, target {color}: automatic trial starts.', flush=True)
                command = [str(ROOT / '.venv/bin/python'), '-m', 'inference.remote_loop',
                    '--task', task, '--instruction', instruction,
                    *(['--target-color',color] if task in ('color_touch','color_transfer') else []),
                    *(['--cube-colors',*colors] if task=='color_transfer' else []),
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
                outcome=summary.get('task_result') or {}
                rows.append(dict(checkpoint=checkpoint, seed=seed, directory=trial.name,
                    phase=summary.get('phase', 'Missing summary'),
                    completed_steps=summary.get('completed_steps', 0),
                    contact=outcome.get('correct_cube_contact',False)
                            if task in ('color_touch','color_transfer') else outcome.get('success', False),
                    error=status.get('error'), returncode=returncode))
                if task in ('color_touch','color_transfer'):
                    rows[-1].update(target_color=color,
                        wrong_contact=outcome.get('wrong_cube_contact',False),
                        outcome=outcome.get('outcome','unmeasured'),
                        first_contact=outcome.get('first_contact'))
                if task=='color_transfer':rows[-1].update(color_pair=pair_name,cube_colors=colors)
                plan['elapsed_seconds']=previous_elapsed+time.monotonic()-started
                write_reports(folder, plan, rows)
                print(f'  {rows[-1]["phase"]}; contact={rows[-1]["contact"]}', flush=True)
                # An unhealthy connection aborts the sweep instead of recording 60 bogus failures.
                try:
                    wait_for_server(tunnel, 30)
                except TimeoutError as exc:
                    raise RuntimeError('Inference server became unavailable between trials (30-second health check).') from exc
    except BaseException as exc:
        error = str(exc) or type(exc).__name__
        collect_server_diagnostics(ssh,folder)
        raise
    finally:
        plan['elapsed_seconds']=previous_elapsed+time.monotonic()-started
        write_reports(folder, plan, rows, error)
        if tunnel is not None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.kill()
                tunnel.wait()


def main(default_task='touch'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh', required=True)
    parser.add_argument('--resume',type=Path,help='Existing local sweep directory; retain recorded outcomes and skip saved trials')
    parser.add_argument('--key', type=Path)
    parser.add_argument('--training-session', required=True)
    parser.add_argument('--task',choices=['touch','color_touch','color_transfer'],default=default_task)
    parser.add_argument('--color-pairs',nargs='+',help='Transfer pairs such as black:purple green:yellow')
    parser.add_argument('--checkpoints', type=int, nargs='+', default=[50, 100, 300, 500])
    parser.add_argument('--scene-seeds', type=int, nargs='+')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--trial-timeout-seconds', type=int, default=180)
    parser.add_argument('--startup-timeout-seconds', type=int, default=900)
    args = parser.parse_args()
    if args.task=='color_transfer':
        from simulation.color_palette import parse_pairs
        try:parse_pairs(args.color_pairs)
        except ValueError as exc:parser.error(str(exc))
    elif args.color_pairs:parser.error('--color-pairs requires the separate color_transfer task')
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
    prefix='scripted_color_transfer' if args.task=='color_transfer' else 'scripted_color_touch' if args.task=='color_touch' else 'scripted_touch'
    name = f'{prefix}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}'
    folder = ROOT / 'outputs' / name
    if args.resume:
        folder=args.resume.resolve()
        if not folder.is_relative_to((ROOT/'outputs').resolve()) or not (folder/'results.json').is_file():
            parser.error('--resume must select a saved sweep inside outputs/')
        name=folder.name
    else:
        folder.mkdir(parents=True)
    control = f'cd /root/robot-openvla && /opt/openvla/bin/python -m cloud.scripted_eval_control --session {name}'
    ready = False
    code = 0
    print('Keep the Mac awake and connected. This dedicated inference pod will be stopped at the end.', flush=True)
    try:
        setup = f'set -e\nif [ -d /root/robot-openvla/.git ]; then git -C /root/robot-openvla pull --ff-only; else git clone {REPOSITORY} /root/robot-openvla; fi\n'
        subprocess.run(ssh + ['bash -s'], input=setup.encode(), check=True, timeout=120)
        if args.task in ('color_touch','color_transfer'):
            # This check is before arming cleanup, so rejecting the training pod
            # cannot stop it through the launcher's finally block.
            subprocess.run(ssh+[control+' --check-only --refuse-training'],check=True,timeout=90)
        ready = True
        if args.task not in ('color_touch','color_transfer'):
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
                           if p.name in ('REPORT.md', 'results.json','statistics.json') or p.name.startswith('cloud_server_failure_')}
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
