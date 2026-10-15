"""Detached checkpoint-6000 continuation, 600 matched trials, HF delivery and stop."""
import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
import uuid

from common.paths import PROJECT_ROOT
from cloud.bowl_chain_plan import make_continuation_plan, timeout_budget, CONTINUATION_CHECKPOINTS
from cloud.chain_timing import Timings, utc
from training.detached_run import preflight, write_status

SOURCE_SESSION = 'panda-train-color-bowl-20261015-094416-4344d31f'
TRAINING_CONFIG = 'training/color_place_bowl_600_continue_12000.json'


def select_source(selected=SOURCE_SESSION):
    from cloud.color_bowl_sweep import PROFILE, resolve_training
    from cloud.checkpoint_ready import check_ready
    profile = json.loads(PROFILE.read_text())
    source = resolve_training(profile, selected=selected)
    adapter = check_ready(source/'runs/training/checkpoint_006000')
    saved = json.loads((adapter/'training_config.json').read_text())
    provenance = saved.get('dataset_provenance', {})
    if (saved.get('checkpoint_update') != 6000 or
        provenance.get('checksums_sha256') != profile['source_dataset_sha256'] or
        provenance.get('dataset_revision') != profile['source_dataset_revision']):
        raise ValueError('Source is not the verified 600-demo checkpoint 6000')
    return source, adapter, profile


def launch(selected, global_volume):
    from cloud.scripted_eval_control import active_training
    from cloud.color_bowl_training import resolve_upload
    from huggingface_hub import HfApi
    if active_training():
        raise ValueError('Launch on a separate idle A40 pod')
    if not global_volume.is_absolute():
        raise ValueError('Recovery volume path must be absolute')
    source, adapter, sweep_profile = select_source(selected)
    training_profile = json.loads((PROJECT_ROOT/'inference/training_profiles/color_place_bowl_600.json').read_text())
    pod, key = preflight()
    token = getpass.getpass('Hugging Face WRITE token (hidden): ').strip()
    if not token or '{{' in token:
        raise ValueError('A resolved HF write token is required')
    HfApi(token=token).whoami()
    revision = resolve_upload(training_profile, token)
    if revision != sweep_profile['source_dataset_revision']:
        # A newer HF commit may contain identical files. Keep the exact original
        # pinned dataset and verify it again in the worker, rather than changing data.
        print('Using the original pinned dataset revision for this continuation.', flush=True)
    name = 'panda-train-color-bowl-cont-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
    session = Path('/workspace/panda-training/sessions')/name
    local = Path('/root/panda-chain-work')/name
    session.mkdir(parents=True, exist_ok=False)
    local.mkdir(parents=True, exist_ok=False)
    lock = Path('/root/panda-long-chain.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip()
    code = local/'code'
    subprocess.run(['git', 'clone', '--shared', str(PROJECT_ROOT), str(code)], check=True)
    subprocess.run(['git', '-C', str(code), 'checkout', '--detach', commit], check=True)
    sweep_name = 'scripted_color_bowl_continue_'+name.removeprefix('panda-train-color-bowl-cont-').replace('-', '_')
    config = {k: v for k, v in training_profile.items() if k != 'file_sizes'}
    config.update(training_config=TRAINING_CONFIG, continuation_chain=True, estimated_seconds=22*3600,
        source_training_session=source.name, source_adapter=str(adapter), initial_update=6000,
        dataset_revision=sweep_profile['source_dataset_revision'],
        session=str(session), local=str(local), base_volume='/workspace/openvla-7b', base=str(local/'base'),
        dataset=str(local/training_profile['dataset_bundle']), training_run=str(session/'runs/training'),
        sweep=str(local/'scripted-evaluations'/sweep_name), sweep_repo=sweep_profile['sweep_repo'],
        global_volume=str(global_volume), timeout_budget=timeout_budget(22*3600),
        pod_id=pod, pod_key=key, hf_token=token, code_commit=commit)
    env = os.environ.copy()
    env.pop('HF_TOKEN', None)
    env.pop('RUNPOD_API_KEY', None)
    with (local/'supervisor.log').open('a', buffering=1) as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', 'cloud.bowl_chain', '--supervise'],
            cwd=code, env=env, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, pass_fds=(lock.fileno(),))
        child.stdin.write(json.dumps(config).encode())
        child.stdin.close()
    print('Detached colored bowl continuation chain started. Mac may disconnect.', flush=True)
    print('Session:', name, flush=True)
    print('Source: checkpoint 6000 from', source.name, '(fresh optimizer).', flush=True)
    print('Training: 6000 additional updates; cumulative checkpoints 8000, 10000, 12000.', flush=True)
    print('Sweep:', sweep_name, '— 600 matched trials, then verified HF delivery.', flush=True)
    print('Watch setup/stages:', local/'console.log', flush=True)
    print('Watch training:', session/'training.log', flush=True)
    print('Watch sweep:', session/'sweep.log', flush=True)
    print('Status/timings/checkpoints:', session, flush=True)
    print('Estimated budget 22 h; inner limit 25 h; independent outer limit 30 h.', flush=True)
    print('Checkpoints retained on volume. Supervisor requests pod stop after completion, failure or timeout.', flush=True)
    return 0


def worker(config):
    from huggingface_hub import HfApi
    from cloud.bowl_chain_worker import (global_probe, setup_simulator, download_dataset,
        stage_base, run_command, verify_training)
    from cloud.color_bowl_training import train
    from cloud.chain_archive import backup_training, backup_sweep
    from cloud.color_bowl_sweep import publish_reports
    from cloud.archive_sweep import digest
    from cloud.scripted_touch_session import write_reports
    from cloud.checkpoint_ready import check_ready

    session, local, sweep = map(Path, (config['session'], config['local'], config['sweep']))
    timer = Timings(session)
    def stage(name):
        print('Stage: '+name, flush=True)
        write_status(session/'status.json', dict(phase=name, started_at_utc=config['started_at_utc'],
            stage_started_at_utc=utc(), timeout_budget=config['timeout_budget'], code_commit=config['code_commit']))
        return timer.stage(name)

    with stage('storage_preflight'):
        if shutil.disk_usage(local).free < 110*1024**3:
            raise ValueError('Need 110 GiB actually free on container disk; use 200 GB container disk')
        if shutil.disk_usage(session).free < 5*1024**3:
            raise ValueError('Need 5 GiB free on main volume for three checkpoints and reports')
        global_probe(config['global_volume'])
    api = HfApi(token=config['hf_token'])
    with stage('source_checkpoint_and_hf_preflight'):
        source, adapter, profile = select_source(config['source_training_session'])
        if str(adapter) != config['source_adapter']:
            raise ValueError('Source checkpoint changed since launch')
        for repo, kind in ((config['checkpoint_repo'], 'model'), (config['sweep_repo'], 'dataset')):
            api.create_repo(repo, repo_type=kind, private=True, exist_ok=True)
            if not api.repo_info(repo, repo_type=kind).private:
                raise ValueError('HF backups must be private')
            api.upload_file(repo_id=repo, repo_type=kind,
                path_or_fileobj=json.dumps(dict(session=session.name)).encode(),
                path_in_repo='preflight/'+session.name+'.json', commit_message='Check continuation backup write access')
    os.environ.update(HF_XET_CHUNK_CACHE_SIZE_BYTES='0', HF_XET_CACHE=str(local/'xet-cache'))
    with stage('simulator_setup'):
        config['sim_python'] = setup_simulator(local)
    with stage('dataset_download_and_verification'):
        download_dataset(config)
    with stage('model_and_initial_adapter_staging'):
        receipt = stage_base(config['base_volume'], config['base'])
        write_status(session/'model_staging.json', receipt)
        initial_adapter = local/'initial_adapter/checkpoint_006000'
        shutil.copytree(adapter, initial_adapter)
        for file in adapter.rglob('*'):
            if file.is_file() and digest(file) != digest(initial_adapter/file.relative_to(adapter)):
                raise ValueError('Initial adapter staging checksum mismatch')
        check_ready(initial_adapter)
        parent_sha = digest(initial_adapter/'adapter_model.safetensors')
    provenance = dict(dataset_repo=config['dataset_repo'], dataset_revision=config['dataset_revision'],
        checksums_sha256=config['checksums_sha256'], bundle=config['dataset_bundle'],
        training_episodes=600, training_examples=43344, all_training=True,
        base_model_volume_path=config['base_volume'], code_commit=config['code_commit'],
        initial_training_session=source.name, initial_checkpoint=str(adapter), initial_checkpoint_update=6000,
        initial_adapter_sha256=parent_sha, additional_updates=6000, total_updates=12000,
        continuation_mode='adapter_weights_fresh_optimizer')
    write_status(session/'provenance.json', provenance)
    plan = make_continuation_plan(session.name, source.name)
    sweep.mkdir(parents=True, exist_ok=False)
    write_reports(sweep, plan, [])
    safe = {k: v for k, v in config.items() if k not in ('hf_token', 'pod_key')}
    safe.update(source_dataset_sha256=config['checksums_sha256'],
        source_dataset_revision=config['dataset_revision'], verify_adapter_copy=True)
    write_status(local/'worker_config.json', safe)
    env = os.environ.copy()
    env.pop('HF_TOKEN', None)
    env.pop('RUNPOD_API_KEY', None)
    env.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
    with stage('scene_initialization_and_render_preflight'):
        run_command([config['sim_python'], '-u', '-m', 'cloud.bowl_chain_sweep', '--preflight', sweep/'results.json'], session/'preflight.log', env=env)
    with stage('training_continuation'):
        train([sys.executable, '-u', '-m', 'training.train_panda', '--dataset', config['dataset'],
            '--model', config['base'], '--adapter', initial_adapter, '--initial-update', '6000',
            '--base-staging-receipt', session/'model_staging.json',
            '--config', PROJECT_ROOT/TRAINING_CONFIG, '--output-root', session/'runs',
            '--run-name', 'training', '--provenance', session/'provenance.json'], session/'training.log', env)
        completion = verify_training(config['training_run'], 6000, CONTINUATION_CHECKPOINTS, initial_update=6000)
        write_status(session/'training_completion.json', completion)
    # Back up the three new checkpoints before the long sweep. Volume copies stay.
    with stage('checkpoint_hf_backup_and_verification'):
        backup_training(session, config['checkpoint_repo'], config['hf_token'], local, CONTINUATION_CHECKPOINTS)
    with stage('simulator_sweep'):
        run_command([sys.executable, '-u', '-m', 'cloud.bowl_chain_sweep', '--config', local/'worker_config.json'], session/'sweep.log', env=env)
    with stage('sweep_packing_upload_and_verification'):
        backup_sweep(sweep, config['sweep_repo'], config['hf_token'], local, session)
    (session/'RUN_CONTINUATION_REPORT.md').write_text(
        '# Colored bowl continuation and matched sweep\n\n'
        f'Parent: `{source.name}/runs/training/checkpoint_006000`.\n\n'
        '600 identical demonstrations; batch 8, learning rate 0.0001, rank 32. '
        'Adapter weights continued with a fresh optimizer; 6000 additional updates. '
        'Cumulative checkpoints 8000, 10000, 12000 remain on volume and are verified on HF.\n\n'
        '600 sweep trials: 200 per model, matching the first colored bowl sweep. '
        'Sampling off, up to 120 actions; single, two and four cubes, seen and unseen colors.\n\n'
        '[Stage timings](TIMINGS.md) · [Sweep results](sweep-reports/REPORT.md)\n')
    write_status(session/'worker_completion.json', dict(completed=True, finished_at_utc=utc(),
        training_complete=True, last_update=12000, checkpoint_backup_verified=True, sweep_backup_verified=True))
    with stage('final_report_upload_and_verification'):
        publish_reports(sweep, local, session, config['sweep_repo'], config['hf_token'])
    publish_reports(sweep, local, session, config['sweep_repo'], config['hf_token'])
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--training-session', default=SOURCE_SESSION)
    parser.add_argument('--global-volume', type=Path, default=Path('/workspace-global'))
    args = parser.parse_args()
    if args.launch:
        return launch(args.training_session, args.global_volume)
    return worker(json.load(sys.stdin))


if __name__ == '__main__':
    raise SystemExit(main())
