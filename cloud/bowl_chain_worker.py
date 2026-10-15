"""Stages for the cloud-only bowl chain. No stage independently stops the pod."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
from common.paths import PROJECT_ROOT
from cloud.chain_timing import Timings,utc
from training.detached_run import write_status
from cloud.bowl_chain_plan import make_plan


def run_command(command,log,*,env=None):
    print('Running:', ' '.join(str(v) for v in command),flush=True)
    with Path(log).open('a',buffering=1) as out:
        subprocess.run([str(v) for v in command],cwd=PROJECT_ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)


def setup_simulator(local):
    local=Path(local);python=local/'sim/bin/python'
    if not python.exists():subprocess.run([sys.executable,'-m','venv',str(local/'sim')],check=True)
    subprocess.run([str(python),'-m','pip','install','-r',str(PROJECT_ROOT/'requirements/requirements-sim-linux.txt')],check=True)
    assets=PROJECT_ROOT/'mujoco_menagerie'
    if not (assets/'franka_emika_panda/panda.xml').exists():
        revision=subprocess.check_output(['git','ls-tree','HEAD','mujoco_menagerie'],cwd=PROJECT_ROOT,text=True).split()[2]
        if not (assets/'.git').exists():subprocess.run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/google-deepmind/mujoco_menagerie.git',str(assets)],check=True)
        subprocess.run(['git','-C',str(assets),'sparse-checkout','init','--cone'],check=True)
        subprocess.run(['git','-C',str(assets),'sparse-checkout','set','franka_emika_panda'],check=True)
        subprocess.run(['git','-C',str(assets),'fetch','--depth','1','origin',revision],check=True)
        subprocess.run(['git','-C',str(assets),'checkout','--detach','FETCH_HEAD'],check=True)
    probe=[str(python),'-c','import ctypes; ctypes.CDLL("libEGL.so.1"); ctypes.CDLL("libGL.so.1")']
    if subprocess.run(probe,capture_output=True).returncode:
        subprocess.run(['apt-get','update'],check=True)
        subprocess.run(['apt-get','install','-y','libegl1','libgl1','libgles2','libopengl0'],check=True)
    subprocess.run(probe,check=True)
    return str(python)


def global_probe(root):
    import uuid
    root=Path(root)
    subprocess.run(['mountpoint','-q',str(root)],check=True)
    if root.resolve()==Path('/workspace').resolve():raise ValueError('Global recovery and checkpoint mounts must be distinct')
    marker=root/f'.panda-chain-probe-{uuid.uuid4().hex}';content=os.urandom(4096)
    try:
        marker.write_bytes(content)
        if marker.read_bytes()!=content:raise ValueError('Global volume write/read test failed')
    finally:marker.unlink(missing_ok=True)


def stage_base(source,destination):
    import hashlib
    from cloud.archive_sweep import digest
    source=Path(source);destination=Path(destination);destination.mkdir(parents=True,exist_ok=False);records=[]
    if not (source/'config.json').is_file():raise ValueError('Missing base export')
    for p in sorted(source.rglob('*')):
        if '.cache' in p.relative_to(source).parts:continue
        if p.is_symlink():raise ValueError('Base export symlink unsupported')
        if not p.is_file():continue
        relative=p.relative_to(source);target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
        with p.open('rb') as src,target.open('wb') as dst:
            while data:=src.read(8*1024**2):dst.write(data);h.update(data)
        if target.stat().st_size!=p.stat().st_size or digest(target)!=h.hexdigest():raise ValueError('Staged model failed integrity check')
        records.append(dict(path=relative.as_posix(),bytes=target.stat().st_size,sha256=h.hexdigest()))
    return dict(source=str(source),destination=str(destination),files=records)


def download_dataset(config):
    from huggingface_hub import snapshot_download
    from cloud.download_training_dataset import verify_bundle
    from cloud.archive_sweep import digest
    destination=Path(config['dataset']);destination.parent.mkdir(parents=True,exist_ok=True)
    snapshot_download(config['dataset_repo'],repo_type='dataset',revision=config['dataset_revision'],token=config['hf_token'],local_dir=str(destination),allow_patterns=['SHA256SUMS','*.json','PACKAGE_REPORT.md','rlds/**'])
    checks=(destination/'SHA256SUMS').read_text()
    if digest(destination/'SHA256SUMS')!=config['checksums_sha256']:raise ValueError('Dataset checksum manifest differs from prepared revision')
    verify_bundle(destination,checks)
    return destination


def verify_training(run,expected_updates=3000,checkpoints=(0,500,1000,2000,3000),initial_update=0):
    from cloud.checkpoint_ready import check_ready
    run=Path(run);metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    updates=[r['update'] for r in metrics if 'train_loss' in r]
    if updates!=list(range(initial_update+1,initial_update+expected_updates+1)):raise ValueError(f'Training did not complete exactly {expected_updates} updates from {initial_update}')
    if any('validation_loss' in r for r in metrics):raise ValueError('Unexpected validation in all-training experiment')
    for cp in checkpoints:check_ready(run/f'checkpoint_{cp:06d}')
    for name in ('loss.png','loss_smoothed.png','experiment.json','model_info.json'):
        if not (run/name).is_file():raise ValueError('Missing training report: '+name)
    return dict(training_complete=True,last_update=initial_update+expected_updates,validation_enabled=False,
        initial_update=initial_update,additional_updates=expected_updates,
        training_update_seconds=sum(r.get('training_update_wall_seconds',0) for r in metrics),
        checkpoint_seconds=sum(json.loads(line)['seconds'] for line in (run/'checkpoint_timings.jsonl').read_text().splitlines()))


def write_summary(session,config,training=None):
    session=Path(session)
    lines=['# Bowl training and color placement chain','',f'Session: `{session.name}`.',
        f'Dataset: `{config["dataset_repo"]}@{config["dataset_revision"]}`. All 300 trajectories / 21,906 examples are training data.',
        'Training: 3,000 updates; batch 8; accumulation 1; learning rate 0.0001; LoRA rank 32. Checkpoints 0, 500, 1000, 2000, 3000.',
        'Sweep: base plus four trained checkpoints; 1,000 fresh-reset trials, at most 120 greedy action predictions each.',
        'Single red/black; two red+black/black+purple; four red+green+black+purple/green+black+purple+orange. Blue bowl always present.',
        'Checkpoints remain on the network volume. Detailed recordings are backed up to HF. Global storage is used only for recovery.',
        '', '[Stage timings](TIMINGS.md) · [Sweep results](sweep-reports/REPORT.md)', '']
    if training:lines+=['Training-only update wall time: %.2f seconds.'%training['training_update_seconds'], 'Checkpoint save wall time: %.2f seconds.'%training['checkpoint_seconds']]
    lines+=['','| Backup timing | Seconds |','|---|---:|']
    for filename,label in (('checkpoint_hf_receipt.json','Checkpoints'),('sweep_hf_receipt.json','Sweep')):
        if (session/filename).exists():
            receipt=json.loads((session/filename).read_text())
            for field in ('container_copy_seconds','packing_seconds','upload_seconds','verification_seconds'):
                if field in receipt:lines.append(f'| {label}: {field.replace("_seconds", "").replace("_", " ")} | {receipt[field]:.2f} |')
    (session/'RUN_CHAIN_REPORT.md').write_text('\n'.join(lines)+'\n')


def worker(config):
    from huggingface_hub import HfApi
    session=Path(config['session']);local=Path(config['local']);timings=Timings(session)
    api=HfApi(token=config['hf_token']);env=os.environ.copy();env.pop('RUNPOD_API_KEY',None);env['HF_TOKEN']=config['hf_token'];env['HF_XET_CHUNK_CACHE_SIZE_BYTES']='0';env['HF_XET_CACHE']=str(local/'xet-cache')
    os.environ.update(HF_XET_CHUNK_CACHE_SIZE_BYTES='0',HF_XET_CACHE=str(local/'xet-cache'))
    def stage(name):
        print('Stage: '+name,flush=True)
        write_status(session/'status.json',dict(phase=name,started_at_utc=config.get('started_at_utc',utc()),stage_started_at_utc=utc(),
            elapsed_seconds=time.monotonic()-config.get('started_monotonic',time.monotonic()),timeout_budget=config.get('timeout_budget'),pod_id=config['pod_id'],code_commit=config['code_commit']))
        return timings.stage(name)
    with stage('storage_preflight'):
        if shutil.disk_usage(local).free<110*1024**3:raise ValueError('Need 110 GiB free on container disk after image startup')
        if shutil.disk_usage(session).free<5*1024**3:raise ValueError('Need 5 GiB free on network volume for retained checkpoints/reports')
        global_probe(config['global_volume'])
    with stage('hf_access_preflight'):
        api.whoami()
        for repo,kind in ((config['checkpoint_repo'],'model'),(config['sweep_repo'],'dataset')):
            api.create_repo(repo,repo_type=kind,private=True,exist_ok=True)
            if not api.repo_info(repo,repo_type=kind).private:raise ValueError('Backups require private repositories')
            api.upload_file(repo_id=repo,repo_type=kind,path_or_fileobj=json.dumps(dict(session=session.name,at_utc=utc())).encode(),path_in_repo=f'preflight/{session.name}.json',commit_message='Verify chained-run backup write access')
    with stage('simulator_setup'):config['sim_python']=setup_simulator(local)
    with stage('dataset_download_and_verification'):download_dataset(config)
    with stage('model_staging'):
        receipt=stage_base(config['base_volume'],config['base']);write_status(session/'model_staging.json',receipt)
    provenance=dict(dataset_repo=config['dataset_repo'],dataset_revision=config['dataset_revision'],checksums_sha256=config['checksums_sha256'],bundle='training_place_bowl_v2_300',training_episodes=300,training_examples=21906,all_training=True,base_model_volume_path=config['base_volume'],code_commit=config['code_commit'])
    write_status(session/'provenance.json',provenance)
    plan=make_plan(session.name);sweep=Path(config['sweep']);sweep.mkdir(parents=True,exist_ok=False)
    from cloud.scripted_touch_session import write_reports
    write_reports(sweep,plan,[]);write_summary(session,config)
    safe={k:v for k,v in config.items() if k not in ('hf_token','pod_key')};write_status(local/'worker_config.json',safe)
    sim_env={k:v for k,v in env.items() if k!='HF_TOKEN'};sim_env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    with stage('scene_initialization_and_render_preflight'):
        run_command([config['sim_python'],'-u','-m','cloud.bowl_chain_sweep','--preflight',sweep/'results.json'],session/'preflight.log',env=sim_env)
    with stage('training'):
        run_command([sys.executable,'-u','-m','training.train_panda','--dataset',config['dataset'],'--model',config['base'],'--config',PROJECT_ROOT/'training/place_bowl_300_3000.json','--output-root',session/'runs','--run-name','training','--provenance',session/'provenance.json'],session/'training.log',env=sim_env)
        training=verify_training(config['training_run']);write_status(session/'training_completion.json',training)
    with stage('simulator_sweep'):
        run_command([sys.executable,'-u','-m','cloud.bowl_chain_sweep','--config',local/'worker_config.json'],session/'sweep.log',env=sim_env)
    from cloud.chain_archive import backup_training,backup_sweep
    with stage('checkpoint_hf_backup_and_verification'):backup_training(session,config['checkpoint_repo'],config['hf_token'],local)
    with stage('sweep_packing_upload_and_verification'):backup_sweep(sweep,config['sweep_repo'],config['hf_token'],local,session)
    write_summary(session,config,training)
    write_status(session/'worker_completion.json',dict(completed=True,finished_at_utc=utc()))
    # Publish final reports to a pinned revision and verify their copied snapshot.
    from cloud.chain_archive import verify_files
    from cloud.archive_sweep import digest
    final_staging=local/'final-reports';final_staging.mkdir()
    def copy_reports():
        for p in session.iterdir():
            if p.is_file() and p.suffix in ('.json','.jsonl','.md'):shutil.copyfile(p,final_staging/p.name)
        if (session/'sweep-reports').exists():shutil.copytree(session/'sweep-reports',final_staging/'sweep-reports',dirs_exist_ok=True)
        return [dict(path=p.relative_to(final_staging).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in final_staging.rglob('*') if p.is_file()]
    prefix='chains/'+session.name
    with stage('final_report_upload_and_verification'):
        records=copy_reports()
        commit=api.upload_folder(repo_id=config['checkpoint_repo'],repo_type='model',folder_path=final_staging,path_in_repo=prefix,commit_message='Final chained-run timing and backup reports')
        verify_files(config['checkpoint_repo'],'model',commit.oid,prefix,records,config['hf_token'],local)
    # Republish the completed timing stage after its first snapshot has been verified.
    for name in ('stage_timings.jsonl','TIMINGS.md'):shutil.copyfile(session/name,final_staging/name)
    commit=api.upload_folder(repo_id=config['checkpoint_repo'],repo_type='model',folder_path=final_staging,path_in_repo=prefix,allow_patterns=['stage_timings.jsonl','TIMINGS.md'],commit_message='Completed final report-upload timings')
    records=[dict(path=name,bytes=(final_staging/name).stat().st_size,sha256=digest(final_staging/name)) for name in ('stage_timings.jsonl','TIMINGS.md')]
    verify_files(config['checkpoint_repo'],'model',commit.oid,prefix,records,config['hf_token'],local)
    return 0


if __name__=='__main__':
    config=json.load(sys.stdin)
    raise SystemExit(worker(config))
