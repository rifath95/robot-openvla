"""Prepared 600-demo training, persistent checkpoints, verified HF backup and stop."""
import argparse,fcntl,getpass,hashlib,json,os,subprocess,sys,time,uuid
from datetime import datetime,timezone
from pathlib import Path
from common.paths import PROJECT_ROOT
from cloud.chain_timing import Timings,utc
from cloud.bowl_chain_plan import timeout_budget
from training.detached_run import preflight,write_status


def resolve_upload(profile,token):
    """Require the complete prepared inventory before pinning a single HF commit."""
    from huggingface_hub import HfApi,hf_hub_download
    import tempfile
    info=HfApi(token=token).repo_info(profile['dataset_repo'],repo_type='dataset',files_metadata=True)
    sizes={f.rfilename:f.size for f in info.siblings}
    if any(sizes.get(name)!=size for name,size in profile['file_sizes'].items()):
        raise ValueError('Dataset upload is incomplete or differs from the prepared bundle; training not started. The pod still bills.')
    with tempfile.TemporaryDirectory(prefix='panda-upload-check-') as cache:
        path=Path(hf_hub_download(profile['dataset_repo'],'SHA256SUMS',repo_type='dataset',revision=info.sha,token=token,cache_dir=cache))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=profile['checksums_sha256']:raise ValueError('Uploaded checksums differ; training not started')
    return info.sha


def launch(profile_path,token=None,download_window=120):
    from cloud.scripted_eval_control import active_training
    if active_training():raise ValueError('Use a separate pod with no active training')
    if not 0<=download_window<=3600:raise ValueError('Invalid report download window')
    profile=json.loads(Path(profile_path).read_text());pod,key=preflight()
    token=(token if token is not None else getpass.getpass('Hugging Face WRITE token (hidden): ')).strip()
    if not token or '{{' in token:raise ValueError('A resolved HF write token is required')
    revision=resolve_upload(profile,token)
    revision=profile.get('dataset_revision_pinned',revision)
    name='panda-train-color-bowl-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
    session=Path('/workspace/panda-training/sessions')/name;local=Path('/root/panda-training-work')/name
    session.mkdir(parents=True,exist_ok=False);local.mkdir(parents=True,exist_ok=False)
    lock=Path('/root/panda-long-chain.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,text=True).strip();code=local/'code'
    subprocess.run(['git','clone','--shared',str(PROJECT_ROOT),str(code)],check=True)
    subprocess.run(['git','-C',str(code),'checkout','--detach',commit],check=True)
    config={k:v for k,v in profile.items() if k!='file_sizes'}
    config.update(dataset_revision=revision,session=str(session),local=str(local),base_volume='/workspace/openvla-7b',base=str(local/'base'),dataset=str(local/profile['dataset_bundle']),training_run=str(session/'runs/training'),sweep=str(local/'no-sweep'),global_volume=None,training_only=True,download_grace_seconds=download_window,
        timeout_budget=timeout_budget(profile.get('estimated_seconds',14400)),pod_id=pod,pod_key=key,hf_token=token,code_commit=commit)
    env=os.environ.copy();env.pop('RUNPOD_API_KEY',None);env.pop('HF_TOKEN',None)
    with (local/'supervisor.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-u','-m','cloud.bowl_chain','--supervise'],cwd=code,env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
        child.stdin.write(json.dumps(config).encode());child.stdin.close()
    print('Cloud session started: '+name,flush=True)
    print('Detached training/download/verified-backup supervisor started. Mac may disconnect.',flush=True)
    print('Dataset revision:',revision,flush=True)
    print('Watch:',local/'console.log',flush=True)
    print('Persistent reports/checkpoints:',session,flush=True)
    print(f'Inner limit {config["timeout_budget"]["inner_seconds"]/3600:g} hours; independent outer limit {config["timeout_budget"]["outer_seconds"]/3600:g} hours. Checkpoints retained on volume.',flush=True)
    return 0


def train(command,log,env,*,local_log=None):
    from contextlib import ExitStack
    with ExitStack() as stack:
        local=stack.enter_context(Path(local_log).open('a',buffering=1)) if local_log else None
        streams=[]
        for path in (Path(log),Path(log).parent/'console.log'):
            try:streams.append(stack.enter_context(path.open('a',buffering=1)))
            except OSError:
                if local is None:raise
                print(f'Volume logging unavailable: {path}; retaining container log.',flush=True)
        process=subprocess.Popen([str(x) for x in command],cwd=PROJECT_ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
        try:
            for line in process.stdout:
                if local is not None:local.write(line)
                for stream in list(streams):
                    try:stream.write(line)
                    except OSError:
                        if local is None:raise
                        streams.remove(stream)
                        try:stream.close()
                        except OSError:pass
                        print('Volume log write failed; container logging continues while checkpoint rescue completes.',flush=True)
                print(line,end='',flush=True)
            code=process.wait()
            if code:raise subprocess.CalledProcessError(code,command)
        finally:
            if process.poll() is None:
                from cloud.bowl_chain import kill_group
                kill_group(process.pid);process.wait(timeout=10)
            process.stdout.close()


def verify_computation(run,profile,checkpoints,allow_graceful_partial=False):
    from cloud.bowl_chain_worker import verify_training
    run=Path(run);marker=run/'compute_completion.json'
    if allow_graceful_partial and marker.is_file():
        final=json.loads(marker.read_text());last=final['last_update']
        if final.get('training_complete') is False:
            if (type(last) is not int or not 0<last<profile['max_steps'] or
                final.get('run_update')!=last or final.get('reason')!='computation_limit'):
                raise ValueError('Invalid graceful training-stop record')
            actual=sorted(set([u for u in checkpoints if u<=last]+[last]))
            completion=verify_training(run,last,actual)
            completion.update(training_complete=False,requested_updates=profile['max_steps'],
                compute_stop_reason='computation_limit',preserved_updates=last)
            return completion,actual
    return verify_training(run,profile['max_steps'],checkpoints),checkpoints


def worker(config):
    import shutil
    from huggingface_hub import HfApi
    from cloud.bowl_chain_worker import download_dataset,stage_base,verify_training
    from cloud.chain_archive import backup_training,verify_files
    from cloud.archive_sweep import digest
    session=Path(config['session']);local=Path(config['local']);timings=Timings(session)
    profile=json.loads((PROJECT_ROOT/config['training_config']).read_text());checkpoints=profile['checkpoint_updates']
    def stage(name):
        print('Stage: '+name,flush=True)
        with (session/'console.log').open('a') as log:log.write('Stage: '+name+'\n')
        write_status(session/'status.json',dict(phase='training',stage=name,started_at_utc=config['started_at_utc'],stage_started_at_utc=utc(),timeout_budget=config['timeout_budget']))
        return timings.stage(name)
    with stage('storage_preflight'):
        if shutil.disk_usage(local).free<40*1024**3:raise ValueError('Need 40 GiB free on container disk')
        if shutil.disk_usage(session).free<config.get('minimum_volume_free_gib',5)*1024**3:raise ValueError('Insufficient free space on network volume for checkpoints and resume state')
    api=HfApi(token=config['hf_token'])
    with stage('hf_write_preflight'):
        api.create_repo(config['checkpoint_repo'],repo_type='model',private=True,exist_ok=True)
        if not api.repo_info(config['checkpoint_repo'],repo_type='model').private:raise ValueError('Checkpoint backup requires a private model repository')
        api.upload_file(repo_id=config['checkpoint_repo'],repo_type='model',path_or_fileobj=json.dumps(dict(session=session.name)).encode(),path_in_repo=f'preflight/{session.name}.json',commit_message='Check prepared training backup write access')
    os.environ.update(HF_XET_CHUNK_CACHE_SIZE_BYTES='0',HF_XET_CACHE=str(local/'xet-cache'))
    with stage('dataset_download_and_verification'):download_dataset(config)
    with stage('model_staging'):write_status(session/'model_staging.json',stage_base(config['base_volume'],config['base']))
    initial_adapter=None;initial_identity=None
    if config.get('initial_adapter'):
        from training.initial_adapter import stage_initial_adapter
        with stage('original_initial_adapter_download_and_verification'):
            initial_adapter,initial_identity=stage_initial_adapter(config['initial_adapter'],local,config,token=config['hf_token'])
            write_status(session/'initial_adapter_source.json',initial_identity)
    provenance=dict(dataset_repo=config['dataset_repo'],dataset_revision=config['dataset_revision'],checksums_sha256=config['checksums_sha256'],bundle=config['dataset_bundle'],training_episodes=600,training_examples=43344,all_training=True,base_model_volume_path=config['base_volume'],code_commit=config['code_commit'])
    if initial_identity:provenance.update(initial_adapter=initial_identity,initialization_mode='restored_initial_lora')
    write_status(session/'provenance.json',provenance)
    env=os.environ.copy();env.pop('HF_TOKEN',None);env.pop('RUNPOD_API_KEY',None)
    with stage('training'):
        train([sys.executable,'-u','-m','training.train_panda','--dataset',config['dataset'],'--model',config['base'],'--config',PROJECT_ROOT/config['training_config'],'--output-root',session/'runs','--run-name','training','--provenance',session/'provenance.json',*(['--adapter',str(initial_adapter),'--base-staging-receipt',str(session/'model_staging.json')] if initial_adapter else [])],session/'training.log',env)
        completion,checkpoints=verify_computation(config['training_run'],profile,checkpoints,config.get('deadline_policy')=='compute_then_delivery_v1')
        if profile.get('save_final_training_state',True):
            from training.resume_state import verify_state
            verify_state(Path(config['training_run'])/f'checkpoint_{completion["last_update"]:06d}',completion['last_update'])
            completion['final_training_state_verified']=True
        write_status(session/'training_completion.json',completion)
    with stage('checkpoint_hf_backup_and_verification'):backup_training(session,config['checkpoint_repo'],config['hf_token'],local,checkpoints)
    start_label='Original checkpoint-zero LoRA initialization' if initial_adapter else 'Fresh base'
    lines=['# Colored cube-to-bowl training', '', '600 training demonstrations / 43,344 action examples; all four training colors balanced.',
        f'{start_label}, {completion["last_update"]:,}/{profile["max_steps"]:,} requested updates; microbatch 8, accumulation 1, learning rate 0.0001, rank 32; no validation split.',
        f'Checkpoints {checkpoints} retained on volume and verified on HF; final checkpoint includes optimizer, RNG and input-stream state.',
        f'Dataset: `{config["dataset_repo"]}@{config["dataset_revision"]}`.',
        f'Training-update wall time: {completion["training_update_seconds"]:.2f} seconds.',
        f'Checkpoint-save wall time: {completion["checkpoint_seconds"]:.2f} seconds.',
        '', '[Stage timings](TIMINGS.md) · [Smoothed loss](runs/training/loss_smoothed.png)']
    if not completion['training_complete']:lines[2:2]=['**INCOMPLETE: computation limit reached; completed updates and resumable state preserved.**','']
    (session/'RUN_TRAINING_REPORT.md').write_text('\n'.join(lines)+'\n')
    write_status(session/'worker_completion.json',dict(training_complete=completion['training_complete'],work_complete=completion['training_complete'],checkpoint_backup_verified=True,delivery_verified=True,finished_at_utc=utc()))
    with stage('final_report_hf_backup'):
        staging=local/'final-reports';staging.mkdir()
        for p in session.iterdir():
            if p.is_file() and p.suffix in ('.json','.jsonl','.md'):shutil.copyfile(p,staging/p.name)
        records=[dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in staging.iterdir()]
        prefix='training-reports/'+session.name
        commit=api.upload_folder(repo_id=config['checkpoint_repo'],repo_type='model',folder_path=staging,path_in_repo=prefix,commit_message='Prepared colored bowl training reports')
        verify_files(config['checkpoint_repo'],'model',commit.oid,prefix,records,config['hf_token'],local)
    # Include the completed upload duration in the last verified timing snapshot.
    records=[]
    for name in ('stage_timings.jsonl','TIMINGS.md'):
        shutil.copyfile(session/name,staging/name);p=staging/name;records.append(dict(path=name,bytes=p.stat().st_size,sha256=digest(p)))
    commit=api.upload_folder(repo_id=config['checkpoint_repo'],repo_type='model',folder_path=staging,path_in_repo=prefix,allow_patterns=['stage_timings.jsonl','TIMINGS.md'],commit_message='Complete training stage timings')
    verify_files(config['checkpoint_repo'],'model',commit.oid,prefix,records,config['hf_token'],local)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--launch',action='store_true');p.add_argument('--token-stdin',action='store_true');p.add_argument('--download-window-seconds',type=int,default=120)
    p.add_argument('--profile',type=Path,default=PROJECT_ROOT/'inference/training_profiles/color_place_bowl_600.json');a=p.parse_args()
    if a.launch:
        token=json.load(sys.stdin)['hf_token'] if a.token_stdin else None
        try:return launch(a.profile,token,a.download_window_seconds)
        except (OSError,ValueError,RuntimeError) as exc:raise SystemExit(str(exc))
    return worker(json.load(sys.stdin))


if __name__=='__main__':raise SystemExit(main())
