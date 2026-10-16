"""Detached 800-trial colored bowl sweep, verified HF delivery and pod stop."""
import argparse,fcntl,getpass,json,os,re,shutil,subprocess,sys,uuid
from datetime import datetime,timezone
from pathlib import Path
from common.paths import PROJECT_ROOT
from cloud.bowl_chain_plan import make_colored_plan,make_uninterrupted_plan,timeout_budget
from cloud.chain_timing import Timings,utc
from training.detached_run import preflight,write_status

PROFILE=PROJECT_ROOT/'inference/sweep_profiles/color_place_bowl_600.json'


def resolve_training(profile,root=Path('/workspace/panda-training/sessions'),selected=None):
    if selected and not re.fullmatch(r'panda-train-color-bowl-[A-Za-z0-9_-]+',selected):raise ValueError('Invalid source training session')
    candidates=[root/selected] if selected else sorted(root.glob('panda-train-color-bowl-*'))
    matches=[]
    for folder in candidates:
        try:
            config=json.loads((folder/'chain_config.json').read_text());source=json.loads((folder/'provenance.json').read_text())
        except (OSError,ValueError):continue
        if (config.get('training_only') is True and config.get('training_config')==profile['source_training_config']
            and source.get('dataset_repo')==profile['source_dataset_repo']
            and source.get('dataset_revision')==profile['source_dataset_revision']
            and source.get('checksums_sha256')==profile['source_dataset_sha256']
            and source.get('training_episodes')==600 and source.get('training_examples')==43344):matches.append(folder)
    if len(matches)!=1:raise ValueError('Need one matching 600-demo training session; use --training-session if ambiguous')
    return matches[0]


def launch(global_volume,selected=None,profile_path=PROFILE):
    from cloud.scripted_eval_control import active_training
    from huggingface_hub import HfApi
    if active_training():raise ValueError('Use a separate pod; this pod is training')
    pod,key=preflight()
    if not global_volume.is_absolute():raise ValueError('Global recovery mount must be absolute')
    token=getpass.getpass('Hugging Face WRITE token (hidden): ').strip()
    if not token or '{{' in token:raise ValueError('A resolved HF write token is required')
    HfApi(token=token).whoami()
    profile_path=Path(profile_path).resolve()
    profile=json.loads(profile_path.read_text());name='panda-train-bowl-sweep-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
    session=Path('/workspace/panda-training/reports/sweep-sessions')/name;local=Path('/root/panda-chain-work')/name
    session.mkdir(parents=True,exist_ok=False);local.mkdir(parents=True,exist_ok=False)
    lock=Path('/root/panda-long-chain.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,text=True).strip();code=local/'code'
    subprocess.run(['git','clone','--shared',str(PROJECT_ROOT),str(code)],check=True)
    subprocess.run(['git','-C',str(code),'checkout','--detach',commit],check=True)
    sweep_name='scripted_color_bowl_models_'+name.removeprefix('panda-train-bowl-sweep-').replace('-','_')
    config=dict(session=str(session),local=str(local),sweep=str(local/'scripted-evaluations'/sweep_name),base_volume='/workspace/openvla-7b',base=str(local/'base'),global_volume=str(global_volume),selected_training_session=selected,sweep_profile=str(profile_path.relative_to(PROJECT_ROOT)),sweep_only=True,sweep_repo=profile['sweep_repo'],timeout_budget=timeout_budget(profile.get('estimated_seconds',20*3600)),pod_id=pod,pod_key=key,hf_token=token,code_commit=commit)
    env=os.environ.copy();env.pop('RUNPOD_API_KEY',None);env.pop('HF_TOKEN',None)
    with (local/'supervisor.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-u','-m','cloud.bowl_chain','--supervise'],cwd=code,env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
        child.stdin.write(json.dumps(config).encode());child.stdin.close()
    print('Detached 800-trial colored bowl sweep started. Mac may disconnect.',flush=True)
    print('Sweep:',sweep_name,flush=True);print('Watch:',session/'sweep.log',flush=True);print('Setup:',local/'console.log',flush=True);print('Persistent status/timings:',session,flush=True)
    print(f'Inner limit: {config["timeout_budget"]["inner_seconds"]/3600:g} hours; independent outer limit: {config["timeout_budget"]["outer_seconds"]/3600:g} hours. HF upload and verification included.',flush=True)
    return 0


def publish_reports(sweep,local,session,repo,token):
    """Update the small verified report manifest without reuploading archive parts."""
    from huggingface_hub import HfApi
    from cloud.archive_sweep import digest
    from cloud.chain_archive import verify_files
    staging=Path(local)/'sweep-backup';reports=staging/'reports';updated=[]
    for source in Path(session).iterdir():
        if source.is_file() and source.suffix in ('.json','.jsonl','.md'):
            target=reports/source.name;shutil.copyfile(source,target)
            updated.append(dict(path='reports/'+source.name,bytes=target.stat().st_size,sha256=digest(target)))
    manifest=staging/'manifest.json';payload=json.loads(manifest.read_text());paths={r['path'] for r in updated}
    payload['files']=[r for r in payload['files'] if r['path'] not in paths]+updated
    manifest.write_text(json.dumps(payload,indent=2)+'\n')
    api=HfApi(token=token);commit=api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=staging,path_in_repo=Path(sweep).name,allow_patterns=['reports/*','manifest.json'],commit_message='Complete colored bowl sweep timings and provenance')
    updated.append(dict(path='manifest.json',bytes=manifest.stat().st_size,sha256=digest(manifest)))
    verify_files(repo,'dataset',commit.oid,Path(sweep).name,updated,token,local)
    receipt_path=Path(session)/'sweep_hf_receipt.json';receipt=json.loads(receipt_path.read_text());receipt.update(revision=commit.oid,final_reports_verified_at_utc=utc())
    write_status(receipt_path,receipt)


def worker(config):
    from huggingface_hub import HfApi
    from cloud.bowl_chain_worker import global_probe,setup_simulator,stage_base,run_command
    from cloud.chain_archive import backup_sweep
    from cloud.scripted_touch_session import write_reports
    profile=json.loads((PROJECT_ROOT/config.get('sweep_profile',str(PROFILE))).read_text());session=Path(config['session']);local=Path(config['local']);sweep=Path(config['sweep']);timer=Timings(session)
    def stage(name):
        print('Stage: '+name,flush=True)
        write_status(session/'status.json',dict(phase=name,started_at_utc=config['started_at_utc'],stage_started_at_utc=utc(),timeout_budget=config['timeout_budget'],code_commit=config['code_commit']))
        return timer.stage(name)
    with stage('storage_preflight'):
        if shutil.disk_usage(local).free<110*1024**3:raise ValueError('Need 110 GiB actually free on container disk')
        if shutil.disk_usage(session).free<1*1024**3:raise ValueError('Need 1 GiB free on network volume for compact reports')
        global_probe(config['global_volume'])
    api=HfApi(token=config['hf_token'])
    with stage('hf_write_preflight'):
        if not api.repo_info(config['sweep_repo'],repo_type='dataset').private:raise ValueError('Sweep backup requires a private repository')
        api.upload_file(repo_id=config['sweep_repo'],repo_type='dataset',path_or_fileobj=json.dumps(dict(session=session.name)).encode(),path_in_repo='preflight/'+session.name+'.json',commit_message='Check sweep archive write access')
    checkpoints=[6000,8000,10000,12000] if profile.get('uninterrupted') else [1000,2000,4000,6000]
    with stage('source_training_selection'):
        source=resolve_training(profile,selected=config.get('selected_training_session'))
        print('Source training session:',source.name,flush=True)
        config.update(training_run=str(source/'runs/training'),source_status=str(source/'status.json'),checkpoint_wait_seconds=profile['checkpoint_wait_seconds'],source_dataset_sha256=profile['source_dataset_sha256'],source_dataset_revision=profile['source_dataset_revision'],verify_adapter_copy=True)
        write_status(session/'source_training.json',dict(training_session=source.name,dataset_repo=profile['source_dataset_repo'],dataset_revision=profile['source_dataset_revision'],checksums_sha256=profile['source_dataset_sha256'],checkpoints=checkpoints))
    with stage('simulator_setup'):config['sim_python']=setup_simulator(local)
    with stage('model_staging'):write_status(session/'model_staging.json',stage_base(config['base_volume'],config['base']))
    plan=(make_uninterrupted_plan(source.name) if profile.get('uninterrupted') else make_colored_plan(source.name));sweep.mkdir(parents=True,exist_ok=False);write_reports(sweep,plan,[])
    safe={k:v for k,v in config.items() if k not in ('hf_token','pod_key')};write_status(local/'worker_config.json',safe)
    env=os.environ.copy();env.pop('HF_TOKEN',None);env.pop('RUNPOD_API_KEY',None);env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    with stage('scene_initialization_and_render_preflight'):run_command([config['sim_python'],'-u','-m','cloud.bowl_chain_sweep','--preflight',sweep/'results.json'],session/'preflight.log',env=env)
    with stage('simulator_sweep'):run_command([sys.executable,'-u','-m','cloud.bowl_chain_sweep','--config',local/'worker_config.json'],session/'sweep.log',env=env)
    os.environ.update(HF_XET_CHUNK_CACHE_SIZE_BYTES='0',HF_XET_CACHE=str(local/'xet-cache'))
    with stage('sweep_packing_upload_and_verification'):backup_sweep(sweep,config['sweep_repo'],config['hf_token'],local,session)
    (session/'RUN_SWEEP_REPORT.md').write_text(f'# Colored bowl sweep delivery\n\n800 trials; exact prior layouts; checkpoints {checkpoints}.\nResults uploaded and verified on HF before pod stop.\nHistorical base results use the old 300-demo action normalization.\n[Stage timings](TIMINGS.md)\n')
    with stage('final_report_upload_and_verification'):publish_reports(sweep,local,session,config['sweep_repo'],config['hf_token'])
    publish_reports(sweep,local,session,config['sweep_repo'],config['hf_token'])
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--launch',action='store_true');p.add_argument('--training-session');p.add_argument('--global-volume',type=Path,default=Path('/workspace-global'));p.add_argument('--profile',type=Path,default=PROFILE);a=p.parse_args()
    if a.launch:return launch(a.global_volume,a.training_session,a.profile)
    return worker(json.load(sys.stdin))


if __name__=='__main__':raise SystemExit(main())
