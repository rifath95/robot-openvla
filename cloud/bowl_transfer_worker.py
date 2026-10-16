"""Run bowl trials on the GPU; retain recordings on volume for a transfer pod."""
import json
from pathlib import Path
import shutil
import subprocess
import time
from common.paths import PROJECT_ROOT


def run(directory,sim_python,*,cache_root=Path('/root/panda-bowl-transfer-work')):
    from cloud.bowl_chain_plan import validate_saved
    from cloud.bowl_chain_worker import stage_base
    from cloud.bowl_chain_sweep import run as sweep
    from training.detached_run import write_status
    directory=Path(directory);payload=json.loads((directory/'results.json').read_text());validate_saved(payload)
    plan=payload['plan'];remaining=len(validate_saved(payload))-len(payload['trials'])
    if shutil.disk_usage(directory).free<max(2,35*remaining/800)*1024**3:
        raise ValueError('Need 35 GiB free on network volume for a new 800-trial bowl sweep')
    local=Path(cache_root)/(directory.name+'_'+str(time.time_ns()))
    local.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(local).free<25*1024**3:raise ValueError('Need 25 GiB free on container disk for staged weights')
    # Probe graphics before model staging; use the same frozen scene preflight.
    import os
    env=os.environ.copy();env.pop('RUNPOD_API_KEY',None);env.pop('HF_TOKEN',None)
    env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    subprocess.run([str(sim_python),'-m','cloud.bowl_chain_sweep','--preflight',str(directory/'results.json')],cwd=PROJECT_ROOT,env=env,check=True,timeout=180)
    write_status(directory/'model_staging.json',stage_base('/workspace/openvla-7b',local/'base'))
    profile=json.loads((PROJECT_ROOT/'inference/sweep_profiles/color_place_bowl_600_uninterrupted.json').read_text())
    config=dict(sweep=str(directory),session=str(directory),local=str(local),base=str(local/'base'),
        training_run=str(Path('/workspace/panda-training/sessions')/plan['training_session']/'runs/training'),
        source_status=str(Path('/workspace/panda-training/sessions')/plan['training_session']/'status.json'),
        checkpoint_wait_seconds=profile['checkpoint_wait_seconds'],source_dataset_sha256=profile['source_dataset_sha256'],
        source_dataset_revision=profile['source_dataset_revision'],verify_adapter_copy=True,sim_python=str(sim_python))
    result=sweep(config)
    print('Recordings retained on network volume. Upload later with cloud.archive_sweep on a cheaper transfer pod.',flush=True)
    return result
