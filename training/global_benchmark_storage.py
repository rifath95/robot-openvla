"""Stage global benchmark inputs locally; publish reports with regular copies."""
import argparse
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import sys
import time

from cloud.download_training_dataset import verify_bundle
from cloud.prepare_cloud_model import MODEL_ID,REVISION
from training.detached_run import ROOT,api_request
from training.pod_environment import recover_pod_environment

LOCAL_ROOT=ROOT/'.benchmark-local'


def global_preflight(volume):
    volume=Path(volume)
    if str(volume) not in ('/workspace','/workspace-global'):
        raise RuntimeError('Global volume must be mounted at /workspace or /workspace-global')
    mount=subprocess.run(['findmnt','-n','-o','TARGET,FSTYPE','-T',str(volume)],
                         text=True,capture_output=True,check=True).stdout.split()
    if len(mount)!=2 or mount[0]!=str(volume) or mount[1] not in ('fuse.geesefs','fuse.gee','fuse.geese'):
        raise RuntimeError('Expected a mounted RunPod global volume; refusing container-disk storage')
    pod_id,key=recover_pod_environment()
    pod=api_request(pod_id,key)
    if pod.get('id')!=pod_id:
        raise RuntimeError('RunPod API did not identify this benchmark pod')
    return pod_id,key


def stage_inputs(volume,dataset,local_root=LOCAL_ROOT):
    started=time.monotonic();volume=Path(volume);dataset=Path(dataset)
    if dataset.parent!=volume/'panda-training' or dataset.name not in ('training_v1','training_v2'):
        raise ValueError('Dataset must be a verified named bundle on the selected global volume')
    source=volume/'openvla-7b'
    marker=json.loads((source/'export_complete.json').read_text())
    if marker.get('model')!=MODEL_ID or marker.get('revision')!=REVISION or not marker.get('files'):
        raise ValueError('Global volume does not contain the completed pinned base-model export')
    sizes=marker['files']
    for name,size in sizes.items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or (source/name).is_symlink():
            raise ValueError('Invalid model export path')
        if (source/name).stat().st_size!=size:raise ValueError('Incomplete model export: '+name)
    verify_bundle(dataset,(dataset/'SHA256SUMS').read_text())
    model=local_root/'inputs/openvla-7b';local_dataset=local_root/'inputs'/dataset.name
    required=sum(sizes.values())+sum(p.stat().st_size for p in dataset.rglob('*') if p.is_file())
    local_root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(local_root).free<required+2*2**30:
        raise ValueError('Need space for staged model/dataset plus 2 GiB on the container disk')
    model.mkdir(parents=True,exist_ok=True)
    cached=ROOT/'.cache-openvla-server/download/models--openvla--openvla-7b/snapshots'/REVISION
    read_source=cached if all((cached/name).is_file() and (cached/name).stat().st_size==size for name,size in sizes.items()) else source
    print('Staging global model and dataset on container disk...',flush=True)
    for name in sizes:
        target=model/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(read_source/name,target)
        if target.stat().st_size!=sizes[name]:raise ValueError('Incomplete staged model: '+name)
    shutil.copyfile(source/'export_complete.json',model/'export_complete.json')
    if local_dataset.exists():shutil.rmtree(local_dataset)
    shutil.copytree(dataset,local_dataset,copy_function=shutil.copyfile)
    verify_bundle(local_dataset,(dataset/'SHA256SUMS').read_text())
    metadata=dict(source_storage='global_volume',volume_mount=str(volume),source_model=str(source),
                  source_dataset=str(dataset),training_reads='container_disk',
                  model_staging_reads='local_pinned_download_cache' if read_source==cached else 'global_volume',
                  input_staging_seconds=time.monotonic()-started,staged_bytes=required)
    return model,local_dataset,metadata


def copy_reports(source,destination):
    source=Path(source);destination=Path(destination)
    if not source.exists():return 0
    count=0
    for path in source.rglob('*'):
        if not path.is_file():continue
        if path.is_symlink():raise ValueError('Refusing symlink in report publication')
        if path.suffix not in ('.json','.jsonl','.md','.log','.png') and not path.name.startswith('events.out.tfevents.'):
            continue
        if any(p.startswith('checkpoint_') for p in path.relative_to(source).parts):continue
        target=destination/path.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        if target.stat().st_size!=path.stat().st_size:raise ValueError('Incomplete report copy')
        count+=1
    return count


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--volume',type=Path)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--max-batch',type=int,default=64)
    p.add_argument('--controlled',action='store_true')
    p.add_argument('--sync-source',type=Path)
    p.add_argument('--sync-destination',type=Path)
    args=p.parse_args()
    if args.sync_source:
        if args.sync_destination is None:p.error('Provide --sync-destination')
        print('Published report files:',copy_reports(args.sync_source,args.sync_destination),flush=True)
        return
    if args.volume is None or args.dataset is None:p.error('Provide --volume and --dataset')
    model,dataset,metadata=stage_inputs(args.volume,args.dataset)
    metadata_path=LOCAL_ROOT/'input_staging.json'
    metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
    os.environ['PANDA_BENCHMARK_STAGING_METADATA']=str(metadata_path)
    command=[sys.executable,'-u',"-m", 'training.benchmark_training_batches',
             '--dataset',str(dataset),'--model',str(model),
             '--output-root',str(LOCAL_ROOT/'batch-benchmarks'),'--max-batch',str(args.max_batch)]
    if not args.controlled:command.append('--find-capacity')
    os.execv(sys.executable,command)


if __name__=='__main__':main()
