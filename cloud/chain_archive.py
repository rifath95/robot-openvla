"""Chain-only backups: no pod stopping or deletion of persistent checkpoints."""
import hashlib,json,os,shutil,tempfile,time
from pathlib import Path
from cloud.archive_sweep import package,digest
from cloud.chain_timing import utc


def verify_files(repo,repo_type,revision,prefix,records,token,local):
    from huggingface_hub import hf_hub_download
    # One isolated cache per file; verified downloads never accumulate to the full sweep size.
    for record in records:
        with tempfile.TemporaryDirectory(prefix='hf-verify-',dir=local) as tmp:
            file=Path(hf_hub_download(repo_id=repo,repo_type=repo_type,revision=revision,filename=prefix+'/'+record['path'],token=token,cache_dir=tmp))
            if file.stat().st_size!=record['bytes'] or digest(file)!=record['sha256']:raise ValueError('HF backup verification failed')
        print('HF verified:',record['path'],flush=True)


def backup_training(session,repo,token,local):
    from huggingface_hub import HfApi
    from cloud.checkpoint_ready import check_ready
    started=time.monotonic()
    source=Path(session);run=source/'runs/training'
    for cp in (0,500,1000,2000,3000):check_ready(run/f'checkpoint_{cp:06d}')
    staging=Path(local)/'checkpoint-backup';staging.mkdir(parents=True,exist_ok=False)
    records=[]
    for parent,dirs,files in os.walk(source):
        dirs[:]=[d for d in dirs if d not in ('upstream','adapter-tmp','.cache','sweep-reports')]
        for name in files:
            p=Path(parent)/name;rel=p.relative_to(source)
            if p.is_symlink():raise ValueError('Refusing a checkpoint symlink')
            checkpoint=any(s.startswith('checkpoint_') for s in rel.parts)
            if not checkpoint and p.suffix not in ('.json','.jsonl','.md','.png','.log','.txt') and not name.startswith('events.out.tfevents.'):continue
            target=staging/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
            records.append(dict(path=rel.as_posix(),bytes=target.stat().st_size,sha256=digest(target)))
    manifest=staging/'BACKUP_MANIFEST.json';manifest.write_text(json.dumps(dict(session=source.name,files=records),indent=2)+'\n')
    records.append(dict(path=manifest.name,bytes=manifest.stat().st_size,sha256=digest(manifest)))
    api=HfApi(token=token);prefix='sessions/'+source.name
    if not api.repo_info(repo,repo_type='model').private:raise ValueError('Checkpoint backup repository must be private')
    upload_started=time.monotonic()
    commit=api.upload_folder(repo_id=repo,repo_type='model',folder_path=staging,path_in_repo=prefix,commit_message=f'Preserve {source.name} checkpoints and reports')
    verification_started=time.monotonic()
    verify_files(repo,'model',commit.oid,prefix,records,token,local)
    receipt=dict(verified=True,repo_id=repo,repo_type='model',revision=commit.oid,path_in_repo=prefix,verified_bytes=sum(r['bytes'] for r in records),files=records,verified_at_utc=utc(),volume_checkpoints_retained=True,container_copy_seconds=upload_started-started,upload_seconds=verification_started-upload_started,verification_seconds=time.monotonic()-verification_started)
    (source/'checkpoint_hf_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def backup_sweep(source,repo,token,local,receipt_dir):
    from huggingface_hub import HfApi
    from cloud.sweep_plan import validate_saved
    source=Path(source);payload=json.loads((source/'results.json').read_text());specs=validate_saved(payload)
    if payload.get('launcher_error') or len(payload['trials'])!=len(specs):raise ValueError('Cannot archive an incomplete sweep as complete')
    staging=Path(local)/'sweep-backup';staging.mkdir(parents=True,exist_ok=False)
    api=HfApi(token=token);prefix=source.name
    if not api.repo_info(repo,repo_type='dataset').private:raise ValueError('Sweep backup repository must be private')
    def upload_part(path,entry):
        started=time.monotonic()
        api.upload_file(repo_id=repo,repo_type='dataset',path_or_fileobj=path,path_in_repo=prefix+'/'+entry['path'],commit_message=f'{prefix}: {path.name}')
        with (Path(receipt_dir)/'archive_part_timings.jsonl').open('a') as f:f.write(json.dumps(dict(part=entry['path'],event='upload',seconds=time.monotonic()-started,bytes=entry['bytes']))+'\n')
    start=time.monotonic();packing_start=[start]
    original_upload=upload_part
    def timed_part(path,entry):
        with (Path(receipt_dir)/'archive_part_timings.jsonl').open('a') as f:f.write(json.dumps(dict(part=entry['path'],event='packing',seconds=time.monotonic()-packing_start[0],bytes=entry['bytes']))+'\n')
        original_upload(path,entry);packing_start[0]=time.monotonic()
    package(source,staging,payload,on_part=timed_part)
    # The final revision contains every uploaded part plus the report manifest.
    final_upload_started=time.monotonic()
    commit=api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=staging,path_in_repo=prefix,allow_patterns=['reports/*','manifest.json'],commit_message=f'Complete {prefix} archive')
    manifest=json.loads((staging/'manifest.json').read_text());records=manifest['files']+[dict(path='manifest.json',bytes=(staging/'manifest.json').stat().st_size,sha256=digest(staging/'manifest.json'))]
    verification_start=time.monotonic();verify_files(repo,'dataset',commit.oid,prefix,records,token,local)
    receipt=dict(verified=True,repo_id=repo,repo_type='dataset',revision=commit.oid,path_in_repo=prefix,verified_bytes=sum(r['bytes'] for r in records),trials=manifest['trials'],source_revision=manifest['source_revision'],verified_at_utc=utc(),packing_and_upload_seconds=verification_start-start,verification_seconds=time.monotonic()-verification_start)
    parts=[json.loads(line) for line in (Path(receipt_dir)/'archive_part_timings.jsonl').read_text().splitlines()]
    receipt.update(packing_seconds=sum(r['seconds'] for r in parts if r['event']=='packing'),upload_seconds=sum(r['seconds'] for r in parts if r['event']=='upload')+verification_start-final_upload_started)
    (Path(receipt_dir)/'sweep_hf_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def recover_to_global(source,global_directory):
    """Regular file copies only: no locks/links/atomic rename on object-backed storage."""
    source=Path(source);destination=Path(global_directory)/source.name
    destination.mkdir(parents=True,exist_ok=False);records=[]
    for p in sorted(source.rglob('*')):
        if p.is_symlink():raise ValueError('Refusing recovery symlink')
        if not p.is_file():continue
        target=destination/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
        with p.open('rb') as src,target.open('wb') as dst:
            shutil.copyfileobj(src,dst,length=8*1024**2)
            dst.flush()
        sha=digest(p)
        if target.stat().st_size!=p.stat().st_size or digest(target)!=sha:raise ValueError('Global recovery checksum failed')
        records.append(dict(path=p.relative_to(source).as_posix(),bytes=p.stat().st_size,sha256=sha))
    receipt=dict(verified=True,source=str(source),destination=str(destination),files=records,verified_at_utc=utc())
    # A complete receipt is published only after every file is verified.
    (destination/'RECOVERY_COMPLETE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def prepare_recovery_archive(source,local):
    """Preserve raw or partial runs in a few archives rather than thousands of object writes."""
    import tarfile
    from cloud.archive_sweep import add_checked
    source=Path(source);local=Path(local)
    staged=local/'sweep-backup'
    manifest=staged/'manifest.json'
    if manifest.exists():
        try:saved=json.loads(manifest.read_text())
        except ValueError:saved={}
        if saved.get('source_revision')==digest(source/'results.json'):
            for entry in saved['files']:
                p=staged/entry['path']
                if not p.is_file() or p.stat().st_size!=entry['bytes'] or digest(p)!=entry['sha256']:raise ValueError('Recovery packing cache corrupted')
            return staged
    # Only incomplete, regenerable container packing is removed to make recovery space.
    if staged.exists():shutil.rmtree(staged)
    destination=local/'recovery-package';destination.mkdir(parents=True,exist_ok=False)
    required=sum(p.stat().st_size for p in source.rglob('*') if p.is_file())
    if shutil.disk_usage(local).free<required+2*1024**3:raise ValueError('Insufficient container space to pack emergency recovery')
    target=destination/'recordings.tar.gz'
    with tarfile.open(target,'w:gz',compresslevel=1) as archive:add_checked(archive,source,Path(source.name))
    (destination/'RECOVERY_INFO.json').write_text(json.dumps(dict(source_name=source.name,format='full_source_tar_gz',note='May contain a partial sweep; inspect results.json before resuming. Extract with path/link checks. Checkpoints remain on main volume.'),indent=2)+'\n')
    return destination
