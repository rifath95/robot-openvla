"""Last-resort verified checkpoint copies; no HF calls and no source deletion."""
import json
import re
from pathlib import Path

from cloud.archive_sweep import digest
from cloud.chain_archive import recover_to_global
from cloud.chain_timing import utc
from cloud.checkpoint_ready import check_ready


def record(config, name, value):
    """Local receipt first: a full network volume must not block recovery."""
    data=json.dumps(value,indent=2)+'\n'
    (Path(config['local'])/name).write_text(data)
    try:(Path(config['session'])/name).write_text(data)
    except OSError:pass


def hf_covers(config, folder, receipt):
    if not isinstance(receipt,dict):return False
    if (receipt.get('verified') is not True or receipt.get('repo_id')!=config.get('checkpoint_repo')
        or receipt.get('path_in_repo')!='sessions/'+Path(config['session']).name or not receipt.get('revision')):
        return False
    try:records={row['path']:row for row in receipt.get('files',[])}
    except (KeyError,TypeError):return False
    for path in folder.rglob('*'):
        if path.is_symlink():return False
        if not path.is_file():continue
        relative='runs/training/'+folder.name+'/'+path.relative_to(folder).as_posix()
        entry=records.get(relative)
        if not isinstance(entry,dict) or entry.get('bytes')!=path.stat().st_size or entry.get('sha256')!=digest(path):return False
    return True


def recover_checkpoints(config):
    if not config.get('checkpoint_recovery') or not config.get('global_volume'):
        return dict(enabled=False)
    session=Path(config['session'])
    primary=Path(config['training_run'])
    emergency=Path(config['checkpoint_recovery_dir'])
    try:receipt=json.loads((session/'checkpoint_hf_receipt.json').read_text())
    except (OSError,ValueError):receipt={}
    copies=[];covered=[];unavailable=[];errors=[]
    destination=Path(config['global_volume'])/'panda-chain-recovery'/session.name/'checkpoints'
    updates=set(config['recovery_checkpoint_updates'])
    # The trainer can publish one final unscheduled checkpoint on a runtime
    # limit. Preserve it too, rather than discarding the latest completed save.
    for root in (emergency,primary):
        try:
            for folder in root.glob('checkpoint_*'):
                match=re.fullmatch(r'checkpoint_([0-9]{6})',folder.name)
                if match and config.get('initial_update',0)<int(match[1])<=max(updates):updates.add(int(match[1]))
        except OSError:pass
    for cp in sorted(updates):
        name=f'checkpoint_{cp:06d}'
        selected=None
        # A freshly rescued local copy may supersede an incomplete volume save.
        for folder in (emergency/name,primary/name):
            try:
                check_ready(folder)
                saved=json.loads((folder/'training_config.json').read_text())
                provenance=saved.get('dataset_provenance',{})
                if saved.get('checkpoint_update')!=cp:raise ValueError('Wrong checkpoint update')
                if config.get('checksums_sha256') and provenance.get('checksums_sha256')!=config['checksums_sha256']:
                    raise ValueError('Wrong checkpoint dataset')
                if config.get('dataset_revision') and provenance.get('dataset_revision')!=config['dataset_revision']:
                    raise ValueError('Wrong checkpoint dataset revision')
                selected=folder
                break
            except (OSError,ValueError,KeyError,TypeError):continue
        if selected is None:
            unavailable.append(cp)
            continue
        try:
            if hf_covers(config,selected,receipt):
                covered.append(cp)
                print(f'Checkpoint {cp} already verified on HF; no global copy.',flush=True)
                continue
            print(f'Last-resort checkpoint recovery: {selected} -> {destination/name}',flush=True)
            copied=recover_to_global(selected,destination)
            copies.append(dict(update=cp,**copied))
        except Exception as exc:
            errors.append(dict(update=cp,error=type(exc).__name__+': '+str(exc)))
    result=dict(enabled=True,verified=bool(copies or covered) and not errors,
        global_copies=copies,hf_verified_updates=covered,unavailable_updates=unavailable,
        errors=errors,expected_updates=config['recovery_checkpoint_updates'],finished_at_utc=utc(),
        note='Only complete checkpoints are recoverable; unavailable updates were not saved or are incomplete. Sources retained.')
    # A global index is emitted only after copying and read-back verification.
    if copies:
        (destination/'CHECKPOINT_RECOVERY.json').write_text(json.dumps(result,indent=2)+'\n')
    record(config,'checkpoint_global_recovery_receipt.json',result)
    if errors:raise RuntimeError('Some checkpoint recovery copies failed; see local/global receipts')
    return result
