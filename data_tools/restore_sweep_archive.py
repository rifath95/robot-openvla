"""Restore verified HF archive parts into the ordinary local sweep/replay layout."""

from common.output_layout import output_path,resolve_output_path
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import tarfile
from common.paths import PROJECT_ROOT
from cloud.sweep_plan import validate_saved


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(4*1024**2):h.update(block)
    return h.hexdigest()


def restore(source,destination=None):
    source=resolve_output_path(source).resolve();manifest=json.loads((source/'manifest.json').read_text())
    records=json.loads((source/'reports/results.json').read_text());specs=validate_saved(records)
    partial=(manifest.get('complete') is False and manifest.get('expected_trials')==len(specs)
        and manifest.get('stop_reason')=='inner_compute_deadline'
        and records['plan'].get('compute_stop_reason')=='inner_compute_deadline'
        and not records.get('launcher_error') and manifest['trials']<len(specs))
    if len(records['trials'])!=manifest['trials'] or (len(specs)!=manifest['trials'] and not partial):raise ValueError('Incomplete archive trial inventory')
    name=manifest['sweep']
    if not name.startswith('scripted_') or Path(name).name!=name:raise ValueError('Invalid sweep name')
    destination=Path(destination).resolve() if destination else output_path(name)
    if not destination.is_relative_to((PROJECT_ROOT/'outputs').resolve()):raise ValueError('Restore under the project outputs directory')
    destination.mkdir(parents=True,exist_ok=True)
    progress=destination/'restore_progress.json'
    done=json.loads(progress.read_text()) if progress.exists() else {}
    total_files=0;total_bytes=0
    from common.sweep_layout import validate_trial_directory
    for name in specs:validate_trial_directory(name)
    from common.sweep_layout import prepare_trial_restore
    prepare_trial_restore(destination,specs)
    for entry in manifest['files']:
        relative=PurePosixPath(entry['path'])
        if not relative.parts or relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in ('archives','reports'):raise ValueError('Unsafe manifest path')
        path=source/str(relative)
        if path.is_symlink() or not path.resolve().is_relative_to(source):raise ValueError('Unsafe source file')
        if path.stat().st_size!=entry['bytes']:raise ValueError('Incomplete download: '+str(relative))
        if relative.parts[0]=='archives' and done.get(str(relative))==entry['sha256']:
            print('Already restored:',relative,flush=True);continue
        print('Verifying and restoring:',relative,flush=True)
        if sha256(path)!=entry['sha256']:raise ValueError('Checksum mismatch: '+str(relative))
        if relative.parts[0]=='archives':
            with tarfile.open(path,'r|gz') as archive:
                for member in archive:
                    member_path=PurePosixPath(member.name)
                    identity='/'.join(member_path.parts[:2]) if member_path.parts[:1]==('trials',) else '/'.join(member_path.parts[:1])
                    parent_only=member.name=='trials' and member.isdir()
                    target=destination/str(member_path)
                    if (not member_path.parts or member_path.is_absolute() or '..' in member_path.parts
                        or (identity not in specs and not parent_only) or not (member.isfile() or member.isdir())
                        or not target.resolve().is_relative_to(destination)):
                        raise ValueError('Unsafe archive entry: '+member.name)
                    total_files+=1;total_bytes+=member.size
                    if total_files>1500000 or total_bytes>100*1024**3:raise ValueError('Archive exceeds restore limits')
                    # Extract old archives directly into trials/ as well, avoiding a
                    # duplicate tree when a previous restore was already organized.
                    if not parent_only and member_path.parts[0]!='trials':
                        member.name='trials/'+member.name
                    if not (destination/member.name).resolve().is_relative_to(destination):
                        raise ValueError('Unsafe archive entry: '+member.name)
                    archive.extract(member,path=destination,filter='data')
            done[str(relative)]=entry['sha256'];progress.write_text(json.dumps(done,indent=2)+'\n')
        else:
            if len(relative.parts)!=2 or path.suffix not in ('.json','.jsonl','.md','.log','.txt'):raise ValueError('Unexpected report file')
            target=destination/relative.name
            if target.is_symlink() or not target.resolve().is_relative_to(destination):raise ValueError('Unsafe report destination')
            shutil.copyfile(path,target)
    from data_tools.organize_sweep_trials import organize
    organize(destination,apply=True)
    restored=json.loads((destination/'results.json').read_text())
    if any(not (destination/r['directory']/'summary.json').is_file() for r in restored['trials']):raise ValueError('Restored trial missing')
    from analysis.replay_launcher import refresh
    refresh(destination)
    print(f"Restored {manifest['trials']} trials to {destination}",flush=True)
    return destination


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive_folder',type=Path)
    p.add_argument('--destination',type=Path);a=p.parse_args();restore(a.archive_folder,a.destination)

if __name__=='__main__':main()
