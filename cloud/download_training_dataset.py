"""Download the verified Hugging Face training bundle onto the network volume."""

from common.paths import PROJECT_ROOT
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess


REPOSITORY='Maxalderone/robot-openvla-panda'
REVISION='8ffd54b1dd27022102aba0866b65a7a9ac4eefc7'
V1_REVISION='f653bb505352c358fa7ea478e2338929d3222a68'
BOWL_REPOSITORY='Maxalderone/robot-openvla-panda-bowl'
BOWL_REVISION='6750b3dad36b0d238c5e101ec70e64a2c36bbc62'
TOUCH_REPOSITORY='Maxalderone/robot-openvla-panda-touch'
TOUCH_REVISION='4db62eb5225e9d02df450565f1987eff65dd71c4'
COLOR_TOUCH_REPOSITORY='Maxalderone/robot-openvla-panda-color-touch'
COLOR_TOUCH_REVISION='48d4da5e692b3f13c774e8cf547a8668115fae09'


def bundle_source(bundle, repo=None, revision=None):
    if bundle=='training_four_color_touch_v1':
        if not revision:
            raise ValueError('The four-color bundle requires --revision from its completed upload')
        return repo or 'Maxalderone/robot-openvla-panda-four-color-touch', revision, ''
    if bundle=='training_color_touch_v1':
        return repo or COLOR_TOUCH_REPOSITORY, revision or COLOR_TOUCH_REVISION, ''
    if bundle=='training_touch_v1':
        return repo or TOUCH_REPOSITORY, revision or TOUCH_REVISION, ''
    if bundle=='training_bowl_v2_30':
        if not revision:
            raise ValueError('The pilot bundle requires --revision from its completed upload')
        return repo or 'Maxalderone/robot-openvla-panda-bowl-v2', revision, ''
    if bundle=='training_bowl_v1':
        return repo or BOWL_REPOSITORY, revision or BOWL_REVISION, ''
    if bundle not in ('training_v1','training_v2'):
        raise ValueError('Unsupported training bundle')
    return repo or REPOSITORY, revision or (V1_REVISION if bundle=='training_v1' else REVISION), bundle+'/'


def load_hf_token(environ=None,container_environment=Path('/proc/1/environ')):
    """SSH may omit the HF_TOKEN injected into the container's PID 1.

    Recover only HF_TOKEN in memory, without sourcing shell code or persisting
    credentials. An explicitly provided shell token takes precedence.
    """
    environ=os.environ if environ is None else environ
    token=environ.get('HF_TOKEN','')
    source='shell environment'
    if not token:
        try:
            entries=container_environment.read_bytes().split(b'\0')
            token=next((entry.split(b'=',1)[1].decode() for entry in entries
                        if entry.startswith(b'HF_TOKEN=')), '')
        except (OSError,UnicodeError):
            token=''
        source='container environment'
    token=token.strip()
    if not token:
        raise ValueError('HF_TOKEN is missing. Set it to the Hugging Face read secret on the pod.')
    if '{{' in token or 'RUNPOD_SECRET' in token:
        raise ValueError('HF_TOKEN contains an unresolved RunPod secret placeholder.')
    environ['HF_TOKEN']=token
    return source


def verify_bundle(directory, checksums):
    if (directory/'SHA256SUMS').read_text()!=checksums:
        raise ValueError('Bundle checksums differ from the selected Hugging Face revision')
    count=0
    for line in checksums.splitlines():
        digest,name=line.split('  ',1)
        path=PurePosixPath(name)
        if (not re.fullmatch(r'[0-9a-f]{64}',digest) or path.is_absolute()
            or '..' in path.parts):
            raise ValueError('Invalid checksum entry')
        target=directory.joinpath(*path.parts)
        if not target.resolve().is_relative_to(directory.resolve()) or target.is_symlink():
            raise ValueError('Checksum target escapes the dataset directory')
        hasher=hashlib.sha256()
        with target.open('rb') as stream:
            while chunk:=stream.read(1024**2):hasher.update(chunk)
        if hasher.hexdigest()!=digest:
            raise ValueError(f'Checksum failed: {name}')
        count+=1
        print(f'{name}: OK',flush=True)
    if not count or not json.loads((directory/'verification.json').read_text()).get('passed'):
        raise ValueError('Bundle must have files and a passing dataset verification')
    return count


def finalize_download(bundle,destination,checksums):
    if destination.exists():
        raise ValueError('Destination already exists; verify it instead of overwriting')
    count=verify_bundle(bundle,checksums)
    bundle.rename(destination)
    return count


def copy_verified_bundle(bundle,destination,checksums):
    """Global storage: regular copies, no locks, links, or atomic directory rename."""
    if destination.exists():
        raise ValueError('Destination already exists; verify it instead of overwriting')
    verify_bundle(bundle,checksums)
    destination.mkdir(parents=True)
    files=sorted(p for p in bundle.rglob('*') if p.is_file())
    # Publish the verification record last. Only one preparation pod should write.
    files.sort(key=lambda p:p.relative_to(bundle).as_posix()=='verification.json')
    for source in files:
        target=destination/source.relative_to(bundle)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    return verify_bundle(destination,checksums)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',help='Override the selected bundle’s Hugging Face repository')
    p.add_argument('--bundle',choices=('training_v1','training_v2','training_bowl_v1','training_bowl_v2_30','training_touch_v1','training_color_touch_v1','training_four_color_touch_v1'),default='training_v2')
    p.add_argument('--revision',help='Override the selected bundle’s pinned completed upload')
    p.add_argument('--volume-root',type=Path,default=Path('/workspace'))
    p.add_argument('--global-volume',action='store_true',help='Download on container disk and copy regular files to global storage')
    args=p.parse_args()
    if not args.volume_root.is_absolute():p.error('--volume-root must be an absolute mounted path')
    subprocess.run(['mountpoint','-q',str(args.volume_root)],check=True)
    try:
        source=load_hf_token()
    except ValueError as exc:
        p.error(str(exc))
    print(f'Hugging Face authentication loaded from {source}.',flush=True)
    from huggingface_hub import HfApi,hf_hub_download,snapshot_download
    # Resolve any supplied tag to a commit once, so files cannot mix revisions.
    repository,requested_revision,prefix=bundle_source(args.bundle,args.repo,args.revision)
    revision=HfApi().repo_info(repository,repo_type='dataset',revision=requested_revision).sha
    cache=PROJECT_ROOT/'.cache-dataset-download'/args.bundle
    download_options={'cache_dir':str(cache/'hub')} if args.global_volume else {}
    expected=Path(hf_hub_download(repository,prefix+'SHA256SUMS',
        repo_type='dataset',revision=revision,**download_options)).read_text()
    parent=args.volume_root/'panda-training';parent.mkdir(parents=True,exist_ok=True)
    destination=parent/args.bundle
    if destination.exists():
        count=verify_bundle(destination,expected)
        print(f'Existing bundle verified: {destination} ({count} checksums)',flush=True)
    else:
        staging=cache/'snapshot' if args.global_volume else parent/f'.hf-download-{args.bundle}'
        print(f'Downloading {repository}@{revision} into a resumable staging directory...',flush=True)
        patterns=[prefix+'**'] if prefix else ['SHA256SUMS','*.json','PACKAGE_REPORT.md','rlds/**']
        snapshot_download(repository,repo_type='dataset',revision=revision,
            allow_patterns=patterns,local_dir=staging,**download_options)
        publish=copy_verified_bundle if args.global_volume else finalize_download
        count=publish(staging/args.bundle if prefix else staging,destination,expected)
        print(f'Verified downloaded bundle: {destination} ({count} checksums)',flush=True)
    (parent/f'{args.bundle}_hf_source.json').write_text(json.dumps(dict(
        repository=repository,revision=revision,bundle=str(destination),verified_files=count,
        publication='regular_copy' if args.global_volume else 'directory_rename'),indent=2)+'\n')
    print('Dataset ready. This command does not start training or stop the pod.',flush=True)


if __name__=='__main__':main()
