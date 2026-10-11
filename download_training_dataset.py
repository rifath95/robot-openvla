"""Download the verified Hugging Face training bundle onto the network volume."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess


REPOSITORY='Maxalderone/robot-openvla-panda'
REVISION='8ffd54b1dd27022102aba0866b65a7a9ac4eefc7'


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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default=REPOSITORY)
    p.add_argument('--revision',default=REVISION,help='Pinned completed upload; change explicitly for future bundles')
    args=p.parse_args()
    subprocess.run(['mountpoint','-q','/workspace'],check=True)
    try:
        source=load_hf_token()
    except ValueError as exc:
        p.error(str(exc))
    print(f'Hugging Face authentication loaded from {source}.',flush=True)
    from huggingface_hub import HfApi,hf_hub_download,snapshot_download
    # Resolve any supplied tag to a commit once, so files cannot mix revisions.
    revision=HfApi().repo_info(args.repo,repo_type='dataset',revision=args.revision).sha
    expected=Path(hf_hub_download(args.repo,'training_v2/SHA256SUMS',
        repo_type='dataset',revision=revision)).read_text()
    parent=Path('/workspace/panda-training');parent.mkdir(parents=True,exist_ok=True)
    destination=parent/'training_v2'
    if destination.exists():
        count=verify_bundle(destination,expected)
        print(f'Existing bundle verified: {destination} ({count} checksums)',flush=True)
    else:
        staging=parent/'.hf-download-training_v2'
        print(f'Downloading {args.repo}@{revision} into a resumable staging directory...',flush=True)
        snapshot_download(args.repo,repo_type='dataset',revision=revision,
            allow_patterns=['training_v2/**'],local_dir=staging)
        count=finalize_download(staging/'training_v2',destination,expected)
        print(f'Verified downloaded bundle: {destination} ({count} checksums)',flush=True)
    (parent/'training_v2_hf_source.json').write_text(json.dumps(dict(
        repository=args.repo,revision=revision,bundle=str(destination),verified_files=count),indent=2)+'\n')
    print('Dataset ready. This command does not start training or stop the pod.',flush=True)


if __name__=='__main__':main()
