"""Populate mounted global storage with the pinned base model and original dataset."""

from common.paths import PROJECT_ROOT
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--volume-root',type=Path,default=Path('/workspace'),
                   help='Actual global-volume mount from findmnt; sometimes /workspace-global')
    args=p.parse_args()
    if not args.volume_root.is_absolute():p.error('Use an absolute mounted volume path')
    subprocess.run(['mountpoint','-q',str(args.volume_root)],check=True)
    root=PROJECT_ROOT
    print('Preparing base weights and training_v1 on:',args.volume_root,flush=True)
    print('No model is loaded into GPU memory; only one preparation pod should write these files.',flush=True)
    subprocess.run([sys.executable,"-m", 'cloud.prepare_cloud_model',
                    '--export-dir',str(args.volume_root/'openvla-7b')],check=True)
    subprocess.run([sys.executable,str(root/'download_training_dataset.py'),
                    '--bundle','training_v1','--global-volume',
                    '--volume-root',str(args.volume_root)],check=True)
    print('Global benchmark inputs ready. This does not start benchmarking or stop the pod.',flush=True)


if __name__=='__main__':main()
