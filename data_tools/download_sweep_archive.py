"""Download a pinned HF sweep with one overall byte progress bar and resumable cache."""

import argparse
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import time


def retained_bytes(records):
    """Count each selected file once, including its matching partial download."""
    total = 0
    for final, partial, size in records:
        final = Path(final)
        partial = Path(partial)
        try:
            count = final.stat().st_size
            if count != size:
                count = partial.stat().st_size if partial.exists() else 0
        except FileNotFoundError:
            try:
                count = partial.stat().st_size
            except FileNotFoundError:
                count = 0
        total += min(count, size)
    return total


def download(repo, revision, folder, filenames, workers):
    # Set before importing HF: regular HTTP writes partial files sequentially,
    # allowing accurate disk-byte progress and standard HTTP range resumption.
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    from huggingface_hub import snapshot_download
    snapshot_download(repo, repo_type='dataset', revision=revision,
                      local_dir=folder, allow_patterns=filenames,
                      max_workers=workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', required=True)
    parser.add_argument('--repo', default='Maxalderone/robot-openvla-simulator-results')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--local-dir', type=Path,
                        default=Path('outputs/archives/hf-sweeps/hf-sweep-archives'))
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    if not args.sweep.startswith('scripted_') or Path(args.sweep).name != args.sweep:
        parser.error('Use a single scripted_... sweep name')
    from huggingface_hub import HfApi
    from huggingface_hub._local_folder import get_local_download_paths
    from tqdm import tqdm

    print('Checking archive inventory on Hugging Face...', flush=True)
    info = HfApi().repo_info(args.repo, repo_type='dataset',
                           revision=args.revision, files_metadata=True)
    files = [f for f in info.siblings if f.rfilename.startswith(args.sweep + '/')]
    if not files or not any(f.rfilename == args.sweep + '/manifest.json' for f in files):
        parser.error('No completed archive manifest found at this revision')
    records = []
    for f in files:
        relative = PurePosixPath(f.rfilename)
        if relative.is_absolute() or '..' in relative.parts or f.size is None:
            raise ValueError('Unsafe or unsized archive file')
        paths = get_local_download_paths(args.local_dir, f.rfilename)
        etag = f.lfs.sha256 if f.lfs else f.blob_id
        if not etag:
            raise ValueError('Missing file identity: ' + f.rfilename)
        records.append((str(paths.file_path), str(paths.incomplete_path(etag)), f.size))
    total = sum(row[2] for row in records)
    print(f'{len(files)} files; {total / 1024**3:.2f} GiB; revision {info.sha}', flush=True)
    print('Keep this folder/cache. Ctrl+C stops safely; rerun the same command to resume.', flush=True)
    process = multiprocessing.get_context('spawn').Process(
        target=download, args=(args.repo, info.sha, str(args.local_dir),
                               [f.rfilename for f in files], args.workers))
    process.start()
    interrupted = False
    try:
        with tqdm(total=total, initial=retained_bytes(records), desc='Total download',
                  unit='B', unit_scale=True, unit_divisor=1024,
                  mininterval=0.5) as bar:
            changed = time.monotonic()
            while process.is_alive():
                count = retained_bytes(records)
                if count != bar.n:
                    bar.update(count - bar.n)
                    changed = time.monotonic()
                bar.set_postfix_str('waiting for network' if time.monotonic() - changed > 30 else '')
                time.sleep(0.5)
            process.join()
            bar.update(retained_bytes(records) - bar.n)
    except KeyboardInterrupt:
        interrupted = True
        process.terminate()
        print('\nStopped. Downloaded files and resumable partial files are retained.', flush=True)
    finally:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
    if interrupted:
        return 130
    if process.exitcode:
        print('Download interrupted or failed. Rerun this command to resume.', flush=True)
        return 1
    print('Download complete. You can now run data_tools.restore_sweep_archive.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
