"""Download a pinned model on local disk, then export regular files to persistent storage."""

import argparse
import json
from pathlib import Path
import shutil

MODEL_ID = "openvla/openvla-7b"
REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).with_name(".cache-openvla-server") / "download")
    parser.add_argument("--export-dir", type=Path, required=True)
    args = parser.parse_args()
    cache = args.cache_dir.resolve()
    destination = args.export_dir.resolve()
    if cache == destination or cache in destination.parents or destination in cache.parents:
        parser.error("Download cache and export directory must be separate")
    marker = destination / "export_complete.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        if record.get("revision") != REVISION or record.get("model") != MODEL_ID:
            raise RuntimeError("Destination contains a different model export; choose another directory")
        if all((destination / name).is_file() and (destination / name).stat().st_size == size
               for name, size in record["files"].items()):
            print(f"Completed export already exists: {destination}")
            return
        marker.unlink()
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(
        MODEL_ID, revision=REVISION, cache_dir=str(cache),
        allow_patterns=["*.json", "*.py", "*.safetensors", "*.model"], max_workers=2,
    ))
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for source in sorted(snapshot.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(snapshot)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)  # Dereference HF cache links; no locks/renames on Global storage.
        if target.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"Incomplete copy: {relative}")
        files[str(relative)] = target.stat().st_size
        print(f"Exported {relative}", flush=True)
    marker.write_text(json.dumps({"model": MODEL_ID, "revision": REVISION, "files": files}, indent=2) + "\n")
    print(f"Completed export: {destination}. Download cache remains on local disk: {cache}")
    print("Do not load the export until this command completes. Only one pod should write it.")


if __name__ == "__main__":
    main()
