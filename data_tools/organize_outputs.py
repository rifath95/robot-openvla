"""Move flat local output entries into categories without copying or deleting data."""
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone
from common.paths import OUTPUTS_ROOT
from common.output_layout import OUTPUT_CATEGORIES,output_path


def migration_plan():
    moves=[]
    for source in sorted(OUTPUTS_ROOT.iterdir()):
        if source.name.startswith('.') or source.name in (*OUTPUT_CATEGORIES,'README.md'):continue
        if source.is_symlink():raise ValueError('Inspect an output symlink before moving: '+str(source))
        destination=output_path(source.name)
        if not destination.resolve().is_relative_to(OUTPUTS_ROOT.resolve()):raise ValueError('Destination escapes outputs: '+str(destination))
        if destination.exists():raise ValueError('Destination already exists: '+str(destination))
        moves.append((source,destination))
    return moves


def organize(apply=False):
    moves=migration_plan()
    for source,destination in moves:print(f'{source.relative_to(OUTPUTS_ROOT)} -> {destination.relative_to(OUTPUTS_ROOT)}')
    if not apply:return moves
    receipt=OUTPUTS_ROOT/'diagnostics/organization/output_layout_migration.json'
    receipt.parent.mkdir(parents=True,exist_ok=True)
    history=json.loads(receipt.read_text()) if receipt.exists() else []
    for source,destination in moves:
        stat=source.stat();destination.parent.mkdir(parents=True,exist_ok=True)
        source.rename(destination)
        moved=destination.stat()
        if (stat.st_dev,stat.st_ino)!=(moved.st_dev,moved.st_ino):raise RuntimeError('Move identity changed')
        history.append(dict(old=str(source),new=str(destination),moved_at_utc=datetime.now(timezone.utc).isoformat()))
        temporary=receipt.with_suffix('.tmp');temporary.write_text(json.dumps(history,indent=2)+'\n');temporary.replace(receipt)
    print(f'Moved {len(moves)} entries. No data deleted. Receipt: {receipt}')
    return moves


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply',action='store_true')
    organize(p.parse_args().apply)

if __name__=='__main__':main()
