"""Group local sweep recordings under trials/, preserving reports and old paths."""
import argparse
import json
from pathlib import Path
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from common.paths import OUTPUTS_ROOT
from common.sweep_layout import validate_trial_directory


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def rewrite_links(text, names):
    def path(value):
        parts = value.split('/')
        result = []
        for part in parts:
            if part in names and (not result or result[-1] != 'trials'):
                result.append('trials')
            result.append(part)
        return '/'.join(result)

    def link(match):
        url = urlsplit(match[1])
        query = [(key, path(value) if key in ('run', 'trial') else value)
                 for key, value in parse_qsl(url.query, keep_blank_values=True)]
        return '](' + urlunsplit((url.scheme, url.netloc, path(url.path),
                                  urlencode(query) if url.query else '', url.fragment)) + ')'
    return re.sub(r'\]\(([^\n)]+)\)', link, text)


def organize(folder, *, apply=False):
    """Resume-safe rename with a saved original index; never overwrite a recording."""
    folder = Path(folder)
    if folder.is_symlink():
        raise ValueError('Refusing a symlink sweep')
    payload = json.loads((folder / 'results.json').read_text())
    journal = folder / '.trial_layout_migration.json'
    previous = json.loads(journal.read_text()) if journal.exists() else None
    names = set(previous['trial_names']) if previous else set()
    for row in payload['trials']:
        relative = validate_trial_directory(row['directory'])
        names.add(relative.name)
        source = folder / relative.name
        target = folder / 'trials' / relative.name
        if not source.is_dir() and not target.is_dir():
            raise ValueError(f'Missing recorded trial: {relative}')
    # Include interrupted/unindexed trial recordings, but leave unrelated assets alone.
    for child in folder.iterdir():
        if child.name.startswith(('checkpoint_', 'base_model')) and child.is_dir():
            names.add(child.name)
    if (folder / 'trials').is_symlink():
        raise ValueError('Refusing a symlink trials directory')
    moves = []
    for name in sorted(names):
        validate_trial_directory(name)
        source, target = folder / name, folder / 'trials' / name
        if source.is_symlink() or target.is_symlink():
            raise ValueError('Refusing a symlink trial')
        if source.exists() and target.exists():
            raise ValueError(f'Trial destination already exists: {target}')
        if source.exists():
            moves.append((source, target))
    from cloud.sweep_plan import validate_saved
    prepared=json.loads(json.dumps(payload))
    if 'task' not in prepared['plan'] and prepared['plan'].get('instruction')=='touch the red cube.':
        prepared['plan']['task']='touch'
    prepared['plan']['trial_directory']='trials'
    for row in prepared['trials']:
        row['directory']='trials/'+Path(row['directory']).name
    validate_saved(prepared)
    if not apply:
        return len(moves)
    if not previous:
        write_json(journal, dict(trial_names=sorted(names), original_results=payload, complete=False))
    (folder / 'trials').mkdir(exist_ok=True)
    for source, target in moves:
        before = source.stat()
        source.rename(target)
        after = target.stat()
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise RuntimeError('Recording identity changed during rename')
    # Only derived Markdown links change; retain custom tables, prose and original metadata.
    reports = list(folder.glob('*.md'))
    reports += [folder / 'trials' / name / 'RUN_REPORT.md' for name in names]
    for report in reports:
        if report.is_symlink():
            raise ValueError('Refusing a symlink report')
        if report.is_file():
            text = report.read_text()
            updated = rewrite_links(text, names)
            if updated != text:
                temporary = report.with_suffix('.md.tmp')
                temporary.write_text(updated)
                temporary.replace(report)
    if 'task' not in payload['plan'] and payload['plan'].get('instruction')=='touch the red cube.':
        payload['plan']['task']='touch'
    payload['plan']['trial_directory'] = 'trials'
    for row in payload['trials']:
        row['directory'] = 'trials/' + Path(row['directory']).name
    from cloud.sweep_plan import validate_saved
    validate_saved(payload)
    write_json(folder / 'results.json', payload)
    original = previous['original_results'] if previous else json.loads(journal.read_text())['original_results']
    write_json(journal, dict(trial_names=sorted(names), original_results=original, complete=True))
    return len(moves)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', nargs='*', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    folders = args.folders or sorted((OUTPUTS_ROOT / 'sweeps').glob('scripted_*'))
    for folder in folders:
        if (folder / 'results.json').is_file():
            count = organize(folder, apply=args.apply)
            print(f'{folder.name}: {count} trial folders {"moved" if args.apply else "to move"}', flush=True)


if __name__ == '__main__':
    main()
