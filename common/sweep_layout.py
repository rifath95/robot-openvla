"""Versioned trial paths; old saved plans keep their original identities."""
from pathlib import Path, PurePosixPath


def trial_directory(plan, name):
    layout = plan.get('trial_directory', '')
    if layout not in ('', 'trials'):
        raise ValueError('Unknown sweep trial directory layout')
    relative = PurePosixPath(name)
    if len(relative.parts) != 1 or relative.name != name or name in ('', '.', '..'):
        raise ValueError('Invalid trial name')
    return f'trials/{name}' if layout else name


def validate_trial_directory(name):
    relative = PurePosixPath(name)
    if (relative.is_absolute() or '..' in relative.parts or relative.as_posix() != name
            or len(relative.parts) not in (1, 2)
            or (len(relative.parts) == 2 and relative.parts[0] != 'trials')
            or name in ('', '.', 'trials')):
        raise ValueError('Invalid saved trial directory')
    return relative


def prepare_trial_restore(folder, specs):
    """Move previously extracted legacy trials before resuming into nested paths."""
    folder = Path(folder)
    if (folder / 'trials').is_symlink():
        raise ValueError('Unsafe archive destination: trials symlink')
    moves = []
    for relative in specs:
        name = validate_trial_directory(relative).name
        source, target = folder / name, folder / 'trials' / name
        if source.is_symlink() or target.is_symlink():
            raise ValueError('Unsafe archive destination: trial symlink')
        if source.exists() and target.exists():
            raise ValueError('Conflicting legacy and nested trial recordings')
        if source.exists():
            if not source.is_dir():
                raise ValueError('Unsafe archive destination: trial is not a directory')
            moves.append((source, target))
    if moves:
        (folder / 'trials').mkdir(exist_ok=True)
        for source, target in moves:
            source.rename(target)
