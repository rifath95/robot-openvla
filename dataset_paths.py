"""Default locations for locally generated fine-tuning data."""
from pathlib import Path

DATASETS_ROOT = Path(__file__).resolve().parent / 'datasets'
PANDA_DATASET_ROOT = DATASETS_ROOT / 'panda_pick_place'


def write_trajectory_index(folder, results):
    folder = Path(folder).resolve()
    lines = ['# Watch collected trajectories', '',
             'Open an episode’s replay.py using the repository .venv/bin/python interpreter. '
             'Space starts/pauses/resumes; Esc or closing the window exits. '
             'Endpoint-only recordings use approximate interpolated motion.', '',
             '| Episode | Split | Playback script |', '| --- | --- | --- |']
    for row in results:
        if not row.get('passed'):
            continue
        episode = Path(row['episode']).resolve()
        relative = (episode / 'replay.py').relative_to(folder)
        lines.append(f"| {row['id']} | {row['split']} | [replay.py]({relative.as_posix()}) |")
    (folder / 'WATCH_TRAJECTORIES.md').write_text('\n'.join(lines) + '\n')
