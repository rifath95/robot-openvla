"""Default locations for locally generated fine-tuning data."""

from common.paths import PROJECT_ROOT
from pathlib import Path
import re

DATASETS_ROOT = PROJECT_ROOT / 'datasets'
PANDA_DATASET_ROOT = DATASETS_ROOT / 'panda_pick_place'


def write_trajectory_index(folder, results):
    folder = Path(folder).resolve()
    lines = ['# Watch collected trajectories', '',
             'Open replay.py for recorded pose playback, or replay_actions.py to execute saved actions through physics. '
             'Use the repository .venv/bin/python interpreter. '
             'Space starts/pauses/resumes; Esc or closing the window exits. '
             'Endpoint-only recordings use approximate interpolated motion.', '',
             'All accepted trajectories are listed below. This index is refreshed as more are collected.']
    accepted = [row for row in results if row.get('passed')]
    def number_order(row):
        return re.sub(r'\d+', lambda match: match.group().zfill(12), row['id'])
    for split, title in (('train','Training'), ('validation','Validation'), ('test','Test')):
        rows = sorted((row for row in accepted if row['split'] == split), key=number_order)
        lines.extend(['', f'## {title}', '', f'{len(rows)} trajectories.', '',
                      '| Episode | Pose replay | Action replay |', '| --- | --- | --- |'])
        for row in rows:
            episode = Path(row['episode']).resolve()
            relative = (episode / 'replay.py').relative_to(folder)
            actions=episode/'replay_actions.py'
            action_link=(f'[replay_actions.py]({actions.relative_to(folder).as_posix()})'
                         if actions.is_file() else '—')
            lines.append(f"| {row['id']} | [replay.py]({relative.as_posix()}) | {action_link} |")
    (folder / 'WATCH_TRAJECTORIES.md').write_text('\n'.join(lines) + '\n')
