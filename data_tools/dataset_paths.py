"""Default locations for locally generated fine-tuning data."""

from common.paths import PROJECT_ROOT
from pathlib import Path
import re
from common.replay_links import replay_url

DATASETS_ROOT = PROJECT_ROOT / 'datasets'
PANDA_DATASET_ROOT = DATASETS_ROOT / 'panda_pick_place'


def write_trajectory_index(folder, results):
    folder = Path(folder).resolve()
    lines = ['# Watch collected trajectories', '',
             'Click ▶ Replay for recorded motion, or ▶ Action Replay to execute saved actions through physics. '
             'Keep the local launcher running: `.venv/bin/python -m analysis.replay_launcher`. '
             'Playback starts automatically. Space pauses/resumes; Esc or closing the window exits. '
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
            episode.relative_to(folder)
            pose=replay_url(episode);action=replay_url(episode,'actions')
            pose_link=f'[▶ Replay]({pose})' if pose else '—'
            action_link=(f'[▶ Action Replay]({action})' if action and (episode/'transitions.jsonl').is_file() else '—')
            lines.append(f"| {row['id']} | {pose_link} | {action_link} |")
    (folder / 'WATCH_TRAJECTORIES.md').write_text('\n'.join(lines) + '\n')
