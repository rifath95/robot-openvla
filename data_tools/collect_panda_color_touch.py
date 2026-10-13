"""First ten paired red/blue touch demonstrations for user review."""
import json
from pathlib import Path
import numpy as np
from data_tools.color_touch_demonstrations import record_episode, replay_episode
from data_tools.dataset_paths import DATASETS_ROOT, write_trajectory_index
from data_tools.panda_demonstrations import write_json

DATASET_ROOT = DATASETS_ROOT / 'panda_color_touch'
# Predeclared scenes: each is executed once per target instruction.
PAIRS = [((.43, -.12), (.64, .12)), ((.64, -.12), (.43, .12)),
         ((.46, .07), (.63, -.09)), ((.58, .13), (.44, -.08)),
         ((.66, .035), (.43, -.14))]


def collect():
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    counts = {'train': 0, 'validation': 0, 'test': 0}
    for pair, (red, blue) in enumerate(PAIRS):
        split = 'train' if pair < 3 else 'validation' if pair == 3 else 'test'
        first_image = None
        first_state = None
        for color in ('red', 'blue'):
            counts[split] += 1
            identifier = f'{split}_{counts[split]:03d}'
            folder = DATASET_ROOT / split / identifier
            spec = dict(id=identifier, split=split, pair_id=pair,
                        target_color=color, cube_positions={'red': list(red), 'blue': list(blue)})
            print(f'Collecting {identifier}: touch the {color} cube.', flush=True)
            if folder.exists():
                raise ValueError('Destination exists; reviewed episodes are never overwritten')
            metadata = record_episode(folder, spec)
            replay = replay_episode(folder)
            image = (folder / 'step_0001/before.png').read_bytes()
            with np.load(folder / 'initial_state.npz') as state:
                qpos = state['qpos'].copy()
            if first_image is None:
                first_image, first_state = image, qpos
            elif image != first_image or not np.array_equal(qpos, first_state):
                raise RuntimeError('Paired instruction trials do not have identical starting scenes')
            results.append(dict(id=identifier, split=split, episode=str(folder.resolve()),
                pair_id=pair, target_color=color, cube_positions=spec['cube_positions'],
                instruction=f'touch the {color} cube.', transitions=metadata['transition_count'],
                episode_sha256=replay['episode_sha256'], passed=replay['passed']))
            write_json(DATASET_ROOT / 'manifest.json', dict(episodes=results,
                split_counts={s: sum(r['split']==s for r in results) for s in counts},
                ready_for_finetuning=False))
            write_trajectory_index(DATASET_ROOT, results)
    write_json(DATASET_ROOT / 'collection_report.json', dict(
        phase='awaiting_user_review', successful_trajectories=len(results),
        paired_initial_images_and_states_identical=True,
        split_by_scene_pair=True, results=results))
    return results


if __name__ == '__main__':
    collect()
