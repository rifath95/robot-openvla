"""Expand paired red/blue touch demonstrations without replacing reviewed data."""

from common.output_layout import output_path
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from data_tools.color_touch_demonstrations import record_episode, replay_episode
from data_tools.dataset_paths import DATASETS_ROOT, write_trajectory_index
from data_tools.panda_demonstrations import write_json
from data_tools.prepare_panda_dataset import fingerprint
from common.paths import PROJECT_ROOT

DATASET_ROOT = DATASETS_ROOT / 'panda_color_touch'
# Predeclared scenes: each is executed once per target instruction.
PAIRS = [((.43, -.12), (.64, .12)), ((.64, -.12), (.43, .12)),
         ((.46, .07), (.63, -.09)), ((.58, .13), (.44, -.08)),
         ((.66, .035), (.43, -.14))]
SEED = 20261013


def pair_spec(pair):
    if pair < 0:
        raise ValueError('Pair index must be nonnegative')
    if pair < len(PAIRS):
        red, blue = PAIRS[pair]
        split = 'train' if pair < 3 else 'validation' if pair == 3 else 'test'
    else:
        split = ['train']*4 + ['validation', 'test']
        split = split[(pair-len(PAIRS)) % 6]
        rng = np.random.default_rng([SEED, pair])
        # Reject overlapping/nearby placements before physics collection. Failed
        # demonstrations never trigger new position draws or easier replacements.
        for _ in range(10000):
            red = rng.uniform([.40, -.16], [.68, .16])
            blue = rng.uniform([.40, -.16], [.68, .16])
            if np.linalg.norm(red-blue) >= .12:
                break
        else:
            raise RuntimeError('Could not declare separated cube positions')
    return dict(pair_id=pair, split=split,
                cube_positions={'red': list(red), 'blue': list(blue)})


def collect(limit=10):
    if type(limit) is not int or limit < 2 or limit % 2:
        raise ValueError('Choose a positive even number of trajectories')
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    path = DATASET_ROOT / 'manifest.json'
    results = json.loads(path.read_text())['episodes'] if path.exists() else []
    if len(results) > limit:
        raise ValueError('Requested limit is smaller than the preserved dataset')
    for result in results:
        folder = Path(result['episode'])
        if not result['passed'] or fingerprint(folder) != result['episode_sha256']:
            raise ValueError('Previously reviewed demonstration changed')
    write_json(DATASET_ROOT / 'collection_plan.json', dict(seed=SEED,
        requested_trajectories=limit, paired_scenes=[pair_spec(i) for i in range(limit//2)],
        fixed_initial_arm=True, fixed_cube_orientation=True, minimum_cube_separation_metres=.12,
        failed_demonstrations='Stop and retain diagnostics; never replace a failed position'))
    counts = {'train': 0, 'validation': 0, 'test': 0}
    for pair in range(limit//2):
        plan = pair_spec(pair)
        split = plan['split']
        first_image = None
        first_state = None
        for color in ('red', 'blue'):
            counts[split] += 1
            identifier = f'{split}_{counts[split]:03d}'
            folder = DATASET_ROOT / split / identifier
            spec = dict(id=identifier, split=split, pair_id=pair,
                        target_color=color, cube_positions=plan['cube_positions'])
            print(f'Collecting {identifier}: touch the {color} cube.', flush=True)
            prior = next((r for r in results if r['split']==split and r['id']==identifier), None)
            if prior:
                if any(prior[k]!=spec[k] for k in ('pair_id','target_color','cube_positions')):
                    raise ValueError('Previously reviewed collection scenario differs from plan')
                metadata = json.loads((folder/'episode.json').read_text())
                replay = json.loads((folder/'replay.json').read_text())
            else:
                if folder.exists():
                    raise ValueError('Destination exists; reviewed episodes are never overwritten')
                staging_root = output_path('color_touch_collection_checks')
                staging_root.mkdir(parents=True, exist_ok=True)
                temporary = Path(tempfile.mkdtemp(prefix=identifier+'-', dir=staging_root))/'episode'
                try:
                    metadata = record_episode(temporary, spec)
                    replay = replay_episode(temporary)
                except Exception as exc:
                    write_json(DATASET_ROOT/'collection_failure.json', dict(
                        scenario=spec, error=str(exc), diagnostics=str(temporary)))
                    raise
                folder.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(temporary, folder)
            image = (folder / 'step_0001/before.png').read_bytes()
            with np.load(folder / 'initial_state.npz') as state:
                qpos = state['qpos'].copy()
            if first_image is None:
                first_image, first_state = image, qpos
            elif image != first_image or not np.array_equal(qpos, first_state):
                raise RuntimeError('Paired instruction trials do not have identical starting scenes')
            if not prior:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=10)
    collect(parser.parse_args().limit)
