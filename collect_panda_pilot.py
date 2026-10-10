"""Collect a small predeclared pick/place pilot; no cloud or fine-tuning."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from dataset_paths import PANDA_DATASET_ROOT, write_trajectory_index

ROOT = Path(__file__).resolve().parent


def pilot_scenarios():
    # Split by entire starting scene, never by neighboring frames of one episode.
    specifications = [
        ('train', .50, -.02, .10), ('train', .52, .02, .10),
        ('train', .54, -.03, .12), ('train', .56, .00, .08),
        ('train', .58, .03, .10), ('train', .60, -.01, .12),
        ('validation', .51, .01, .11), ('validation', .57, -.02, .09),
        ('test', .53, .00, .13), ('test', .59, -.03, .11),
    ]
    return [dict(id=f'{split}_{index:02d}', split=split, cube_xy=[x, y],
                 destination_offset_xy=[0, distance],
                 instruction=f'pick up the red cube and place it {distance*100:g} cm in the positive world Y direction')
            for index, (split, x, y, distance) in enumerate(specifications, 1)]


def collect(folder):
    from panda_demonstrations import record_episode, replay_episode
    from prepare_panda_dataset import prepare_dataset
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    plan = pilot_scenarios()
    (folder / 'collection_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    results = []
    train = []
    def save():
        report = dict(stage='Phase 3 pilot collection', planned_episodes=len(plan), results=results,
                      successful_episodes=sum(row['passed'] for row in results),
                      ready_for_finetuning=False,
                      note='Pilot verifies varied collection and episode-level splits; it is not a sufficient training corpus. Validation/test episodes must not enter training or normalization.')
        (folder / 'collection_report.json').write_text(json.dumps(report, indent=2) + '\n')
        write_trajectory_index(folder, results)
    save()
    for spec in plan:
        episode = folder / spec['split'] / spec['id']
        try:
            print(f"Collecting {spec['id']}: cube {spec['cube_xy']}", flush=True)
            record_episode(episode, cube_xy=spec['cube_xy'],
                           destination_offset=spec['destination_offset_xy'],
                           instruction=spec['instruction'], scenario=spec)
            replay = replay_episode(episode)
            results.append(dict(id=spec['id'], split=spec['split'], episode=str(episode.resolve()),
                                passed=True, transitions=replay['replayed_transitions'],
                                placement_error_metres=replay['placement_xy_error_metres'],
                                lift_metres=replay['maximum_cube_lift_metres']))
            if spec['split'] == 'train':
                train.append(episode)
        except (ValueError, RuntimeError) as exc:
            results.append(dict(id=spec['id'], split=spec['split'], passed=False, error=str(exc)))
        save()
    if train:
        prepare_dataset(train, folder / 'train_staging')
    print(f'Collection summary: {folder / "collection_report.json"}', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PANDA_DATASET_ROOT /
                        datetime.now(timezone.utc).strftime('panda_pilot_%Y%m%d_%H%M%S'))
    args = parser.parse_args()
    collect(args.output_dir)


if __name__ == '__main__':
    main()
