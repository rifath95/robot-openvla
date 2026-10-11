"""Collect 200 broadly varied, replay-verified demonstrations in resumable slots."""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import numpy as np

from data_tools.dataset_paths import PANDA_DATASET_ROOT, write_trajectory_index

SEED = 20261011
SPLITS = ['train'] * 14 + ['validation'] * 3 + ['test'] * 3


def combination_pools():
    # A combination is hand XY quadrant, cube XY quadrant, and initial yaw sign.
    cells = list(itertools.product(range(4), range(4), (-1, 1)))
    np.random.default_rng(SEED).shuffle(cells)
    return dict(train=cells[:22], validation=cells[22:27], test=cells[27:])


def scenario(slot, attempt):
    split = SPLITS[slot % 20]
    pools = combination_pools()
    rng = np.random.default_rng([SEED, slot, attempt])
    # Cycling guarantees each split's declared combinations are represented.
    rank = sum(SPLITS[i % 20] == split for i in range(slot))
    hand_cell, cube_cell, sign = pools[split][rank % len(pools[split])]
    def quadrant(cell, x_ranges, y_ranges):
        return [float(rng.uniform(*x_ranges[cell // 2])),
                float(rng.uniform(*y_ranges[cell % 2]))]
    start_xy = quadrant(hand_cell, ((.38, .51), (.53, .67)), ((-.22, -.03), (.03, .22)))
    cube_xy = quadrant(cube_cell, ((.46, .54), (.56, .64)), ((-.14, -.02), (.02, .14)))
    # Destination is independently sampled inside the reachable table region.
    for _ in range(1000):
        destination = rng.uniform([.46, -.15], [.64, .15])
        offset = np.round(destination - cube_xy, 3)
        if .06 <= np.linalg.norm(offset) <= .16:
            break
    else:
        raise ValueError('Could not sample a placement destination')
    return dict(id=f'{split}_broad_{slot + 1:03d}', split=split,
                slot=slot, attempt=attempt, combination=[hand_cell, cube_cell, sign],
                cube_xy=cube_xy, destination_offset_xy=offset.tolist(),
                initial_tool_position=[*start_xy, float(rng.uniform(.38, .59))],
                initial_yaw_radians=float(sign * rng.uniform(.45, 1.15)),
                grasp_yaw_radians=0.0,
                instruction=f'pick up the red cube and move it {offset[0]*100:g} cm along world X and {offset[1]*100:g} cm along world Y')


def coverage(results):
    if not results:
        return {}
    poses = np.array([row['starting_pose']['actual_position'] for row in results])
    cubes = np.array([row['initial_cube'][:2] for row in results])
    yaws = np.array([row['starting_pose']['requested_yaw_radians'] for row in results])
    return dict(hand_position_min_metres=poses.min(axis=0).tolist(),
                hand_position_max_metres=poses.max(axis=0).tolist(),
                hand_position_span_metres=np.ptp(poses, axis=0).tolist(),
                cube_xy_span_metres=np.ptp(cubes, axis=0).tolist(),
                initial_yaw_range_degrees=np.rad2deg([yaws.min(), yaws.max()]).tolist(),
                mean_initial_hand_cube_distance_metres=float(np.mean([
                    np.linalg.norm(poses[i] - row['initial_cube']) for i, row in enumerate(results)])),
                observed_combinations={split: sorted({tuple(row['combination']) for row in results
                                                     if row['split'] == split}) for split in set(SPLITS)})


def collect_slot(folder, slot, first_attempt, max_attempts):
    from data_tools.panda_demonstrations import record_episode, replay_episode
    folder = Path(folder)
    failures = []
    for attempt in range(first_attempt, max_attempts):
        spec = scenario(slot, attempt)
        episode = folder / spec['split'] / spec['id']
        # Interrupted attempts are preserved; a fresh candidate cannot overwrite them.
        if episode.exists():
            episode = folder / 'attempts' / f"{spec['id']}_attempt_{attempt:02d}"
        if episode.exists():
            continue
        print(f"Slot {slot+1}/200, attempt {attempt+1}: {spec}", flush=True)
        try:
            metadata = record_episode(episode, cube_xy=spec['cube_xy'],
                destination_offset=spec['destination_offset_xy'], instruction=spec['instruction'],
                scenario=spec, initial_tool_position=spec['initial_tool_position'],
                initial_yaw_radians=spec['initial_yaw_radians'], grasp_yaw_radians=0)
            replay = replay_episode(episode)
            result = dict(id=spec['id'], split=spec['split'], slot=slot, attempt=attempt,
                combination=spec['combination'], episode=str(episode), passed=True,
                starting_pose=metadata['starting_pose'], initial_cube=metadata['initial_cube'],
                transitions=replay['replayed_transitions'],
                lift_metres=replay['maximum_cube_lift_metres'],
                placement_error_metres=replay['placement_xy_error_metres'])
            return result, failures
        except (ValueError, RuntimeError) as exc:
            failures.append(dict(slot=slot, attempt=attempt, episode=str(episode),
                                 specification=spec, error=str(exc)))
    return None, failures

def collect(folder, limit=200, max_attempts=30, workers=1):
    from data_tools.panda_demonstrations import write_json
    from data_tools.prepare_panda_dataset import fingerprint, prepare_dataset
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    plan = dict(seed=SEED, target=200, split_counts=dict(train=140, validation=30, test=30),
                combinations=combination_pools(),
                note='Entire hand quadrant / cube quadrant / yaw-sign combinations are withheld. '
                     'Failed attempts remain outside the accepted manifest. Pilot counts toward 200.')
    if (folder / 'collection_plan.json').exists():
        if json.loads((folder / 'collection_plan.json').read_text()) != json.loads(json.dumps(plan)):
            raise ValueError('Existing plan differs; use a new output directory')
    else:
        write_json(folder / 'collection_plan.json', plan)
    report_path = folder / 'collection_report.json'
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    results, failures = previous.get('results', []), previous.get('failures', [])
    for row in results:
        episode = Path(row['episode'])
        replay = json.loads((episode / 'replay.json').read_text())
        if not replay['passed'] or replay['episode_sha256'] != fingerprint(episode):
            raise ValueError(f'Accepted episode changed: {episode}')
    def save(phase):
        write_json(report_path, dict(phase=phase, target=200, successful_episodes=len(results),
                                    results=results, failures=failures, coverage=coverage(results),
                                    ready_for_finetuning=False))
        write_json(folder / 'manifest.json', dict(episodes=results,
                   split_counts={s: sum(r['split'] == s for r in results) for s in set(SPLITS)},
                   ready_for_finetuning=False))
        write_trajectory_index(folder, results)
    save('collecting')
    jobs = []
    for slot in range(limit):
        if any(row['slot'] == slot for row in results):
            continue
        prior = [row['attempt'] for row in failures if row['slot'] == slot]
        first_attempt = max(prior, default=-1) + 1
        jobs.append((folder, slot, first_attempt, max_attempts))
    from concurrent.futures import ProcessPoolExecutor, as_completed
    def accept(outcome):
        result, rejected = outcome
        failures.extend(rejected)
        if result is None:
            save('blocked_slot')
            raise RuntimeError('A slot exhausted its attempts; inspect failures')
        results.append(result)
        results.sort(key=lambda row: row['slot'])
        save('collecting')
    if workers == 1:
        for job in jobs:
            accept(collect_slot(*job))
    else:
        # Independent process-local MuJoCo/graphics contexts; only the parent writes manifests.
        import multiprocessing
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(collect_slot, *job) for job in jobs]
            for future in as_completed(futures):
                accept(future.result())
    if limit >= 20:
        spans = coverage(results)['hand_position_span_metres']
        yaw_range = coverage(results)['initial_yaw_range_degrees']
        if min(spans[:2]) < .20 or spans[2] < .12 or yaw_range[1] - yaw_range[0] < 90:
            save('insufficient_coverage')
            raise RuntimeError('Actual starting-pose coverage is too narrow; inspect before expanding')
    if limit == 200:
        # Keep the new broad combinations separate from the narrow legacy corpus.
        # Training statistics must never use validation/test episodes.
        if not (folder / 'train_staging').exists():
            prepare_dataset([Path(r['episode']) for r in results if r['split'] == 'train'],
                            folder / 'train_staging')
        legacy_path = PANDA_DATASET_ROOT / 'panda_expansion_20261010_104720/combined_manifest.json'
        if legacy_path.exists():
            legacy = json.loads(legacy_path.read_text())['episodes']
            if len(legacy) != 50 or not all(row['passed'] for row in legacy):
                raise ValueError('Expected the original verified 50-trajectory corpus')
            combined = [dict(row, cohort='legacy') for row in legacy] + [
                dict(row, cohort='broad') for row in results]
            if len({row['id'] for row in combined}) != len(combined):
                raise ValueError('Episode IDs collide between cohorts')
            if not (folder / 'combined_train_staging').exists():
                prepare_dataset([Path(r['episode']) for r in combined if r['split'] == 'train'],
                                folder / 'combined_train_staging')
            write_json(folder / 'combined_manifest.json', dict(episodes=combined,
                split_counts={s: sum(r['split'] == s for r in combined) for s in set(SPLITS)},
                note='Legacy and broad cohorts remain distinguishable; keep held-out splits fixed.',
                ready_for_finetuning=False))
            write_trajectory_index(PANDA_DATASET_ROOT, combined)
    save('completed' if len(results) == 200 else 'pilot_verified')
    print(f'Collection report: {report_path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PANDA_DATASET_ROOT /
                        datetime.now(timezone.utc).strftime('panda_broad_%Y%m%d_%H%M%S'))
    parser.add_argument('--limit', type=int, choices=(20, 200), default=200)
    parser.add_argument('--workers', type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    collect(args.output_dir, args.limit, workers=args.workers)


if __name__ == '__main__':
    main()
