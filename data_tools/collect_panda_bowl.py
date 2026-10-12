"""Collect numbered bowl trajectories; the first ten belong to the final dataset."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from common.paths import PROJECT_ROOT
from data_tools.dataset_paths import DATASETS_ROOT, write_trajectory_index
from data_tools.panda_demonstrations import write_json
from data_tools.prepare_panda_dataset import fingerprint
from simulation.bowl_scene import INSTRUCTION, CAMERA_NAME

SEED = 20261012
SPLITS = ['train']*4+['validation','test']
DATASET_ROOT = DATASETS_ROOT/'panda_pick_bowl'


def scenario(slot, attempt):
    split = SPLITS[slot % 6]
    rank = sum(SPLITS[i % 6] == split for i in range(slot))+1
    rng = np.random.default_rng([SEED, slot, attempt])
    # Include fully closed/open and intermediate starts in every split.
    mode = (rank-1) % 4
    gripper = 0 if mode == 0 else 1 if mode == 1 else float(rng.uniform(.1,.9))
    return dict(id=f'{split}_{rank:03d}', split=split, slot=slot, attempt=attempt,
                random_seed=[SEED,slot,attempt], initial_gripper=gripper,
                instruction=INSTRUCTION)


def coverage(results):
    if not results:
        return {}
    values = np.array([[*r['starting_pose']['actual_position'],
                        r['starting_pose']['requested_roll_radians'],
                        r['starting_pose']['requested_pitch_radians'],
                        r['starting_pose']['requested_yaw_radians'],
                        r['starting_pose']['requested_gripper_open_fraction']] for r in results])
    return dict(components=['x','y','z','roll','pitch','yaw','gripper'],
                minimum=values.min(axis=0).tolist(), maximum=values.max(axis=0).tolist(),
                spans=np.ptp(values,axis=0).tolist(),
                cube_xy_span_metres=np.ptp([r['initial_cube'][:2] for r in results],axis=0).tolist(),
                bowl_xy_span_metres=np.ptp([r['bowl_position'][:2] for r in results],axis=0).tolist())


def collect(folder=DATASET_ROOT, limit=10, max_attempts=30):
    from data_tools.bowl_demonstrations import record_episode, replay_episode
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    for split in set(SPLITS):
        (folder/split).mkdir(exist_ok=True)
    plan = dict(seed=SEED, target_successful_trajectories=300,
                target_split_counts=dict(train=200,validation=50,test=50),
                instruction=INSTRUCTION, camera=CAMERA_NAME,
                randomized_start_components=['x','y','z','roll','pitch','yaw','gripper'],
                sampling='Independent split/slot/attempt seeds; reachable, settled, collision-checked starts. '
                         'Cube and bowl vary over the reachable table interior; not every physical-table point is reachable.',
                first_ten='First ten retained in final folders; eight train, one validation, one test.',
                accepted_only='Only successful contact grasps and deterministic physics replays are accepted.')
    plan_path = folder/'collection_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Dataset plan changed; do not mix incompatible demonstrations')
    write_json(plan_path,plan)
    report_path = folder/'collection_report.json'
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    results, failures = previous.get('results',[]), previous.get('failures',[])
    for row in results:
        episode = Path(row['episode'])
        report = json.loads((episode/'replay.json').read_text())
        if not report['passed'] or report['episode_sha256'] != fingerprint(episode):
            raise ValueError(f'Previously accepted episode changed: {episode}')
    def save(phase):
        write_json(report_path,dict(phase=phase, successful_trajectories=len(results), target=300,
                                   results=results, failures=failures, coverage=coverage(results),
                                   ready_for_finetuning=False))
        write_json(folder/'manifest.json',dict(episodes=results,
                   split_counts={split:sum(row['split']==split for row in results) for split in set(SPLITS)},
                   instruction=INSTRUCTION, ready_for_finetuning=False))
        write_trajectory_index(folder,results)
    checks = PROJECT_ROOT/'outputs/bowl_collection_checks'
    checks.mkdir(parents=True,exist_ok=True)
    save('collecting')
    for slot in range(limit):
        if any(row['slot']==slot for row in results):
            continue
        start = max((r['attempt'] for r in failures if r['slot']==slot),default=-1)+1
        for attempt in range(start,max_attempts):
            spec = scenario(slot,attempt)
            destination = folder/spec['split']/spec['id']
            if destination.exists():
                raise ValueError(f'Unindexed episode already exists: {destination}')
            temporary = checks/f'{spec["id"]}_attempt_{attempt:03d}'
            if temporary.exists():
                temporary = checks/f'{spec["id"]}_attempt_{attempt:03d}_{datetime.now(timezone.utc):%H%M%S%f}'
            print(f'Slot {slot+1}/{limit}, attempt {attempt+1}: {spec["id"]}',flush=True)
            try:
                metadata = record_episode(temporary,spec)
                replay = replay_episode(temporary)
                shutil.move(str(temporary),str(destination))
                result = dict(id=spec['id'], split=spec['split'], slot=slot, attempt=attempt,
                              episode=str(destination), passed=True, transitions=replay['replayed_transitions'],
                              starting_pose=metadata['starting_pose'], initial_cube=metadata['initial_cube'],
                              bowl_position=metadata['scenario']['scene']['bowl_position'],
                              lift_metres=replay['maximum_cube_lift_metres'],
                              bowl_center_distance_metres=replay['bowl_center_distance_metres'])
                results.append(result)
                results.sort(key=lambda row:row['slot'])
                save('collecting')
                print(f'Accepted {spec["id"]}: {result["transitions"]} steps',flush=True)
                break
            except (ValueError,RuntimeError) as exc:
                failures.append(dict(slot=slot, attempt=attempt, specification=spec,
                                     diagnostic_directory=str(temporary),error=str(exc)))
                save('retrying')
                print(f'Rejected attempt: {exc}',flush=True)
        else:
            save('attempts_exhausted')
            raise RuntimeError(f'Slot {slot+1} exhausted attempts; inspect diagnostics before retrying')
    save('completed' if len(results)==300 else 'awaiting_user_review')
    print(f'Created {len(results)} successful trajectories. Review {folder/"WATCH_TRAJECTORIES.md"}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DATASET_ROOT)
    parser.add_argument('--limit',type=int,choices=(10,300),default=10)
    parser.add_argument('--max-attempts',type=int,default=30)
    args = parser.parse_args()
    collect(args.output_dir,args.limit,args.max_attempts)


if __name__ == '__main__':
    main()
