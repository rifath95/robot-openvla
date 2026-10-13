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


def scenario(slot, attempt, coverage_target=None,startup_policy='legacy'):
    split = SPLITS[slot % 6]
    rank = sum(SPLITS[i % 6] == split for i in range(slot))+1
    rng = np.random.default_rng([SEED, slot, attempt])
    # Include fully closed/open and intermediate starts in every split.
    mode = (rank-1) % 4
    gripper = 0 if mode == 0 else 1 if mode == 1 else float(rng.uniform(.1,.9))
    result=dict(id=f'{split}_{rank:03d}', split=split, slot=slot, attempt=attempt,
                random_seed=[SEED,slot,attempt], initial_gripper=gripper,
                random_cube_yaw=slot>=10,
                instruction=INSTRUCTION)
    if coverage_target:result['coverage_target']=coverage_target
    if startup_policy not in ('legacy','state_dependent'):
        raise ValueError('Unknown demonstration startup policy')
    if startup_policy!='legacy':
        result.update(startup_policy=startup_policy,random_cube_yaw=True)
    return result


def coverage(results):
    if not results:
        return {}
    values = np.array([[*r['starting_pose']['actual_position'],
                        r['starting_pose']['requested_roll_radians'],
                        r['starting_pose']['requested_pitch_radians'],
                        r['starting_pose']['requested_yaw_radians'],
                        r['starting_pose']['requested_gripper_open_fraction']] for r in results])
    cube_yaws=[r.get('cube_yaw_radians',0) for r in results]
    return dict(components=['x','y','z','roll','pitch','yaw','gripper'],
                cube_yaw_range_degrees=np.rad2deg([min(cube_yaws),max(cube_yaws)]).tolist(),
                minimum=values.min(axis=0).tolist(), maximum=values.max(axis=0).tolist(),
                spans=np.ptp(values,axis=0).tolist(),
                cube_xy_span_metres=np.ptp([r['initial_cube'][:2] for r in results],axis=0).tolist(),
                bowl_xy_span_metres=np.ptp([r['bowl_position'][:2] for r in results],axis=0).tolist())


def collect_slot(folder,slot,first_attempt,max_attempts,target,startup_policy='legacy'):
    from data_tools.bowl_demonstrations import record_episode, replay_episode
    folder=Path(folder)
    checks=PROJECT_ROOT/'outputs/bowl_collection_checks'
    rejected=[]
    for attempt in range(first_attempt,max_attempts):
        spec=scenario(slot,attempt,target,startup_policy)
        destination=folder/spec['split']/spec['id']
        if destination.exists():raise ValueError(f'Unindexed episode already exists: {destination}')
        temporary=checks/f'{spec["id"]}_attempt_{attempt:03d}'
        if temporary.exists():temporary=checks/f'{temporary.name}_{datetime.now(timezone.utc):%H%M%S%f}'
        print(f'Slot {slot+1}/300, attempt {attempt+1}: {spec["id"]}',flush=True)
        try:
            metadata=record_episode(temporary,spec)
            replay=replay_episode(temporary)
            result=dict(id=spec['id'],split=spec['split'],slot=slot,attempt=attempt,
                        episode=str(destination),passed=True,transitions=replay['replayed_transitions'],
                        starting_pose=metadata['starting_pose'],initial_cube=metadata['initial_cube'],
                        bowl_position=metadata['scenario']['scene']['bowl_position'],
                        cube_yaw_radians=metadata['scenario']['scene']['requested_cube_yaw_radians'],
                        lift_metres=replay['maximum_cube_lift_metres'],
                        bowl_center_distance_metres=replay['bowl_center_distance_metres'])
            # The parent publishes accepted episodes only after recording the outcome.
            return result,rejected,str(temporary)
        except (ValueError,RuntimeError) as exc:
            rejected.append(dict(slot=slot,attempt=attempt,specification=spec,
                                 diagnostic_directory=str(temporary),error=str(exc)))
            # Retain compact diagnostics; rejected render sequences are not training data.
            steps=sorted(temporary.glob('step_*'))
            if steps:
                for source,name in ((steps[0]/'before.png','failure_start.png'),
                                    (steps[-1]/'after.png','failure_last.png'),
                                    (steps[-1]/'before_state.npz','failure_last_state.npz')):
                    if source.is_file():shutil.copyfile(source,temporary/name)
                for step in steps:shutil.rmtree(step)
            print(f'Rejected {spec["id"]}: {exc}',flush=True)
    return None,rejected,None


def collect(folder=DATASET_ROOT, limit=10, max_attempts=30, workers=1,startup_policy='legacy'):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    for split in set(SPLITS):
        (folder/split).mkdir(exist_ok=True)
    plan = dict(seed=SEED, target_successful_trajectories=300,
                target_split_counts=dict(train=200,validation=50,test=50),
                instruction=INSTRUCTION, camera=CAMERA_NAME,
                randomized_start_components=['x','y','z','roll','pitch','yaw','gripper'],
                cube_orientation=dict(initial_fixed_orientation_slots=10,
                                      subsequent_yaw_range_degrees=[-180,180],
                                      grasp='nearest edge-aligned yaw modulo the cube square symmetry'),
                coverage_policy='Slots 21-300 use immutable balanced per-split assignments in coverage_plan.json; '
                                'retries retain their assigned combination and settled starts are checked.',
                sampling='Independent split/slot/attempt seeds; reachable, settled, collision-checked starts. '
                         'Cube and bowl vary over the reachable table interior; not every physical-table point is reachable.',
                first_ten='First ten retained in final folders; eight train, one validation, one test.',
                accepted_only='Only successful contact grasps and deterministic physics replays are accepted.')
    if startup_policy!='legacy':
        plan.update(startup_policy=startup_policy,controller_policy='holding_correction_v1')
        plan['cube_orientation']['initial_fixed_orientation_slots']=0
    plan_path = folder/'collection_plan.json'
    if plan_path.exists():
        previous_plan=json.loads(plan_path.read_text())
        compatible=[{k:v for k,v in plan.items() if k not in omitted}
                    for omitted in ((),('coverage_policy',),('cube_orientation','coverage_policy'))]
        if previous_plan not in compatible:
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
    coverage_plan=None
    if limit>20:
        from data_tools.bowl_coverage import build_plan,write_summary
        coverage_path=folder/'coverage_plan.json'
        expected=build_plan([row for row in results if row['slot']<20])
        if coverage_path.exists() and json.loads(coverage_path.read_text())!=expected:
            raise ValueError('Fixed coverage assignments changed; do not replace hard configurations')
        coverage_plan=expected
        write_json(coverage_path,coverage_plan)
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
    jobs=[]
    for slot in range(limit):
        if any(row['slot']==slot for row in results):continue
        start=max((r['attempt'] for r in failures if r['slot']==slot),default=-1)+1
        jobs.append((folder,slot,start,max_attempts,
                     coverage_plan['assignments'].get(str(slot)) if coverage_plan else None,startup_policy))
    def accept(outcome):
        result,rejected,temporary=outcome
        failures.extend(rejected)
        if result is None:
            save('attempts_exhausted')
            raise RuntimeError('A fixed coverage slot exhausted attempts; inspect diagnostics rather than replacing it')
        shutil.move(temporary,result['episode'])
        results.append(result)
        results.sort(key=lambda row:row['slot'])
        save('collecting')
        print(f'Accepted {result["id"]}; total {len(results)}/{limit}',flush=True)
    if workers==1:
        for job in jobs:accept(collect_slot(*job))
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
        iterator=iter(jobs)
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            pending={pool.submit(collect_slot,*job) for job in [next(iterator,None) for _ in range(workers)] if job}
            while pending:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    accept(future.result())
                    job=next(iterator,None)
                    if job:pending.add(pool.submit(collect_slot,*job))
    save('completed' if len(results)==300 else 'awaiting_user_review')
    if coverage_plan:
        checked=write_summary(folder,results,coverage_plan)
        if not checked['passed']:
            save('coverage_failed')
            raise RuntimeError('Accepted configurations do not match their fixed coverage targets')
    print(f'Created {len(results)} successful trajectories. Review {folder/"WATCH_TRAJECTORIES.md"}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DATASET_ROOT)
    parser.add_argument('--limit',type=int,choices=(10,20,300),default=10)
    parser.add_argument('--max-attempts',type=int,default=30)
    parser.add_argument('--workers',type=int,choices=(1,2,4),default=1)
    parser.add_argument('--startup-policy',choices=('legacy','state_dependent'),default='legacy')
    args = parser.parse_args()
    collect(args.output_dir,args.limit,args.max_attempts,args.workers,args.startup_policy)


if __name__ == '__main__':
    main()
