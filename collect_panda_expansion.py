"""Collect 40 additional verified pick/place demonstrations with fixed splits."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from collect_panda_pilot import pilot_scenarios
from dataset_paths import PANDA_DATASET_ROOT, write_trajectory_index


def expansion_scenarios():
    rng = np.random.default_rng(20261010)
    existing = {tuple(row['cube_xy']) for row in pilot_scenarios()}
    plan = []
    directions = [(0,.10),(0,-.10),(.065,0),(-.065,0),(.045,.075),(-.045,-.075),(.045,-.075),(-.045,.075)]
    for i in range(40):
        split = 'train' if i < 30 else 'validation' if i < 35 else 'test'
        while True:
            xy = np.round([rng.uniform(.525,.575),rng.uniform(-.04,.04)], 4)
            if tuple(xy) not in existing:
                existing.add(tuple(xy)); break
        offset = directions[i % len(directions)]
        yaw = [-.18,-.09,.09,.18][i % 4]
        approach = [[.025,0],[-.025,0],[0,.025],[0,-.025]][(i // 4) % 4]
        plan.append(dict(id=f'{split}_extra_{i+1:02d}', split=split, cube_xy=xy.tolist(),
                         destination_offset_xy=list(offset), grasp_yaw_radians=yaw,
                         approach_offset_xy=approach, initial_joint1_radians=[-.08,0,.08][i % 3],
                         instruction=f'pick up the red cube and move it {offset[0]*100:g} cm along world X and {offset[1]*100:g} cm along world Y'))
    return plan


def collect(folder, pilot):
    from panda_demonstrations import record_episode, replay_episode
    from prepare_panda_dataset import prepare_dataset, fingerprint
    pilot = Path(pilot).resolve()
    previous = json.loads((pilot / 'collection_report.json').read_text())['results']
    if len(previous) != 10 or not all(row['passed'] for row in previous):
        raise ValueError('Expected the verified ten-trajectory pilot')
    for row in previous:
        episode = Path(row['episode'])
        if json.loads((episode/'replay.json').read_text())['episode_sha256'] != fingerprint(episode):
            raise ValueError('Pilot episode changed since verification')
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    plan = expansion_scenarios()
    (folder/'collection_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    results=[]
    def save():
        report=dict(planned_episodes=40, successful_episodes=sum(row['passed'] for row in results),
                    ready_for_finetuning=False, results=results,
                    note='Expert data validation, not OpenVLA performance. Failed attempts are excluded and retained. Splits are fixed before collection.')
        (folder/'collection_report.json').write_text(json.dumps(report,indent=2)+'\n')
        write_trajectory_index(folder,results)
    save()
    for spec in plan:
        episode=folder/spec['split']/spec['id']
        try:
            print(f"Collecting {spec['id']} | offset {spec['destination_offset_xy']} | yaw {spec['grasp_yaw_radians']}",flush=True)
            record_episode(episode,cube_xy=spec['cube_xy'],destination_offset=spec['destination_offset_xy'],
                           instruction=spec['instruction'],scenario=spec,grasp_yaw_radians=spec['grasp_yaw_radians'],
                           approach_offset_xy=spec['approach_offset_xy'],initial_joint1_radians=spec['initial_joint1_radians'])
            r=replay_episode(episode)
            results.append(dict(id=spec['id'],split=spec['split'],episode=str(episode),passed=True,
                                transitions=r['replayed_transitions'],lift_metres=r['maximum_cube_lift_metres'],
                                placement_error_metres=r['placement_xy_error_metres']))
        except (ValueError,RuntimeError) as exc:
            results.append(dict(id=spec['id'],split=spec['split'],episode=str(episode),passed=False,error=str(exc)))
        save()
    accepted=previous+[row for row in results if row['passed']]
    train=[Path(row['episode']) for row in accepted if row['split']=='train']
    prepare_dataset(train,folder/'combined_train_staging')
    (folder/'combined_manifest.json').write_text(json.dumps(dict(episodes=accepted,
        split_counts={split:sum(row['split']==split for row in accepted) for split in ('train','validation','test')},
        ready_for_finetuning=False),indent=2)+'\n')
    if all(Path(row['episode']).is_relative_to(PANDA_DATASET_ROOT.resolve()) for row in accepted):
        write_trajectory_index(PANDA_DATASET_ROOT, accepted)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir',type=Path,default=PANDA_DATASET_ROOT/'panda_pilot_20261010_101733')
    parser.add_argument('--output-dir',type=Path,default=PANDA_DATASET_ROOT/datetime.now(timezone.utc).strftime('panda_expansion_%Y%m%d_%H%M%S'))
    args=parser.parse_args();collect(args.output_dir,args.pilot_dir)

if __name__=='__main__':main()
