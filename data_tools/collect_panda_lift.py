"""Collect ten replay-verified grasp/lift demonstrations for review before expansion."""

from common.output_layout import output_path
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from common.paths import PROJECT_ROOT
from data_tools.dataset_paths import DATASETS_ROOT, write_trajectory_index
from data_tools.panda_demonstrations import write_json
from data_tools.prepare_panda_dataset import fingerprint
from data_tools.lift_demonstrations import record_episode, replay_episode
from simulation.lift_scene import INSTRUCTION, CAMERA_NAME, START_POSITION, CUBE_X_BOUNDS, CUBE_Y_BOUNDS

SEED=20261014
SPLITS=['train']*4+['validation','test']
DATASET_ROOT=DATASETS_ROOT/'panda_lift_cube'


def scenario(slot):
    split=SPLITS[slot%len(SPLITS)]
    rank=sum(SPLITS[i%len(SPLITS)]==split for i in range(slot))+1
    rng=np.random.default_rng([SEED,slot])
    if slot<10:
        # Ten fixed spatial cells; failed locations are not replaced by easier ones.
        xy=[.42+.06*(slot%5)+rng.uniform(-.008,.008),
            (-.12 if slot<5 else .12)+rng.uniform(-.008,.008)]
    else:
        xy=[rng.uniform(*CUBE_X_BOUNDS),rng.uniform(*CUBE_Y_BOUNDS)]
    return dict(id=f'{split}_{rank:03d}',split=split,slot=slot,
                random_seed=[SEED,slot],cube_xy=xy,instruction=INSTRUCTION)


def collect(folder=DATASET_ROOT,limit=10):
    if type(limit) is not int or limit<1:raise ValueError('limit must be positive')
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    plan=dict(seed=SEED,task='lift_cube',instruction=INSTRUCTION,camera=CAMERA_NAME,
              fixed_arm_position=list(START_POSITION),fixed_arm_orientation='downward scene home reference',
              gripper='open at start; closes around cube; closed while lifting',cube_orientation_wxyz=[1,0,0,0],
              cube_x_bounds=list(CUBE_X_BOUNDS),cube_y_bounds=list(CUBE_Y_BOUNDS),
              split_pattern=SPLITS,success='cube center rises 5 cm above start with continuous bilateral grasp for 1 simulator second',
              expert_policy='approach above cube; descend; one closing action; continue lifting until verified; no idle tail',
              first_ten_spatial_cells='five x positions, two y rows, seeded 8 mm jitter',
              sampling='predeclared positions; never replace a failed location automatically')
    plan_path=folder/'collection_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError('Collection plan changed; do not mix demonstrations')
    write_json(plan_path,plan)
    path=folder/'collection_report.json'
    prior=json.loads(path.read_text()) if path.exists() else {}
    results=prior.get('results',[]);failures=prior.get('failures',[])
    for result in results:
        episode=Path(result['episode']);replay=json.loads((episode/'replay.json').read_text())
        if not replay['passed'] or replay['episode_sha256']!=fingerprint(episode):
            raise ValueError('Previously accepted demonstration changed')
    def save(phase):
        write_json(path,dict(phase=phase,successful_trajectories=len(results),requested_limit=limit,
                            results=results,failures=failures,ready_for_finetuning=False))
        write_json(folder/'manifest.json',dict(episodes=results,instruction=INSTRUCTION,
                   split_counts={split:sum(r['split']==split for r in results) for split in set(SPLITS)},
                   ready_for_finetuning=False))
        write_trajectory_index(folder,results)
    save('collecting')
    checks=output_path('lift_collection_checks');checks.mkdir(parents=True,exist_ok=True)
    for slot in range(limit):
        if any(r['slot']==slot for r in results):continue
        spec=scenario(slot);destination=folder/spec['split']/spec['id']
        if destination.exists():raise ValueError('Episode destination already exists')
        temp=Path(tempfile.mkdtemp(prefix=spec['id']+'-',dir=checks))/'episode'
        print(f"Collecting {spec['id']}, cube XY={spec['cube_xy']}",flush=True)
        try:
            metadata=record_episode(temp,spec);report=replay_episode(temp)
        except Exception as exc:
            failures.append(dict(spec,error=str(exc),diagnostics=str(temp)))
            save('failed_fixed_location')
            raise
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(temp,destination)
        results.append(dict(id=spec['id'],split=spec['split'],slot=slot,episode=str(destination),
                            passed=True,transitions=metadata['transition_count'],cube_xy=spec['cube_xy'],
                            episode_sha256=report['episode_sha256']))
        results.sort(key=lambda r:r['slot']);save('collecting')
        print(f"Accepted {spec['id']}: {metadata['transition_count']} actions; {len(results)}/{limit}",flush=True)
    save('awaiting_user_review')
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=DATASET_ROOT)
    p.add_argument('--limit',type=int,default=10)
    args=p.parse_args();collect(args.output_dir,args.limit)


if __name__=='__main__':main()
