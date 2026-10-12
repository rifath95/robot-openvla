"""Fixed per-episode coverage quotas; failed attempts retain their assigned bins."""
import json
from pathlib import Path

import numpy as np

from data_tools.bowl_demonstrations import orientation_delta

BIN_COUNTS = dict(cube_region=4, bowl_region=4, hand_region=4, hand_height=2,
                  hand_distance=2, roll=3, pitch=3, yaw=4, cube_rotation=4, gripper=3)
SPLIT_COUNTS = dict(train=200, validation=50, test=50)
SEED = 20261012


def quadrant(xy, midpoint):
    return 2*int(xy[0]>=midpoint)+int(xy[1]>=0)


def classify(scene):
    start=scene['starting_pose']
    position=np.array(start['actual_position'])
    rotation=orientation_delta(np.array(scene['reference_quaternion_wxyz']),
                               np.array(start['actual_quaternion_wxyz']))
    cube=np.array(scene['cube_position'])
    cube_yaw=float(scene.get('requested_cube_yaw_radians',0))
    if 'cube_quaternion_wxyz' in scene:
        w,x,y,z=scene['cube_quaternion_wxyz']
        cube_yaw=float(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    cube_angle=(cube_yaw+np.pi/4)%(np.pi/2)-np.pi/4
    if abs(cube_angle)<1e-4:cube_angle=0.0  # Stable category for numerically axis-aligned cubes.
    grip=start.get('actual_gripper_open_fraction',start['requested_gripper_open_fraction'])
    return dict(cube_region=quadrant(cube[:2],.55), bowl_region=quadrant(scene['bowl_position'][:2],.55),
                hand_region=quadrant(position[:2],.53), hand_height=int(position[2]>=.49),
                hand_distance=int(np.linalg.norm(position-cube)>=.30),
                roll=int(np.clip(np.searchsorted([-.85/3,.85/3],rotation[0]),0,2)),
                pitch=int(np.clip(np.searchsorted([-.85/3,.85/3],rotation[1]),0,2)),
                yaw=int(np.clip(np.searchsorted([-1.1,0,1.1],rotation[2]),0,3)),
                cube_rotation=int(np.clip(np.floor((cube_angle+np.pi/4)/(np.pi/8)),0,3)),
                gripper=0 if grip<.05 else 2 if grip>.95 else 1)


def build_plan(existing):
    """Balance each marginal per split, including the twenty preserved examples."""
    from data_tools.collect_panda_bowl import SPLITS
    if {row['slot'] for row in existing if row['slot']<20}!=set(range(20)):
        raise ValueError('Collect and review the first twenty trajectories before planning the remaining coverage')
    assignments={}
    quotas={}
    old_bins={row['slot']:classify(json.loads((Path(row['episode'])/'episode.json').read_text())['scenario']['scene'])
              for row in existing if row['slot']<20}
    for split_index,(split,total) in enumerate(SPLIT_COUNTS.items()):
        slots=[slot for slot in range(20,300) if SPLITS[slot%6]==split]
        quotas[split]={}
        for dimension_index,(dimension,count) in enumerate(BIN_COUNTS.items()):
            rng=np.random.default_rng([SEED,split_index,dimension_index,918])
            desired=np.full(count,total//count,dtype=int)
            desired[rng.permutation(count)[:total%count]]+=1
            used=np.bincount([old_bins[row['slot']][dimension] for row in existing
                              if row['slot']<20 and row['split']==split],minlength=count)
            remaining=desired-used
            if np.any(remaining<0) or sum(remaining)!=len(slots):
                raise ValueError('Preserved examples cannot satisfy the declared coverage quotas')
            values=np.repeat(np.arange(count),remaining)
            rng.shuffle(values)
            quotas[split][dimension]=desired.tolist()
            for slot,value in zip(slots,values):
                assignments.setdefault(str(slot),{})[dimension]=int(value)
    return dict(version=1, preserved_slots=20, bin_counts=BIN_COUNTS,
                quotas_by_split=quotas, assignments=assignments,
                note='Balanced marginal bins, not exhaustive Cartesian combinations. '
                     'Every retry and internal resample retains the complete assigned combination.')


def sample(rng,target):
    def uniform_bin(edges,index):
        low,high=edges[index:index+2]
        return float(rng.uniform(low+.001,high-.001))
    def xy(name,low,high,mid):
        cell=target[name]
        return np.array([uniform_bin([low[0],mid,high[0]],cell//2),
                         uniform_bin([low[1],0,high[1]],cell%2)])
    angle=uniform_bin(np.linspace(-np.pi/4,np.pi/4,5),target['cube_rotation'])
    cube_yaw=float((angle+int(rng.integers(-2,2))*np.pi/2+np.pi)%(2*np.pi)-np.pi)
    grip=0 if target['gripper']==0 else 1 if target['gripper']==2 else float(rng.uniform(.1,.9))
    return dict(cube_xy=xy('cube_region',[.38,-.22],[.72,.22],.55),
                bowl_xy=xy('bowl_region',[.40,-.21],[.70,.21],.55),
                position=np.r_[xy('hand_region',[.36,-.24],[.70,.24],.53),
                               uniform_bin([.38,.49,.60],target['hand_height'])],
                roll=uniform_bin(np.linspace(-.85,.85,4),target['roll']),
                pitch=uniform_bin(np.linspace(-.85,.85,4),target['pitch']),
                yaw=uniform_bin(np.linspace(-2.2,2.2,5),target['yaw']),
                cube_yaw=cube_yaw,gripper=grip)


def audit(results,plan):
    counts={split:{dim:[0]*size for dim,size in BIN_COUNTS.items()} for split in SPLIT_COUNTS}
    mismatches=[]
    combinations=set()
    continuous=[]
    for row in results:
        scene=json.loads((Path(row['episode'])/'episode.json').read_text())['scenario']['scene']
        actual=classify(scene)
        start=scene['starting_pose']
        angles=orientation_delta(np.array(scene['reference_quaternion_wxyz']),
                                 np.array(start['actual_quaternion_wxyz']))
        distance=float(np.linalg.norm(np.array(start['actual_position'])-scene['cube_position']))
        continuous.append([*scene['cube_position'][:2],*scene['bowl_position'][:2],
                           *start['actual_position'],*angles,distance])
        for dimension,value in actual.items():counts[row['split']][dimension][value]+=1
        target=plan['assignments'].get(str(row['slot']))
        if target:combinations.add(tuple(target[dimension] for dimension in BIN_COUNTS))
        if target and actual!=target:mismatches.append(dict(id=row['id'],target=target,actual=actual))
    complete=len(results)==300
    values=np.array(continuous)
    return dict(accepted_counts_by_split=counts, quotas_by_split=plan['quotas_by_split'],
                unique_assigned_combinations=len(combinations),
                continuous_features=['cube_x','cube_y','bowl_x','bowl_y','hand_x','hand_y','hand_z',
                                     'roll','pitch','yaw','hand_cube_distance'],
                continuous_minimum=values.min(axis=0).tolist(),
                continuous_maximum=values.max(axis=0).tolist(),
                assignment_mismatches=mismatches, complete=complete,
                passed=not mismatches and (not complete or counts==plan['quotas_by_split']))


def write_summary(folder,results,plan):
    """Persist both machine-readable coverage checks and a readable dataset summary."""
    folder=Path(folder)
    checked=audit(results,plan)
    (folder/'coverage_audit.json').write_text(json.dumps(checked,indent=2)+'\n')
    splits={split:sum(row['split']==split for row in results) for split in SPLIT_COUNTS}
    steps={split:sum(row['transitions'] for row in results if row['split']==split) for split in SPLIT_COUNTS}
    replays=[json.loads((Path(row['episode'])/'replay.json').read_text()) for row in results]
    lines=['# Bowl dataset collection summary','',
           '**Instruction:** pick up the red cube and drop it in the blue bowl.','',
           '| Split | Successful trajectories | Recorded steps |', '| --- | ---: | ---: |']
    lines.extend(f'| {split} | {splits[split]} | {steps[split]} |' for split in SPLIT_COUNTS)
    lines.extend(['',f'Total: **{len(results)} trajectories / {sum(steps.values())} steps**.',
                  'The original twenty trajectories are retained. All later examples have fixed coverage assignments.',
                  'This is the recorded demonstration corpus; RLDS packaging and fine-tuning have not started.',
                  '', '## Success and replay checks','',
                  f'- All sequential physics replays passed: **{all(row["passed"] for row in replays)}**.',
                  f'- Maximum replay state difference: **{max(row["maximum_qpos_difference"] for row in replays):.3g}** (limit 1e-6).',
                  f'- Minimum cube lift: **{min(row["maximum_cube_lift_metres"] for row in replays)*100:.2f} cm** (required 8 cm).',
                  f'- Maximum final cube–bowl-center distance: **{max(row["bowl_center_distance_metres"] for row in replays)*100:.2f} cm** (limit 4.5 cm).',
                  f'- Fixed assignments match settled starts: **{not checked["assignment_mismatches"]}**.',
                  f'- Complete coverage quotas passed: **{checked["complete"] and checked["passed"]}**.',
                  f'- Distinct complete assigned combinations: **{checked["unique_assigned_combinations"]}**.',
                  '', '## Accepted counts by category','',
                  'Each list is ordered by category number from zero upward. Counts must equal the corresponding fixed quota.',
                  'Region categories are low-X/negative-Y, low-X/positive-Y, high-X/negative-Y, high-X/positive-Y.',
                  'Height and distance categories are low/high and near/far. Angular categories increase with angle.',
                  'Gripper categories are closed/intermediate/open. See the task guide for exact boundaries.',
                  '', '| Dimension | Training | Validation | Test |', '| --- | --- | --- | --- |'])
    for dimension in BIN_COUNTS:
        values=[' / '.join(map(str,checked['accepted_counts_by_split'][split][dimension])) for split in SPLIT_COUNTS]
        lines.append('| '+dimension+' | '+' | '.join(values)+' |')
    lines.extend(['','These are balanced marginal categories, not every possible Cartesian combination. '
                  'Uniformity within a category is not guaranteed: unreachable starts and failed physical motions are rejected within that same assigned combination.',
                  '', '## Actual feature ranges','', '| Feature | Minimum | Maximum |','| --- | ---: | ---: |'])
    for feature,low,high in zip(checked['continuous_features'],checked['continuous_minimum'],checked['continuous_maximum']):
        lines.append(f'| {feature} | {low:.4f} | {high:.4f} |')
    lines.extend(['','Positions/distances are metres; angles are radians relative to the home hand orientation.',
                  '', '## Replay','',
                  'Use [WATCH_TRAJECTORIES.md](WATCH_TRAJECTORIES.md) to open any trajectory from Training, Validation, or Test. '
                  'The index is ordered by episode number. Each replay uses the saved bowl position and motion frames.',
                  'Rejected attempts are outside this corpus under `outputs/bowl_collection_checks/`.'])
    (folder/'COLLECTION_SUMMARY.md').write_text('\n'.join(lines)+'\n')
    return checked
