"""Balanced 600-episode one/two-cube placement dataset; no cloud access or upload."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
from data_tools.dataset_paths import DATASETS_ROOT,write_trajectory_index
from data_tools.panda_demonstrations import write_json
from data_tools.prepare_panda_dataset import fingerprint
from simulation.color_place_bowl_scene import validate_layout

COLORS=('red','blue','green','yellow')
ROOT=DATASETS_ROOT/'panda_color_place_bowl'
SOURCE=DATASETS_ROOT/'panda_place_bowl/manifest.json'


def pair_layout(seed):
    # Immutable seeds, without substituting easier layouts after execution failures.
    rng=np.random.default_rng([20261015,seed,61])
    for _ in range(10000):
        xy=rng.uniform([.405,-.155],[.675,.155],size=(3,2)).tolist()
        layout=dict(cubes=dict(cube=xy[0],cube_1=xy[1]),bowl=xy[2])
        try:validate_layout(layout,2)
        except ValueError:continue
        return layout
    raise RuntimeError('Could not construct separated workspace layout')


def build_plan(source=SOURCE):
    source=Path(source);original=json.loads(source.read_text())['episodes']
    reused=sorted((r for r in original if r['split']=='train'),key=lambda r:r['id'])[:50]
    if len(reused)!=50 or [r['id'] for r in reused]!=[f'train_{n:03d}' for n in range(1,51)]:
        raise ValueError('Expected the first 50 numbered training demonstrations')
    plan=[]
    for row in reused:
        metadata=json.loads((Path(row['episode'])/'episode.json').read_text())
        if metadata['instruction']!='pick up the red cube and place it in the blue bowl.':raise ValueError('Wrong reuse task')
        layout=dict(cubes=dict(cube=row['cube_positions']['red']),bowl=row['bowl_position'])
        validate_layout(layout,1)
        plan.append(dict(id=row['id'],split='train',cube_count=1,target_color='red',cube_colors=dict(cube='red'),layout=layout,
            reused_from=row['episode'],source_sha256=row['episode_sha256'],layout_id='single_'+row['id']))
    for color in COLORS[1:]:
        for index,row in enumerate(reused):
            plan.append(dict(id=f'train_{len(plan)+1:03d}',split='train',cube_count=1,target_color=color,
                cube_colors=dict(cube=color),layout=plan[index]['layout'],layout_id=plan[index]['layout_id']))
    for a,b in itertools.combinations(COLORS,2):
        # Opposite edges get one extra paired layout: every color has 100 targets.
        count=34 if (a,b) in (('red','blue'),('green','yellow')) else 33
        for index in range(count):
            layout_index=(len(plan)-200)//2;seed=60000+layout_index
            layout=pair_layout(seed);names=dict(zip(('cube','cube_1'),(a,b) if index%2==0 else (b,a)))
            for target in (a,b):
                plan.append(dict(id=f'train_{len(plan)+1:03d}',split='train',cube_count=2,target_color=target,
                    cube_colors=names,layout=layout,layout_id=f'pair_{seed}',layout_seed=seed))
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if len(plan)!=600 or [r['id'] for r in plan]!=[f'train_{n:03d}' for n in range(1,601)]:raise ValueError('Expected 600 contiguous episode IDs')
    if Counter(r['cube_count'] for r in plan)!={1:200,2:400}:raise ValueError('Wrong scene ratio')
    for count,per_color in ((1,50),(2,100)):
        if Counter(r['target_color'] for r in plan if r['cube_count']==count)!=dict.fromkeys(COLORS,per_color):raise ValueError('Unbalanced targets')
    paired={}
    for row in plan:
        validate_layout(row['layout'],row['cube_count'])
        names=row['cube_colors']
        if set(names.values())-set(COLORS) or row['target_color'] not in names.values() or len(set(names.values()))!=row['cube_count']:raise ValueError('Invalid color assignment')
        if row['cube_count']==2:paired.setdefault(row['layout_id'],[]).append(row)
    if len(paired)!=200:raise ValueError('Need 200 counterfactual pairs')
    for rows in paired.values():
        if len(rows)!=2 or rows[0]['layout']!=rows[1]['layout'] or rows[0]['cube_colors']!=rows[1]['cube_colors'] or {r['target_color'] for r in rows}!=set(rows[0]['cube_colors'].values()):raise ValueError('Unpaired targets')


def record_spec(payload):
    from data_tools.bowl_demonstrations import record_episode,replay_episode
    spec,temporary=payload;temporary=Path(temporary)
    if spec.get('reused_from'):
        src=Path(spec['reused_from'])
        if fingerprint(src)!=spec['source_sha256']:raise ValueError('Reused source changed')
        shutil.copytree(src,temporary)
        metadata=json.loads((temporary/'episode.json').read_text());replay=json.loads((temporary/'replay.json').read_text())
        if not metadata['success'] or not metadata['training_eligible'] or not replay['passed'] or fingerprint(temporary)!=spec['source_sha256']:raise ValueError('Reuse copy not verified')
    else:
        scenario=dict(spec,random_seed=[20261015,int(spec['id'].split('_')[1]),70],initial_gripper=1,startup_policy='state_dependent')
        metadata=record_episode(temporary,scenario,scene_profile='panda_color_place_bowl_v1');replay=replay_episode(temporary)
    return metadata,replay


def collect(folder=ROOT,source=SOURCE,limit=600,workers=1):
    from concurrent.futures import ProcessPoolExecutor
    from contextlib import nullcontext
    import multiprocessing
    if type(workers) is not int or not 1<=workers<=4:raise ValueError('Use 1–4 local workers')
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    plan_path=folder/'collection_plan.json'
    if plan_path.exists():plan=json.loads(plan_path.read_text())['episodes']
    else:
        plan=build_plan(source);write_json(plan_path,dict(schema_version=1,episodes=plan,colors=COLORS,total=600,export_policy='all_training'))
    validate_plan(plan)
    if not 1<=limit<=600:raise ValueError('limit must be between 1 and 600')
    manifest=folder/'manifest.json';rows=json.loads(manifest.read_text())['episodes'] if manifest.exists() else []
    for index,row in enumerate(rows):
        if row['id']!=plan[index]['id'] or fingerprint(Path(row['episode']))!=row['episode_sha256']:raise ValueError('Accepted demonstration changed')
    def save():
        write_json(manifest,dict(episodes=rows,export_policy='all_training',training_colors=COLORS,
            reserved_colors=['black','purple','orange','white'],split_counts=dict(train=len(rows),validation=0,test=0),
            scene_counts=dict(Counter(str(r['cube_count']) for r in rows)),target_counts=dict(Counter(r['target_color'] for r in rows)),
            all_successful=len(rows)==600,ready_for_finetuning=False))
        write_trajectory_index(folder,rows)
    remaining=plan[len(rows):limit]
    (folder/'.collecting').mkdir(exist_ok=True)
    for spec in remaining:
        if (folder/'train'/spec['id']).exists() or (folder/'.collecting'/spec['id']).exists():raise ValueError(f'Unindexed/incomplete episode exists: {spec["id"]}; inspect before retry')
    def payloads():
        for spec in remaining:
            if shutil.disk_usage(folder).free<2*1024**3:raise RuntimeError('Less than 2 GiB free; accepted episodes retained')
            yield spec,str(folder/'.collecting'/spec['id'])
    context=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) if workers>1 else nullcontext(None)
    try:
        with context as pool:
            results=pool.map(record_spec,payloads(),chunksize=1) if pool else map(record_spec,payloads())
            for spec,(metadata,replay) in zip(remaining,results):
                temporary=folder/'.collecting'/spec['id'];destination=folder/'train'/spec['id']
                destination.parent.mkdir(parents=True,exist_ok=True);temporary.rename(destination)
                row={k:v for k,v in spec.items() if k!='layout'}
                row.update(episode=str(destination),passed=True,episode_sha256=replay['episode_sha256'],transitions=metadata['transition_count'])
                rows.append(row);save();print('Accepted',len(rows),'/ 600',flush=True)
    except BaseException as exc:
        write_json(folder/'collection_error.json',dict(error=str(exc),diagnostics=str(folder/'.collecting'),policy='No failed layout substitutions; inspect completed/incomplete scratch recordings before retry'))
        save();raise
    save();return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--limit',type=int,default=600);p.add_argument('--workers',type=int,default=1);p.add_argument('--output-dir',type=Path,default=ROOT);p.add_argument('--source-manifest',type=Path,default=SOURCE)
    a=p.parse_args();collect(a.output_dir,a.source_manifest,a.limit,a.workers)
