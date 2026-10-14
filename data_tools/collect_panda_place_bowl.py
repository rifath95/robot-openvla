"""Phase 4 placement demonstrations, retained under one numbered dataset on expansion."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from common.output_layout import output_path
from data_tools.dataset_paths import DATASETS_ROOT,write_trajectory_index
from data_tools.panda_demonstrations import write_json
from data_tools.prepare_panda_dataset import fingerprint
from data_tools.bowl_demonstrations import record_episode,replay_episode
from simulation.place_bowl_scene import INSTRUCTION,positions
SPLITS=['train']*4+['validation','test']
DATASET_ROOT=DATASETS_ROOT/'panda_place_bowl'


def collect(folder=DATASET_ROOT,limit=10):
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'manifest.json';rows=json.loads(manifest.read_text())['episodes'] if manifest.exists() else []
    for r in rows:
        if fingerprint(Path(r['episode']))!=r['episode_sha256']:raise ValueError('Accepted episode changed')
    checks=output_path('place_bowl_collection_checks');checks.mkdir(parents=True,exist_ok=True)
    def save():
        write_json(manifest,dict(episodes=rows,instruction=INSTRUCTION,
            split_counts={k:sum(r['split']==k for r in rows) for k in ('train','validation','test')},ready_for_finetuning=False))
        write_trajectory_index(folder,rows)
    for slot in range(limit):
        if any(r['slot']==slot for r in rows):continue
        split=SPLITS[slot%6];rank=sum(SPLITS[i%6]==split for i in range(slot))+1
        name=f'{split}_{rank:03d}';destination=folder/split/name
        if destination.exists():raise ValueError('Unindexed destination exists')
        spec=dict(id=name,split=split,slot=slot,random_seed=[20261014,slot,5],initial_gripper=1,
            random_cube_yaw=False,startup_policy='state_dependent',instruction=INSTRUCTION,**positions(slot))
        temp=Path(tempfile.mkdtemp(prefix=name+'-',dir=checks))/'episode'
        print('Collecting',name,spec['cube_xy'],spec['bowl_xy'],flush=True)
        meta=record_episode(temp,spec,scene_profile='panda_place_bowl_v1');replay=replay_episode(temp)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.move(temp,destination)
        rows.append(dict(id=name,split=split,slot=slot,episode=str(destination),passed=True,
            transitions=meta['transition_count'],episode_sha256=replay['episode_sha256'],
            cube_positions={'red':spec['cube_xy']},bowl_position=spec['bowl_xy']))
        save();print('Accepted',name,len(rows),'/',limit,flush=True)
    save()
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--limit',type=int,default=10);a=p.parse_args();collect(limit=a.limit)
