"""Preserve the reviewed 50 and add 250 paired four-color touch demonstrations."""
import json
import shutil
import tempfile
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
from common.paths import PROJECT_ROOT
from data_tools.collect_panda_color_touch import pair_spec
from data_tools.color_touch_demonstrations import record_episode, replay_episode
from data_tools.dataset_paths import DATASETS_ROOT, write_trajectory_index
from data_tools.panda_demonstrations import write_json
from data_tools.prepare_panda_dataset import fingerprint

ROOT = DATASETS_ROOT/'panda_four_color_touch'
COLORS = ('red','blue','green','yellow')


def expansion_plan():
    plans=[]
    # Existing red/blue scenes already fill their quota of 25. Each remaining
    # unordered color pair receives 25 scenes. Each color is requested 75 times.
    quotas=[(17,4,4),(17,4,4),(17,4,4),(16,4,5),(16,5,4)]
    pairs=[p for p in combinations(COLORS,2) if p!=('red','blue')]
    for group,(colors,quota) in enumerate(zip(pairs,quotas)):
        for i in range(25):
            number=25+group*25+i
            positions=pair_spec(number)['cube_positions']
            names=dict(zip(('red','blue'), colors if i%2==0 else colors[::-1]))
            split='train' if i<quota[0] else 'validation' if i<sum(quota[:2]) else 'test'
            plans.append(dict(pair_id=number,split=split,cube_positions=positions,cube_colors=names))
    return plans


def collect():
    ROOT.mkdir(parents=True,exist_ok=True)
    source=json.loads((DATASETS_ROOT/'panda_color_touch/manifest.json').read_text())['episodes']
    if len(source)!=50:raise ValueError('Expected the reviewed original 50')
    manifest=ROOT/'manifest.json'
    results=json.loads(manifest.read_text())['episodes'] if manifest.exists() else []
    for row in source:
        old=Path(row['episode']);new=ROOT/row['split']/row['id']
        if fingerprint(old)!=row['episode_sha256']:raise ValueError('Original changed')
        if not new.exists():shutil.copytree(old,new)
        if fingerprint(new)!=row['episode_sha256']:raise ValueError('Preserved copy changed')
        if not any(r['id']==row['id'] for r in results):results.append(dict(row,episode=str(new.resolve())))
    for row in results:
        if fingerprint(Path(row['episode']))!=row['episode_sha256']:raise ValueError('Saved episode changed')
    plans=expansion_plan()
    write_json(ROOT/'collection_plan.json',dict(total_trajectories=300,colors=COLORS,
        preserved_trajectories=50,new_scenes=plans,failed_demonstrations='Stop; never replace a failed scene'))
    counts=Counter(r['split'] for r in source)
    for plan in plans:
        first=None
        for slot in ('red','blue'):
            split=plan['split'];counts[split]+=1;identifier=f'{split}_{counts[split]:03d}'
            spec=dict(plan,id=identifier,target_color=slot)
            folder=ROOT/split/identifier
            prior=next((r for r in results if r['id']==identifier),None)
            if prior:
                metadata=json.loads((folder/'episode.json').read_text())
                if any(metadata['scenario'][k]!=plan[k] for k in ('pair_id','split','cube_positions','cube_colors')):
                    raise ValueError('Plan changed')
            else:
                staging=PROJECT_ROOT/'outputs/four_color_collection_checks';staging.mkdir(parents=True,exist_ok=True)
                temporary=Path(tempfile.mkdtemp(prefix=identifier+'-',dir=staging))/'episode'
                print(f"Collecting {identifier}: touch the {plan['cube_colors'][slot]} cube.",flush=True)
                try:
                    metadata=record_episode(temporary,spec);replay=replay_episode(temporary)
                except Exception as exc:
                    write_json(ROOT/'collection_failure.json',dict(scenario=spec,error=str(exc),diagnostics=str(temporary)))
                    raise
                if folder.exists():raise ValueError('Refusing to overwrite episode')
                folder.parent.mkdir(parents=True,exist_ok=True);shutil.move(temporary,folder)
                results.append(dict(id=identifier,split=split,episode=str(folder.resolve()),pair_id=plan['pair_id'],
                    target_color=plan['cube_colors'][slot],cube_colors=plan['cube_colors'],cube_positions=plan['cube_positions'],
                    instruction=metadata['instruction'],transitions=metadata['transition_count'],
                    episode_sha256=replay['episode_sha256'],passed=replay['passed']))
            image=(folder/'step_0001/before.png').read_bytes()
            with np.load(folder/'initial_state.npz') as state:qpos=state['qpos'].copy()
            if first is None:first=(image,qpos)
            elif image!=first[0] or not np.array_equal(qpos,first[1]):raise ValueError('Paired scenes differ')
            write_json(manifest,dict(episodes=results,split_counts=dict(Counter(r['split'] for r in results)),ready_for_finetuning=False))
    assert len(results)==300
    assert Counter(r['split'] for r in results)=={'train':200,'validation':50,'test':50}
    assert Counter(r['target_color'] for r in results)==dict.fromkeys(COLORS,75)
    write_trajectory_index(ROOT,results)
    write_json(ROOT/'collection_report.json',dict(successful_trajectories=300,all_action_replays_passed=True,
        paired_initial_images_and_states_identical=True,split_by_scene_pair=True,
        split_counts=dict(Counter(r['split'] for r in results)),color_counts=dict(Counter(r['target_color'] for r in results))))
    return results


if __name__=='__main__':collect()
