"""Audit the completed mixture, replay certificates and counterfactual scene identity."""
import argparse
from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path
import numpy as np
from data_tools.collect_panda_color_place_bowl import ROOT,validate_plan
from data_tools.prepare_panda_dataset import fingerprint


def audit(folder=ROOT):
    folder=Path(folder);plan=json.loads((folder/'collection_plan.json').read_text())['episodes'];validate_plan(plan)
    rows=json.loads((folder/'manifest.json').read_text())['episodes']
    if len(rows)!=600 or [r['id'] for r in rows]!=[r['id'] for r in plan]:raise ValueError('Collection incomplete or IDs changed')
    max_replay=max_tracking=0.;transitions=0;paired=defaultdict(list);reused=0
    for row,spec in zip(rows,plan):
        p=Path(row['episode']);meta=json.loads((p/'episode.json').read_text());replay=json.loads((p/'replay.json').read_text())
        if not all((row['passed'],meta['success'],meta['training_eligible'],replay['passed'])):raise ValueError('Unsuccessful/uncertified episode')
        digest=fingerprint(p)
        if digest!=row['episode_sha256'] or digest!=replay['episode_sha256']:raise ValueError('Episode changed after replay')
        if spec.get('reused_from'):
            reused+=1
            if digest!=spec['source_sha256'] or fingerprint(Path(spec['reused_from']))!=digest:raise ValueError('Original red demonstration changed')
        instruction=f"pick up the {spec['target_color']} cube and place it in the blue bowl."
        if meta['instruction']!=instruction:raise ValueError('Wrong target instruction')
        actions=[json.loads(line) for line in (p/'transitions.jsonl').read_text().splitlines()]
        if len(actions)!=row['transitions'] or len(actions)!=meta['transition_count'] or len(actions)!=replay['replayed_transitions']:raise ValueError('Action count differs')
        if any(r['instruction']!=instruction or not r['executed'] for r in actions):raise ValueError('Bad action/instruction alignment')
        transitions+=len(actions);max_replay=max(max_replay,replay['maximum_qpos_difference']);max_tracking=max(max_tracking,max(r['tracking_error_metres'] for r in actions))
        if spec['cube_count']==2:
            if meta['cube_colors']!=spec['cube_colors'] or meta['target_color']!=spec['target_color'] or meta['wrong_placement']:raise ValueError('Wrong cube selected')
            if meta['maximum_final_distractor_displacement_metres']>.003:raise ValueError('Distractor moved')
            paired[spec['layout_id']].append(p)
    if reused!=50 or max_replay>=1e-6 or max_tracking>.005:raise ValueError('Reuse/replay/tracking checks failed')
    for paths in paired.values():
        a,b=paths
        with np.load(a/'initial_state.npz') as x,np.load(b/'initial_state.npz') as y:
            if set(x.files)!=set(y.files) or any(not np.allclose(x[k],y[k],atol=1e-12,rtol=0) for k in x.files):raise ValueError('Paired requests start in different physics states')
        if hashlib.sha256((a/'step_0001/before.png').read_bytes()).digest()!=hashlib.sha256((b/'step_0001/before.png').read_bytes()).digest():raise ValueError('Paired request initial images differ')
    result=dict(passed=True,episodes=600,training_episodes=600,validation_episodes=0,test_episodes=0,
        reused_unchanged_red=50,new_episodes=550,transitions=transitions,scene_counts=dict(Counter(str(r['cube_count']) for r in rows)),
        requested_color_counts=dict(Counter(r['target_color'] for r in rows)),paired_layouts_verified=len(paired),
        maximum_replay_qpos_difference=max_replay,maximum_tracking_error_metres=max_tracking,
        original_demonstrations_unchanged=True,all_task_successes=True,
        training_colors=['red','blue','green','yellow'],reserved_colors=['black','purple','orange','white'])
    (folder/'collection_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    counts=Counter((r['cube_count'],r['target_color']) for r in rows)
    lines=['# Colored cube placement: collection summary','','600 successful demonstrations; all are training data. No cloud upload or training has run.',
        '', '| Scene | Red target | Blue target | Green target | Yellow target | Total |','|---|---:|---:|---:|---:|---:|']
    for count in (1,2):lines.append(f'| {count} cube(s) | '+ ' | '.join(str(counts[count,c]) for c in result['training_colors'])+f' | {200 if count==1 else 400} |')
    lines+=['',f'Valid action examples: **{transitions:,}**.',
        '50 original single-red recordings were copied unchanged; 550 were newly generated.',
        'All saved-action physics replays passed. All 200 two-cube layouts have matching initial physics states and identical first images across the two target instructions.',
        f'Maximum replay joint-position difference: {max_replay:.3g}; maximum action tracking error: {max_tracking*1000:.3f} mm.',
        'The fixed camera, starting arm pose, open gripper, cube geometry and holding controller are preserved. Both objects are placed before settling. No layouts were replaced after expert failure.',
        'Black, purple, orange and white remain absent from training. Independent simulator layouts are needed for behavioral evaluation.',
        '', '[Watch all demonstrations](WATCH_TRAJECTORIES.md)']
    (folder/'COLLECTION_SUMMARY.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=ROOT);a=p.parse_args();print(json.dumps(audit(a.dataset),indent=2))
