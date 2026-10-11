"""Local audit of Panda v2 inputs and saved rollouts; never load model weights."""

from common.paths import PROJECT_ROOT
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from simulation.panda_actions import action_contract,to_controller_action,from_controller_action
from data_tools.prepare_training_dataset import load_episodes,normalize,training_statistics

ROOT=PROJECT_ROOT


def audit(source,dataset,processor_dir,rollouts,output):
    os.environ.setdefault('HF_MODULES_CACHE',str(ROOT/'.cache-openvla/modules'))
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from data_tools.check_training_inputs import official_batch_transform
    from training.upstream.prompting import PurePromptBuilder
    from inference.openvla_prompt import action_prompt

    torch.set_num_threads(2)
    print('Verifying source fingerprints against the v2 bundle...',flush=True)
    episodes=load_episodes(source,verify_images=False)
    manifest=json.loads((dataset/'manifest.json').read_text())
    expected={e['id']:(e['split'],e['transitions'],e['sha256']) for e in manifest['episodes']}
    actual={e['id']:(e['split'],len(e['rows']),e['fingerprint']) for e in episodes}
    source_matches=actual==expected
    statistics=json.loads((dataset/'dataset_statistics.json').read_text())
    stats_match=statistics==training_statistics(episodes)
    contract_match=manifest['action_contract']==action_contract()
    train=[e for e in episodes if e['split']=='train']
    instructions=sorted({r['instruction'] for e in train for r in e['rows']})
    processor=AutoProcessor.from_pretrained(str(processor_dir),local_files_only=True,trust_remote_code=True)
    transform=official_batch_transform(processor)
    # Exercise the official batch transform with images from every stage and both
    # training/validation splits, selected evenly through each stage population.
    groups={}
    for e in episodes:
        if e['split'] not in ('train','validation'):continue
        for r in e['rows']:groups.setdefault((e['split'],r['stage']),[]).append((e,r))
    samples=[]
    for group,population in sorted(groups.items()):
        samples.extend(population[i] for i in np.linspace(0,len(population)-1,min(4,len(population)),dtype=int))
    checked=[]
    print(f'Checking official processor tensors and tokens on {len(samples)} images...',flush=True)
    for e,r in samples:
        with Image.open(e['path']/r['observation']) as im:
            image=im.convert('RGB')
            batch=transform({'dataset_name':b'panda_grasp_v1',
                'action':normalize(r['action'],statistics)[None,:],
                'observation':{'image_primary':np.asarray(image)[None,...]},
                'task':{'language_instruction':r['instruction'].encode()}})
            inference=processor(action_prompt(r['instruction']),image)
        ids=inference['input_ids'][0]
        if ids[-1]!=29871:ids=torch.cat((ids,ids.new_tensor([29871])))
        prefix=batch['input_ids'][batch['labels']==-100]
        checked.append(dict(episode_id=e['id'],split=e['split'],stage=r['stage'],index=r['index'],
            token_prefix_equal=torch.equal(prefix,ids),
            pixels_equal=torch.equal(batch['pixel_values'],inference['pixel_values'][0]),
            supervised_tokens=int((batch['labels']!=-100).sum())))
    prompt_checks=[]
    for instruction in instructions:
        builder=PurePromptBuilder('openvla')
        builder.add_turn('human',f'What action should the robot take to {instruction.lower()}?')
        prompt_checks.append(processor.tokenizer(builder.get_prompt()).input_ids==
                             processor.tokenizer(action_prompt(instruction)).input_ids)
    scene_hash=hashlib.sha256(b''.join((ROOT/n).read_bytes() for n in
        ('scene.xml','mujoco_menagerie/franka_emika_panda/panda.xml'))).hexdigest()
    scene_matches=sum(json.loads((e['path']/'episode.json').read_text())['scene_xml_sha256']==scene_hash
                      for e in episodes)
    stage_counts=Counter(r['stage'] for e in train for r in e['rows'])
    all_actions=np.asarray([r['action'] for e in train for r in e['rows']])
    mapping_error=max(float(np.max(np.abs(from_controller_action(to_controller_action(a))-a)))
                      for a in all_actions)
    references=[(e,r) for e in train for r in e['rows']
                if r['stage'] in ('approach','approach_via_offset','lower_to_grasp','close')]
    reference_offsets=np.asarray([np.asarray(r['control_position_before'])-r['cube_position'] for e,r in references])
    results=[]
    for rollout in rollouts:
        startup=json.loads((rollout/'startup_info.json').read_text())
        summary=json.loads((rollout/'summary.json').read_text())
        rows=summary['steps'];matches=[]
        for step in (1,10,20,50,100):
            r=next((r for r in rows if r['step']==step),None)
            if r is None:continue
            state=np.load(rollout/f'step_{step:03d}/scene_state.npz')
            cube=state['qpos'][9:12]
            offset=np.asarray(r['before_position'])-cube
            distances=np.linalg.norm(reference_offsets-offset,axis=1)
            index=int(np.argmin(distances));e,expert=references[index]
            q1=np.asarray(r['before_quaternion_wxyz']);q2=np.asarray(expert['quaternion_before_wxyz'])
            angle=2*np.arccos(np.clip(abs(np.dot(q1,q2))/(np.linalg.norm(q1)*np.linalg.norm(q2)),0,1))
            matches.append(dict(step=step,relative_grasp_point=offset.tolist(),raw_action=r['raw_action'],
                reference_episode=e['id'],reference_index=expert['index'],reference_stage=expert['stage'],
                relative_position_distance_mm=float(distances[index]*1000),
                orientation_difference_degrees=float(np.rad2deg(angle)),
                reference_action=expert['action'],reference_instruction=expert['instruction'],
                reference_image=str(e['path']/expert['observation']),rollout_image=str(rollout/f'step_{step:03d}/before.png')))
        results.append(dict(rollout=str(rollout),adapter=startup.get('adapter_checkpoint'),
            saved_stats_match=startup.get('action_statistics')==statistics['panda_grasp_v1']['action'],
            action_contract_match=json.loads((rollout/'action_contract.json').read_text())==action_contract(),
            closed_commands=sum(r['raw_action'][6]<.5 for r in rows),
            completed_steps=summary['completed_steps'],matches=matches))
    result=dict(weights_loaded=False,cloud_requests=0,source_manifest=str(source),dataset=str(dataset),
        processor_dir=str(processor_dir),source_matches_bundle=source_matches,
        training_statistics_recomputed_match=stats_match,action_contract_matches=contract_match,
        scene_matches=scene_matches,episodes=len(episodes),stage_counts=dict(stage_counts),
        training_examples=len(all_actions),training_closed_commands=int((all_actions[:,6]<.5).sum()),
        action_mapping_max_error=mapping_error,distinct_training_instructions=len(instructions),
        short_pickup_instruction_in_training='pick up the red cube' in instructions,
        all_instruction_prompts_equal=all(prompt_checks),checked_inputs=checked,rollouts=results,
        reference_matching_limit='Nearest training approach/grasp position relative to cube only; different orientation, instruction, joints and appearance can remain. Reference action is not ground truth for a rollout state.')
    output.mkdir(parents=True,exist_ok=False)
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    render_report(result,output)
    print(f'Saved local audit: {output}',flush=True)
    return result


def render_report(result,output):
    lines=['# Local Panda v2 pipeline and stalled-state audit','',
        'No model weights loaded, no generated model predictions, no cloud access and no robot movement.','',
        '| Check | Result |','| --- | --- |',
        f"| Source fingerprints match prepared bundle | {result['source_matches_bundle']} |",
        f"| Recomputed training-only statistics | {result['training_statistics_recomputed_match']} |",
        f"| Dataset action contract | {result['action_contract_matches']} |",
        f"| Source scenes match current scene | {result['scene_matches']}/{result['episodes']} |",
        f"| Sampled pixel tensors match | {sum(r['pixels_equal'] for r in result['checked_inputs'])}/{len(result['checked_inputs'])} |",
        f"| Sampled prompt token prefixes match | {sum(r['token_prefix_equal'] for r in result['checked_inputs'])}/{len(result['checked_inputs'])} |",
        f"| All training instruction prompts match | {result['all_instruction_prompts_equal']} ({result['distinct_training_instructions']} instructions) |",
        f"| Controller mapping max absolute label difference | {result['action_mapping_max_error']:.3g} |",
        f"| Exact short pickup instruction in training | {result['short_pickup_instruction_in_training']} |",'','## Training stage coverage','',
        '| Stage | Examples | Fraction |','| --- | --- | --- |']
    for stage,count in result['stage_counts'].items():lines.append(f"| {stage} | {count} | {100*count/result['training_examples']:.2f}% |")
    lines+=['',f"Closed-gripper target commands: {result['training_closed_commands']}/{result['training_examples']}. Closed targets also occur during lift/transfer, not only the close stage.",
        '', '## Saved rollouts','']
    for run in result['rollouts']:
        lines += [f"### {Path(run['rollout']).name}",'',f"Adapter: `{run['adapter']}`.",
            f"Normalization matches v2: {run['saved_stats_match']}. Action contract matches: {run['action_contract_match']}. Closed commands: {run['closed_commands']}/{run['completed_steps']}.",'',
            '| Step | Nearest expert stage | Position difference mm | Orientation difference degrees | Reference action dx/dy/dz mm | Model dx/dy/dz mm |',
            '| --- | --- | --- | --- | --- | --- |']
        for m in run['matches']:
            ref=', '.join(f'{1000*x:.2f}' for x in m['reference_action'][:3])
            live=', '.join(f'{1000*x:.2f}' for x in m['raw_action'][:3])
            lines.append(f"| {m['step']} | {m['reference_stage']} | {m['relative_position_distance_mm']:.2f} | {m['orientation_difference_degrees']:.2f} | {ref} | {live} |")
    lines+=['', '## Interpretation and limits','',
        'The saved trials used the short instruction "pick up the red cube". The training demonstrations use longer pick-and-place instructions including a destination. This is an instruction-distribution difference; it does not establish that the shorter instruction caused failure.',
        result['reference_matching_limit'],
        'The next GPU diagnostic should generate predictions on exact recorded train/validation examples with their original instructions, independently of teacher forcing. Compare approach, close and first-lift outputs against recorded targets. Also compare the short instruction with a recorded full instruction on the identical image. No such new predictions were made by this audit.','']
    lines+=['Visual comparison: [scene_comparison.png](scene_comparison.png). The position-nearest demonstration is a reference, not the correct action for the live state.','']
    (output/'AUDIT_REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    fig,axes=plt.subplots(len(result['rollouts']),3,figsize=(15,4*len(result['rollouts'])),squeeze=False)
    for axes_row,run in zip(axes,result['rollouts']):
        match=run['matches'][-1]
        episode=Path(match['reference_image']).parent.parent
        close=next(json.loads(line) for line in (episode/'transitions.jsonl').read_text().splitlines()
                   if json.loads(line)['stage']=='close')
        panels=[(match['rollout_image'],f"{Path(run['adapter']).name}: step {match['step']}"),
                (match['reference_image'],'Position-nearest demo (different state)'),
                (episode/close['observation'],'Successful demo: before close command')]
        for ax,(path,title) in zip(axes_row,panels):
            with Image.open(path) as im:ax.imshow(np.asarray(im))
            ax.set_title(title,fontsize=10);ax.axis('off')
    fig.tight_layout();fig.savefig(output/'scene_comparison.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'datasets/panda_pick_place/panda_broad_20261011/combined_manifest.json')
    parser.add_argument('--dataset',type=Path,default=ROOT/'datasets/panda_pick_place/training_v2')
    parser.add_argument('--processor',type=Path,required=True)
    parser.add_argument('--rollout',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.source,args.dataset,args.processor,args.rollout,args.output)
