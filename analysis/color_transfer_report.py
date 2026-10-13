"""Separate task success from selection accuracy and paired evidence."""
from collections import defaultdict,Counter


def metrics(rows,comparisons=1):
    correct=sum(r['outcome']=='correct_cube' for r in rows)
    wrong=sum(r['outcome']=='wrong_cube' for r in rows)
    chosen=correct+wrong
    blocks=defaultdict(int)
    for r in rows:
        blocks[r['seed']]+=1 if r['outcome']=='correct_cube' else -1 if r['outcome']=='wrong_cube' else 0
    # Exact one-sided label-swap reference distribution. Swapping the two target
    # labels reverses a scene's correct-minus-wrong score. Block by seed to keep
    # paired instructions and color-pair measurements dependent within a scene.
    distribution=Counter({0:1})
    for score in blocks.values():
        following=Counter()
        for value,count in distribution.items():
            following[value+score]+=count;following[value-score]+=count
        distribution=following
    observed=sum(blocks.values())
    p=sum(count for value,count in distribution.items() if value>=observed)/sum(distribution.values()) if blocks and chosen else None
    pairs=defaultdict(list)
    for r in rows:pairs[(r['seed'],r['color_pair'])].append(r)
    return dict(trials=len(rows),correct=correct,wrong=wrong,chosen=chosen,
        no_contact=sum(r['outcome']=='no_contact' for r in rows),
        both=sum(r['outcome']=='both_cubes' for r in rows),
        unmeasured=sum(r['outcome']=='unmeasured' for r in rows),
        overall_success=correct/len(rows) if rows else None,
        choice_accuracy=correct/chosen if chosen else None,
        paired_scenes_both_correct=sum(len(rs)==2 and all(r['outcome']=='correct_cube' for r in rs) for rs in pairs.values()),
        measured_scene_blocks=len(blocks),informative_scene_blocks=sum(score!=0 for score in blocks.values()),
        one_sided_label_swap_p=p,bonferroni_p=min(1.,p*comparisons) if p is not None else None)


def write_report(folder,plan,rows,error=None):
    groups=defaultdict(list)
    training_colors=set(plan.get('training_colors',[]))
    def model_label(checkpoint):
        return 'Pretrained base' if checkpoint==0 and plan.get('include_base_model') else str(checkpoint)
    for row in rows:
        groups[(row['checkpoint'],'all pairs')].append(row)
        groups[(row['checkpoint'],row['color_pair'])].append(row)
        if training_colors:
            colors=set(row['color_pair'].split(':'))
            category='seen pairs' if colors<=training_colors else 'unseen pairs' if not colors.intersection(training_colors) else 'mixed pairs'
            groups[(row['checkpoint'],category)].append(row)
    comparisons=len(plan['checkpoints'])*(len(plan['color_pairs'])+(3 if training_colors else 1))
    percent=lambda x:'—' if x is None else f'{100*x:.1f}%'
    number=lambda x:'—' if x is None else f'{x:.4f}'
    lines=['# Four-color model comparison' if training_colors else '# Unseen-color transfer evaluation','',
        'Sampling off. Both instructions from identical starting images. Color-to-position assignment alternates across scenes.',
        'Overall success counts all trials. Choice accuracy excludes neither/both/unmeasured contacts; inspect those counts before interpreting it.',
        'Greater than 50% alone is not proof. Label-swap p values assume exchangeability under a color-unrelated choice null, block by scene seed, and are exploratory. Bonferroni adjustment covers the planned checkpoint/pair and aggregate comparisons. Do not pool checkpoints as independent trials.',
        'These tests do not establish causal attribution to pretraining without an appropriate baseline.','',
        '| Checkpoint | Pair | Trials | Correct | Wrong | Neither | Both | Unmeasured | Overall success | Choice accuracy | Both targets correct in scene | Label-swap p | Adjusted p |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    import json
    measures=[]
    for (checkpoint,pair),rs in sorted(groups.items()):
        m=metrics(rs,comparisons);measures.append(dict(checkpoint=checkpoint,color_pair=pair,**m))
        lines.append(f'| {model_label(checkpoint)} | {pair} | {m["trials"]} | {m["correct"]} | {m["wrong"]} | '
            f'{m["no_contact"]} | {m["both"]} | {m["unmeasured"]} | {percent(m["overall_success"])} | '
            f'{percent(m["choice_accuracy"])} | {m["paired_scenes_both_correct"]} | '
            f'{number(m["one_sided_label_swap_p"])} | {number(m["bonferroni_p"])} |')
    if training_colors:lines+=['','Seen pairs use colors present in fine-tuning. Unseen pairs use reserved colors. Keep their aggregate results separate.',
        'Pretrained base uses unchanged pretrained neural weights with the same robot action-normalization statistics as the adapters; no LoRA weights are applied.']
    lines+=['',f'Checkpoint preparation and trial wall time: {plan.get("elapsed_seconds",0):.1f} seconds.',
        '', '| Checkpoint | Pair | Seed | Request | Outcome | Phase | Details |','|---|---|---:|---|---|---|---|']
    for r in rows:lines.append(f'| {model_label(r["checkpoint"])} | {r["color_pair"]} | {r["seed"]} | {r["target_color"]} | '
        f'{r["outcome"]} | {r["phase"]} | [Run]({r["directory"]}/RUN_REPORT.md) |')
    if error:lines+=['',f'Launcher ended early: {error}']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (folder/'statistics.json').write_text(json.dumps(measures,indent=2)+'\n')
