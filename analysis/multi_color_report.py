"""Multi-cube selection reports: task success, target familiarity and confusion."""
from collections import defaultdict,Counter
import json


def metrics(rows):
    counts=Counter(r['outcome'] for r in rows)
    correct=counts['correct_cube'];wrong=counts['wrong_cube'];chosen=correct+wrong
    return dict(trials=len(rows),correct=correct,wrong=wrong,no_contact=counts['no_contact'],
        multiple=counts['multiple_cubes'],unmeasured=counts['unmeasured'],
        clean_correct=sum(r['outcome']=='correct_cube' and r['phase']=='Completed' for r in rows),
        success_rate=correct/len(rows) if rows else None,choice_accuracy=correct/chosen if chosen else None)


def write_report(folder,plan,rows,error=None):
    groups=defaultdict(list);seen=set(plan['training_colors'])
    for r in rows:
        cp,n,palette=r['checkpoint'],r['cube_count'],r['color_set']
        for category in ('all targets','seen targets' if r['target_color'] in seen else 'unseen targets'):
            groups[cp,n,palette,category].append(r)
    label=lambda cp:'Pretrained base' if cp==0 and plan.get('include_base_model') else str(cp)
    pct=lambda x:'—' if x is None else f'{100*x:.1f}%'
    lines=['# Multi-cube color selection sweep','',
        'Every visible color is requested separately from the same initial scene. Sampling off; no training occurs.',
        'Success counts all trials. Choice accuracy counts only single-cube contacts; no-contact and multiple-contact outcomes remain separate.',
        'The color-independent uniform-choice reference is 1/N: 25% for four cubes, 20% for five, 16.7% for six. This is not a formal significance test; trials share scenes and checkpoints.',
        'Seen/unseen labels refer to the requested target color, including scenes with a mixture of target and distractor familiarity.','',
        '| Model | Cubes | Palette | Targets | Trials | Correct | Wrong | No contact | Multiple | Unmeasured | Clean correct | Success | Choice accuracy | Chance reference |',
        '|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    measures=[]
    for (cp,n,palette,category),rs in sorted(groups.items()):
        m=metrics(rs);measures.append(dict(checkpoint=cp,cube_count=n,color_set=palette,target_category=category,**m))
        lines.append(f'| {label(cp)} | {n} | {palette} | {category} | {m["trials"]} | {m["correct"]} | {m["wrong"]} | {m["no_contact"]} | {m["multiple"]} | {m["unmeasured"]} | {m["clean_correct"]} | {pct(m["success_rate"])} | {pct(m["choice_accuracy"])} | {pct(1/n)} |')
    scenes=defaultdict(list)
    for r in rows:scenes[r['checkpoint'],r['cube_count'],r['color_set'],r['seed']].append(r)
    lines+=['','## Complete-scene instruction switching','',
        'A scene counts only when all N target instructions have saved trials. This tests whether the model can select every cube by changing the instruction.',
        '', '| Model | Cubes | Palette | Fully tested scenes | All targets correct |','|---|---:|---|---:|---:|']
    scene_groups=defaultdict(list)
    for (cp,n,palette,seed),rs in scenes.items():
        if len(rs)==n and len({r['target_color'] for r in rs})==n:scene_groups[cp,n,palette].append(rs)
    for (cp,n,palette),sets in sorted(scene_groups.items()):
        lines.append(f'| {label(cp)} | {n} | {palette} | {len(sets)} | {sum(all(r["outcome"]=="correct_cube" for r in rs) for rs in sets)} |')
    confusion=Counter((r['checkpoint'],r['cube_count'],r['target_color'],r.get('first_contact')) for r in rows if r.get('first_contact') not in (None,'multiple'))
    lines+=['','## Requested → first contacted color','', '| Model | Cubes | Requested | First contacted | Trials |','|---|---:|---|---|---:|']
    for (cp,n,target,touched),number in sorted(confusion.items()):lines.append(f'| {label(cp)} | {n} | {target} | {touched} | {number} |')
    lines+=['',f'Preparation and trial wall time: {plan.get("elapsed_seconds",0):.1f} seconds.','',
        '| Model | Cubes | Palette | Seed | Request | Outcome | Phase | Details |','|---|---:|---|---:|---|---|---|---|']
    for r in rows:lines.append(f'| {label(r["checkpoint"])} | {r["cube_count"]} | {r["color_set"]} | {r["seed"]} | {r["target_color"]} | {r["outcome"]} | {r["phase"]} | [Run]({r["directory"]}/RUN_REPORT.md) |')
    if error:lines+=['',f'Launcher ended early: {error}']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n');(folder/'statistics.json').write_text(json.dumps(measures,indent=2)+'\n')
