"""Fixed resumable trial plans and timeout budgets for simulator sweeps."""
import json
import math
from pathlib import Path


def timeout_budget(estimated_seconds):
    if not math.isfinite(estimated_seconds) or estimated_seconds<=0:raise ValueError('Estimate must be positive')
    inner=math.ceil(estimated_seconds)+3600
    return dict(estimated_seconds=math.ceil(estimated_seconds),inner_seconds=inner,outer_seconds=inner+3600)


def trial_specs(plan):
    task=plan['task']
    if task not in ('touch','color_touch','color_transfer'):raise ValueError('Unsupported scripted task')
    for checkpoint in plan['checkpoints']:
        if task=='color_transfer':
            pairs=plan['color_pairs']
        else:pairs=[['red','blue']] if task=='color_touch' else [['red']]
        for pair_index,pair in enumerate(pairs):
            for index,seed in enumerate(plan['scene_seeds']):
                colors=pair[::-1] if (index+pair_index)%2 else pair
                for target in pair:
                    suffix=f'_{target}' if task!='touch' else ''
                    if task=='color_transfer':suffix=f'_{"-".join(pair)}'+suffix
                    prefix='base_model' if checkpoint==0 and plan.get('include_base_model') else f'checkpoint_{checkpoint:06d}'
                    name=f'{prefix}_seed_{seed}{suffix}'
                    yield dict(checkpoint=checkpoint,seed=seed,target_color=target,
                        cube_colors=colors if task=='color_transfer' else None,
                        color_pair=':'.join(pair) if task=='color_transfer' else None,directory=name)


def validate_saved(payload):
    plan=payload['plan'];rows=payload['trials']
    if (plan.get('sampling') is not False or type(plan['steps']) is not int or not 1<=plan['steps']<=1000
        or not plan['training_session'].startswith('panda-train-')):raise ValueError('Invalid sweep plan')
    import re
    if not re.fullmatch(r'panda-train-[A-Za-z0-9-]+',plan['training_session']):raise ValueError('Invalid training session')
    for field in ('checkpoints','scene_seeds'):
        values=plan[field]
        if not values or len(set(values))!=len(values) or any(type(x)is not int or x<0 for x in values):raise ValueError('Invalid plan indices')
    if plan['task']=='color_transfer':
        colors={'red','blue','black','purple','green','yellow','orange','white','pink','brown','cyan'}
        pairs=plan['color_pairs']
        if not pairs or any(len(p)!=2 or len(set(p))!=2 or not set(p)<=colors for p in pairs):raise ValueError('Invalid color pairs')
    if plan.get('include_base_model') and (0 not in plan['checkpoints'] or plan.get('action_statistics_checkpoint',50) not in plan['checkpoints']):
        raise ValueError('Base comparison requires baseline index 0 and its statistics reference checkpoint')
    specs={s['directory']:s for s in trial_specs(plan)}
    if len({r['directory'] for r in rows})!=len(rows):raise ValueError('Duplicate trial records')
    for row in rows:
        spec=specs.get(row['directory'])
        if spec is None or any(row.get(k)!=spec[k] for k in ('checkpoint','seed')):raise ValueError('Trial outside saved plan')
        if plan['task']!='touch' and row.get('target_color')!=spec['target_color']:raise ValueError('Wrong trial target')
        if plan['task']=='color_transfer' and any(row.get(k)!=spec[k] for k in ('cube_colors','color_pair')):raise ValueError('Trial colors differ')
    return specs


def estimate_remaining(payload):
    """Conservative unbatched estimate: observed full-sweep wall time per trial."""
    specs=validate_saved(payload)
    remaining=len(specs)-len(payload['trials'])
    elapsed=float(payload['plan'].get('elapsed_seconds',0))
    seconds_per_trial=max(60,elapsed/max(1,len(payload['trials'])))
    # Include fresh staging, model setup, and switching overhead separately.
    checkpoints={s['checkpoint'] for name,s in specs.items() if name not in {r['directory'] for r in payload['trials']}}
    estimate=remaining*seconds_per_trial+600+120*len(checkpoints)
    return dict(remaining_trials=remaining,seconds_per_trial=seconds_per_trial,**timeout_budget(estimate))


def validate_scene_positions(actual, expected, tolerance=1e-12):
    """Reject changed scenes while accepting sub-picometre arithmetic rounding."""
    if len(actual)!=len(expected):raise ValueError('Saved scene count differs from this checkout')
    for generated,saved in zip(actual,expected):
        if set(generated)!= {'red','blue'} or set(saved)!= {'red','blue'}:raise ValueError('Invalid cube identities')
        for slot in ('red','blue'):
            if len(generated[slot])!=2 or len(saved[slot])!=2:raise ValueError('Invalid cube position')
            for a,b in zip(generated[slot],saved[slot]):
                if not math.isfinite(a) or not math.isfinite(b) or abs(a-b)>tolerance:
                    raise ValueError('Saved scene positions differ from this checkout')
