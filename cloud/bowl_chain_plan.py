"""Frozen 1,000-trial bowl placement plan; pure validation also used on restoration."""
import math
from simulation.color_place_bowl_scene import positions,validate_layout,slots
GROUPS=(('single_seen',['red'],20),('single_unseen',['black'],20),
        ('two_red_present',['red','black'],20),('two_red_absent',['black','purple'],20),
        ('four_red_present',['red','green','black','purple'],10),('four_red_absent',['green','black','purple','orange'],10))
CHECKPOINTS=[0,500,1000,2000,3000]
COLORED_CHECKPOINTS=[1000,2000,4000,6000]
COLORED_VARIANT='colored_bowl_600_v1'
CONTINUATION_VARIANT='colored_bowl_600_continue_12000_v1'
CONTINUATION_CHECKPOINTS=[8000,10000,12000]
UNINTERRUPTED_VARIANT='colored_bowl_600_uninterrupted_12000_v1'
UNINTERRUPTED_CHECKPOINTS=[6000,8000,10000,12000]


def timeout_budget(estimate=25*3600):
    if type(estimate) not in (int,float) or not math.isfinite(estimate) or estimate<=0:raise ValueError('Invalid estimate')
    return dict(estimated_seconds=math.ceil(estimate),inner_buffer_seconds=3*3600,outer_buffer_seconds=5*3600,
                inner_seconds=math.ceil(estimate)+3*3600,outer_seconds=math.ceil(estimate)+8*3600)


def validate_layout_copy(actual,expected):
    if set(actual)!=set(expected) or set(actual.get('cubes',{}))!=set(expected['cubes']):raise ValueError('Saved layout differs')
    for a,b in [(actual['bowl'],expected['bowl'])]+[(actual['cubes'][s],xy) for s,xy in expected['cubes'].items()]:
        if len(a)!=len(b) or any(type(x) not in (int,float) or not math.isfinite(x) or abs(x-y)>1e-12 for x,y in zip(a,b)):raise ValueError('Saved layout differs')


def make_plan(session):
    groups=[dict(name=name,colors=list(colors),scene_seeds=list(range(1000,1000+n)),
                 layouts=[positions(s,len(colors)) for s in range(1000,1000+n)]) for name,colors,n in GROUPS]
    return dict(task='multi_place_bowl',training_session=session,include_base_model=True,action_statistics_checkpoint=500,
                checkpoints=list(CHECKPOINTS),sampling=False,steps=120,groups=groups,
                timeout_budget=timeout_budget(),estimated_seconds=25*3600,elapsed_seconds=0)


def make_colored_plan(session):
    import json
    from common.paths import PROJECT_ROOT
    profile=json.loads((PROJECT_ROOT/'inference/sweep_profiles/color_place_bowl_600.json').read_text())
    plan=make_plan(session)
    plan.update(variant=COLORED_VARIANT,groups=profile['groups'],checkpoints=list(COLORED_CHECKPOINTS),include_base_model=False,
        action_statistics_checkpoint=1000,training_colors=['red','blue','green','yellow'],
        historical_baseline=profile['historical_baseline'],layout_sha256=profile['layout_sha256'],
        estimated_seconds=20*3600,timeout_budget=timeout_budget(20*3600))
    return plan


def make_continuation_plan(session,source_session):
    plan=make_colored_plan(session)
    plan.update(variant=CONTINUATION_VARIANT,checkpoints=list(CONTINUATION_CHECKPOINTS),
        action_statistics_checkpoint=8000,initial_training_session=source_session,
        initial_checkpoint_update=6000,additional_updates=6000,
        continuation_mode='adapter_weights_fresh_optimizer',
        estimated_seconds=22*3600,timeout_budget=timeout_budget(22*3600))
    return plan


def make_uninterrupted_plan(session):
    plan=make_colored_plan(session)
    plan.update(variant=UNINTERRUPTED_VARIANT,checkpoints=list(UNINTERRUPTED_CHECKPOINTS),
        action_statistics_checkpoint=6000,estimated_seconds=12*3600,timeout_budget=timeout_budget(12*3600))
    return plan


def trial_specs(plan):
    for cp in plan['checkpoints']:
        for g in plan['groups']:
            colors=g['colors'];count=len(colors)
            for index,(seed,layout) in enumerate(zip(g['scene_seeds'],g['layouts'])):
                offset=index%count;rotated=colors[offset:]+colors[:offset]
                for color in colors:
                    label='base_model' if cp==0 else f'checkpoint_{cp:06d}'
                    yield dict(checkpoint=cp,seed=seed,target_color=color,cube_colors=rotated,
                        cube_count=count,color_set=g['name'],color_pair=None,layout=layout,
                        directory=f'{label}_{g["name"]}_{count}c_seed_{seed}_{color}')


def validate_saved(payload):
    plan=payload['plan'];rows=payload['trials']
    uninterrupted=plan.get('variant')==UNINTERRUPTED_VARIANT
    continuation=plan.get('variant')==CONTINUATION_VARIANT
    colored=plan.get('variant') in (COLORED_VARIANT,CONTINUATION_VARIANT,UNINTERRUPTED_VARIANT)
    if plan.get('variant') not in (None,COLORED_VARIANT,CONTINUATION_VARIANT,UNINTERRUPTED_VARIANT):raise ValueError('Unknown bowl sweep variant')
    expected_checkpoints=UNINTERRUPTED_CHECKPOINTS if uninterrupted else CONTINUATION_CHECKPOINTS if continuation else COLORED_CHECKPOINTS if colored else CHECKPOINTS
    if (plan.get('checkpoints')!=expected_checkpoints
        or plan.get('include_base_model') is not (False if colored else True)
        or plan.get('sampling') is not False or plan.get('steps')!=120):raise ValueError('Altered chain sweep')
    if plan.get('action_statistics_checkpoint')!=(6000 if uninterrupted else 8000 if continuation else 1000 if colored else 500):raise ValueError('Wrong base normalization reference')
    if colored:
        import hashlib,json
        from common.paths import PROJECT_ROOT
        profile=json.loads((PROJECT_ROOT/'inference/sweep_profiles/color_place_bowl_600.json').read_text())
        sha=hashlib.sha256(json.dumps(plan['groups'],sort_keys=True).encode()).hexdigest()
        if (sha!=profile['layout_sha256'] or plan.get('layout_sha256')!=sha or plan.get('historical_baseline')!=profile['historical_baseline']
            or plan.get('training_colors')!=['red','blue','green','yellow']):raise ValueError('Changed comparison layouts or baseline provenance')
    import re
    if continuation and (plan.get('initial_checkpoint_update')!=6000 or plan.get('additional_updates')!=6000
        or plan.get('continuation_mode')!='adapter_weights_fresh_optimizer'
        or not re.fullmatch('panda-train-color-bowl-[A-Za-z0-9_-]+',plan.get('initial_training_session',''))):raise ValueError('Invalid continuation provenance')
    if not re.fullmatch('panda-train-[A-Za-z0-9_-]+',plan.get('training_session','')):raise ValueError('Invalid session')
    if len(plan['groups'])!=len(GROUPS):raise ValueError('Wrong group inventory')
    for g,(name,colors,n) in zip(plan['groups'],GROUPS):
        if g['name']!=name or g['colors']!=colors or g['scene_seeds']!=list(range(1000,1000+n)) or len(g['layouts'])!=n:raise ValueError('Altered trial groups')
        for seed,layout in zip(g['scene_seeds'],g['layouts']):
            validate_layout(layout,len(colors));validate_layout_copy(layout,positions(seed,len(colors)))
    specs={s['directory']:s for s in trial_specs(plan)}
    if len(specs)!=(600 if continuation else 800 if colored else 1000) or len({r['directory'] for r in rows})!=len(rows):raise ValueError('Invalid trial inventory')
    for r in rows:
        if r['directory'] not in specs or any(r.get(k)!=specs[r['directory']][k] for k in ('checkpoint','seed','target_color','cube_colors','cube_count','color_set')):raise ValueError('Trial differs from plan')
    return specs
