"""Create a fresh cloud sweep from one reviewable profile, without connecting."""

from common.output_layout import output_path
import json
from pathlib import Path
from datetime import datetime,timezone
import uuid
from common.paths import PROJECT_ROOT
from cloud.sweep_plan import validate_saved,trial_specs


def create(profile):
    config=json.loads(Path(profile).read_text())
    plan=dict(config,task=config.get('task','color_transfer'),sampling=False,elapsed_seconds=0)
    known=json.loads((PROJECT_ROOT/config['dataset_manifest']).read_text())['episodes']
    if plan['task']=='multi_color_touch':
        from simulation.multi_color_scene import positions
        plan['multi_cube_positions']={str(n):[positions(seed,n) for seed in plan['scene_seeds']] for n in plan['cube_counts']}
        layouts=[v for group in plan['multi_cube_positions'].values() for v in group]
    else:
        from data_tools.collect_panda_color_touch import pair_spec
        plan['cube_positions']=[pair_spec(seed)['cube_positions'] for seed in plan['scene_seeds']]
        layouts=plan['cube_positions']
    for layout in layouts:
        for episode in known:
            for xy in layout.values():
                if any(sum((a-b)**2 for a,b in zip(xy,old))<1e-12 for old in episode['cube_positions'].values()):
                    raise ValueError('Evaluation position appears in the collected dataset')
    payload=dict(plan=plan,trials=[],launcher_error=None);validate_saved(payload)
    prefix='scripted_multi_color_models' if plan['task']=='multi_color_touch' else 'scripted_four_color_models'
    name=f'{prefix}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}'
    folder=output_path(name,root=PROJECT_ROOT/'outputs');folder.mkdir(parents=True)
    from cloud.scripted_touch_session import write_reports
    write_reports(folder,plan,[])
    print(f'Prepared {len(list(trial_specs(plan)))} trials in {folder}',flush=True)
    return folder
