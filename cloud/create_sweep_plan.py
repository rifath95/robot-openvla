"""Create a fresh cloud sweep from one reviewable profile, without connecting."""
import json
from pathlib import Path
from datetime import datetime,timezone
import uuid
from common.paths import PROJECT_ROOT
from cloud.sweep_plan import validate_saved,trial_specs


def create(profile):
    config=json.loads(Path(profile).read_text())
    from data_tools.collect_panda_color_touch import pair_spec
    plan=dict(config,task='color_transfer',sampling=False,elapsed_seconds=0)
    known=json.loads((PROJECT_ROOT/config['dataset_manifest']).read_text())['episodes']
    plan['cube_positions']=[pair_spec(seed)['cube_positions'] for seed in plan['scene_seeds']]
    for positions in plan['cube_positions']:
        for episode in known:
            for xy in positions.values():
                if any(sum((a-b)**2 for a,b in zip(xy,old))<1e-12 for old in episode['cube_positions'].values()):
                    raise ValueError('Evaluation position appears in the collected dataset')
    payload=dict(plan=plan,trials=[],launcher_error=None);validate_saved(payload)
    name=f'scripted_four_color_models_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}'
    folder=PROJECT_ROOT/'outputs'/name;folder.mkdir(parents=True)
    from cloud.scripted_touch_session import write_reports
    write_reports(folder,plan,[])
    print(f'Prepared {len(list(trial_specs(plan)))} trials in {folder}',flush=True)
    return folder
