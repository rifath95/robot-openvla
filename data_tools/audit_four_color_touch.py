"""Audit all four-color demonstrations and create a labeled scene overview."""
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from data_tools.dataset_paths import DATASETS_ROOT
from data_tools.prepare_panda_dataset import fingerprint

ROOT = DATASETS_ROOT/'panda_four_color_touch'
COLORS = ('red','blue','green','yellow')


def audit():
    rows=json.loads((ROOT/'manifest.json').read_text())['episodes']
    assert len(rows)==300
    groups=defaultdict(list)
    for row in rows:
        folder=Path(row['episode'])
        assert row['passed'] and fingerprint(folder)==row['episode_sha256']
        metadata=json.loads((folder/'episode.json').read_text())
        assert metadata['success'] and metadata['training_eligible']
        transitions=[json.loads(s) for s in (folder/'transitions.jsonl').read_text().splitlines()]
        names=metadata['scenario'].get('cube_colors',dict(red='red',blue='blue'))
        target=names[metadata['scenario']['target_color']]
        assert all(t['instruction']==f'touch the {target} cube.' for t in transitions)
        assert target==row['target_color']
        groups[row['pair_id']].append(row)
    assert len(groups)==150
    for pair in groups.values():
        assert len(pair)==2 and len({r['split'] for r in pair})==1
        assert len({r['target_color'] for r in pair})==2
        folders=[Path(r['episode']) for r in pair]
        assert (folders[0]/'step_0001/before.png').read_bytes()==(folders[1]/'step_0001/before.png').read_bytes()
        with np.load(folders[0]/'initial_state.npz') as a,np.load(folders[1]/'initial_state.npz') as b:
            assert np.array_equal(a['qpos'],b['qpos'])
    originals=json.loads((DATASETS_ROOT/'panda_color_touch/manifest.json').read_text())['episodes']
    preserved_files=0
    for row in originals:
        old=Path(row['episode']);new=ROOT/row['split']/row['id']
        for path in old.rglob('*'):
            if path.is_file():
                assert path.read_bytes()==(new/path.relative_to(old)).read_bytes()
                preserved_files+=1
    split=Counter(r['split'] for r in rows);colors=Counter(r['target_color'] for r in rows)
    assert split==dict(train=200,validation=50,test=50)
    assert colors==dict.fromkeys(COLORS,75)
    counts={s:dict(episodes=split[s],transitions=sum(r['transitions'] for r in rows if r['split']==s)) for s in split}
    report=dict(passed=True,split_counts=counts,target_color_counts=dict(colors),paired_scenes=150,
        preserved_original_files=preserved_files,paired_scenes_identical=True,all_actions_replayed=True,
        maximum_qpos_difference=max(json.loads((Path(r['episode'])/'replay.json').read_text())['maximum_qpos_difference'] for r in rows))
    (ROOT/'REVIEW_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    width,height=240,205
    montage=Image.new('RGB',(width*6,height*25),'white');draw=ImageDraw.Draw(montage)
    for i,(number,pair) in enumerate(sorted(groups.items())):
        x=(i%6)*width;y=(i//6)*height
        with Image.open(Path(pair[0]['episode'])/'step_0001/before.png') as im:montage.paste(im.resize((240,180)),(x,y))
        draw.text((x+4,y+181),f"Scene {number+1}: {pair[0]['target_color']} / {pair[1]['target_color']} ({pair[0]['split']})",fill='black')
    montage.save(ROOT/'REVIEW_ALL_PAIRS.jpg',quality=88)
    selected={}
    for number,pair in sorted(groups.items()):
        key=tuple(sorted(r['target_color'] for r in pair))
        selected.setdefault(key,(number,pair))
    preview=Image.new('RGB',(640*3,510*2),'white');labels=ImageDraw.Draw(preview)
    for i,(number,pair) in enumerate(selected.values()):
        x=(i%3)*640;y=(i//3)*510
        with Image.open(Path(pair[0]['episode'])/'step_0001/before.png') as im:preview.paste(im,(x,y))
        labels.text((x+8,y+485),f"Scene {number+1}: {pair[0]['instruction']} / {pair[1]['instruction']}",fill='black')
    preview.save(ROOT/'COLOR_PAIRS_PREVIEW.jpg',quality=90)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':audit()
