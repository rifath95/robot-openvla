"""Build portable TFDS/RLDS episodes, with fixed splits and training-only stats."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from panda_actions import CONVENTION_ID, action_contract
from prepare_panda_dataset import fingerprint


def load_episodes(manifest):
    entries = json.loads(Path(manifest).read_text())['episodes']
    episodes, seen = [], set()
    for entry in entries:
        path = Path(entry['episode']).resolve()
        if path in seen or entry['split'] not in ('train', 'validation', 'test') or not entry['passed']:
            raise ValueError('Duplicate, invalid split, or unsuccessful episode')
        seen.add(path)
        metadata = json.loads((path/'episode.json').read_text())
        replay = json.loads((path/'replay.json').read_text())
        if (metadata['action_contract'] != action_contract() or not metadata['success']
                or not metadata['training_eligible'] or not replay['passed']
                or replay['episode_sha256'] != fingerprint(path)):
            raise ValueError(f'Unverified or incompatible episode: {path}')
        rows = [json.loads(line) for line in (path/'transitions.jsonl').read_text().splitlines()]
        if len(rows) != metadata['transition_count'] or len(rows) != replay['replayed_transitions']:
            raise ValueError('Transition count mismatch')
        for i, row in enumerate(rows, 1):
            if row['index'] != i or not row['executed'] or row['instruction'] != metadata['instruction']:
                raise ValueError('Invalid transition')
            image = (path/row['observation']).resolve()
            if not image.is_relative_to(path):
                raise ValueError('Image outside episode')
            with Image.open(image) as im:
                if im.mode != 'RGB' or im.size != (640, 480):
                    raise ValueError('Unexpected camera image')
        episodes.append(dict(id=entry['id'], split=entry['split'], path=path, rows=rows,
                             fingerprint=replay['episode_sha256']))
    if len({ep['id'] for ep in episodes}) != len(episodes):
        raise ValueError('Episode IDs must be unique')
    if any(not any(ep['split'] == split for ep in episodes) for split in ('train','validation','test')):
        raise ValueError('All three splits are required')
    return episodes


def training_statistics(episodes):
    actions = np.array([r['action'] for e in episodes if e['split']=='train' for r in e['rows']])
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError('Expected finite seven-dimensional actions')
    stats = {k:fn(actions, axis=0).tolist() for k,fn in
             [('mean',np.mean),('std',np.std),('min',np.min),('max',np.max)]}
    lo, hi = np.quantile(actions,.01,axis=0), np.quantile(actions,.99,axis=0)
    # Degenerate dimensions map to zero in training and a constant in inference.
    # Keep mask=True so the model's ordinary unnormalizer restores that constant.
    constant = (hi-lo < 1e-8)
    lo[constant] = hi[constant] = np.median(actions,axis=0)[constant]
    stats.update(q01=lo.tolist(), q99=hi.tolist(), mask=[True]*6+[False])
    return {CONVENTION_ID:dict(action=stats, num_transitions=len(actions),
        num_trajectories=sum(e['split']=='train' for e in episodes))}


def normalize(action, statistics):
    s = statistics[CONVENTION_ID]['action']
    x,lo,hi,mask = [np.asarray(v) for v in (action,s['q01'],s['q99'],s['mask'])]
    span=hi-lo
    scaled = 2*(x-lo)/np.where(span>1e-8,span,1)-1
    scaled = np.where(span>1e-8,scaled,0)
    return np.where(mask,np.clip(scaled,-1,1),x).astype(np.float32)


def build(manifest, output):
    os.environ.setdefault('TFDS_DISABLE_GCS','1')
    import tensorflow as tf
    import tensorflow_datasets as tfds
    episodes=load_episodes(manifest)
    output=Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a new output directory; do not overwrite a prepared dataset')
    statistics=training_statistics(episodes)

    class PandaPickPlace(tfds.core.GeneratorBasedBuilder):
        VERSION=tfds.core.Version('1.0.0')

        def _info(self):
            return tfds.core.DatasetInfo(builder=self,description='Replay-verified Panda pick-and-place expert episodes.',features=tfds.features.FeaturesDict({
                'episode_id':tfds.features.Text(),
                'steps':tfds.features.Dataset({
                    'observation':{'image':tfds.features.Image(shape=(480,640,3),encoding_format='png')},
                    'action':tfds.features.Tensor(shape=(7,),dtype=np.float32),
                    'language_instruction':tfds.features.Text(),
                    'is_first':np.bool_, 'is_last':np.bool_, 'is_terminal':np.bool_,
                    'is_action_valid':np.bool_,
                    'reward':np.float32, 'discount':np.float32,
                })}))

        def _split_generators(self, dl_manager):
            return {split:self._generate_examples(split) for split in ('train','validation','test')}

        def _generate_examples(self, split):
            for episode in episodes:
                if episode['split']!=split: continue
                steps=[]
                for i,row in enumerate(episode['rows']):
                    last=i==len(episode['rows'])-1
                    steps.append(dict(observation={'image':str(episode['path']/row['observation'])},
                        action=np.asarray(row['action'],np.float32), language_instruction=row['instruction'],
                        is_first=i==0,is_last=False,is_terminal=False,is_action_valid=True,
                        reward=float(last),discount=float(not last)))
                # RLDS terminal observations have no valid outgoing action. Keep
                # all demonstrated actions and append the final after-image.
                steps.append(dict(observation={'image':str(episode['path']/episode['rows'][-1]['next_observation'])},
                    action=np.zeros(7,np.float32),language_instruction=episode['rows'][-1]['instruction'],
                    is_first=False,is_last=True,is_terminal=True,is_action_valid=False,reward=0.,discount=0.))
                yield episode['id'],dict(episode_id=episode['id'],steps=steps)

    tf.config.set_visible_devices([], 'GPU')
    builder=PandaPickPlace(data_dir=str(output/'rlds'))
    builder.download_and_prepare(download_config=tfds.download.DownloadConfig(try_download_gcs=False))
    (output/'dataset_statistics.json').write_text(json.dumps(statistics,indent=2)+'\n')
    report=dict(schema_version=1, dataset_name='panda_pick_place', action_key=CONVENTION_ID,
        builder_directory='rlds/panda_pick_place/1.0.0',action_contract=action_contract(),
        normalization='Training-only q01/q99, clipped to [-1,1]; gripper stays [0,1].',
        episodes=[dict(id=e['id'],split=e['split'],transitions=len(e['rows']),sha256=e['fingerprint']) for e in episodes],
        split_counts={s:dict(episodes=sum(e['split']==s for e in episodes),
            transitions=sum(len(e['rows']) for e in episodes if e['split']==s)) for s in ('train','validation','test')},
        training_executed=False)
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    verify(output)
    return report


def verify(output):
    import tensorflow_datasets as tfds
    output=Path(output)
    report=json.loads((output/'manifest.json').read_text())
    stats=json.loads((output/'dataset_statistics.json').read_text())
    builder=tfds.builder_from_directory(str(output/report['builder_directory']))
    seen=set(); counts={}
    for split in ('train','validation','test'):
        n=0;steps=0
        for ep in tfds.as_numpy(builder.as_dataset(split=split,shuffle_files=False)):
            key=ep['episode_id'].decode()
            if key in seen: raise ValueError('Cross-split episode leakage')
            seen.add(key);n+=1
            rows=list(ep['steps'])
            for i,row in enumerate(rows):
                if row['is_first']!=(i==0) or row['is_last']!=(i==len(rows)-1):
                    raise ValueError('Invalid episode boundary')
                if row['observation']['image'].shape!=(480,640,3) or not row['language_instruction']:
                    raise ValueError('Image/instruction mismatch')
                if not np.isfinite(normalize(row['action'],stats)).all():
                    raise ValueError('Invalid normalization')
            if rows[-1]['is_action_valid'] or not rows[-1]['is_terminal']:
                raise ValueError('Terminal observation must have no outgoing training action')
            steps+=sum(bool(row['is_action_valid']) for row in rows)
        counts[split]=dict(episodes=n,transitions=steps)
    if counts!=report['split_counts']: raise ValueError('Export count mismatch')
    result=dict(passed=True,split_counts=counts, portable=True)
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    files=sorted(p for p in output.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (output/'SHA256SUMS').write_text(''.join(
        hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(output).as_posix()+'\n' for p in files))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('datasets/panda_pick_place/panda_expansion_20261010_104720/combined_manifest.json'))
    p.add_argument('--output-dir',type=Path,default=Path('datasets/panda_pick_place/training_v1'))
    p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    report=verify(args.output_dir) if args.verify_only else build(args.manifest,args.output_dir)
    print(json.dumps(dict(output_dir=str(args.output_dir),split_counts=report['split_counts'],
                         training_executed=False),indent=2))
