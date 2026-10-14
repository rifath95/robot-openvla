"""Build portable TFDS/RLDS episodes, with fixed splits and training-only stats."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from simulation.panda_actions import CONVENTION_ID, action_contract
from data_tools.prepare_panda_dataset import fingerprint


def load_episodes(manifest, *, verify_images=True, training_only=False):
    if verify_images:
        from PIL import Image
    entries = json.loads(Path(manifest).read_text())['episodes']
    episodes, seen = [], set()
    for entry in entries:
        path = Path(entry['episode']).resolve()
        if path in seen or entry['split'] not in ('train', 'validation', 'test') or not entry['passed']:
            raise ValueError('Duplicate, invalid split, or unsuccessful episode')
        seen.add(path)
        metadata = json.loads((path/'episode.json').read_text())
        scenario = metadata.get('scenario') or {}
        if scenario.get('id',entry['id']) != entry['id'] or scenario.get('split',entry['split']) != entry['split']:
            raise ValueError('Source manifest changes the predeclared episode identity or split')
        replay = json.loads((path/'replay.json').read_text())
        contract = metadata['action_contract']
        if (contract != action_contract(camera=metadata.get('camera', 'workspace_camera')) or not metadata['success']
                or not metadata['training_eligible'] or not replay['passed']
                or replay['episode_sha256'] != fingerprint(path)):
            raise ValueError(f'Unverified or incompatible episode: {path}')
        if episodes and contract != episodes[0]['action_contract']:
            raise ValueError('All episodes in a training bundle must use the same camera/action contract')
        rows = [json.loads(line) for line in (path/'transitions.jsonl').read_text().splitlines()]
        if len(rows) != metadata['transition_count'] or len(rows) != replay['replayed_transitions']:
            raise ValueError('Transition count mismatch')
        for i, row in enumerate(rows, 1):
            if row['index'] != i or not row['executed'] or row['instruction'] != metadata['instruction']:
                raise ValueError('Invalid transition')
            image = (path/row['observation']).resolve()
            if not image.is_relative_to(path):
                raise ValueError('Image outside episode')
            if verify_images:
                with Image.open(image) as im:
                    if im.mode != 'RGB' or im.size != (640, 480):
                        raise ValueError('Unexpected camera image')
        episodes.append(dict(id=entry['id'], split=entry['split'], path=path, rows=rows,
                             action_contract=contract,
                             fingerprint=replay['episode_sha256'],
                             cohort=entry.get('cohort', 'legacy'),
                             combination=(metadata.get('scenario') or {}).get('combination')))
    if len({ep['id'] for ep in episodes}) != len(episodes):
        raise ValueError('Episode IDs must be unique')
    if not training_only and any(not any(ep['split'] == split for ep in episodes) for split in ('train','validation','test')):
        raise ValueError('All three splits are required')
    combinations = {}
    for episode in episodes:
        if episode['combination'] is not None:
            combination = tuple(episode['combination'])
            previous = combinations.setdefault(combination, episode['split'])
            if previous != episode['split']:
                raise ValueError('Broad starting-pose combination crosses dataset splits')
    if training_only:
        for episode in episodes:
            episode['source_split']=episode['split'];episode['split']='train'
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


def build(manifest, output, *, training_only=False):
    os.environ.setdefault('TFDS_DISABLE_GCS','1')
    import tensorflow as tf
    import tensorflow_datasets as tfds
    episodes=load_episodes(manifest,training_only=training_only)
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
            return {split:self._generate_examples(split) for split in (('train',) if training_only else ('train','validation','test'))}

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
        builder_directory='rlds/panda_pick_place/1.0.0',action_contract=episodes[0]['action_contract'],
        normalization='Training-only q01/q99, clipped to [-1,1]; gripper stays [0,1].',
        source_manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        episodes=[dict(id=e['id'],split=e['split'],transitions=len(e['rows']),sha256=e['fingerprint'],
                       cohort=e['cohort'], combination=e['combination'],source_split=e.get('source_split',e['split'])) for e in episodes],
        split_counts={s:dict(episodes=sum(e['split']==s for e in episodes),
            transitions=sum(len(e['rows']) for e in episodes if e['split']==s)) for s in ('train','validation','test')},
        training_executed=False,training_only=training_only)
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    verify(output)
    return report


def write_package_report(output):
    output = Path(output)
    report = json.loads((output / 'manifest.json').read_text())
    verification = json.loads((output / 'verification.json').read_text())
    counts = verification['split_counts']
    input_check = json.loads((output/'input_check.json').read_text()) if (output/'input_check.json').exists() else {}
    table = ['# Training dataset package', '',
             f'Bundle: **{output.name}**. Dataset schema: **{report["dataset_name"]} / 1.0.0**.', '',
             '| Split | Trajectories | Valid action examples |', '|---|---:|---:|']
    table.extend(f'| {s} | {counts[s]["episodes"]} | {counts[s]["transitions"]} |'
                 for s in ('train', 'validation', 'test'))
    table.extend(['', 'RGB PNG images are embedded in TFRecords. Instructions remain text and '
                  'actions remain seven float32 values; tokenization happens during training.', '',
                  'Each trajectory has one additional terminal after-image with no valid outgoing '
                  'action. Terminal records are excluded from the example counts above.', '',
                  'Normalization statistics use training episodes only. Source episode hashes, '
                  'cohort labels and broad scene combinations are recorded in manifest.json.', '',
                  f'Portable export/count checks passed: **{verification["passed"]}**.',
                  f'All source images/actions/instructions checked: '
                  f'**{input_check.get("all_source_images_actions_instructions_match", False)}**.',
                  f'Official processor batches checked: **{bool(input_check.get("official_processor_batches"))}**.', '',
                  'See verification.json and input_check.json for detailed checks; SHA256SUMS '
                  'covers the upload files. No cloud upload or training was performed by packaging.', ''])
    (output/'PACKAGE_REPORT.md').write_text('\n'.join(table))


def write_upload_checksums(output):
    output = Path(output)
    files=sorted(p for p in output.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (output/'SHA256SUMS').write_text(''.join(
        hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(output).as_posix()+'\n' for p in files))


def verify(output):
    import tensorflow_datasets as tfds
    output=Path(output)
    report=json.loads((output/'manifest.json').read_text())
    stats=json.loads((output/'dataset_statistics.json').read_text())
    builder=tfds.builder_from_directory(str(output/report['builder_directory']))
    seen=set(); counts={}
    for split in ('train','validation','test'):
        n=0;steps=0
        if split not in builder.info.splits:
            counts[split]=dict(episodes=0,transitions=0);continue
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
    write_package_report(output)
    write_upload_checksums(output)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('datasets/panda_pick_place/panda_expansion_20261010_104720/combined_manifest.json'))
    p.add_argument('--output-dir',type=Path,default=Path('datasets/panda_pick_place/training_v1'))
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--training-only',action='store_true',help='Explicitly place every source episode into training, with no held-out splits')
    args=p.parse_args()
    report=verify(args.output_dir) if args.verify_only else build(args.manifest,args.output_dir,training_only=args.training_only)
    print(json.dumps(dict(output_dir=str(args.output_dir),split_counts=report['split_counts'],
                         training_executed=False),indent=2))
