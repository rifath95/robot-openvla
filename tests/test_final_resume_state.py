"""Actual AdamW and shuffled TF input continuation, plus frozen sweep profile."""
import copy,json,os,random,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from training.resume_state import save_state,restore_state,verify_state
from training.panda_rlds import PandaRLDSDataset

class FinalResumeStateTests(unittest.TestCase):
    def test_optimizer_rng_and_shuffled_stream_continue_identically(self):
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        def data():
            d=PandaRLDSDataset.__new__(PandaRLDSDataset)
            options=tf.data.Options();options.threading.private_threadpool_size=1
            d.dataset=tf.data.Dataset.range(100).shuffle(16,seed=42).repeat().with_options(options)
            d._tf_iterator=iter(d.dataset)
            return d
        def step(m,o,d):
            sample=float(next(d._tf_iterator).numpy())/100
            noise=random.random()+np.random.random()+torch.rand(()).item()
            x=torch.tensor([[sample,noise]],dtype=torch.float32)
            loss=m(x).square().mean();loss.backward();o.step();o.zero_grad()
        m=torch.nn.Linear(2,1);o=torch.optim.AdamW(m.parameters(),lr=.001);d=data()
        for _ in range(12):step(m,o,d)
        cfg=SimpleNamespace(batch_size=1,grad_accumulation_steps=1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);saved=copy.deepcopy(m.state_dict());save_state(root,m,o,d,12,cfg)
            self.assertEqual(verify_state(root,12)['update'],12)
            step(m,o,d);expected=copy.deepcopy(m.state_dict());next_sample=next(d._tf_iterator).numpy()
            m2=torch.nn.Linear(2,1);m2.load_state_dict(saved);o2=torch.optim.AdamW(m2.parameters(),lr=.001);d2=data()
            # Restore lazily like the real dataset __iter__ implementation.
            with patch.dict(os.environ,{'PANDA_INITIAL_UPDATE':'12'}):restore_state(root,m2,o2,d2,cfg)
            tf.train.Checkpoint(iterator=d2._tf_iterator).read(d2._restore_prefix).assert_consumed()
            step(m2,o2,d2)
            for k,v in expected.items():torch.testing.assert_close(v,m2.state_dict()[k],rtol=0,atol=0)
            self.assertEqual(next_sample,next(d2._tf_iterator).numpy())
            with self.assertRaises(ValueError):verify_state(root,13)
            with (root/'training_state/state.pt').open('ab') as f:f.write(b'corrupt')
            with self.assertRaises(ValueError):verify_state(root,12)

    def test_relocated_tfrecord_iterator_uses_safe_staging_alias(self):
        import tensorflow as tf
        from training.resume_state import ensure_input_alias
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'staged/old/bundle';new=root/'current';old.mkdir(parents=True);new.mkdir()
            for i in range(2):
                with tf.io.TFRecordWriter(str(old/f'{i}.tf')) as writer:
                    for k in range(20):writer.write(str(i*20+k).encode())
            def stream(folder):
                options=tf.data.Options();options.threading.private_threadpool_size=1
                files=tf.data.Dataset.from_tensor_slices([str(folder/'0.tf'),str(folder/'1.tf')]).shuffle(2,seed=7)
                return files.interleave(lambda f:tf.data.TFRecordDataset(f),cycle_length=2).shuffle(5,seed=7).repeat().with_options(options)
            iterator=iter(stream(old))
            for _ in range(8):next(iterator)
            tf.train.Checkpoint(iterator=iterator).write(str(root/'input'))
            expected=[next(iterator).numpy() for _ in range(25)]
            import shutil
            for f in old.iterdir():shutil.copy(f,new/f.name)
            shutil.rmtree(old)
            with patch('training.resume_state.INPUT_CACHE_ROOTS',(root/'staged',)):
                ensure_input_alias(str(old),new)
            other=iter(stream(new));tf.train.Checkpoint(iterator=other).read(str(root/'input')).assert_consumed()
            self.assertEqual(expected,[next(other).numpy() for _ in range(25)])
            with self.assertRaises(ValueError):ensure_input_alias('/unrelated/location',new)

    def test_final_metadata_is_not_published_when_state_save_fails(self):
        from training.runtime import write_adapter_files
        m=SimpleNamespace(module=SimpleNamespace(save_pretrained=lambda folder:None))
        processor=SimpleNamespace(save_pretrained=lambda folder:None)
        dataset=SimpleNamespace(dataset_statistics={});cfg=SimpleNamespace(batch_size=1,grad_accumulation_steps=1)
        with tempfile.TemporaryDirectory() as tmp,patch('training.resume_state.save_state',side_effect=OSError('disk full')):
            folder=Path(tmp)/'checkpoint_000010'
            with patch.dict(os.environ,{'PANDA_SAVE_FINAL_STATE':'1','PANDA_INITIAL_UPDATE':'0'}):
                with self.assertRaises(OSError):write_adapter_files(folder,m,processor,cfg,dataset,10,10,optimizer=object(),final=True)
            self.assertFalse((folder/'training_config.json').exists())

    def test_frozen_uninterrupted_plan_and_profile(self):
        from cloud.bowl_chain_plan import make_uninterrupted_plan,make_colored_plan,validate_saved
        plan=make_uninterrupted_plan('panda-train-color-bowl-test')
        self.assertEqual(plan['groups'],make_colored_plan('panda-train-color-bowl-test')['groups'])
        self.assertEqual(plan['checkpoints'],[6000,8000,10000,12000])
        self.assertEqual(len(validate_saved(dict(plan=plan,trials=[]))),800)
        damaged=copy.deepcopy(plan);damaged['checkpoints']=[1000,2000,4000,6000]
        with self.assertRaises(ValueError):validate_saved(dict(plan=damaged,trials=[]))
        profile=json.loads(Path('training/color_place_bowl_600_uninterrupted_12000.json').read_text())
        self.assertEqual(profile['checkpoint_updates'],plan['checkpoints']);self.assertEqual(profile['max_steps'],12000)
        self.assertTrue(profile['save_final_training_state'])

    def test_generated_final_save_includes_optimizer_and_baseline_does_not(self):
        from training.integrate_upstream import integrated_source
        s=integrated_source()
        self.assertIn('optimizer=optimizer, final=finished',s)
        self.assertIn('save_adapter(vla, processor, cfg, vla_dataset, 0)',s)
        self.assertLess(s.index('training_iterator = iter(dataloader)'),s.index('restore_if_requested(vla'))
        self.assertLess(s.index('restore_if_requested(vla'),s.index('enumerate(training_iterator)'))

if __name__=='__main__':unittest.main()
