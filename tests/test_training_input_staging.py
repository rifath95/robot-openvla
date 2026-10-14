"""Container training uses verified copies without moving persistent sources or outputs."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from training.input_staging import stage_inputs

class InputStagingTests(unittest.TestCase):
    def sources(self,root):
        model=root/'volume/base';dataset=root/'volume/training_place_bowl_v1'
        model.mkdir(parents=True);dataset.mkdir()
        (model/'config.json').write_text('{}');(model/'model.safetensors').write_bytes(b'base-weights')
        (dataset/'verification.json').write_text('{"passed":true}')
        (dataset/'data.tfrecord').write_bytes(b'images-instructions-actions')
        checksums=''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in sorted(dataset.iterdir()))
        (dataset/'SHA256SUMS').write_text(checksums)
        return model,dataset

    def test_copy_verified_and_sources_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,dataset=self.sources(root)
            local_model,local_dataset,receipt=stage_inputs(model,dataset,root/'container/session')
            self.assertEqual(local_dataset.name,dataset.name)
            self.assertEqual((local_model/'model.safetensors').read_bytes(),(model/'model.safetensors').read_bytes())
            self.assertEqual((local_dataset/'data.tfrecord').read_bytes(),(dataset/'data.tfrecord').read_bytes())
            self.assertTrue(model.exists() and dataset.exists())
            self.assertEqual(receipt['sources']['base'],str(model.resolve()))
            stage_inputs(model,dataset,root/'container/session')
            (local_dataset/'data.tfrecord').write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError,'corrupted'):stage_inputs(model,dataset,root/'container/session')

    def test_launcher_reads_local_inputs_but_keeps_output_root_persistent(self):
        import io
        import os
        from training import train_panda
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,dataset=self.sources(root)
            config=json.loads(Path('training/smoke.json').read_text());config['stage_inputs_on_container']=True
            profile=root/'profile.json';profile.write_text(json.dumps(config))
            local_model=root/'container/base';local_dataset=root/'container'/dataset.name
            receipt=dict(sources=dict(base=str(model),dataset=str(dataset)))
            outputs=root/'volume/sessions/test/runs'
            argv=['train','--dataset',str(dataset),'--model',str(model),'--config',str(profile),
                '--output-root',str(outputs),'--run-name','training','--batch-policy','config']
            with patch('sys.argv',argv),patch('training.input_staging.stage_inputs',return_value=(local_model,local_dataset,receipt)), \
                 patch('training.fetch_upstream.fetch'),patch('training.train_panda.subprocess.run') as run, \
                 patch('sys.stdout',new_callable=io.StringIO):
                train_panda.main()
            command=run.call_args.args[0];env=run.call_args.kwargs['env']
            self.assertEqual(command[command.index('--vla_path')+1],str(local_model.resolve()))
            self.assertEqual(command[command.index('--data_root_dir')+1],str(local_dataset.resolve()))
            self.assertEqual(env['PANDA_RUN_DIR'],str((outputs/'training').resolve()))
            self.assertEqual(env['PANDA_SOURCE_MODEL'],str(model.resolve()))
            self.assertTrue((outputs/'training/input_staging.json').is_file())
            self.assertNotIn('--stage_inputs_on_container',command)

    def test_bad_source_checksum_never_publishes_ready_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,dataset=self.sources(root);(dataset/'data.tfrecord').write_bytes(b'corrupt')
            destination=root/'container/session'
            with self.assertRaisesRegex(ValueError,'Checksum failed'):stage_inputs(model,dataset,destination)
            self.assertFalse((destination/'staging.json').exists())

    def test_insufficient_local_space_and_overlap_rejected(self):
        from collections import namedtuple
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,dataset=self.sources(root)
            with self.assertRaisesRegex(ValueError,'separate container'):stage_inputs(model,dataset,model/'cache')
            Usage=namedtuple('Usage','total used free')
            with patch('training.input_staging.shutil.disk_usage',return_value=Usage(1024,1000,24)):
                with self.assertRaisesRegex(ValueError,'Container disk'):stage_inputs(model,dataset,root/'container/session')
            self.assertFalse((root/'container/session/staging.json').exists())

if __name__=='__main__':unittest.main()
