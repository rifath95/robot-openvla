"""Pinned initial matrices, zero B validation, provenance and train command."""
import hashlib,json,os,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import torch
from safetensors.torch import save_file
from training.initial_adapter import stage_initial_adapter

class OriginalInitializationTests(unittest.TestCase):
    def fixture(self,root,*,bad_b=False,bad_update=False):
        source=root/'source';source.mkdir()
        tensors={'base.layer.lora_A.weight':torch.randn(2,4),'base.layer.lora_B.weight':torch.ones(3,2) if bad_b else torch.zeros(3,2)}
        save_file(tensors,source/'adapter_model.safetensors')
        values={'adapter_config.json':dict(peft_type='LORA',r=2,lora_dropout=0.),
            'training_config.json':dict(checkpoint_update=1 if bad_update else 0,dataset_provenance=dict(checksums_sha256='sha',dataset_revision='rev')),
            'dataset_statistics.json':{},'processor_config.json':{},'preprocessor_config.json':{}}
        for name,value in values.items():(source/name).write_text(json.dumps(value))
        cfg=root/'cfg.json';cfg.write_text(json.dumps(dict(lora_rank=2,lora_dropout=0.)))
        entries=[dict(path=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in source.iterdir()]
        spec=dict(checkpoint_update=0,path_in_repo='sessions/original/checkpoint_000000',files=entries,repo_id='owner/checkpoints',revision='pinned',source_training_session='original')
        profile=dict(training_config=str(cfg),checksums_sha256='sha',dataset_revision='rev')
        def download(repo,filename,**kw):
            self.assertEqual(kw['revision'],'pinned');self.assertEqual(kw['token'],'PRIVATE')
            target=Path(kw['local_dir'])/filename;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/Path(filename).name,target)
            return str(target)
        return source,tensors,spec,profile,download

    def test_initial_weights_are_identical_and_source_is_pinned(self):
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,tensors,spec,profile,download=self.fixture(root)
            with patch('huggingface_hub.hf_hub_download',side_effect=download):
                folder,identity=stage_initial_adapter(spec,root,profile,token='PRIVATE')
            restored=load_file(folder/'adapter_model.safetensors')
            for k,v in tensors.items():torch.testing.assert_close(v,restored[k],rtol=0,atol=0)
            self.assertTrue(identity['exact_initial_adapter']);self.assertEqual(identity['zero_b_matrices_verified'],1)

    def test_nonzero_b_wrong_update_and_corrupt_inventory_rejected(self):
        for damage in ('b','update','hash'):
            with self.subTest(damage=damage),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source,tensors,spec,profile,download=self.fixture(root,bad_b=damage=='b',bad_update=damage=='update')
                if damage=='hash':spec['files'][0]['sha256']='wrong'
                with patch('huggingface_hub.hf_hub_download',side_effect=download),self.assertRaises(ValueError):
                    stage_initial_adapter(spec,root,profile,token='PRIVATE')

    def test_training_worker_stages_original_adapter_and_passes_base_receipt(self):
        from cloud.color_bowl_training import worker
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);session=root/'session';local=root/'local';session.mkdir();local.mkdir();adapter=local/'initial';adapter.mkdir()
            cfg=dict(session=str(session),local=str(local),training_config='training/color_place_bowl_600_uninterrupted_12000.json',
                hf_token='PRIVATE',checkpoint_repo='owner/checkpoints',dataset_repo='owner/data',dataset_revision='rev',checksums_sha256='sha',
                dataset_bundle='bundle',dataset=str(local/'data'),base=str(local/'base'),base_volume='/workspace/openvla-7b',
                training_run=str(session/'runs/training'),started_at_utc='now',code_commit='code',timeout_budget={},initial_adapter={'checkpoint_update':0})
            api=Mock();api.repo_info.return_value.private=True;api.upload_folder.return_value.oid='commit'
            completion=dict(training_complete=True,last_update=12000,training_update_seconds=1,checkpoint_seconds=1)
            def train(command,log,env):
                self.assertEqual(command[command.index('--adapter')+1],str(adapter))
                self.assertEqual(command[command.index('--base-staging-receipt')+1],str(session/'model_staging.json'))
            def backup(*args):(session/'checkpoint_hf_receipt.json').write_text('{"verified":true}')
            with patch('huggingface_hub.HfApi',return_value=api),patch('shutil.disk_usage',return_value=type('D',(),{'free':100*1024**3})()),patch('cloud.bowl_chain_worker.download_dataset'),patch('cloud.bowl_chain_worker.stage_base',return_value={}),patch('training.initial_adapter.stage_initial_adapter',return_value=(adapter,{'exact_initial_adapter':True})) as initial,patch('cloud.color_bowl_training.train',side_effect=train),patch('cloud.color_bowl_training.verify_computation',return_value=(completion,[6000,8000,10000,12000])),patch('training.resume_state.verify_state'),patch('cloud.chain_archive.backup_training',side_effect=backup),patch('cloud.chain_archive.verify_files'):
                self.assertEqual(worker(cfg),0)
            self.assertEqual(initial.call_args.kwargs['token'],'PRIVATE')
            provenance=json.loads((session/'provenance.json').read_text());self.assertEqual(provenance['initialization_mode'],'restored_initial_lora')

    def test_saved_initial_adapter_is_loaded_trainable_without_reinitialization(self):
        from training.continuation import attach_lora,initialization_mode
        with patch.dict(os.environ,{'PANDA_INITIAL_ADAPTER':'original-zero','PANDA_INITIAL_ADAPTER_UPDATE':'0','PANDA_RESUME_CHECKPOINT':''}),patch('peft.PeftModel.from_pretrained',return_value='loaded') as load,patch('peft.get_peft_model') as random_init:
            self.assertEqual(attach_lora('base','config'),'loaded');load.assert_called_once_with('base','original-zero',is_trainable=True)
            random_init.assert_not_called();self.assertEqual(initialization_mode(),'restored_initial_lora')

if __name__=='__main__':unittest.main()
