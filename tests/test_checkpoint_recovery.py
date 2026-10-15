"""ENOSPC checkpoint rescue and last-resort global delivery, without a real pod."""
import errno
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock, MagicMock, patch
from types import SimpleNamespace

from cloud.archive_sweep import digest
from cloud.checkpoint_ready import check_ready
from cloud.checkpoint_recovery import recover_checkpoints


def fixture_checkpoint(folder,update):
    folder.mkdir(parents=True)
    for name in ('adapter_config.json','dataset_statistics.json','processor_config.json','preprocessor_config.json'):
        (folder/name).write_text('{}')
    (folder/'training_config.json').write_text(json.dumps(dict(checkpoint_update=update,
        dataset_provenance=dict(checksums_sha256='data',dataset_revision='revision'))))
    header=json.dumps(dict(weight=dict(dtype='F32',shape=[1],data_offsets=[0,4]))).encode()
    (folder/'adapter_model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'1234')
    check_ready(folder)


def config(root):
    session=root/'session';local=root/'local'
    session.mkdir();local.mkdir()
    return dict(session=str(session),local=str(local),training_run=str(session/'runs/training'),
        checkpoint_recovery=True,checkpoint_recovery_dir=str(local/'checkpoint-recovery'),
        recovery_checkpoint_updates=[8000,10000,12000],global_volume=str(root/'global'),
        checkpoint_repo='owner/weights',sweep_repo='owner/sweeps',sweep=str(local/'no-sweep'),
        checksums_sha256='data',dataset_revision='revision')


class CheckpointRecoveryTests(unittest.TestCase):
    def test_unscheduled_final_checkpoint_is_also_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=config(Path(tmp));cfg['initial_update']=6000
            fixture_checkpoint(Path(cfg['training_run'])/'checkpoint_009123',9123)
            result=recover_checkpoints(cfg)
            self.assertEqual([r['update'] for r in result['global_copies']],[9123])

    def test_failed_hf_upload_copies_complete_volume_checkpoints_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=config(Path(tmp));primary=Path(cfg['training_run'])
            fixture_checkpoint(primary/'checkpoint_008000',8000)
            (primary/'checkpoint_010000').mkdir()
            (primary/'checkpoint_010000/adapter_model.safetensors').write_bytes(b'incomplete')
            (Path(cfg['session'])/'checkpoint_hf_receipt.json').write_text('{"verified":false}')
            result=recover_checkpoints(cfg)
            self.assertTrue(result['verified'])
            self.assertEqual([r['update'] for r in result['global_copies']],[8000])
            self.assertEqual(result['unavailable_updates'],[10000,12000])
            target=Path(result['global_copies'][0]['destination'])
            check_ready(target)
            self.assertEqual(digest(target/'adapter_model.safetensors'),digest(primary/'checkpoint_008000/adapter_model.safetensors'))
            self.assertTrue((target/'RECOVERY_COMPLETE.json').is_file())
            self.assertTrue((primary/'checkpoint_008000').is_dir())

    def test_verified_hf_checkpoint_is_not_copied_to_global(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=config(Path(tmp));primary=Path(cfg['training_run'])
            records=[]
            for cp in cfg['recovery_checkpoint_updates']:
                folder=primary/f'checkpoint_{cp:06d}';fixture_checkpoint(folder,cp)
                for p in folder.iterdir():
                    records.append(dict(path='runs/training/'+folder.name+'/'+p.name,bytes=p.stat().st_size,sha256=digest(p)))
            receipt=dict(verified=True,repo_id=cfg['checkpoint_repo'],revision='verified-commit',
                path_in_repo='sessions/session',files=records)
            (Path(cfg['session'])/'checkpoint_hf_receipt.json').write_text(json.dumps(receipt))
            with patch('cloud.checkpoint_recovery.recover_to_global') as copy:
                result=recover_checkpoints(cfg)
                copy.assert_not_called()
            self.assertEqual(result['hf_verified_updates'],[8000,10000,12000])
            self.assertFalse(Path(cfg['global_volume']).exists())

    def test_local_checkpoint_rescue_survives_full_volume_report_writes(self):
        from cloud.bowl_chain import recovery
        with tempfile.TemporaryDirectory() as tmp:
            cfg=config(Path(tmp))
            fixture_checkpoint(Path(cfg['checkpoint_recovery_dir'])/'checkpoint_008000',8000)
            original=Path.write_text
            def full_volume(path,*args,**kwargs):
                if path.is_relative_to(Path(cfg['session'])):raise OSError(errno.ENOSPC,'Volume full')
                return original(path,*args,**kwargs)
            with patch.object(Path,'write_text',full_volume):
                self.assertEqual(recovery(cfg),0)
            local=Path(cfg['local'])/'checkpoint_global_recovery_receipt.json'
            result=json.loads(local.read_text())
            self.assertTrue(result['verified'])
            self.assertTrue(Path(result['global_copies'][0]['destination']).is_dir())

    def test_global_failure_never_claims_verified_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=config(Path(tmp));fixture_checkpoint(Path(cfg['training_run'])/'checkpoint_008000',8000)
            with patch('cloud.checkpoint_recovery.recover_to_global',side_effect=OSError('Global volume unavailable')):
                with self.assertRaises(RuntimeError):recover_checkpoints(cfg)
            result=json.loads((Path(cfg['local'])/'checkpoint_global_recovery_receipt.json').read_text())
            self.assertFalse(result['verified'])
            self.assertEqual(result['errors'][0]['update'],8000)

    def test_verified_sweep_does_not_skip_checkpoint_recovery(self):
        from cloud.bowl_chain import recovery
        with tempfile.TemporaryDirectory() as tmp:
            cfg=config(Path(tmp));fixture_checkpoint(Path(cfg['training_run'])/'checkpoint_008000',8000)
            sweep=Path(cfg['sweep']);sweep.mkdir();(sweep/'results.json').write_text('{}')
            receipt=dict(verified=True,repo_id=cfg['sweep_repo'],path_in_repo=sweep.name,
                source_revision=digest(sweep/'results.json'))
            (Path(cfg['session'])/'sweep_hf_receipt.json').write_text(json.dumps(receipt))
            with patch('cloud.chain_archive.prepare_recovery_archive') as pack:
                self.assertEqual(recovery(cfg),0)
                pack.assert_not_called()
            result=json.loads((Path(cfg['local'])/'checkpoint_global_recovery_receipt.json').read_text())
            self.assertEqual([r['update'] for r in result['global_copies']],[8000])

    def test_supervisor_recovers_even_when_volume_timing_finalization_fails(self):
        from cloud.bowl_chain import supervise
        with tempfile.TemporaryDirectory() as tmp:
            cfg=config(Path(tmp))
            cfg.update(pod_id='pod',pod_key='secret',hf_token='private-token',code_commit='commit',
                timeout_budget=dict(inner_seconds=100,outer_seconds=1000))
            guard=Mock(pid=100,stdin=Mock());worker=Mock(pid=200,stdin=Mock());recovery=Mock(pid=300,stdin=Mock())
            worker.wait.return_value=1;worker.poll.return_value=1;recovery.wait.return_value=0
            from training.detached_run import write_status
            def full_final_status(path,value):
                if Path(path)==Path(cfg['session'])/'status.json' and value.get('phase')=='failed':
                    raise OSError(errno.ENOSPC,'No space for final status')
                return write_status(path,value)
            with patch('cloud.bowl_chain.subprocess.Popen',side_effect=[guard,worker,recovery]), \
                patch('cloud.bowl_chain.interrupted_timings',side_effect=OSError(errno.ENOSPC,'Full volume')), \
                patch('cloud.bowl_chain.write_status',side_effect=full_final_status), \
                patch('cloud.bowl_chain.stop_request',return_value=True) as stop:
                self.assertEqual(supervise(cfg),1)
                stop.assert_called_once()
            payload=json.loads(recovery.stdin.write.call_args.args[0])
            self.assertTrue(payload['checkpoint_recovery'])
            self.assertNotIn('hf_token',payload)
            self.assertNotIn('pod_key',payload)
            saved=json.loads((Path(cfg['local'])/'status.json').read_text())
            self.assertEqual(saved['recovery_exit_code'],0)

    def test_volume_log_failure_does_not_kill_checkpoint_rescue(self):
        from cloud.color_bowl_training import train
        class FullStream:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def write(self,*args):raise OSError(errno.ENOSPC,'Full volume')
            def close(self):pass
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);volume=root/'volume';volume.mkdir();local=root/'training.log'
            original=Path.open
            def open_path(path,*args,**kwargs):
                return FullStream() if path.parent==volume else original(path,*args,**kwargs)
            process=Mock();process.stdout=MagicMock()
            process.stdout.__iter__.return_value=iter(['Checkpoint rescued locally\n'])
            process.wait.return_value=0;process.poll.return_value=0
            with patch.object(Path,'open',open_path),patch('cloud.color_bowl_training.subprocess.Popen',return_value=process), \
                patch('cloud.bowl_chain.kill_group') as kill:
                train(['fake-trainer'],volume/'training.log',{},local_log=local)
                kill.assert_not_called()
            self.assertIn('Checkpoint rescued locally',local.read_text())

    def test_real_adapter_is_reserialized_locally_after_enospc(self):
        import torch
        from peft import LoraConfig,get_peft_model,PeftModel
        from tests.test_training_continuation import TinyModel
        from training.runtime import save_adapter,write_adapter_files
        torch.manual_seed(23)
        base=TinyModel();base_state={k:v.clone() for k,v in base.state_dict().items()}
        model=get_peft_model(base,LoraConfig(r=2,target_modules='all-linear'))
        x=torch.ones(2,4)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        model(x).square().sum().backward();optimizer.step();model.eval()
        expected=model(x).detach()
        with tempfile.TemporaryDirectory() as tmp:
            cfg=config(Path(tmp));primary=Path(cfg['training_run']);primary.mkdir(parents=True)
            env=dict(PANDA_INITIAL_UPDATE='6000',PANDA_INITIAL_ADAPTER='parent',PANDA_INPUT_STAGING='',
                PANDA_CHECKPOINT_RECOVERY_DIR=cfg['checkpoint_recovery_dir'],PANDA_SAVE_CHECKPOINTS='1',
                PANDA_PROVENANCE=json.dumps(dict(checksums_sha256='data',dataset_revision='revision',base_model_volume_path='/base')))
            def write(folder,*args):
                if folder.is_relative_to(primary):
                    folder.mkdir();(folder/'adapter_model.safetensors').write_bytes(b'partial')
                    raise OSError(errno.ENOSPC,'No space left')
                return write_adapter_files(folder,*args)
            processor=Mock()
            def processor_save(folder):
                for name in ('processor_config.json','preprocessor_config.json'):(folder/name).write_text('{}')
            processor.save_pretrained.side_effect=processor_save
            metrics=SimpleNamespace(directory=primary,model_details={},plot=Mock())
            with patch.dict(os.environ,env),patch('training.runtime.metrics',metrics),patch('training.runtime.write_adapter_files',side_effect=write):
                with self.assertRaisesRegex(RuntimeError,'complete container copy'):
                    save_adapter(SimpleNamespace(module=model),processor,SimpleNamespace(max_steps=6000),
                        SimpleNamespace(dataset_statistics={}),2000)
            with self.assertRaises(ValueError):check_ready(primary/'checkpoint_008000')
            result=recover_checkpoints(cfg)
            target=Path(result['global_copies'][0]['destination'])
            check_ready(target)
            fresh=TinyModel();fresh.load_state_dict(base_state)
            restored=PeftModel.from_pretrained(fresh,str(target));restored.eval()
            torch.testing.assert_close(restored(x),expected)
            self.assertTrue(json.loads((Path(cfg['checkpoint_recovery_dir'])/'RECOVERY_REQUEST.json').read_text())['complete'])


if __name__=='__main__':unittest.main()
