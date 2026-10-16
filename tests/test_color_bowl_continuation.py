"""Cumulative numbering, frozen 600-trial comparison and detached chain ordering."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from common.paths import PROJECT_ROOT
from cloud.bowl_chain_plan import make_colored_plan, make_continuation_plan, validate_saved, trial_specs
from training.update_numbering import validate_initial_update, cumulative_update


class ContinuationChainTests(unittest.TestCase):
    def test_supervisor_routes_the_new_chain_and_preserves_existing_jobs(self):
        from cloud.bowl_chain import worker_module
        self.assertEqual(worker_module(dict(continuation_chain=True)), 'cloud.color_bowl_continuation')
        self.assertEqual(worker_module(dict(training_only=True)), 'cloud.color_bowl_training')
        self.assertEqual(worker_module(dict(sweep_only=True)), 'cloud.color_bowl_sweep')
        self.assertEqual(worker_module({}), 'cloud.bowl_chain_worker')
        with self.assertRaises(ValueError):
            worker_module(dict(continuation_chain=True,training_only=True))

    def test_six_hundred_trial_archive_restores_with_replay_report(self):
        from cloud.archive_sweep import package
        from cloud.scripted_touch_session import write_reports
        from data_tools.restore_sweep_archive import restore
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/'outputs') as tmp:
            root=Path(tmp)
            source=root/'scripted_continue_test';source.mkdir()
            staging=root/'archive';staging.mkdir()
            plan=make_continuation_plan('panda-train-color-bowl-cont-test','panda-train-color-bowl-parent')
            rows=[]
            for spec in trial_specs(plan):
                row={k:v for k,v in spec.items() if k!='layout'}
                row.update(phase='Completed',outcome='no_lift',placement_success=False)
                trial=source/spec['directory'];trial.mkdir()
                (trial/'summary.json').write_text('{}')
                rows.append(row)
            write_reports(source,plan,rows)
            package(source,staging,dict(plan=plan,trials=rows))
            restored=restore(staging,root/'restored')
            self.assertEqual(len(json.loads((restored/'results.json').read_text())['trials']),600)
            self.assertIn('fresh optimizer',(restored/'REPORT.md').read_text())
            self.assertIn('12000',(restored/'REPORT.md').read_text())

    def test_training_verification_requires_the_whole_cumulative_range(self):
        from cloud.bowl_chain_worker import verify_training
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[dict(update=i,train_loss=1.,training_update_wall_seconds=2.) for i in range(6001,12001)]
            (root/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            for name in ('loss.png','loss_smoothed.png','experiment.json','model_info.json'):
                (root/name).write_text('fixture')
            (root/'checkpoint_timings.jsonl').write_text('{"seconds":1}\n')
            with patch('cloud.checkpoint_ready.check_ready'):
                result=verify_training(root,6000,[8000,10000,12000],initial_update=6000)
                self.assertEqual(result['last_update'],12000)
                self.assertEqual(result['additional_updates'],6000)
                rows.pop()
                (root/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaises(ValueError):
                    verify_training(root,6000,[8000,10000,12000],initial_update=6000)

    def test_exact_same_scenes_and_only_three_new_checkpoints(self):
        before = make_colored_plan('panda-train-color-bowl-parent')
        plan = make_continuation_plan('panda-train-color-bowl-cont-new', 'panda-train-color-bowl-parent')
        self.assertEqual(plan['groups'], before['groups'])
        self.assertEqual(plan['layout_sha256'], before['layout_sha256'])
        self.assertEqual(plan['checkpoints'], [8000, 10000, 12000])
        self.assertEqual(len(validate_saved(dict(plan=plan, trials=[]))), 600)
        for cp in plan['checkpoints']:
            self.assertEqual(sum(s['checkpoint']==cp for s in trial_specs(plan)), 200)
        self.assertEqual(plan['timeout_budget']['inner_seconds'], 27.5*3600)
        self.assertEqual(plan['timeout_budget']['outer_seconds'], 33*3600)
        for field, value in [('checkpoints', [6000, 8000, 12000]), ('initial_checkpoint_update', 0)]:
            bad = copy.deepcopy(plan)
            bad[field] = value
            with self.assertRaises(ValueError):
                validate_saved(dict(plan=bad, trials=[]))

    def test_initial_update_requires_the_exact_parent_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'training_config.json').write_text('{"checkpoint_update":6000}')
            self.assertEqual(validate_initial_update(root, 6000), 6000)
            self.assertEqual(validate_initial_update(None, 0), 0)
            for adapter, update in [(None, 6000), (root, 5000), (root, -1)]:
                with self.assertRaises(ValueError):
                    validate_initial_update(adapter, update)
            with patch.dict(os.environ, {'PANDA_INITIAL_UPDATE':'6000'}):
                self.assertEqual(cumulative_update(1), 6001)
                self.assertEqual(cumulative_update(2000), 8000)
                self.assertEqual(cumulative_update(6000), 12000)

    def test_staged_base_can_continue_but_changed_weights_are_rejected(self):
        from training.continuation import validate_adapter
        from cloud.archive_sweep import digest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root/'adapter'
            base = root/'base'
            adapter.mkdir()
            base.mkdir()
            (adapter/'adapter_model.safetensors').write_bytes(b'fixture')
            (adapter/'adapter_config.json').write_text('{"peft_type":"LORA","r":32,"lora_dropout":0.0}')
            (adapter/'training_config.json').write_text('{"source_model_path":"/workspace/openvla-7b"}')
            (base/'config.json').write_text('{}')
            receipt = root/'copy.json'
            receipt.write_text(json.dumps(dict(source='/workspace/openvla-7b', destination=str(base),
                files=[dict(path='config.json', bytes=2, sha256=digest(base/'config.json'))])))
            cfg = dict(lora_rank=32, lora_dropout=0.)
            with self.assertRaises(ValueError):
                validate_adapter(adapter, cfg, base)
            self.assertEqual(validate_adapter(adapter, cfg, base, staging_receipt=receipt), adapter.resolve())
            (base/'config.json').write_text('changed')
            with self.assertRaises(ValueError):
                validate_adapter(adapter, cfg, base, staging_receipt=receipt)

    def test_worker_orders_continuation_backup_sweep_and_delivery_and_aborts_on_training_failure(self):
        from cloud.color_bowl_continuation import worker
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                local, session, source = root/'local', root/'session', root/'panda-train-color-bowl-parent'
                for p in (local, session, source):p.mkdir()
                adapter = source/'checkpoint_006000'
                adapter.mkdir()
                (adapter/'adapter_model.safetensors').write_bytes(b'learned-weights')
                config = dict(session=str(session), local=str(local), sweep=str(local/'scripted_continue'),
                    source_training_session=source.name, source_adapter=str(adapter),
                    base_volume='/workspace/openvla-7b', base=str(local/'base'), training_run=str(session/'runs/training'),
                    dataset=str(local/'dataset'), dataset_repo='owner/data', dataset_revision='revision',
                    checksums_sha256='dataset-hash', dataset_bundle='training_color_place_bowl_v1_600',
                    checkpoint_repo='owner/checkpoints', sweep_repo='owner/sweeps', global_volume=str(root/'global'),
                    hf_token='secret-hf', pod_key='secret-pod', code_commit='code',
                    checkpoint_recovery=True,checkpoint_recovery_dir=str(local/'checkpoint-recovery'),
                    recovery_checkpoint_updates=[8000,10000,12000],
                    started_at_utc='2026-10-15T00:00:00+00:00', timeout_budget={})
                order = []
                def command(cmd, *args, **kwargs):
                    self.assertNotIn('HF_TOKEN', kwargs['env'])
                    self.assertNotIn('RUNPOD_API_KEY', kwargs['env'])
                    order.append('preflight' if '--preflight' in cmd else 'sweep')
                def train(cmd, log, env, **kwargs):
                    order.append('training')
                    self.assertEqual(str(cmd[cmd.index('--initial-update')+1]), '6000')
                    self.assertIn('--adapter', cmd)
                    self.assertNotIn('HF_TOKEN', env)
                    self.assertNotIn('RUNPOD_API_KEY', env)
                    self.assertEqual(env['PANDA_CHECKPOINT_RECOVERY_DIR'],config['checkpoint_recovery_dir'])
                    self.assertEqual(kwargs['local_log'],local/'training.log')
                    if fail:raise RuntimeError('training failed')
                def backup(*args):
                    self.assertEqual(args[-1], [8000,10000,12000])
                    order.append('checkpoint_backup')
                api=Mock()
                api.repo_info.return_value.private=True
                with patch('cloud.color_bowl_continuation.select_source', return_value=(source,adapter,{})), \
                    patch('huggingface_hub.HfApi', return_value=api), \
                    patch('cloud.bowl_chain_worker.global_probe'), \
                    patch('cloud.bowl_chain_worker.setup_simulator', return_value='/sim/python'), \
                    patch('cloud.bowl_chain_worker.download_dataset'), \
                    patch('cloud.bowl_chain_worker.stage_base', return_value={}), \
                    patch('cloud.checkpoint_ready.check_ready'), \
                    patch('cloud.bowl_chain_worker.run_command', side_effect=command), \
                    patch('cloud.color_bowl_training.train', side_effect=train), \
                    patch('cloud.bowl_chain_worker.verify_training', return_value=dict(last_update=12000)) as verify, \
                    patch('cloud.chain_archive.backup_training', side_effect=backup), \
                    patch('cloud.chain_archive.backup_sweep', side_effect=lambda *a:order.append('sweep_backup')), \
                    patch('cloud.color_bowl_sweep.publish_reports', side_effect=lambda *a:order.append('final_reports')), \
                    patch('cloud.color_bowl_continuation.shutil.disk_usage', return_value=SimpleNamespace(free=200*1024**3)):
                    if fail:
                        with self.assertRaises(RuntimeError):worker(config)
                        self.assertEqual(order, ['preflight','training'])
                    else:
                        worker(config)
                        self.assertEqual(order, ['preflight','training','checkpoint_backup','sweep','sweep_backup','final_reports','final_reports'])
                        verify.assert_called_once_with(config['training_run'],6000,[8000,10000,12000],initial_update=6000)
                        self.assertTrue((session/'worker_completion.json').is_file())
                self.assertEqual((adapter/'adapter_model.safetensors').read_bytes(), b'learned-weights')

    def test_real_cpu_metrics_and_save_use_cumulative_numbers_only(self):
        from training.runtime import Metrics, save_adapter
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = dict(PANDA_RUN_DIR=str(root), PANDA_UPSTREAM_REVISION='test',
                PANDA_VALIDATE_EVERY='100', PANDA_VALIDATION_MAX_BATCHES='0', PANDA_MAX_RUNTIME='25200',
                PANDA_INITIAL_UPDATE='6000', PANDA_INITIAL_ADAPTER='/container/initial_adapter',
                PANDA_CHECKPOINT_UPDATES='[2000,4000,6000]', PANDA_VALIDATION_ENABLED='0',
                PANDA_SAVE_CHECKPOINTS='1', PANDA_INPUT_STAGING='', PANDA_PROVENANCE='')
            with patch.dict(os.environ, env):
                metrics = Metrics()
                metrics.init(name='continuation')
                metrics.model_details = dict(gpu='cpu-fixture')
                try:
                    with patch('training.runtime.torch.cuda.is_available', return_value=False):
                        metrics.log(dict(train_loss=1.0), step=1)
                    self.assertEqual(metrics.rows[0]['update'], 6001)
                    self.assertEqual(metrics.rows[0]['run_update'], 1)
                    saved_model = Mock()
                    saved_model.module.save_pretrained.side_effect=lambda p:(p/'adapter_model.safetensors').write_bytes(b'weights')
                    with patch('training.runtime.metrics', metrics):
                        save_adapter(saved_model, Mock(), SimpleNamespace(max_steps=6000),
                            SimpleNamespace(dataset_statistics={}), 2000)
                    saved=json.loads((root/'checkpoint_008000/training_config.json').read_text())
                    self.assertEqual(saved['checkpoint_update'], 8000)
                    self.assertEqual(saved['run_update'], 2000)
                    self.assertEqual(saved['continuation_mode'], 'adapter_weights_fresh_optimizer')
                    self.assertFalse((root/'checkpoint_002000').exists())
                    experiment=json.loads((root/'experiment.json').read_text())
                    self.assertEqual(experiment['checkpoint_updates'], [8000,10000,12000])
                    self.assertTrue((root/'loss_smoothed.png').is_file())
                finally:
                    metrics.writer.close()

    def test_training_launcher_keeps_six_thousand_local_updates_and_offsets_reports(self):
        import training.train_panda as trainer
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            dataset=root/'dataset'
            adapter=root/'adapter'
            dataset.mkdir();adapter.mkdir()
            (dataset/'verification.json').write_text('{"passed":true}')
            (adapter/'training_config.json').write_text('{"checkpoint_update":6000}')
            cfg=json.loads((PROJECT_ROOT/'training/color_place_bowl_600_continue_12000.json').read_text())
            cfg['expected_gpu']=None
            path=root/'config.json';path.write_text(json.dumps(cfg))
            argv=['train','--dataset',str(dataset),'--config',str(path),'--output-root',str(root/'runs'),
                '--adapter',str(adapter),'--initial-update','6000','--batch-policy','config']
            with patch('sys.argv',argv),patch('training.dataset_profile.validate_dataset'), \
                patch('training.continuation.validate_adapter',return_value=adapter), \
                patch('training.fetch_upstream.fetch'),patch('training.train_panda.subprocess.run') as run:
                trainer.main()
            command=run.call_args.args[0]
            env=run.call_args.kwargs['env']
            self.assertEqual(command[command.index('--max_steps')+1], '6000')
            self.assertEqual(env['PANDA_INITIAL_UPDATE'], '6000')
            self.assertEqual(env['PANDA_INITIAL_ADAPTER'], str(adapter))
            self.assertEqual(json.loads(env['PANDA_CHECKPOINT_UPDATES']), [2000,4000,6000])
            self.assertNotIn('--initial-update', command)


if __name__=='__main__':unittest.main()
