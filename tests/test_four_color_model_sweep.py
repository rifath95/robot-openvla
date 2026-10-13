"""Four-pair plan, true pretrained baseline, shared normalization and holdouts."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,patch
from common.paths import PROJECT_ROOT
from cloud.sweep_plan import trial_specs,validate_saved,timeout_budget
from cloud import sweep_worker

class FourColorSweepTests(unittest.TestCase):
    def test_selected_profile_counts_and_pair_groups(self):
        c=json.loads((PROJECT_ROOT/'inference/sweep_profiles/four_color_all_models.json').read_text())
        self.assertEqual(c['checkpoints'],[0,50,100,300,500,1000])
        self.assertEqual(c['color_pairs'],[['red','blue'],['green','yellow'],['black','purple'],['orange','white']])
        specs=list(trial_specs(dict(c,task='color_transfer')))
        self.assertEqual(len(specs),480)
        self.assertEqual(sum(s['checkpoint']==0 for s in specs),80)
        self.assertTrue(all(s['directory'].startswith('base_model_') for s in specs if s['checkpoint']==0))
        self.assertEqual(timeout_budget(c['estimated_seconds']),dict(estimated_seconds=10800,inner_seconds=14400,outer_seconds=18000))

    def test_report_separates_seen_unseen_and_names_baseline(self):
        from analysis.color_transfer_report import write_report
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            plan=dict(checkpoints=[0],include_base_model=True,training_colors=['red','blue','green','yellow'],
                color_pairs=[['red','blue'],['black','purple']])
            rows=[dict(checkpoint=0,seed=1000,color_pair=pair,target_color=color,outcome='correct_cube',
                       phase='Completed',directory=f'trial_{color}') for pair,colors in [('red:blue',['red','blue']),('black:purple',['black','purple'])] for color in colors]
            write_report(folder,plan,rows)
            text=(folder/'REPORT.md').read_text()
            self.assertIn('Pretrained base',text);self.assertIn('seen pairs',text);self.assertIn('unseen pairs',text)
            stats=json.loads((folder/'statistics.json').read_text())
            self.assertEqual(next(m['trials'] for m in stats if m['color_pair']=='seen pairs'),2)
            self.assertEqual(next(m['trials'] for m in stats if m['color_pair']=='unseen pairs'),2)

    def test_base_server_uses_statistics_without_lora(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'sweep';directory.mkdir()
            plan=dict(task='color_transfer',checkpoints=[0,50],include_base_model=True,action_statistics_checkpoint=50,
                scene_seeds=[1000],cube_positions=[{'red':[.4,.1],'blue':[.6,-.1]}],color_pairs=[['red','blue']],
                steps=50,sampling=False,training_session='panda-train-test',timeout_budget=timeout_budget(100))
            (directory/'results.json').write_text(json.dumps(dict(plan=plan,trials=[])))
            def run(command,**kwargs):
                if 'inference.remote_loop' in command:
                    trial=Path(command[command.index('--output-dir')+1]);trial.mkdir(parents=True)
                    (trial/'summary.json').write_text(json.dumps(dict(phase='Completed',completed_steps=1,
                        task_result=dict(outcome='no_contact',correct_cube_contact=False,wrong_cube_contact=False))))
                return SimpleNamespace(returncode=0)
            server=MagicMock();server.poll.return_value=0
            with patch.object(sweep_worker,'PROJECT_ROOT',root),patch.object(sweep_worker,'stage_weights',return_value=root/'staged') as stage, \
                patch.object(sweep_worker.subprocess,'run',side_effect=run),patch.object(sweep_worker.subprocess,'Popen',return_value=server) as start, \
                patch.object(sweep_worker.subprocess,'check_output',return_value='commit'), \
                patch.object(sweep_worker,'wait_server',return_value=dict(startup_info={},server_port=12345)), \
                patch.object(sweep_worker,'write_reports'):
                sweep_worker.run(directory,'/sim/python')
            commands=[c.args[0] for c in start.call_args_list]
            self.assertIn('--action-statistics',commands[0]);self.assertNotIn('--adapter',commands[0])
            self.assertIn('--adapter',commands[1]);self.assertNotIn('--action-statistics',commands[1])
            self.assertEqual(set(stage.call_args.args[1]),{50})

    def test_baseline_normalization_leaves_weights_unadapted(self):
        from inference.openvla_backend import OpenVLAPredictor
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'config.json').write_text('{}')
            stats={'panda_grasp_v1':{'action':dict(q01=[0]*7,q99=[1]*7)}}
            file=root/'stats.json';file.write_text(json.dumps(stats))
            torch=MagicMock();torch.cuda.is_available.return_value=True;torch.cuda.is_bf16_supported.return_value=True
            torch.cuda.mem_get_info.return_value=(30,45);torch.__version__='test'
            torch.cuda.get_device_properties.return_value=SimpleNamespace(name='test')
            transformers=MagicMock();transformers.__version__='test'
            model=MagicMock();model.eval.return_value=model;model.norm_stats={};model.config=SimpleNamespace()
            transformers.AutoModelForVision2Seq.from_pretrained.return_value=model
            peft=MagicMock()
            with patch.dict('sys.modules',torch=torch,transformers=transformers,peft=peft):
                p=OpenVLAPredictor(root,device='cuda',unnorm_key='panda_grasp_v1',action_statistics=file)
            peft.PeftModel.from_pretrained.assert_not_called()
            model.save_pretrained.assert_not_called()
            self.assertEqual(p.model.norm_stats,stats)
            self.assertEqual(p.startup_info['adapter_merge_seconds'],0)

if __name__=='__main__':unittest.main()
