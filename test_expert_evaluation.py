"""Checks for label comparison, server identity, sampling, and partial outputs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import evaluate_expert_actions as evaluation


STATS = {'panda_grasp_v1': {'action': {'q01': [-.01]*6+[0], 'q99': [.01]*6+[1],
                                     'mask': [True]*6+[False]}}}


class ExpertEvaluationTests(unittest.TestCase):
    def test_physical_and_normalized_errors_and_lift_direction(self):
        result = evaluation.compare_actions([0,0,.01,0,0,0,0], [0,0,-.01,0,0,0,1], STATS)
        self.assertAlmostEqual(result['translation_l2_error_mm'], 20)
        self.assertAlmostEqual(result['normalized_absolute_error'][2], 2)
        self.assertFalse(result['gripper_state_match'])
        self.assertTrue(result['expert_upward'])
        self.assertFalse(result['predicted_upward'])

    def test_direction_aggregate_uses_only_expert_upward_examples(self):
        rows = [dict(comparison=evaluation.compare_actions([0,0,z,0,0,0,0], [0,0,p,0,0,0,0], STATS))
                for z,p in [(0.01,0.01),(0.01,0),(-0.01,-0.01)]]
        result = evaluation.aggregate(rows)
        self.assertEqual(result['expert_upward_examples'], 2)
        self.assertEqual(result['predicted_upward_when_expert_upward'], .5)

    def test_wrong_model_checkpoint_or_normalization_rejected(self):
        info = dict(unnorm_key='panda_grasp_v1', action_statistics=STATS['panda_grasp_v1']['action'],
                    evaluation_provenance={'adapter_checkpoint': '/checkpoint500'})
        health = dict(mode='openvla', model_loaded=True, startup_info=info)
        evaluation.check_server(health, STATS, '/checkpoint500')
        with self.assertRaisesRegex(ValueError, 'adapter'):
            evaluation.check_server(health, STATS, '/checkpoint1000')
        with self.assertRaisesRegex(ValueError, 'normalization'):
            evaluation.check_server(health, {'panda_grasp_v1': {'action': dict(info['action_statistics'], q99=[1]*7)}}, '/checkpoint500')
        with self.assertRaisesRegex(ValueError, 'loaded OpenVLA'):
            evaluation.check_server(dict(health, mode='connection_test'), STATS, '/checkpoint500')

    def test_selection_separates_first_lift_and_preserves_instructions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            stages = ['close','lift','lift','transfer','lift']
            episode = dict(id='a',split='train',fingerprint='verified',path=path,
                           rows=[dict(index=i+1,stage=stage,instruction='original instruction',
                                      observation='input.png',action=[0]*7) for i,stage in enumerate(stages)])
            (path/'manifest.json').write_text(json.dumps({'episodes': [dict(id='a',split='train',transitions=5,sha256='verified')]}))
            (path/'dataset_statistics.json').write_text(json.dumps(STATS))
            with patch.object(evaluation,'load_episodes',return_value=[episode]):
                samples,_,counts = evaluation.prepare_samples(path/'source',path,per_group=1)
                self.assertEqual(counts['first_lift'],2)
                self.assertEqual(counts['lift'],1)
                self.assertEqual(len(samples),4)
                self.assertTrue(all(s['instruction']=='original instruction' for s in samples))
                all_samples,_,_ = evaluation.prepare_samples(path/'source',path,selection='all')
                self.assertEqual(len(all_samples),5)
                episode['fingerprint']='changed'
                with self.assertRaisesRegex(ValueError,'do not match'):
                    evaluation.prepare_samples(path/'source',path)

    def test_failed_request_preserves_completed_predictions_and_report(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'result'
            image=Path(folder)/'input.png';image.write_bytes(b'PNG fixture')
            sample=dict(episode_id='a',split='train',index=1,stage='lift',group='first_lift',
                        instruction='pick up cube',image=str(image),expert_action=[0,0,.01,0,0,0,0])
            health=dict(mode='openvla',model_loaded=True,startup_info={
                'unnorm_key':'panda_grasp_v1','action_statistics':STATS['panda_grasp_v1']['action'],
                'evaluation_provenance':{'adapter_checkpoint':evaluation.CHECKPOINT_500}})
            class Response:
                def __enter__(self):
                    import io
                    return io.StringIO(json.dumps(health))
                def __exit__(self,*args):pass
            with patch('sys.argv',['evaluate_expert_actions.py','--output-dir',str(output)]), \
                 patch.object(evaluation,'prepare_samples',return_value=([sample,sample],STATS,{'first_lift':2})), \
                 patch.object(evaluation,'urlopen',return_value=Response()), \
                 patch.object(evaluation,'request_prediction',side_effect=[dict(unnorm_key='panda_grasp_v1',action=sample['expert_action']),RuntimeError('disconnected')]):
                self.assertEqual(evaluation.main(),1)
            self.assertEqual(len((output/'predictions.jsonl').read_text().splitlines()),1)
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['completed'],1)
            self.assertEqual(summary['phase'],'failed')
            self.assertIn('not teacher-forced',(output/'EVALUATION_REPORT.md').read_text())

    def test_http_evaluation_uses_saved_image_instruction_and_real_mode(self):
        from PIL import Image
        import threading
        from connection_server import Handler, ThreadingHTTPServer
        calls=[]
        class Predictor:
            load_seconds=1
            startup_info=dict(unnorm_key='panda_grasp_v1',action_statistics=STATS['panda_grasp_v1']['action'],
                              evaluation_provenance={'adapter_checkpoint':evaluation.CHECKPOINT_500})
            def predict(self,image,instruction):
                calls.append((image.size,instruction))
                return dict(action=[0,0,.01,0,0,0,0],unnorm_key='panda_grasp_v1',
                            timings_seconds={'model_inference':.01})
        with tempfile.TemporaryDirectory() as folder, ThreadingHTTPServer(('127.0.0.1',0),Handler) as server:
            path=Path(folder); image=path/'image.png';Image.new('RGB',(20,10)).save(image)
            sample=dict(episode_id='a',split='train',index=1,stage='lift',group='first_lift',
                        instruction='original demonstration instruction',image=str(image),expert_action=[0,0,.01,0,0,0,0])
            server.mode='openvla';server.predictor=Predictor();server.prediction_lock=threading.Lock()
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                with patch('sys.argv',['evaluate_expert_actions.py','--server-url',f'http://127.0.0.1:{server.server_port}',
                                       '--output-dir',str(path/'result')]), \
                     patch.object(evaluation,'prepare_samples',return_value=([sample],STATS,{'first_lift':1})):
                    self.assertEqual(evaluation.main(),0)
                self.assertEqual(calls,[((20,10),'original demonstration instruction')])
                result=json.loads((path/'result/predictions.jsonl').read_text())
                self.assertTrue(result['prediction']['model_loaded'])
                self.assertAlmostEqual(result['comparison']['translation_l2_error_mm'],0)
                self.assertIn('request_round_trip',result['prediction']['client_timings_seconds'])
            finally:
                server.shutdown();thread.join(timeout=5)


if __name__=='__main__':
    unittest.main()
