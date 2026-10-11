"""Causal vision-token alignment, label isolation, and diagnostic HTTP/report checks."""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from matched_diagnostic import teacher_forced_metrics
from training.upstream.action_tokenizer import ActionTokenizer
from openvla_backend import OpenVLAPredictor
from connection_client import request_prediction
from connection_server import Handler, ThreadingHTTPServer
import evaluate_expert_actions as evaluation

STATS = {'q01': [-.01]*6+[0], 'q99': [.01]*6+[1], 'mask': [True]*6+[False]}


def oracle(labels, patches=3):
    logits = torch.full((1, labels.shape[1]+patches, 512), -10.)
    for i in range(1, labels.shape[1]):
        if labels[0, i] != -100:
            logits[0, patches+i-1, labels[0, i]] = 10.
    return logits


class MatchedDiagnosticTests(unittest.TestCase):
    def test_causal_shift_vision_offset_and_stop_exclusion(self):
        labels = torch.tensor([[-100]*3 + [500,499,498,497,496,495,494] + [2]])
        tokenizer = ActionTokenizer(SimpleNamespace(vocab_size=512))
        result = teacher_forced_metrics(oracle(labels), labels, tokenizer, STATS, .25)
        self.assertEqual(result['action_token_accuracy'], 1)
        self.assertEqual(result['target_token_ids'], [500,499,498,497,496,495,494])
        self.assertEqual(result['normalized_l1_token_targets'], 0)
        self.assertEqual(result['supervised_tokens'], 8)
        # Deliberately use logits one position too late: alignment must no longer pass.
        shifted = torch.roll(oracle(labels), shifts=1, dims=1)
        self.assertLess(teacher_forced_metrics(shifted,labels,tokenizer,STATS,.25)['action_token_accuracy'],1)
        invalid = oracle(labels);invalid[0,5,:] = -10;invalid[0,5,1] = 10
        self.assertTrue(teacher_forced_metrics(invalid,labels,tokenizer,STATS,.25)['invalid_action_tokens'][0])
        with self.assertRaisesRegex(ValueError,'seven action'):
            teacher_forced_metrics(oracle(labels),labels[:,:-1],tokenizer,STATS,.25)

    def test_generation_receives_no_expert_labels_and_runs_before_teacher_forward(self):
        events=[]
        ids=torch.tensor([1,42,29871,500,499,498,497,496,495,494,2])
        labels=ids.clone();labels[:3]=-100
        pixels=torch.zeros(6,2,2)
        class Processor:
            tokenizer=SimpleNamespace(vocab_size=512)
            def __call__(self,prompt,image):
                self.asserted_prompt=prompt
                return {'input_ids':torch.tensor([[1,42,29871]]),'pixel_values':pixels[None,...]}
        class Model:
            norm_stats={'panda_grasp_v1':{'action':STATS}}
            def __call__(self,**kwargs):
                events.append('teacher')
                self.kwargs=kwargs
                assert not torch.is_grad_enabled()
                return SimpleNamespace(logits=oracle(kwargs['labels']),loss=torch.tensor(.25))
        predictor=OpenVLAPredictor.__new__(OpenVLAPredictor)
        predictor.torch=torch;predictor.device='cpu';predictor.dtype=torch.float32
        predictor.unnorm_key='panda_grasp_v1';predictor.processor=Processor();predictor.model=Model()
        def generate(image,instruction):
            events.append('generated')
            return {'action':[0]*7,'timings_seconds':{'model_inference':.01}}
        predictor.predict=generate
        batch={'input_ids':ids,'labels':labels,'pixel_values':pixels}
        with patch('check_training_inputs.official_batch_transform',return_value=lambda _:batch):
            result=predictor.diagnose(Image.new('RGB',(2,2)),'Move along world Y',[0]*7)
        self.assertEqual(events,['generated','teacher'])
        self.assertEqual(result['teacher_forced']['action_token_accuracy'],1)
        self.assertEqual(predictor.processor.asserted_prompt,'In: What action should the robot take to move along world y?\nOut:')
        self.assertFalse(predictor.model.kwargs['use_cache'])
        # A mismatched prefix must fail, rather than produce misleading paired metrics.
        batch['input_ids']=ids.clone();batch['input_ids'][1]=43
        with patch('check_training_inputs.official_batch_transform',return_value=lambda _:batch):
            with self.assertRaisesRegex(ValueError,'prefixes differ'):
                predictor.diagnose(Image.new('RGB',(2,2)),'Move along world Y',[0]*7)

    def test_matched_http_client_and_report_keep_both_predictions(self):
        calls=[]
        teacher=dict(action=[0,0,.01,0,0,0,0],predicted_token_ids=[500]*7,target_token_ids=[500]*7,
                     token_matches=[True]*7,action_token_accuracy=1.,normalized_l1_token_targets=0.,
                     loss_including_stop=.1,supervised_tokens=8,invalid_action_tokens=[False]*7,
                     prompt_and_pixels_match_generation=True)
        class Predictor:
            load_seconds=1
            startup_info=dict(unnorm_key='panda_grasp_v1',action_statistics=STATS,
                instruction_processing='lowercase to match official RLDSBatchTransform',
                evaluation_provenance={'adapter_checkpoint':evaluation.CHECKPOINT_500})
            def diagnose(self,image,instruction,expert_action):
                calls.append((image.size,instruction,expert_action))
                return dict(action=[0,0,-.01,0,0,0,0],unnorm_key='panda_grasp_v1',
                            teacher_forced=dict(teacher),timings_seconds={'model_inference':.01})
        with tempfile.TemporaryDirectory() as folder, ThreadingHTTPServer(('127.0.0.1',0),Handler) as server:
            root=Path(folder);image=root/'input.png';Image.new('RGB',(20,10)).save(image)
            sample=dict(episode_id='a',split='train',index=1,stage='lift',group='first_lift',
                        instruction='pick up cube',image=str(image),expert_action=[0,0,.01,0,0,0,0])
            server.mode='openvla';server.predictor=Predictor();server.prediction_lock=threading.Lock()
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            url=f'http://127.0.0.1:{server.server_port}'
            try:
                with patch('sys.argv',['evaluate_expert_actions.py','--matched','--server-url',url,
                                       '--output-dir',str(root/'results')]), \
                     patch.object(evaluation,'prepare_samples',return_value=([sample],{'panda_grasp_v1':{'action':STATS}},{'first_lift':1})):
                    self.assertEqual(evaluation.main(),0)
                result=json.loads((root/'results/matched_summary.json').read_text())
                self.assertEqual(result['overall']['generated']['mean_translation_l2_error_mm'],20)
                self.assertEqual(result['overall']['teacher_forced']['mean_translation_l2_error_mm'],0)
                self.assertIn('teacher_forced',(root/'results/predictions.jsonl').read_text())
                self.assertEqual(calls,[((20,10),'pick up cube',sample['expert_action'])])
                self.assertFalse(result['robot_moved'])
                # Malformed diagnostics must not be accepted as ordinary actions.
                with patch.object(server.predictor,'diagnose',return_value=dict(
                        action=[0]*7,unnorm_key='panda_grasp_v1',teacher_forced={},timings_seconds={})):
                    with self.assertRaisesRegex(ValueError,'teacher-forced result'):
                        request_prediction(image,'test',server_url=url,expert_action=sample['expert_action'])
                # If a later request fails, preserve the first complete pair and its report.
                with patch('sys.argv',['evaluate_expert_actions.py','--matched','--server-url',url,
                                       '--output-dir',str(root/'partial')]), \
                     patch.object(evaluation,'prepare_samples',return_value=([sample,sample],{'panda_grasp_v1':{'action':STATS}},{'first_lift':2})), \
                     patch.object(server.predictor,'diagnose',side_effect=[dict(
                         action=[0,0,-.01,0,0,0,0],unnorm_key='panda_grasp_v1',
                         teacher_forced=dict(teacher),timings_seconds={}),RuntimeError('interrupted')]):
                    self.assertEqual(evaluation.main(),1)
                partial=json.loads((root/'partial/matched_summary.json').read_text())
                self.assertEqual(partial['phase'],'failed')
                self.assertEqual(partial['completed'],1)
                self.assertEqual(len((root/'partial/predictions.jsonl').read_text().splitlines()),1)
                with self.assertRaisesRegex(ValueError,'seven finite'):
                    request_prediction(image,'test',server_url=url,expert_action=[float('nan')]*7)
                # A model exception must release the shared prediction lock.
                with patch.object(server.predictor,'diagnose',side_effect=RuntimeError('GPU failure')):
                    with self.assertRaisesRegex(ValueError,'HTTP 500'):
                        request_prediction(image,'test',server_url=url,expert_action=sample['expert_action'])
                self.assertFalse(server.prediction_lock.locked())
                server.predictor=None
                with self.assertRaisesRegex(ValueError,'HTTP 409'):
                    request_prediction(image,'test',server_url=url,expert_action=sample['expert_action'])
            finally:
                server.shutdown();thread.join(timeout=5)


if __name__=='__main__':
    unittest.main()
