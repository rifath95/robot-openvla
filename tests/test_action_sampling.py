import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from inference.action_sampling import generation_options, validate_options, ActionTokenLogitsProcessor
from inference.openvla_backend import OpenVLAPredictor
from cloud.cloud_session import bootstrap_script


class SamplingTests(unittest.TestCase):
    def test_greedy_default_and_validation(self):
        self.assertEqual(generation_options(), {'do_sample':False})
        self.assertEqual(validate_options(generation_options(True,.7,.95)), generation_options(True,.7,.95))
        for options in ({'do_sample':'yes'},{'temperature':0},{'top_p':2},
                        {'top_p':float('nan')},{'temperature':True},{'top_k':50},{'unexpected':1}):
            with self.assertRaises(ValueError):validate_options(options)

    def test_samples_only_action_tokens_and_varies_on_repeated_inputs(self):
        processor=ActionTokenLogitsProcessor(10,4)
        scores=torch.zeros((200,12))
        scores[:,0]=1000  # A nonaction token must never escape the mask.
        masked=processor(torch.zeros((200,1),dtype=torch.long),scores)
        self.assertTrue(torch.isneginf(masked[:,:6]).all())
        self.assertTrue(torch.isneginf(masked[:,10:]).all())
        torch.manual_seed(123)
        tokens=torch.multinomial(torch.softmax(masked,dim=-1),1).flatten()
        self.assertTrue(((tokens>=6)&(tokens<10)).all())
        self.assertGreater(len(tokens.unique()),1)

    def test_transformers_generation_keeps_all_seven_tokens_valid(self):
        from transformers import LlamaConfig, LlamaForCausalLM, LogitsProcessorList
        torch.manual_seed(42)
        model=LlamaForCausalLM(LlamaConfig(vocab_size=64,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,
            max_position_embeddings=32,eos_token_id=2,pad_token_id=0)).eval()
        with torch.inference_mode():
            generated=model.generate(input_ids=torch.tensor([[1,3]]),max_new_tokens=7,
                **generation_options(True,.7,.95),
                logits_processor=LogitsProcessorList([ActionTokenLogitsProcessor(64,16)]))
        self.assertEqual(generated.shape,(1,9))
        self.assertTrue(((generated[:,2:]>=48)&(generated[:,2:]<64)).all())

    def predictor(self):
        predictor=OpenVLAPredictor.__new__(OpenVLAPredictor)
        predictor.torch=torch;predictor.device='cpu';predictor.dtype=torch.float32
        predictor.unnorm_key='panda_grasp_v1';predictor.synchronize=lambda:None
        class Inputs(dict):
            def to(self,*args,**kwargs):return self
        predictor.processor=lambda *args:Inputs(input_ids=torch.tensor([[1,29871]]),
                                                 attention_mask=torch.ones((1,2),dtype=torch.long))
        from unittest.mock import Mock
        predictor.model=Mock(vocab_size=32000,config=SimpleNamespace(n_action_bins=256))
        predictor.model.predict_action.return_value=np.zeros(7)
        return predictor

    def test_backend_passes_sampling_controls_and_mask(self):
        predictor=self.predictor()
        result=predictor.predict(None,'touch the red cube',generation=generation_options(True,.7,.95))
        kwargs=predictor.model.predict_action.call_args.kwargs
        self.assertTrue(kwargs['do_sample'])
        self.assertEqual(kwargs['temperature'],.7)
        self.assertEqual(kwargs['top_p'],.95)
        self.assertEqual(kwargs['top_k'],0)
        scores=torch.zeros((1,32064));kwargs['logits_processor'](torch.ones((1,1),dtype=torch.long),scores)
        self.assertEqual(torch.isfinite(scores).sum().item(),256)
        self.assertTrue(result['generation_options']['action_tokens_only'])
        predictor.predict(None,'touch the red cube')
        kwargs=predictor.model.predict_action.call_args.kwargs
        self.assertFalse(kwargs['do_sample'])
        self.assertNotIn('logits_processor',kwargs)

    def test_http_sampling_round_trip_and_ignored_options_rejected(self):
        from http.server import ThreadingHTTPServer
        import threading
        from PIL import Image
        from cloud.connection_server import Handler
        from cloud.connection_client import request_prediction
        applied=[]
        class Predictor:
            action_sampling_available=True
            def predict(self,image,instruction,*,generation=None):
                applied.append(generation)
                return dict(action=[0,0,0,0,0,0,1],unnorm_key='panda_grasp_v1',
                            generation_options=dict(generation,action_tokens_only=True),timings_seconds={})
        with tempfile.TemporaryDirectory() as tmp, ThreadingHTTPServer(('127.0.0.1',0),Handler) as server:
            path=Path(tmp)/'image.png';Image.new('RGB',(8,8)).save(path)
            server.mode='openvla';server.predictor=Predictor();server.prediction_lock=threading.Lock()
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                options=generation_options(True,.7,.95)
                result=request_prediction(path,'touch cube',server_url=f'http://127.0.0.1:{server.server_port}',generation=options)
                self.assertEqual(applied,[options])
                self.assertTrue(result['generation_options']['action_tokens_only'])
                with patch.object(server.predictor,'predict',return_value=dict(action=[0]*7,timings_seconds={})), \
                     self.assertRaisesRegex(ValueError,'did not confirm'):
                    request_prediction(path,'touch cube',server_url=f'http://127.0.0.1:{server.server_port}',generation=options)
                server.predictor.action_sampling_available=False
                with self.assertRaisesRegex(ValueError,'HTTP 409'):
                    request_prediction(path,'touch cube',server_url=f'http://127.0.0.1:{server.server_port}',generation=options)
            finally:
                server.shutdown();thread.join()

    def test_old_server_capability_required_when_sampling(self):
        script=bootstrap_script(1800,'/workspace/checkpoint',require_sampling=True)
        self.assertIn('--require-sampling',script)
        self.assertNotIn('--require-sampling',bootstrap_script(1800,'/workspace/checkpoint'))


if __name__=='__main__':unittest.main()
