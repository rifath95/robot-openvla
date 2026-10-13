"""In-memory merge uses the device-loaded base and adapter-specific statistics."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,patch
from inference.openvla_backend import OpenVLAPredictor

class InMemoryMergeTests(unittest.TestCase):
    def test_gpu_base_is_loaded_then_merged_without_disk_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';adapter=root/'adapter';base.mkdir();adapter.mkdir()
            (base/'config.json').write_text('{}')
            statistics={'panda_grasp_v1':{'action':{'q01':[0]*7,'q99':[1]*7}}}
            (adapter/'dataset_statistics.json').write_text(json.dumps(statistics))
            model=MagicMock();model.eval.return_value=model;model.norm_stats={};model.config=SimpleNamespace()
            merged=MagicMock();merged.eval.return_value=merged;merged.norm_stats={};merged.config=SimpleNamespace()
            torch=MagicMock();torch.cuda.is_available.return_value=True;torch.cuda.is_bf16_supported.return_value=True
            torch.cuda.mem_get_info.return_value=(30,45);torch.__version__='test'
            torch.cuda.get_device_properties.return_value=SimpleNamespace(name='testgpu')
            transformers=MagicMock();transformers.__version__='test'
            transformers.AutoModelForVision2Seq.from_pretrained.return_value=model
            peft=MagicMock();peft.PeftModel.from_pretrained.return_value.merge_and_unload.return_value=merged
            with patch.dict('sys.modules',torch=torch,transformers=transformers,peft=peft):
                predictor=OpenVLAPredictor(base,device='cuda',unnorm_key='panda_grasp_v1',adapter=adapter)
            self.assertEqual(transformers.AutoModelForVision2Seq.from_pretrained.call_args.kwargs['device_map'],{'':'cuda'})
            peft.PeftModel.from_pretrained.assert_called_once_with(model,str(adapter.resolve()))
            merged.save_pretrained.assert_not_called()
            self.assertEqual(predictor.model.norm_stats,statistics)
            self.assertEqual(predictor.startup_info['evaluation_provenance']['merge_location'],'cuda')

if __name__=='__main__':unittest.main()
