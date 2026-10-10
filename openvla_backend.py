"""Resident OpenVLA predictor used by the HTTP server."""

import math
import json
from pathlib import Path
import time


class OpenVLAPredictor:
    def __init__(self, model_dir, *, device="cuda", unnorm_key="bridge_orig"):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.torch = torch
        self.device = device
        self.unnorm_key = unnorm_key
        folder = Path(model_dir).resolve()
        if not (folder / "config.json").is_file():
            raise ValueError(f"Missing config.json in model directory: {folder}")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; check the NVIDIA driver and CUDA PyTorch installation")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS GPU is unavailable")
        self.dtype = (torch.float32 if device == "cpu" else
                      torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported()
                      else torch.float16)
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            str(folder), trust_remote_code=True, local_files_only=True,
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            str(folder), trust_remote_code=True, local_files_only=True,
            torch_dtype=self.dtype, low_cpu_mem_usage=True,
            attn_implementation="eager", device_map={"": device},
        ).eval()
        self.synchronize()
        self.load_seconds = time.perf_counter() - started
        metrics_path = Path(__file__).resolve().parent / 'outputs/cloud_startup.json'
        self.startup_info = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        import transformers
        self.startup_info.update({
            'model_load_seconds': self.load_seconds,
            'loading_note': 'Processor setup, checkpoint reading, model construction and device transfer, synchronized at completion; not isolated storage-to-VRAM time.',
            'device': device, 'dtype': str(self.dtype), 'attention': 'eager',
            'pytorch_version': torch.__version__, 'transformers_version': transformers.__version__,
        })
        if device == 'cuda':
            properties = torch.cuda.get_device_properties(torch.cuda.current_device())
            free, total = torch.cuda.mem_get_info()
            self.startup_info['gpu'] = {
                'name': properties.name, 'total_vram_bytes': total,
                'free_vram_bytes_after_loading': free,
                'used_vram_bytes_after_loading': total - free,
                'pytorch_allocated_bytes_after_loading': torch.cuda.memory_allocated(),
                'pytorch_reserved_bytes_after_loading': torch.cuda.memory_reserved(),
                'pytorch_peak_allocated_bytes_during_loading': torch.cuda.max_memory_allocated(),
            }
        if unnorm_key not in self.model.norm_stats:
            raise ValueError(f"Unknown action statistics {unnorm_key}; choices: {list(self.model.norm_stats)}")
        stats = self.model.norm_stats[unnorm_key]["action"]
        if len(stats["q01"]) != 7 or len(stats["q99"]) != 7:
            raise ValueError("This server requires a model with seven action components")

    def synchronize(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def predict(self, image, instruction):
        torch = self.torch
        started = time.perf_counter()
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=self.dtype)
        # Match the successful local path: keep the extra training token and mask aligned.
        if not torch.all(inputs["input_ids"][:, -1] == 29871):
            inputs["input_ids"] = torch.cat([
                inputs["input_ids"], inputs["input_ids"].new_full((1, 1), 29871)
            ], dim=1)
            inputs["attention_mask"] = torch.cat([
                inputs["attention_mask"], inputs["attention_mask"].new_ones((1, 1))
            ], dim=1)
        self.synchronize()
        prepared = time.perf_counter()
        with torch.inference_mode():
            raw_action = self.model.predict_action(
                **inputs, unnorm_key=self.unnorm_key, do_sample=False,
            )
        self.synchronize()
        predicted = time.perf_counter()
        action = raw_action.tolist()
        if not isinstance(action, list) or len(action) != 7 or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in action
        ):
            raise RuntimeError("OpenVLA did not return seven finite action values")
        return {
            "action": action,
            "unnorm_key": self.unnorm_key,
            "device": self.device,
            "dtype": str(self.dtype),
            "attention": "eager",
            "action_convention": "Raw OpenVLA action unnormalized with dataset statistics; provisional Bridge-to-Panda mapping is not applied. Gripper is 0 closed / 1 open for bridge_orig.",
            "timings_seconds": {
                "processor_and_device_transfer": prepared - started,
                "model_inference": predicted - prepared,
            },
        }
