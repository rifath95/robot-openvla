"""One-image OpenVLA feasibility test. Does not command the robot.

Run with .venv-openvla/bin/python; the simulator's .venv is separate.
"""

import argparse
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache-openvla"))
os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / ".cache-openvla" / "modules"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--image", type=Path, default=ROOT / "camera_image.png")
    parser.add_argument("--instruction", default="pick up the red cube")
    parser.add_argument("--unnorm-key", default="bridge_orig")
    args = parser.parse_args()

    import torch
    import transformers
    from huggingface_hub import HfApi, snapshot_download

    print(f"PyTorch {torch.__version__}; Transformers {transformers.__version__}", flush=True)
    print(f"Apple GPU available: {torch.backends.mps.is_available()}", flush=True)
    if args.preflight:
        if torch.backends.mps.is_available():
            tensor = torch.ones((32, 32), device="mps", dtype=torch.float16)
            assert (tensor @ tensor).cpu()[0, 0].item() == 32
            print("MPS float16 matrix test passed", flush=True)
        return

    model_id = "openvla/openvla-7b"
    manifest = ROOT / ".cache-openvla" / "model_revision.json"
    if args.download:
        revision = HfApi().model_info(model_id).sha
        folder = snapshot_download(
            model_id, revision=revision,
            allow_patterns=["*.json", "*.py", "*.safetensors", "*.model"],
            max_workers=2,
        )
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"model": model_id, "revision": revision, "path": folder}, indent=2))
        print(f"Downloaded pinned revision {revision}", flush=True)
        return

    if not manifest.exists():
        raise RuntimeError("Download the model first using --download")
    model_path = json.loads(manifest.read_text())["path"]
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple GPU is unavailable")
    # Eager attention avoids the CUDA-only FlashAttention dependency.
    # Device-map loading places each weight on its destination immediately.
    print(f"Loading float16 model on {args.device}…", flush=True)
    start = time.monotonic()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation="eager", device_map={"": args.device},
    ).eval()
    print(f"Loaded in {time.monotonic()-start:.1f} seconds", flush=True)
    if args.unnorm_key not in model.norm_stats:
        raise ValueError(f"Unknown statistics key; choices: {list(model.norm_stats)}")
    image = Image.open(args.image).convert("RGB")
    prompt = f"In: What action should the robot take to {args.instruction}?\nOut:"
    inputs = processor(prompt, image).to(args.device, dtype=torch.float16)
    # predict_action appends this training-time token without extending the
    # processor's attention mask. Supply both together to keep lengths aligned.
    if not torch.all(inputs["input_ids"][:, -1] == 29871):
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], inputs["input_ids"].new_full((1, 1), 29871)], dim=1
        )
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], inputs["attention_mask"].new_ones((1, 1))], dim=1
        )
    print("Image and prompt processed; token and attention-mask lengths aligned", flush=True)
    start = time.monotonic()
    with torch.inference_mode():
        action = model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
    if args.device == "mps":
        torch.mps.synchronize()
    import numpy as np
    if np.asarray(action).shape != (7,) or not np.all(np.isfinite(action)):
        raise RuntimeError(f"Invalid predicted action: {action}")
    result = {"action": action.tolist(), "inference_seconds": time.monotonic()-start,
              "device": args.device, "unnorm_key": args.unnorm_key,
              "instruction": args.instruction, "image": str(args.image)}
    destination = ROOT / "outputs" / "openvla_mac_action.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved {destination}. This action has NOT been sent to the robot.", flush=True)


if __name__ == "__main__":
    main()
