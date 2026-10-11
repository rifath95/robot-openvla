"""Check package imports and a small LoRA update without downloading model weights.

CPU mode runs during image build. CUDA mode checks the actual rented GPU.
This is a dependency smoke test, not a full OpenVLA training validation.
"""
import argparse
from contextlib import nullcontext
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time


def check(device, expect_blackwell_stack=False):
    started=time.monotonic()
    import torch
    import torchvision
    import timm
    import tensorflow as tf
    import tensorflow_datasets
    from PIL import Image
    from peft import LoraConfig,get_peft_model
    from transformers import LlamaConfig,LlamaForCausalLM,AutoModelForVision2Seq,AutoProcessor

    # Dataset decoding uses TensorFlow on CPU; PyTorch owns the training GPU.
    tf.config.set_visible_devices([], 'GPU')
    packages={name:version(name) for name in
              ('torch','torchvision','transformers','tokenizers','peft','accelerate',
               'timm','numpy','tensorflow','tensorflow-datasets','huggingface-hub')}
    if expect_blackwell_stack:
        expected={'torch':'2.14.1','torchvision':'0.29.1'}
        for name,value in expected.items():
            if packages[name].split('+')[0]!=value:
                raise RuntimeError(f'Expected {name} {value}; found {packages[name]}')
        if torch.version.cuda!='13.0':
            raise RuntimeError(f'Expected CUDA 13.0 wheel; found {torch.version.cuda}')
    result=dict(python=platform.python_version(),packages=packages,
                cuda_runtime=torch.version.cuda,device=device,
                scope='package imports, image transform, NMS and tiny Llama LoRA update')
    if device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        properties=torch.cuda.get_device_properties(0)
        result.update(gpu=properties.name,total_vram_gib=properties.total_memory/2**30,
                      compute_capability=list(torch.cuda.get_device_capability()),
                      compiled_architectures=torch.cuda.get_arch_list())
        if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 unavailable')
        matrix=torch.ones((32,32),device=device,dtype=torch.bfloat16)
        if (matrix@matrix)[0,0].item()!=32:raise RuntimeError('BF16 matrix test failed')
    transform=timm.data.create_transform(input_size=(3,224,224),is_training=False)
    if tuple(transform(Image.new('RGB',(640,480))).shape)!=(3,224,224):
        raise RuntimeError('Image transform returned an unexpected shape')
    boxes=torch.tensor([[0.,0.,2.,2.],[0.,0.,2.,2.]],device=device)
    scores=torch.tensor([.9,.8],device=device)
    if torchvision.ops.nms(boxes,scores,.5).tolist()!=[0]:
        raise RuntimeError('TorchVision NMS test failed')
    if int((tf.constant(1)+1).numpy())!=2:raise RuntimeError('TensorFlow CPU test failed')
    torch.manual_seed(0)
    cfg=LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,
                    num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=4,
                    max_position_embeddings=32)
    cfg._attn_implementation='eager'
    model=get_peft_model(LlamaForCausalLM(cfg),LoraConfig(r=4,lora_alpha=8,
                         target_modules=['q_proj','v_proj'],lora_dropout=0.0)).to(device)
    model.train()
    trainable=[p for p in model.parameters() if p.requires_grad]
    before=[p.detach().clone() for p in trainable]
    optimizer=torch.optim.AdamW(trainable,lr=.001)
    tokens=torch.randint(0,64,(2,8),device=device)
    with torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda' else nullcontext():
        loss=model(input_ids=tokens,labels=tokens).loss
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
    loss.backward()
    if not any(p.grad is not None and torch.count_nonzero(p.grad).item() for p in trainable):
        raise RuntimeError('LoRA gradients are missing or zero')
    optimizer.step()
    if not any(not torch.equal(old,p.detach()) for old,p in zip(before,trainable)):
        raise RuntimeError('LoRA parameters did not change')
    if device=='cuda':torch.cuda.synchronize()
    result.update(passed=True,loss=float(loss.detach()),lora_update_passed=True,
                  elapsed_seconds=time.monotonic()-started)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--expect-blackwell-stack',action='store_true')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    result=check(args.device,args.expect_blackwell_stack)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text)
    print(text,flush=True)


if __name__=='__main__':main()
