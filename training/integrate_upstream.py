"""Small, checked edits to the pinned official OpenVLA LoRA training script."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def integrated_source():
    pin=json.loads((ROOT/'upstream/PIN.json').read_text())
    source=(ROOT/'upstream/finetune.py').read_text()
    if hashlib.sha256(source.encode()).hexdigest()!=pin['finetune_sha256']:
        raise ValueError('Pinned upstream script was modified')
    def replace(old,new):
        nonlocal source
        if source.count(old)!=1:raise ValueError(f'Upstream integration anchor changed: {old[:60]}')
        source=source.replace(old,new)
    replace('import wandb','from training.runtime import metrics as wandb\nfrom training.runtime import validate, save_adapter, budget_expired, record_model\nfrom training.panda_rlds import PandaRLDSDataset')
    replace('        vla.print_trainable_parameters()',
            '        vla.print_trainable_parameters()\n        record_model(vla, cfg)')
    replace('vla_dataset = RLDSDataset(', 'vla_dataset = PandaRLDSDataset(')
    replace('gradient_step_idx = batch_idx // cfg.grad_accumulation_steps',
            'gradient_step_idx = (batch_idx + 1) // cfg.grad_accumulation_steps')
    replace('                        "train_loss": smoothened_loss,',
            '                        "train_loss": smoothened_loss,\n                        "learning_rate": cfg.learning_rate,')
    replace('if distributed_state.is_main_process and gradient_step_idx % 10 == 0:',
            'if distributed_state.is_main_process and (batch_idx + 1) % cfg.grad_accumulation_steps == 0:')
    # Adapter-only saves avoid a second 7B model and expensive per-save merges.
    start=source.index('            # Save Model Checkpoint =>>')
    end=source.index('\n\n\nif __name__',start)
    source=source[:start]+'''            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                if gradient_step_idx % int(os.environ["PANDA_VALIDATE_EVERY"]) == 0:
                    validate(vla, cfg, batch_transform, collator, device_id, gradient_step_idx)
                finished = gradient_step_idx >= cfg.max_steps or budget_expired()
                if gradient_step_idx % cfg.save_steps == 0 or finished:
                    save_adapter(vla, processor, cfg, vla_dataset, gradient_step_idx)
                if finished:
                    wandb.plot()
                    break
'''+source[end:]
    # Explicit eager attention: existing inference image needs no flash-attn build.
    replace('        quantization_config=quantization_config,',
            '        quantization_config=quantization_config,\n        attn_implementation="eager",')
    compile(source,'panda_finetune.py','exec')
    return source


if __name__=='__main__':
    output=ROOT/'generated/panda_finetune.py'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(integrated_source())
    print(output)
