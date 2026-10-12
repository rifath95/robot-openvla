"""Print exact benchmark compatibility differences without loading model weights."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

from training.batch_policy import benchmark_context,choose_batch
from training.dataset_profile import select_dataset,validate_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('training/pick_bowl_5000.json'))
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--model',type=Path,default=Path('/workspace/openvla-7b'))
    parser.add_argument('--benchmark-root',type=Path,default=Path('/workspace/panda-training/batch-benchmarks'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    config=json.loads(args.config.read_text());dataset=select_dataset(config,args.dataset)
    validate_dataset(config,dataset)
    context=benchmark_context(config,dataset,args.model)
    _,report=choose_batch(config,context,args.benchmark_root)
    report.update(expected_gpu=config.get('expected_gpu'),dataset=str(dataset),
                  training_started=False,model_weights_loaded=False)
    output=args.output or dataset.parent/'batch-checks'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    print(f'Diagnostic saved: {output}. This check does not start training or stop the pod.')
    return 0 if report['source']=='compatible_benchmark' and (
        not config.get('expected_gpu') or (context or {}).get('gpu')==config['expected_gpu']) else 2


if __name__=='__main__':raise SystemExit(main())
