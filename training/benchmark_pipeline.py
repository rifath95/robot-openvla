"""Run training benchmarks, then batch-one inference inside the same watchdog."""
import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer

from cloud.connection_client import request_prediction
from cloud.connection_server import Handler
from training.detached_run import write_status

WARMUP=5
SAMPLES=20


def sample_requests(image, instruction, url, warmup=WARMUP, samples=SAMPLES):
    rows=[]
    for i in range(warmup+samples):
        response=request_prediction(image,instruction,server_url=url,timeout_seconds=60)
        rows.append(dict(warmup=i<warmup,
                         request_timings_seconds=response['client_timings_seconds'],
                         server_timings_seconds=response['timings_seconds']))
        print(f'Inference request {i+1}/{warmup+samples}',flush=True)
    return rows


def fixture(dataset):
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from PIL import Image
    tf.config.set_visible_devices([], 'GPU')
    manifest=json.loads((dataset/'manifest.json').read_text())
    builder=tfds.builder_from_directory(str(dataset/manifest['builder_directory']))
    episode=next(iter(tfds.as_numpy(builder.as_dataset(split='train',shuffle_files=False))))
    step=next(iter(episode['steps']))
    stream=BytesIO();Image.fromarray(step['observation']['image']).save(stream,format='PNG')
    return stream.getvalue(),step['language_instruction'].decode()


def inference(directory, model, dataset, *, persistent_root=None, wait_seconds=120):
    from inference.openvla_backend import OpenVLAPredictor
    from training.gpu_ecc import collect_ecc
    png,instruction=fixture(dataset)
    image=directory/'input.png';image.write_bytes(png)
    predictor=OpenVLAPredictor(model,device='cuda',unnorm_key='bridge_orig')
    startup=dict(predictor.startup_info)
    # Do not reuse another simulator session's preparation/storage measurements.
    for key in ('model_preparation_seconds','volume_mount','adapter_checkpoint','evaluation_provenance'):
        startup.pop(key,None)
    startup['gpu_ecc']=collect_ecc()
    from training.benchmark_software import collect_software
    startup['software_stack']=collect_software(dict(cuda_runtime=predictor.torch.version.cuda,
                                                   precision=str(predictor.dtype),attention='eager'))
    marker=model/'export_complete.json'
    identity=hashlib.sha256(marker.read_bytes()).hexdigest() if marker.exists() else hashlib.sha256((model/'config.json').read_bytes()).hexdigest()
    with ThreadingHTTPServer(('127.0.0.1',0),Handler) as server:
        server.mode='openvla';server.predictor=predictor;server.prediction_lock=threading.Lock()
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            requests=sample_requests(image,instruction,f'http://127.0.0.1:{server.server_port}')
            report=dict(schema_version=1,warmup_requests=WARMUP,measured_requests=SAMPLES,
                        startup_info=startup,model_identity=identity,image_sha256=hashlib.sha256(png).hexdigest(),
                        image_png_base64=base64.b64encode(png).decode(),instruction=instruction,
                        server_port=server.server_port,cloud_requests=requests,mac_requests=[],
                        live_session_dir=str(directory),
                        source='training benchmark inference follow-up')
            write_status(directory/'inference.json',report)
            status=json.loads((directory/'status.json').read_text())
            status.update(phase='inference_ready',inference_report=str(directory/'inference.json'))
            write_status(directory/'status.json',status)
            if persistent_root:
                from training.global_benchmark_storage import copy_reports
                copy_reports(directory.parent.parent,persistent_root)
            print(f'Inference ready for Mac latency test: {directory}',flush=True)
            deadline=time.monotonic()+wait_seconds
            while time.monotonic()<deadline and not (directory/'inference_client_done').exists():
                time.sleep(.25)
        finally:
            server.shutdown();thread.join(timeout=5)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-dir',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--max-batch',type=int,default=64)
    parser.add_argument('--controlled',action='store_true')
    parser.add_argument('--global-volume',type=Path)
    args=parser.parse_args()
    if args.global_volume:
        command=[sys.executable,'-u','-m','training.global_benchmark_storage','--volume',str(args.global_volume),
                 '--dataset',str(args.dataset),'--max-batch',str(args.max_batch)]
    else:
        command=[sys.executable,'-u','-m','training.benchmark_training_batches',
                 '--dataset',str(args.dataset),'--max-batch',str(args.max_batch)]
    if args.controlled and args.global_volume:command.append('--controlled')
    elif not args.controlled and not args.global_volume:command.append('--find-capacity')
    code=subprocess.run(command).returncode
    if code:return code
    model=Path('/workspace/openvla-7b');dataset=args.dataset;persistent=None
    if args.global_volume:
        from training.global_benchmark_storage import LOCAL_ROOT
        model=LOCAL_ROOT/'inputs/openvla-7b';dataset=LOCAL_ROOT/'inputs'/args.dataset.name
        persistent=args.global_volume/'panda-training/batch-benchmarks'
    return inference(args.session_dir,model,dataset,persistent_root=persistent)


if __name__=='__main__':raise SystemExit(main())
