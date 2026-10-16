"""Persistent checkpoint selector for manual bowl trials; never stops the pod."""
import argparse
import fcntl
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from common.paths import PROJECT_ROOT
from inference.playground_config import CHECKPOINTS, PROFILE_RELATIVE, STAGING_DIRECTORY

PROFILE = PROJECT_ROOT / PROFILE_RELATIVE
STAGING = Path(STAGING_DIRECTORY)


def stage(root=STAGING, selected=None):
    from cloud.color_bowl_sweep import resolve_training
    from cloud.checkpoint_ready import check_ready
    from cloud.bowl_chain_worker import stage_base
    from cloud.archive_sweep import digest
    from cloud.scripted_eval_control import active_training
    if active_training():
        raise ValueError('Use a separate inference pod; this pod is training')
    source = resolve_training(json.loads(PROFILE.read_text()), selected=selected)
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(training_session=source.name, training_path=str(source),
                    checkpoints=list(CHECKPOINTS), training_profile=PROFILE_RELATIVE, base=str(root/'base'))
    marker = root/'staged.json'
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError('Container staging belongs to another training session')
        for cp in CHECKPOINTS:
            check_ready(root/f'adapters/checkpoint_{cp:06d}')
        return identity
    if shutil.disk_usage(root).free < 22*1024**3:
        raise ValueError('Need 22 GiB free on container disk for base and four inference adapters')
    base = root/'base'
    if base.exists():
        raise ValueError(f'Incomplete previous staging; inspect {base} before retrying')
    print('Copying base model once to container disk...', flush=True)
    stage_base('/workspace/openvla-7b', base)
    for cp in CHECKPOINTS:
        adapter = check_ready(source/f'runs/training/checkpoint_{cp:06d}')
        metadata = json.loads((adapter/'training_config.json').read_text())
        profile = json.loads(PROFILE.read_text())
        provenance = metadata.get('dataset_provenance', {})
        if (metadata.get('checkpoint_update') != cp or
            provenance.get('checksums_sha256') != profile['source_dataset_sha256'] or
            provenance.get('dataset_revision') != profile['source_dataset_revision']):
            raise ValueError('Checkpoint identity does not match the 600-demo run')
        target = root/f'adapters/checkpoint_{cp:06d}'
        target.parent.mkdir(exist_ok=True)
        shutil.copytree(adapter, target, ignore=shutil.ignore_patterns('training_state'))
        for path in adapter.rglob('*'):
            if path.is_file() and 'training_state' not in path.relative_to(adapter).parts and digest(path) != digest(target/path.relative_to(adapter)):
                raise ValueError('Adapter staging checksum mismatch')
        check_ready(target)
        print(f'Staged checkpoint {cp} on container disk.', flush=True)
    marker.write_text(json.dumps(identity, indent=2)+'\n')
    return identity


class Models:
    def __init__(self, root, runtime=43200):
        self.root = Path(root)
        self.identity = json.loads((self.root/'staged.json').read_text())
        self.runtime = runtime
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.state = dict(status='idle', model_loaded=False, checkpoint=None,
                          training_session=self.identity['training_session'],
                          training_profile=self.identity.get('training_profile'),checkpoints=list(CHECKPOINTS))
        self.child = None
        self.port = None

    def status(self):
        with self.state_lock:
            result = dict(self.state)
        if result['model_loaded'] and self.child and self.child.poll() is not None:
            result.update(status='error', model_loaded=False, error='Model server exited; select checkpoint again')
        return result

    def update(self, **values):
        with self.state_lock:
            self.state.update(values)

    def stop_child(self):
        if self.child and self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait(timeout=5)
        self.child = None

    def select(self, cp):
        if type(cp) is not int or cp not in CHECKPOINTS:
            raise ValueError('Unknown checkpoint')
        if not self.lock.acquire(blocking=False):
            raise ValueError('Prediction or checkpoint load in progress; wait and retry')
        current = self.status()
        if current['status'] == 'ready' and current['checkpoint'] == cp:
            self.lock.release()
            return
        self.update(status='loading', checkpoint=cp, model_loaded=False, error=None)
        threading.Thread(target=self.load, args=(cp,), daemon=True).start()

    def load(self, cp):
        try:
            self.stop_child()
            ready = self.root/'ready.json'
            ready.unlink(missing_ok=True)
            adapter = self.root/f'adapters/checkpoint_{cp:06d}'
            command = [sys.executable, '-u', '-m', 'cloud.connection_server',
                       '--mode', 'openvla', '--device', 'cuda', '--model-dir', str(self.root/'base'),
                       '--adapter', str(adapter), '--unnorm-key', 'panda_grasp_v1',
                       '--port', '0', '--ready-file', str(ready),
                       '--max-runtime-seconds', str(self.runtime)]
            with (self.root/f'checkpoint_{cp:06d}.log').open('a') as log:
                self.child = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
            from cloud.sweep_worker import wait_server
            health = wait_server(self.child, 0, seconds=1800, ready_file=ready)
            self.port = health['server_port']
            self.update(status='ready', model_loaded=True, checkpoint=cp,
                        model_load_seconds=health.get('model_load_seconds'),
                        startup_info=health.get('startup_info'),
                        base_path=str(self.root/'base'), adapter_path=str(adapter))
        except Exception as exc:
            self.stop_child()
            self.update(status='error', model_loaded=False, error=str(exc))
        finally:
            self.lock.release()

    def predict(self, payload):
        with self.lock:
            state = self.status()
            if not state['model_loaded'] or state['checkpoint'] != payload.pop('checkpoint', None):
                raise ValueError('Selected checkpoint is not ready; prediction cancelled')
            connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=180)
            try:
                connection.request('POST', '/predict', json.dumps(payload), {'Content-Type': 'application/json'})
                response = connection.getresponse()
                result = json.loads(response.read(1024*1024))
                if response.status != 200:
                    raise ValueError(str(result))
                result['checkpoint'] = state['checkpoint']
                return result
            finally:
                connection.close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != '/health':
            return self.reply(404, dict(error='Not found'))
        self.reply(200, self.server.models.status())

    def do_POST(self):
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 4*1024**2:
                raise ValueError('Invalid request size')
            payload = json.loads(self.rfile.read(size))
            if self.path == '/select':
                self.server.models.select(payload.get('checkpoint'))
                self.reply(202, self.server.models.status())
            elif self.path == '/predict':
                self.reply(200, self.server.models.predict(payload))
            else:
                self.reply(404, dict(error='Not found'))
        except Exception as exc:
            self.reply(409, dict(error=str(exc)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--training-session')
    parser.add_argument('--port', type=int, default=8010)
    args = parser.parse_args()
    if args.prepare:
        stage(selected=args.training_session)
        return 0
    lock = (STAGING/'host.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    models = Models(STAGING)
    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        server.models = models
        models.select(CHECKPOINTS[0])
        print('Interactive host ready; checkpoints stay loaded between trials. Pod remains running.', flush=True)
        try:
            server.serve_forever()
        finally:
            # A loading thread must finish before shutdown can release its child.
            with models.lock:
                models.stop_child()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
