"""Receive image/instruction requests; return fixed test values or resident OpenVLA predictions."""

import argparse
import base64
import binascii
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
import os
from pathlib import Path
import threading
import time
import warnings

from PIL import Image, UnidentifiedImageError


MAX_BODY_BYTES = 8 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 4_000_000
ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
TEST_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def send_json(self, status, payload):
        body = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_GET(self):
        if self.path != "/health":
            self.send_json(404, {"error": "Unknown endpoint"})
            return
        self.send_json(200, {"status": "ready", "mode": self.server.mode,
                             "model_loaded": self.server.predictor is not None,
                             "model_load_seconds": getattr(self.server.predictor, "load_seconds", None)})

    def do_POST(self):
        if self.path != "/predict":
            self.send_json(404, {"error": "Unknown endpoint"})
            return
        started = time.perf_counter()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY_BYTES:
                self.send_json(413, {"error": "Body must be between 1 byte and 8 MiB"})
                return
            if self.headers.get_content_type() != "application/json":
                self.send_json(415, {"error": "Expected application/json"})
                return
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete request body")
            received = time.perf_counter()
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            request_id = payload.get("request_id")
            instruction = payload.get("instruction")
            encoded = payload.get("image_png_base64")
            if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
                raise ValueError("request_id must be a nonempty string of at most 128 characters")
            if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 4096:
                raise ValueError("instruction must be nonempty and at most 4096 characters")
            if not isinstance(encoded, str):
                raise ValueError("image_png_base64 must be a string")
            png = base64.b64decode(encoded, validate=True)
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(png)) as image:
                    if image.format != "PNG":
                        raise ValueError("Expected a PNG image")
                    image.load()
                    rgb = image.convert("RGB")
                    width, height = rgb.size
            prepared = time.perf_counter()
        except (ValueError, TypeError, OSError, binascii.Error, UnidentifiedImageError,
                Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            self.send_json(400, {"error": str(exc)})
            return
        inference = {"action": TEST_ACTION,
                     "action_convention": "Panda controller: world XYZ metres, rotation radians, gripper -1 closed / +1 open",
                     "timings_seconds": {"processor_and_device_transfer": None, "model_inference": None}}
        if self.server.predictor is not None:
            if not self.server.prediction_lock.acquire(blocking=False):
                self.send_json(503, {"error": "Model is busy; wait for the current prediction to finish", "request_id": request_id})
                return
            try:
                inference = self.server.predictor.predict(rgb, instruction)
            except Exception as exc:
                self.log_error("Prediction failed: %s", exc)
                self.send_json(500, {"error": "OpenVLA prediction failed; inspect the server log", "request_id": request_id})
                return
            finally:
                self.server.prediction_lock.release()
        finished = time.perf_counter()
        timings = inference.pop("timings_seconds")
        self.send_json(200, {
            "request_id": request_id,
            "mode": self.server.mode,
            "model_loaded": self.server.predictor is not None,
            "instruction": instruction,
            "image": {"width": width, "height": height, "mode": "RGB", "png_bytes": len(png)},
            **inference,
            "action_names": ACTION_NAMES,
            "timings_seconds": {
                "receive_body": received - started,
                "preprocess": prepared - received,
                **timings,
                "server_processing": finished - started,
            },
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-runtime-seconds", type=float, default=900)
    parser.add_argument("--mode", choices=("connection_test", "openvla"), default="connection_test")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--unnorm-key", default="bridge_orig")
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).with_name(".cache-openvla-server"))
    args = parser.parse_args()
    if not math.isfinite(args.max_runtime_seconds) or args.max_runtime_seconds <= 0:
        parser.error("--max-runtime-seconds must be positive")
    if args.host not in ("127.0.0.1", "localhost"):
        parser.error("Use a loopback address and an SSH tunnel; this test server has no authentication")
    predictor = None
    if args.mode == "openvla":
        if args.model_dir is None:
            parser.error("--model-dir is required for --mode openvla")
        cache = args.cache_dir.resolve()
        os.environ["HF_HOME"] = str(cache)
        os.environ["HF_MODULES_CACHE"] = str(cache / "modules")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        print(f"Loading OpenVLA from {args.model_dir} on {args.device}; HTTP readiness follows successful loading.", flush=True)
        from openvla_backend import OpenVLAPredictor
        predictor = OpenVLAPredictor(args.model_dir, device=args.device, unnorm_key=args.unnorm_key)
        print(f"Model loaded in {predictor.load_seconds:.2f}s", flush=True)
    with ThreadingHTTPServer((args.host, args.port), Handler) as server:
        server.mode = args.mode
        server.predictor = predictor
        server.prediction_lock = threading.Lock()
        timer = threading.Timer(args.max_runtime_seconds, server.shutdown)
        timer.daemon = True
        timer.start()
        print(f"{args.mode} server ready at http://{args.host}:{server.server_port}", flush=True)
        print("Robot movement is not executed. Use an SSH tunnel for remote requests.", flush=True)
        print(f"Server stops accepting requests after {args.max_runtime_seconds:g}s of readiness. Loading and an in-flight prediction may extend this. This does NOT terminate a rented pod or stop billing.", flush=True)
        try:
            server.serve_forever(poll_interval=0.1)
        except KeyboardInterrupt:
            pass
        finally:
            timer.cancel()


if __name__ == "__main__":
    main()
