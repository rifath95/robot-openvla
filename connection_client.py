"""Send a fresh MuJoCo image to a test or OpenVLA server; never move the robot."""

import argparse
import base64
from datetime import datetime, timezone
import http.client
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit
import uuid


def capture_png(path):
    import mujoco
    from capture_camera import SCENE_PATH, HEIGHT, WIDTH, save_rgb_png

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "scene_home")
    if home < 0:
        raise ValueError("scene_home keyframe is missing")
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        renderer.update_scene(data, camera="workspace_camera")
        save_rgb_png(path, renderer.render())


def request_prediction(image_path, instruction, *, server_url="http://127.0.0.1:8000",
                       expected_mode="openvla", timeout_seconds=180,
                       request_id=None, progress=None):
    """Request and validate one action, without touching any simulation state."""
    url = urlsplit(server_url)
    if url.scheme != "http" or not url.hostname or url.path not in ("", "/") or url.query or url.fragment or url.username:
        raise ValueError("server_url must be an HTTP origin; use an SSH tunnel for cloud")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if expected_mode not in ("openvla", "connection_test"):
        raise ValueError("Unknown expected server mode")
    request_id = request_id or uuid.uuid4().hex
    started = time.perf_counter()
    body = json.dumps({"request_id": request_id, "instruction": instruction,
                       "image_png_base64": base64.b64encode(Path(image_path).read_bytes()).decode("ascii")}).encode()
    encoded = time.perf_counter()
    if progress:
        progress("Sending image")
    connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout_seconds)
    try:
        connect_started = time.perf_counter()
        connection.connect()
        connected = time.perf_counter()
        connection.request("POST", "/predict", body=body, headers={"Content-Type": "application/json"})
        sent = time.perf_counter()
        if progress:
            progress("Waiting for OpenVLA prediction" if expected_mode == "openvla" else "Waiting for test response")
        response = connection.getresponse()
        headers_received = time.perf_counter()
        raw = response.read(1024 * 1024 + 1)
        downloaded = time.perf_counter()
        if len(raw) > 1024 * 1024:
            raise ValueError("Response exceeds 1 MiB")
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("Expected a response object")
        if response.status != 200:
            raise ValueError(f"Server returned HTTP {response.status}: {result}")
        if result.get("request_id") != request_id or result.get("mode") != expected_mode or result.get("model_loaded") is not (expected_mode == "openvla"):
            raise ValueError("Unexpected server mode or request ID")
        action = result.get("action")
        if not isinstance(action, list) or len(action) != 7 or any(type(x) not in (int, float) or not math.isfinite(x) for x in action):
            raise ValueError("Response must contain seven finite action numbers")
        result["client_timings_seconds"] = {
            "encode_request": encoded - started,
            "connect": connected - connect_started,
            "upload_send": sent - connected,
            "wait_for_response_headers": headers_received - sent,
            "response_body_read": downloaded - headers_received,
            "request_round_trip": downloaded - connected,
            "total_request": downloaded - started,
        }
        result["timing_note"] = "Client send/read durations are local measurements, not isolated one-way network latency. Header wait includes network and server work. Server preprocessing and inference use its own clock."
        return result
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--instruction", default="pick up the red cube")
    parser.add_argument("--image", type=Path, help="Use an existing PNG instead of capturing MuJoCo")
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-mode", choices=("connection_test", "openvla"), default="connection_test")
    args = parser.parse_args()
    url = urlsplit(args.server_url)
    if url.scheme != "http" or not url.hostname or url.path not in ("", "/") or url.query or url.fragment or url.username:
        parser.error("--server-url must be an HTTP origin, e.g. http://127.0.0.1:8000 (use an SSH tunnel for cloud)")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    request_id = uuid.uuid4().hex
    prefix = "remote_inference" if args.expected_mode == "openvla" else "connection_test"
    output = args.output_dir or Path(__file__).with_name("outputs") / f"{prefix}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{request_id[:8]}"
    output.mkdir(parents=True, exist_ok=True)
    def status(phase, **extra):
        (output / "status.json").write_text(json.dumps({"phase": phase, "request_id": request_id, **extra}, indent=2) + "\n")
    started = time.perf_counter()
    status("Capturing image")
    try:
        image_path = output / "input.png"
        if args.image:
            image_path.write_bytes(args.image.read_bytes())
        else:
            capture_png(image_path)
        captured = time.perf_counter()
        result = request_prediction(image_path, args.instruction, server_url=args.server_url,
                                    expected_mode=args.expected_mode, timeout_seconds=args.timeout_seconds,
                                    request_id=request_id, progress=status)
        action = result["action"]
        result["client_timings_seconds"].update(capture_or_copy=captured - started,
                                                 total=time.perf_counter() - started, robot_movement=None)
        result["robot_moved"] = False
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        status("Completed", elapsed_seconds=time.perf_counter() - started)
        print("Received OpenVLA action:" if args.expected_mode == "openvla" else "Received fixed test action:", action)
        print("Server decoded image:", result["image"])
        print(f"Request round trip: {result['client_timings_seconds']['request_round_trip']:.4f}s")
        if args.expected_mode == "openvla":
            print("Server timings:", result["timings_seconds"])
            print("Prediction displayed only; robot movement is not executed.")
        else:
            print("No model inference or robot movement performed.")
        print(f"Image, timings and status saved in {output}")
    except Exception as exc:
        status("Failed", error=str(exc), elapsed_seconds=time.perf_counter() - started)
        raise SystemExit(f"Connection test failed: {exc}\nDetails: {output / 'status.json'}") from exc


if __name__ == "__main__":
    main()
