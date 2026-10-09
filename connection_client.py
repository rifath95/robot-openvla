"""Send a fresh MuJoCo camera image to the test server; never move the robot."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--instruction", default="pick up the red cube")
    parser.add_argument("--image", type=Path, help="Use an existing PNG instead of capturing MuJoCo")
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    url = urlsplit(args.server_url)
    if url.scheme != "http" or not url.hostname or url.path not in ("", "/") or url.query or url.fragment or url.username:
        parser.error("--server-url must be an HTTP origin, e.g. http://127.0.0.1:8000 (use an SSH tunnel for cloud)")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    request_id = uuid.uuid4().hex
    output = args.output_dir or Path(__file__).with_name("outputs") / f"connection_test_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{request_id[:8]}"
    output.mkdir(parents=True, exist_ok=True)
    def status(phase, **extra):
        (output / "status.json").write_text(json.dumps({"phase": phase, "request_id": request_id, **extra}, indent=2) + "\n")
    started = time.perf_counter()
    status("Capturing image")
    connection = None
    try:
        image_path = output / "input.png"
        if args.image:
            image_path.write_bytes(args.image.read_bytes())
        else:
            capture_png(image_path)
        captured = time.perf_counter()
        body = json.dumps({"request_id": request_id, "instruction": args.instruction,
                           "image_png_base64": base64.b64encode(image_path.read_bytes()).decode("ascii")}).encode()
        encoded = time.perf_counter()
        status("Sending image")
        connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=args.timeout_seconds)
        connect_started = time.perf_counter()
        connection.connect()
        connected = time.perf_counter()
        connection.request("POST", "/predict", body=body, headers={"Content-Type": "application/json"})
        sent = time.perf_counter()
        response = connection.getresponse()
        headers_received = time.perf_counter()
        raw = response.read(1024 * 1024 + 1)
        downloaded = time.perf_counter()
        if len(raw) > 1024 * 1024:
            raise ValueError("Response exceeds 1 MiB")
        result = json.loads(raw)
        if response.status != 200:
            raise ValueError(f"Server returned HTTP {response.status}: {result}")
        if result.get("request_id") != request_id or result.get("mode") != "connection_test" or result.get("model_loaded") is not False:
            raise ValueError("Unexpected server mode or request ID")
        action = result.get("action")
        if not isinstance(action, list) or len(action) != 7 or any(type(x) not in (int, float) or not math.isfinite(x) for x in action):
            raise ValueError("Response must contain seven finite action numbers")
        result["client_timings_seconds"] = {
            "capture_or_copy": captured - started,
            "encode_request": encoded - captured,
            "connect": connected - connect_started,
            "upload_send": sent - connected,
            "wait_for_response_headers": headers_received - sent,
            "response_body_read": downloaded - headers_received,
            "request_round_trip": downloaded - connected,
            "total": downloaded - started,
            "robot_movement": None,
        }
        result["timing_note"] = "Client send/read durations are local measurements, not isolated one-way network latency. Header wait includes network and server work. No model tokenization, inference, or robot movement occurs."
        result["robot_moved"] = False
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        status("Completed", elapsed_seconds=time.perf_counter() - started)
        print("Received fixed test action:", action)
        print("Server decoded image:", result["image"])
        print(f"Request round trip: {result['client_timings_seconds']['request_round_trip']:.4f}s")
        print("No model inference or robot movement performed.")
        print(f"Image, timings and status saved in {output}")
    except Exception as exc:
        status("Failed", error=str(exc), elapsed_seconds=time.perf_counter() - started)
        raise SystemExit(f"Connection test failed: {exc}\nDetails: {output / 'status.json'}") from exc
    finally:
        if connection:
            connection.close()


if __name__ == "__main__":
    main()
