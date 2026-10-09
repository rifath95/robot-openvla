# Camera-to-server connection test

This is the first Phase 2 communication test. It sends a PNG camera image and
instruction to an HTTP server, decodes the image there, and returns a **fixed**
seven-number action. It does not load OpenVLA, tokenize the prompt, or move the arm.
No GPU is required for this test.

## Run both ends on your Mac first

In one local VS Code terminal, from this repository:

```bash
.venv-openvla/bin/python connection_server.py
```

The server needs only Pillow, which is already in the OpenVLA environment. A fresh
server environment can install it with `python -m pip install -r requirements-connection-server.txt`.
There is no need to install PyTorch for this server.

Leave that terminal running. In a second local terminal:

```bash
.venv/bin/python connection_client.py
```

The client captures the home scene with the existing MuJoCo camera, sends it with
the instruction `pick up the red cube`, and prints the response. Each run saves
`input.png`, `result.json`, and `status.json` in a new `outputs/connection_test_*`
folder. Status becomes `Completed` on success or `Failed` with an error. Existing
`camera_image.png` and other experiment outputs are not changed.

The expected response is `[0, 0, 0, 0, 0, 0, 1]`: zero position/rotation changes
and an open-gripper value in the Panda controller convention. These numbers are
never executed. They are not model predictions or Bridge-normalized actions.

Optional arguments:

```bash
.venv/bin/python connection_client.py --instruction "inspect the red cube"
.venv/bin/python connection_client.py --image camera_image.png
```

Using `--image` skips simulation and does not require MuJoCo in the client
interpreter. Stop the server with Control+C. By default it also exits after
900 seconds; change this with `--max-runtime-seconds`.

## Later: server on a cloud pod, client on your Mac

After the current code is pushed to GitHub, clone it **on the pod's container
disk**, for example `/root/robot-openvla`, rather than the Global volume mounted
at `/workspace`. Install `requirements-connection-server.txt` there and run:

```bash
python connection_server.py
```

In a **local Mac terminal**, open an SSH tunnel using that running pod's current
IP/port from RunPod's Connect tab. Replace `POD_IP` and `POD_SSH_PORT` below:

```bash
ssh -N -F /dev/null -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -L 127.0.0.1:8000:127.0.0.1:8000 -i ~/.ssh/id_ed25519_maxalderone -p POD_SSH_PORT root@POD_IP
```

Leave the tunnel running and run the same client command on the Mac. Its
`http://127.0.0.1:8000` address now forwards to the remote server. Stop any local
test server before opening the tunnel, so the port is available. The HTTP test
server binds only to loopback and has no authentication; use the encrypted SSH
tunnel rather than exposing an HTTP port publicly. No provider-specific client
code or RunPod API key is needed.

The server runtime limit **only stops the Python server**. It does not stop or
terminate the pod, and billing continues until you stop/terminate the pod in
RunPod. A billing watchdog is not implemented in this connection test.

## Understanding the timings

`result.json` records:

- Client camera capture (or file copy) and base64/JSON encoding.
- Connection setup, request send, waiting for response headers, response body
  read, overall request round trip, and total client elapsed time.
- Server request-body receive and preprocessing (JSON, base64, PNG decode, RGB
  conversion), plus total server processing before response serialization.
- `model_inference: null` and `robot_movement: null`, since neither occurs.

Each side uses its own monotonic clock. The client send duration measures sending
bytes into the network stack; it does not prove the image arrived then. Header
wait includes network transit and server work, and response read may consume
bytes already buffered. These are **not isolated one-way upload/download
latencies**, and local results do not predict cloud latency. Later OpenVLA
integration will separately measure tokenizer/image processing, synchronized GPU
prediction, robot movement, and the next camera capture.

## Current verification and next step

Verified locally on 2026-10-09: a real 640x480 MuJoCo image was uploaded and decoded,
the request ID matched, and seven finite test numbers came back. Invalid image
and instruction requests were rejected; an unreachable server left a failed
status. Automatic server exit also passed. The measured local request round trip
was approximately 18 ms. Saved successful-test artifacts are in
`outputs/connection_test_verified/` (excluded from Git). Cloud connectivity has
not been tested with these scripts yet.

The communication test subsequently passed on a pod, with approximately 1.58
seconds round trip. Real inference is now available as an explicit server mode,
but still needs cloud-GPU validation; follow [REMOTE_INFERENCE.md](REMOTE_INFERENCE.md).
