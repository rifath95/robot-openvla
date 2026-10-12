# Keep the model loaded between simulator trials

Requested and implemented on 2026-10-12 after the checkpoint-2,000 bowl trials
finished and the user stopped the pod.

The shell loop launches a separate `cloud.cloud_session` for each scene seed.
Ctrl+C ends that session and its cleanup stops the model server, releasing GPU
memory. The next seed then requires another model load. This wastes time when
the user wants several trials with the same checkpoint.

The launcher now allows multiple trials with different scene seeds in one cloud session. Keep
the server and loaded model alive between those trials, and save each trial's
seed, outputs, timing, and replay separately. Distinguish ending a local trial
from explicitly ending the cloud session. Explicit session exit should retain
server cleanup, and the server runtime limit must remain effective. Loading a
different checkpoint may still require replacing the loaded model.

Use `--task bowl --scene-seeds 0 1 2` to run a sequence using a single model load.
Each window starts paused. Close the window or press Esc to finish that trial;
the next queued seed starts automatically. Once the queue finishes, Enter repeats
the last seed, an integer selects another seed, and `q` explicitly ends the session.
Ctrl+C also ends the session. Server errors or expiry of its existing runtime limit
can end the session; keeping the model loaded does not disable that limit.

Verified with a launcher test covering queued seeds, invalid input, selecting a
new seed, repeating, and quitting: one bootstrap/model load and one final server
shutdown. The prior bowl-scene test verifies recorded replay geometry and camera.
Model-server shutdown does not stop the rented pod; stop it in RunPod afterward.
