"""Download OpenVLA, then attempt one local action prediction.

Run this file with VS Code's Run Python File button. It always uses the
project's isolated .venv-openvla interpreter and never moves the robot.
"""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
STATUS = OUTPUT / "openvla_mac_status.json"
LOG = OUTPUT / "openvla_mac_test.log"


def status(phase, **details):
    STATUS.write_text(json.dumps({"phase": phase, "log": str(LOG), **details}, indent=2))
    print(phase, flush=True)


def main():
    OUTPUT.mkdir(exist_ok=True)
    interpreter = ROOT / ".venv-openvla" / "bin" / "python"
    script = ROOT / "openvla_mac.py"
    environment = os.environ.copy()
    environment["HF_HUB_DISABLE_XET"] = "1"
    environment["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    environment["PYTHONUNBUFFERED"] = "1"
    with LOG.open("a") as log:
        for phase, arguments in (
            ("Checking Apple GPU", ["--preflight"]),
            ("Downloading checkpoint", ["--download"]),
            ("Loading model and predicting one action", ["--device", "mps"]),
        ):
            status(phase)
            log.write(f"\n--- {phase} ---\n")
            log.flush()
            result = subprocess.run(
                [str(interpreter), str(script), *arguments],
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            )
            if result.returncode:
                status("Failed", failed_stage=phase, returncode=result.returncode)
                print(f"See {LOG} for the exact error.", flush=True)
                return result.returncode
    status("Completed", action_file=str(OUTPUT / "openvla_mac_action.json"))
    print("One action predicted locally. It has not been sent to the robot.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
