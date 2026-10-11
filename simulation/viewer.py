"""macOS MuJoCo GUI interpreter support."""
from pathlib import Path
import os
import sys
import mujoco.viewer
from common.paths import PROJECT_ROOT

def _restart_with_mjpython() -> None:
    """Use MuJoCo's required macOS GUI launcher despite spaces in the path."""
    if sys.platform != "darwin" or mujoco.viewer._MJPYTHON is not None:
        return

    venv_bin = PROJECT_ROOT / ".venv" / "bin"
    python = venv_bin / "python"
    mjpython = venv_bin / "mjpython"
    script = str(Path(sys.argv[0]).resolve())
    module = getattr(sys.modules.get("__main__"), "__spec__", None)
    launch = ["-m", module.name] if module is not None else [script]
    os.execv(
        str(python),
        [str(python), str(mjpython), *launch, *sys.argv[1:]],
    )
