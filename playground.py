"""Press Run/Play on the Mac, paste the pod SSH command, and press Enter."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh', help='Optional: skip the interactive SSH prompt')
    parser.add_argument('--training-session', help='Optional: choose a specific training session')
    args = parser.parse_args(argv)
    python = ROOT / '.venv/bin/python'
    if not python.is_file():
        print('Simulator environment missing: ' + str(python), file=sys.stderr)
        return 1

    print('OpenVLA playground — keep this terminal open while testing.', flush=True)
    try:
        ssh = (args.ssh or '').strip()
        while not ssh:
            ssh = input('Paste the pod SSH command, then press Enter: ').strip()

        # Use the simulator environment regardless of the editor's selected Python
        # or working directory. Never execute pasted SSH text through a shell.
        command = [str(python), '-u', '-m', 'cloud.interactive_bowl_session', '--ssh', ssh]
        if args.training_session:
            command.extend(['--training-session', args.training_session])
        if sys.platform == 'darwin' and Path('/usr/bin/caffeinate').is_file():
            command = ['/usr/bin/caffeinate', '-i'] + command
        return subprocess.run(command, cwd=ROOT).returncode
    except EOFError:
        print('\nRun this file in a terminal so you can paste the SSH command.', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\nPlayground closed. The cloud pod remains running; stop it in RunPod when finished.')
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
