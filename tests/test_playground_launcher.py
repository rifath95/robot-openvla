"""The editor launcher must use the simulator environment and preserve SSH text."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import playground


class PlaygroundLauncherTests(unittest.TestCase):
    def test_editor_prompt_uses_repo_environment_and_argv(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            python = root / '.venv/bin/python'
            python.parent.mkdir(parents=True)
            python.touch()
            ssh = 'ssh root@host -p 22100 -i ~/.ssh/id_ed25519'
            with patch.object(playground, 'ROOT', root), \
                    patch.object(playground.sys, 'platform', 'linux'), \
                    patch('builtins.input', side_effect=['', ssh]), \
                    patch.object(playground.subprocess, 'run') as run:
                run.return_value.returncode = 7
                self.assertEqual(playground.main([]), 7)
                run.assert_called_once_with(
                    [str(python), '-u', '-m', 'cloud.interactive_bowl_session', '--ssh', ssh],
                    cwd=root)

    def test_missing_environment_does_not_connect_or_prompt(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(playground, 'ROOT', Path(folder)), \
                patch('builtins.input') as prompt, \
                patch.object(playground.subprocess, 'run') as run:
            self.assertEqual(playground.main([]), 1)
            prompt.assert_not_called()
            run.assert_not_called()

    def test_mac_keeps_awake_and_passes_optional_session(self):
        with patch.object(playground.Path, 'is_file', return_value=True), \
                patch.object(playground.sys, 'platform', 'darwin'), \
                patch('builtins.input') as prompt, \
                patch.object(playground.subprocess, 'run') as run:
            run.return_value.returncode = 0
            self.assertEqual(playground.main(['--ssh', 'ssh root@host',
                                             '--training-session', 'selected-run']), 0)
            prompt.assert_not_called()
            command = run.call_args.args[0]
            self.assertEqual(command[:2], ['/usr/bin/caffeinate', '-i'])
            self.assertEqual(command[-2:], ['--training-session', 'selected-run'])


if __name__ == '__main__':
    unittest.main()
