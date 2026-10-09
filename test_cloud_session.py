"""Launcher checks without a GPU, SSH connection, or Docker build."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cloud_session


class SessionTests(unittest.TestCase):
    def test_parses_direct_tcp_and_uses_explicit_private_key(self):
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / 'my key'
            key.touch()
            host, port, actual = cloud_session.parse_ssh('ssh root@194.68.245.86 -p 22053 -i ~/.ssh/id_ed25519', key)
            self.assertEqual((host, port, actual), ('root@194.68.245.86', 22053, key))
            arguments = cloud_session.ssh_arguments(host, port, actual)
            self.assertIn(str(key), arguments)
            self.assertIn('PasswordAuthentication=no', arguments)

    def test_shell_commands_and_unknown_options_rejected(self):
        for command in ('ssh root@host -p 22; touch /tmp/no', 'ssh root@host -o ProxyCommand=evil',
                        'ssh root@host -p 70000', 'ssh root@host -i', 'bash evil'):
            with self.subTest(command=command), self.assertRaises(ValueError):
                cloud_session.parse_ssh(command)

    def test_public_key_is_not_accepted_as_private_key(self):
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / 'key.pub'
            key.touch()
            with self.assertRaises(ValueError):
                cloud_session.parse_ssh('ssh root@host -p 22', key)

    def test_readiness_requires_a_real_loaded_model(self):
        tunnel = MagicMock()
        tunnel.poll.return_value = None
        replies = [dict(status='ready', mode='connection_test', model_loaded=False),
                   dict(status='ready', mode='openvla', model_loaded=True)]
        responses = []
        for reply in replies:
            context = MagicMock()
            context.__enter__.return_value = io.StringIO(json.dumps(reply))
            responses.append(context)
        with patch('cloud_session.urlopen', side_effect=responses), patch('cloud_session.time.sleep'):
            result = cloud_session.wait_for_server(tunnel, 5)
        self.assertTrue(result['model_loaded'])

    def test_setup_only_opens_tunnel_and_cleans_up_on_interrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / 'key'
            key.touch()
            tunnel = MagicMock()
            tunnel.poll.return_value = None
            argv = ['cloud_session.py', '--ssh', f'ssh root@host -p 22053 -i {key}', '--setup-only']
            with patch('sys.argv', argv), patch('cloud_session.subprocess.run') as run, \
                 patch('cloud_session.subprocess.Popen', return_value=tunnel) as popen, \
                 patch('cloud_session.socket.socket'), \
                 patch('cloud_session.wait_for_server', return_value={'model_load_seconds': 1}), \
                 patch('cloud_session.time.sleep', side_effect=KeyboardInterrupt):
                self.assertEqual(cloud_session.main(), 0)
            self.assertIn('-L', popen.call_args.args[0])
            self.assertIn('127.0.0.1:8000:127.0.0.1:8000', popen.call_args.args[0])
            payload = run.call_args_list[1].kwargs['input'].decode()
            self.assertIn('git clone https://github.com/rifath95/robot-openvla.git', payload)
            self.assertIn('bash scripts/cloud_start.sh 1800', payload)
            self.assertIn('cloud_stop.sh', run.call_args_list[-1].args[0][-1])
            tunnel.terminate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
