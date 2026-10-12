"""Launcher checks without a GPU, SSH connection, or Docker build."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cloud.cloud_session as cloud_session


class SessionTests(unittest.TestCase):
    def test_multiple_seeds_share_one_server_until_explicit_quit(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'.venv/bin').mkdir(parents=True)
            (root/'.venv/bin/python').touch()
            key=root/'key';key.touch()
            argv=['cloud_session.py','--ssh',f'ssh root@host -p 22 -i {key}',
                  '--task','bowl','--scene-seeds','0','1','2']
            tunnel=MagicMock();tunnel.poll.return_value=None
            with patch('sys.argv',argv),patch.object(cloud_session,'ROOT',root), \
                 patch('cloud.cloud_session.socket.socket'), \
                 patch('cloud.cloud_session.subprocess.run',return_value=SimpleNamespace(returncode=0)) as run, \
                 patch('cloud.cloud_session.subprocess.Popen',return_value=tunnel), \
                 patch('cloud.cloud_session.wait_for_server',return_value={'model_load_seconds':1}) as ready, \
                 patch('builtins.input',side_effect=['invalid','3','','q']):
                self.assertEqual(cloud_session.main(),0)
            commands=[c.args[0] for c in run.call_args_list]
            trials=[c for c in commands if 'inference.remote_loop' in c]
            self.assertEqual([c[c.index('--scene-seed')+1] for c in trials],['0','1','2','3','3'])
            self.assertEqual(sum('input' in c.kwargs for c in run.call_args_list),1)
            self.assertEqual(sum('cloud_stop.sh' in c[-1] for c in commands),0)
            self.assertEqual(ready.call_count,5)
            tunnel.terminate.assert_called_once()

    def test_adapter_bootstrap_merges_before_starting_panda_server(self):
        script = cloud_session.bootstrap_script(1800, '/workspace/run/checkpoint_000020')
        self.assertIn('--adapter /workspace/run/checkpoint_000020', script)
        self.assertIn('mktemp -d /root/panda-evaluation-', script)
        self.assertIn('bash scripts/cloud_start.sh 1800 "$merged_dir/model" panda_grasp_v1', script)
        self.assertLess(script.index('training.merge_adapter'), script.index('bash scripts/cloud_start.sh'))
        self.assertLess(script.index('cloud.server_reuse'),script.index('bash scripts/cloud_stop.sh'))
        self.assertIn('--record',script)

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
        with patch('cloud.cloud_session.urlopen', side_effect=responses), patch('cloud.cloud_session.time.sleep'):
            result = cloud_session.wait_for_server(tunnel, 5)
        self.assertTrue(result['model_loaded'])

    def test_setup_only_disconnect_keeps_server_and_closes_tunnel(self):
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / 'key'
            key.touch()
            tunnel = MagicMock()
            tunnel.poll.return_value = None
            argv = ['cloud_session.py', '--ssh', f'ssh root@host -p 22053 -i {key}', '--setup-only']
            with patch('sys.argv', argv), patch('cloud.cloud_session.subprocess.run') as run, \
                 patch('cloud.cloud_session.subprocess.Popen', return_value=tunnel) as popen, \
                 patch('cloud.cloud_session.socket.socket'), \
                 patch('cloud.cloud_session.wait_for_server', return_value={'model_load_seconds': 1}), \
                 patch('cloud.cloud_session.time.sleep', side_effect=KeyboardInterrupt):
                self.assertEqual(cloud_session.main(), 0)
            self.assertIn('-L', popen.call_args.args[0])
            self.assertIn('127.0.0.1:8000:127.0.0.1:8000', popen.call_args.args[0])
            payload = run.call_args_list[1].kwargs['input'].decode()
            self.assertIn('git clone https://github.com/rifath95/robot-openvla.git', payload)
            self.assertIn('bash scripts/cloud_start.sh 1800', payload)
            self.assertFalse(any('cloud_stop.sh' in call.args[0][-1] for call in run.call_args_list))
            tunnel.terminate.assert_called_once()

    def test_explicit_stop_only_does_not_bootstrap_or_open_tunnel(self):
        with tempfile.TemporaryDirectory() as tmp:
            key=Path(tmp)/'key';key.touch()
            with patch('sys.argv',['cloud_session.py','--ssh',f'ssh root@host -i {key}','--stop-server']), \
                 patch('cloud.cloud_session.subprocess.run') as run, \
                 patch('cloud.cloud_session.subprocess.Popen') as popen:
                run.return_value.returncode=0
                self.assertEqual(cloud_session.main(),0)
                self.assertIn('cloud_stop.sh',run.call_args.args[0][-1])
                run.assert_called_once();popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
