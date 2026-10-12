"""Watchdog checks use harmless local children and mocked pod API calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import training.benchmark_supervisor as supervisor
from training.benchmark_supervisor import supervise,kill_benchmark_tree
from training.pod_environment import recover_pod_environment


class PodEnvironmentTests(unittest.TestCase):
    def test_recovers_only_missing_control_variables(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'environ'
            path.write_bytes(b'RUNPOD_POD_ID=pod123\0RUNPOD_API_KEY=secret\0HF_TOKEN=other\0PATH=/bin\0')
            env={'RUNPOD_POD_ID':'explicit'}
            self.assertEqual(recover_pod_environment(env,path),('explicit','secret'))
            self.assertEqual(env,{'RUNPOD_POD_ID':'explicit','RUNPOD_API_KEY':'secret'})

    def test_missing_and_placeholder_errors_identify_variable_without_key(self):
        for env,message in [({},'RUNPOD_POD_ID'),({'RUNPOD_POD_ID':'pod123'},'RUNPOD_API_KEY'),
                            ({'RUNPOD_POD_ID':'pod123','RUNPOD_API_KEY':'{{ RUNPOD_SECRET_private }}'},'placeholder')]:
            with self.assertRaisesRegex(RuntimeError,message) as error:
                recover_pod_environment(env,Path('/missing'))
            self.assertNotIn('RUNPOD_SECRET_private',str(error.exception))


class SupervisorTests(unittest.TestCase):
    def test_training_watcher_waits_past_inference_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'status.json').write_text('{}')
            statuses=['{"phase":"inference_ready"}', '{"phase":"awaiting_sync","outcome":"completed"}']
            with patch.object(Path,'read_text',side_effect=statuses), \
                 patch('training.benchmark_supervisor.time.sleep') as sleep:
                self.assertEqual(supervisor.watch(folder,inference_handoff=False),0)
            sleep.assert_called_once()

    def run_child(self,source,hard_limit=10):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            with patch('training.benchmark_supervisor.stop_pod',return_value=0) as stop:
                result=supervise([sys.executable,'-c',source],directory,'pod123','secret',
                                 hard_limit=hard_limit,sync_grace=0)
            status=json.loads((directory/'status.json').read_text())
            stop.assert_called_once()
            self.assertNotIn('secret',(directory/'status.json').read_text())
            return result,status

    def test_completed_and_failed_children_both_request_stop(self):
        for source,outcome,code in [('print("done")','completed',0),('raise SystemExit(3)','failed',1)]:
            result,status=self.run_child(source)
            self.assertEqual(result,code)
            self.assertEqual(status['outcome'],outcome)
            self.assertEqual(status['stop_state'],'accepted')

    def test_outer_timeout_kills_local_child_and_requests_stop(self):
        result,status=self.run_child('import time; time.sleep(30)',hard_limit=.05)
        self.assertEqual(result,1)
        self.assertEqual(status['outcome'],'wall_limit_reached')
        self.assertEqual(status['stop_state'],'accepted')

    def test_setup_failure_still_requests_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            with patch('training.benchmark_supervisor.subprocess.Popen',side_effect=OSError('launch failed')), \
                 patch('training.benchmark_supervisor.stop_pod',return_value=0) as stop:
                self.assertEqual(supervise(['missing'],directory,'pod123','secret',sync_grace=0),1)
            stop.assert_called_once()
            self.assertEqual(json.loads((directory/'status.json').read_text())['outcome'],'wrapper_failed')

    def test_unwritable_status_does_not_prevent_stop_attempt(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch('training.benchmark_supervisor.write_status',side_effect=OSError('disk full')), \
             patch('training.benchmark_supervisor.stop_pod',return_value=0) as stop:
            self.assertEqual(supervise(['unused'],Path(folder),'pod123','secret',sync_grace=0),1)
        stop.assert_called_once()

    def test_kills_candidate_and_nested_worker_groups_but_not_other_jobs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            # Outer benchmark 100 -> separate candidate group 200 -> nested worker 201.
            for pid,parent,group in [(100,1,100),(200,100,200),(201,200,200),(300,1,300)]:
                path=root/str(pid);path.mkdir();(path/'stat').write_text(f'{pid} (python child) S {parent} {group} 0')
            process=SimpleNamespace(pid=100,wait=lambda **kwargs:None)
            with patch('training.benchmark_supervisor.os.killpg') as kill, \
                 patch('training.benchmark_supervisor.os.getpgrp',return_value=999):
                kill_benchmark_tree(process,root)
            self.assertEqual({call.args[0] for call in kill.call_args_list},{100,200})

    def test_launcher_detaches_worker_with_no_ssh_streams_and_keeps_lock_in_child(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch('sys.argv',['supervisor','--launch']), \
             patch.object(supervisor,'BENCHMARK_ROOT',Path(folder)), \
             patch.object(supervisor,'preflight',return_value=('pod123','secret')), \
             patch.object(supervisor,'watch',return_value=0), \
             patch.object(supervisor.subprocess,'Popen') as launch:
            self.assertEqual(supervisor.main(),0)
            kwargs=launch.call_args.kwargs
            self.assertTrue(kwargs['start_new_session'])
            self.assertEqual(kwargs['stdin'],subprocess.DEVNULL)
            self.assertEqual(len(kwargs['pass_fds']),1)
            self.assertNotIn('secret',str(launch.call_args.args))

    def test_stop_failure_is_reported_in_session_status(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            with patch('training.benchmark_supervisor.stop_pod',return_value=1):
                self.assertEqual(supervise([sys.executable,'-c','pass'],directory,'pod123','secret',sync_grace=0),1)
            self.assertEqual(json.loads((directory/'status.json').read_text())['stop_state'],'failed')


if __name__=='__main__':unittest.main()
