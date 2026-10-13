"""Reserve a dynamic loopback port before loading; publish only when ready."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock,patch
from urllib.request import urlopen
from common.paths import PROJECT_ROOT

class DynamicPortTests(unittest.TestCase):
    def test_bound_socket_precedes_model_loading(self):
        from cloud import connection_server
        with tempfile.TemporaryDirectory() as tmp:
            ready=Path(tmp)/'ready.json';events=[]
            server=MagicMock();server.server_port=12345
            manager=MagicMock();manager.__enter__.return_value=server
            def bind(*args):events.append('bound');return manager
            def load(*args,**kwargs):
                self.assertEqual(events,['bound']);self.assertFalse(ready.exists())
                events.append('loaded');return MagicMock(load_seconds=1)
            with patch('sys.argv',['server','--mode','openvla','--model-dir',tmp,'--port','0','--ready-file',str(ready)]), \
                patch.object(connection_server,'ThreadingHTTPServer',side_effect=bind), \
                patch('inference.openvla_backend.OpenVLAPredictor',side_effect=load), \
                patch.object(connection_server.threading,'Timer'):
                connection_server.main()
            self.assertEqual(json.loads(ready.read_text())['port'],12345)
            self.assertEqual(events,['bound','loaded'])

    def test_real_dynamic_port_can_coexist_with_an_occupied_listener(self):
        with tempfile.TemporaryDirectory() as tmp,socket.socket() as occupied:
            occupied.bind(('127.0.0.1',0));occupied.listen()
            ready=Path(tmp)/'ready.json'
            process=subprocess.Popen([sys.executable,'-u','-m','cloud.connection_server','--port','0',
                '--ready-file',str(ready),'--max-runtime-seconds','30'],cwd=PROJECT_ROOT,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                deadline=time.monotonic()+10
                while not ready.exists() and time.monotonic()<deadline:
                    if process.poll() is not None:self.fail('Test server exited')
                    time.sleep(.05)
                record=json.loads(ready.read_text())
                self.assertNotEqual(record['port'],occupied.getsockname()[1])
                with urlopen(f"http://127.0.0.1:{record['port']}/health",timeout=2) as response:
                    self.assertEqual(json.load(response)['mode'],'connection_test')
            finally:
                process.terminate();process.wait(timeout=5)

if __name__=='__main__':unittest.main()
