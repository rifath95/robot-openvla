"""Small independent SSH heartbeats; simulator requests cannot block cleanup leases."""
import shlex
import subprocess
import threading


class Heartbeat:
    def __init__(self,ssh,remote):
        self.ssh=ssh;self.remote=remote;self.stopped=threading.Event();self.thread=None
    def tick(self):
        try:subprocess.run(self.ssh+['touch '+shlex.quote(self.remote+'/heartbeat')],stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,check=True,timeout=20)
        except (OSError,subprocess.SubprocessError):pass
    def loop(self):
        while not self.stopped.is_set():
            self.tick();self.stopped.wait(15)
    def __enter__(self):
        self.tick();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stopped.set();self.thread.join(timeout=2)
