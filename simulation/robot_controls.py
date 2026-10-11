"""Local button panel; robot commands never pass through viewer shortcuts."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import queue
import secrets
import threading
import webbrowser


COMMANDS = {f"{axis}{sign}" for axis in ("x", "y", "z", "roll", "pitch", "yaw")
            for sign in ("+", "-")} | {"open", "close"}


@contextmanager
def control_panel(pending_actions):
    token = secrets.token_urlsafe(24)
    page = """<!doctype html><meta charset="utf-8">
<title>Panda controls</title>
<style>body{font:18px system-ui;max-width:520px;margin:40px auto;background:#eef2f6;color:#182535}
button{font:inherit;padding:14px;margin:5px;min-width:140px;cursor:pointer}
#status{min-height:3em}h1{font-size:28px}</style>
<h1>Panda controls</h1><p>Each click moves 2.5 cm or rotates 0.1 radians in world coordinates.</p>
<div id="buttons"></div><p id="status">Ready. Keep the simulator open beside this panel.</p>
<p>Use these buttons for the robot. MuJoCo keys control only the viewer.</p>
<script>
const commands=['x+','x-','y+','y-','z+','z-','roll+','roll-','pitch+','pitch-','yaw+','yaw-','open','close'];
for(const command of commands){
 const button=document.createElement('button');button.textContent=command;
 button.onclick=async()=>{
  button.disabled=true;
  try{const response=await fetch('/command',{method:'POST',headers:{'X-Control-Token':'TOKEN'},body:command});
   document.querySelector('#status').textContent=await response.text();
  }catch(error){document.querySelector('#status').textContent='Disconnected. Restart the demo and use its new panel.';}
  finally{button.disabled=false;}
 };document.querySelector('#buttons').appendChild(button);
 if(command.endsWith('-'))document.querySelector('#buttons').appendChild(document.createElement('br'));
}
</script>""".replace("TOKEN", token).encode()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, content_type="text/plain; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/":
                self.reply(404, b"Not found")
                return
            self.reply(200, page, "text/html; charset=utf-8")

        def do_POST(self):
            if self.path != "/command" or self.headers.get("X-Control-Token") != token:
                self.reply(403, b"Forbidden")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16:
                    raise ValueError
                command = self.rfile.read(length).decode("ascii")
                if command not in COMMANDS:
                    raise ValueError
            except (ValueError, UnicodeError):
                self.reply(400, b"Invalid command")
                return
            try:
                pending_actions.put_nowait(command)
            except queue.Full:
                self.reply(409, b"A command is pending. Try again shortly.")
                return
            self.reply(200, f"Queued {command}. See the terminal for IK results.".encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Robot controls: {url}", flush=True)
    try:
        webbrowser.open(url)
        yield url
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
