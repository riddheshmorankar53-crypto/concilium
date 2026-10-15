"""A client that opens a TCP connection to the TLS board and never handshakes must not stall the others.
Before the fix the board did every handshake in its one serving thread, so the demo's crashed member could stall
it for ~2 h (TCP keepalive) and every participant timed out (TLS=1 TASK=cnn, 2026-10-11).
    python board_tls_check.py"""
import socket
import ssl
import tempfile
import threading
import time
import urllib.error
import urllib.request

import deploy

d = tempfile.mkdtemp()
deploy.self_signed(d, "localhost")
deploy.PORT = 0
server = deploy.board_server(f"{d}/cert.pem", f"{d}/key.pem", "127.0.0.1")
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()
silent = socket.create_connection(("127.0.0.1", port))  # never sends a ClientHello
time.sleep(0.5)
ctx = ssl.create_default_context(cafile=f"{d}/cert.pem")
ctx.check_hostname = False  # the demo certificate names localhost, not 127.0.0.1
t = time.time()
try:
    urllib.request.urlopen(f"https://127.0.0.1:{port}/nothing", timeout=10, context=ctx)
except urllib.error.HTTPError as e:  # 404: the board answered
    assert e.code == 404, e
print(f"board answered next to a silent client in {time.time() - t:.1f}s")
assert time.time() - t < 5, "a silent client must not stall the board"
