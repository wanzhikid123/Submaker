import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from submaker.config import Profile
from submaker.llm import Client
from submaker.llm import APIError
from submaker.media import Cancelled
import pytest
import time


def test_actual_http_sends_effort_and_preserves_custom_path():
    captured = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))) )
            data = json.dumps({"choices":[{"finish_reason":"stop","message":{"content":'{"ok":true}'}}],"usage":{"total_tokens":9}}).encode()
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1",0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        profile = Profile(url=f"http://127.0.0.1:{server.server_port}/custom/v1", model="test-model", effort="high", json_mode=True)
        usages = []
        answer = Client(usage=usages.append, key_provider=lambda _: "").request(profile, "test", {"text":"data"})
        assert answer == {"ok": True}
        assert captured[0][0] == "/custom/v1/chat/completions"
        assert captured[0][1]["reasoning_effort"] == "high"
        assert captured[0][1]["response_format"] == {"type":"json_object"}
        assert usages == [{"total_tokens":9}]
    finally:
        server.shutdown()
        server.server_close()


def test_in_flight_http_request_can_be_paused_without_waiting_for_300s():
    entered = threading.Event()
    release = threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            entered.set()
            release.wait(3)
            try:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{}')
            except (BrokenPipeError, ConnectionResetError):
                pass
        def log_message(self,*args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1",0),Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever,daemon=True).start()
    cancel = threading.Event()
    def stop():
        assert entered.wait(2)
        cancel.set()
    threading.Thread(target=stop,daemon=True).start()
    profile=Profile(url=f"http://127.0.0.1:{server.server_port}/v1",model="m")
    began=time.monotonic()
    try:
        with pytest.raises(Cancelled):
            Client(key_provider=lambda _: "").request(profile,"test",{},cancel)
        assert time.monotonic()-began < 2
    finally:
        release.set()
        server.shutdown()
        server.server_close()


def test_authentication_failure_opens_circuit_until_explicit_retry():
    calls=[]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            calls.append(1)
            self.send_response(401)
            self.end_headers()
        def log_message(self,*args):
            pass
    server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    profile=Profile(url=f"http://127.0.0.1:{server.server_port}/v1",model="m")
    client=Client(key_provider=lambda _:"")
    try:
        for _ in range(2):
            with pytest.raises(APIError):
                client.request(profile,"test",{})
        assert len(calls)==1
        client.reset_faults()
        with pytest.raises(APIError):
            client.request(profile,"test",{})
        assert len(calls)==2
    finally:
        server.shutdown()
        server.server_close()
