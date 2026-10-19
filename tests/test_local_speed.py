import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from submaker.local_speed import TokenSpeedMonitor, TokenSpeedSampler


def slots(task=1, decoded=1, active=True):
    return [{"id": 0, "id_task": task, "is_processing": active,
             "next_token": [{"n_decoded": decoded}]}]


def test_speed_uses_output_deltas_and_excludes_prefill_and_other_requests():
    sampler = TokenSpeedSampler()
    sampler.update(slots(decoded=0), now=0)
    assert sampler.snapshot(now=0) == {"tokens_per_second": 0.0, "speed_phase": "prefill"}
    sampler.update(slots(decoded=20), now=20)
    assert sampler.snapshot(now=20)["tokens_per_second"] is None
    sampler.update(slots(decoded=53), now=21)
    sampler.update(slots(decoded=86), now=22)
    assert sampler.snapshot(now=22)["tokens_per_second"] == 33
    sampler.update(slots(task=2, decoded=2000), now=23)
    assert sampler.snapshot(now=23)["tokens_per_second"] is None
    sampler.update(slots(task=2, decoded=2030), now=24)
    assert sampler.snapshot(now=24)["tokens_per_second"] == 30
    sampler.update(slots(task=2, decoded=1), now=25)
    assert sampler.snapshot(now=25)["tokens_per_second"] is None


def test_speed_smoothing_stall_idle_and_stale_samples_never_show_old_speed():
    sampler = TokenSpeedSampler()
    for now, count in enumerate((1, 34, 67, 100, 100, 100, 100)):
        sampler.update(slots(decoded=count), now=now)
    assert sampler.snapshot(now=6)["tokens_per_second"] == 0
    sampler.update(slots(decoded=100, active=False), now=7)
    assert sampler.snapshot(now=7) == {"tokens_per_second": 0.0, "speed_phase": "idle"}
    assert sampler.snapshot(now=11)["tokens_per_second"] is None
    sampler.update(slots(task=3, decoded=10), now=12)
    sampler.update(slots(task=3, decoded=43), now=13)
    assert sampler.snapshot(now=13)["tokens_per_second"] == 33
    sampler.unavailable()
    assert sampler.snapshot()["tokens_per_second"] is None


@pytest.mark.parametrize("rows", [None, {}, [{"is_processing": True}],
                                   slots(decoded=-1), slots(decoded=float("nan")), slots(decoded=True)])
def test_missing_or_invalid_engine_counters_show_unavailable(rows):
    sampler = TokenSpeedSampler()
    sampler.update(rows, now=1)
    assert sampler.snapshot(now=1)["tokens_per_second"] is None


def test_monitor_reads_real_http_counters_off_thread_and_handles_unavailable_engine():
    counts, headers, methods = [], [], []
    fail = threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            counts.append(time.monotonic())
            headers.append(self.headers.get("Authorization"))
            methods.append((self.command, self.path))
            payload = json.dumps(slots(decoded=len(counts)*33)).encode()
            self.send_response(503 if fail.is_set() else 200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monitor = TokenSpeedMonitor(f"http://127.0.0.1:{server.server_port}/slots", "test-local-token")
    try:
        deadline = time.monotonic()+4
        while monitor.snapshot()["tokens_per_second"] is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert 20 < monitor.snapshot()["tokens_per_second"] < 45
        assert set(headers) == {"Bearer test-local-token"}
        assert set(methods) == {("GET", "/slots")}
        fail.set()
        deadline = time.monotonic()+3
        while monitor.snapshot()["tokens_per_second"] is not None and time.monotonic() < deadline:
            time.sleep(.02)
        assert monitor.snapshot()["tokens_per_second"] is None
    finally:
        monitor.stop()
        monitor.thread.join(timeout=2)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert not monitor.thread.is_alive()


@pytest.mark.parametrize("url", ["https://127.0.0.1/slots", "http://example.com/slots", "http://127.0.0.1/api/generate"])
def test_monitor_rejects_non_runner_urls(url):
    with pytest.raises(ValueError):
        TokenSpeedMonitor(url)
