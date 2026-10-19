"""Read numeric generation counters on loopback; all samples stay in RAM."""
from collections import deque
import threading
import time

import httpx


class TokenSpeedSampler:
    def __init__(self):
        self.lock = threading.Lock()
        self.points = deque(maxlen=16)
        self.task = None
        self.value = None
        self.phase = "unavailable"
        self.updated = None

    def update(self, rows, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if not isinstance(rows, list):
                self._reset(None, "unavailable", now)
                return
            active = next((r for r in rows if isinstance(r, dict) and r.get("is_processing")), None)
            if active is None:
                self._reset(0.0, "idle", now)
                return
            task = (active.get("id"), active.get("id_task"))
            counts = active.get("next_token", [])
            if not isinstance(counts, list) or not counts:
                self._reset(None, "unavailable", now)
                return
            decoded = [r.get("n_decoded") for r in counts if isinstance(r, dict)]
            if not decoded or any(type(n) is not int or n < 0 for n in decoded):
                self._reset(None, "unavailable", now)
                return
            count = sum(decoded)
            if not count:
                self._reset(0.0, "prefill", now)
                return
            if self.task != task or not self.points or count < self.points[-1][1]:
                self.points.clear()
                self.task = task
            self.points.append((now, count))
            while len(self.points) > 2 and self.points[1][0] <= now-3:
                self.points.popleft()
            seconds = now-self.points[0][0]
            self.value = (count-self.points[0][1])/seconds if seconds > 0 else None
            self.phase = "generating" if self.value is not None else "sampling"
            self.updated = now

    def _reset(self, value, phase, now):
        self.points.clear()
        self.task, self.value, self.phase, self.updated = None, value, phase, now

    def unavailable(self):
        with self.lock:
            self._reset(None, "unavailable", time.monotonic())

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.updated is None or now-self.updated > 3:
                return {"tokens_per_second": None, "speed_phase": "unavailable"}
            return {"tokens_per_second": self.value, "speed_phase": self.phase}


class TokenSpeedMonitor:
    def __init__(self, url, token=""):
        # Callers provide only a loopback address of their owned model runner.
        address = httpx.URL(url)
        if address.scheme != "http" or address.host != "127.0.0.1" or address.path != "/slots":
            raise ValueError("Token counters must use the owned loopback runner")
        self.url, self.token = url, token
        self.sampler = TokenSpeedSampler()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._poll, name="local-token-speed", daemon=True)
        self.thread.start()

    def _poll(self):
        headers = {"Authorization": "Bearer "+self.token} if self.token else {}
        with httpx.Client(trust_env=False, timeout=.75, follow_redirects=False) as client:
            while not self.stopped.is_set():
                started = time.monotonic()
                try:
                    response = client.get(self.url, headers=headers)
                    response.raise_for_status()
                    self.sampler.update(response.json())
                except (httpx.HTTPError, ValueError, TypeError):
                    self.sampler.unavailable()
                self.stopped.wait(max(.1, 1.0-(time.monotonic()-started)))

    def snapshot(self):
        return self.sampler.snapshot()

    def stop(self):
        self.stopped.set()
        self.sampler.unavailable()
