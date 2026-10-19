from __future__ import annotations

from collections import deque
from concurrent.futures import Future
import threading


class FairTranslationExecutor:
    """Round-robin job queues; a film cannot reserve every API waiting thread."""

    def __init__(self, max_workers=8):
        self.condition = threading.Condition()
        self.queues = {}
        self.order = deque()
        self.running = {}
        self.limits = {}
        self.closed = False
        self.workers = [threading.Thread(target=self._worker, name=f"translate_{i}", daemon=True) for i in range(max_workers)]
        for worker in self.workers:
            worker.start()

    def submit(self, job, function, max_parallel=2, prepare=None):
        future = Future()
        with self.condition:
            if self.closed:
                raise RuntimeError("翻译调度已关闭")
            if job not in self.queues:
                self.queues[job] = deque()
                self.running[job] = 0
                self.order.append(job)
            self.limits[job] = max(1, min(len(self.workers), max_parallel))
            self.queues[job].append((future, function, prepare))
            self.condition.notify_all()
        return future

    def ensure_capacity(self, count):
        with self.condition:
            if self.closed:
                return
            for i in range(len(self.workers), min(128, count)):
                worker = threading.Thread(target=self._worker, name=f"translate_{i}", daemon=True)
                self.workers.append(worker)
                worker.start()
            self.condition.notify_all()

    def stats(self, job):
        with self.condition:
            return {"queued": sum(not f.cancelled() for f, *_ in self.queues.get(job, ())), "running": self.running.get(job, 0)}

    def _next(self):
        for _ in range(len(self.order)):
            if not self.order:
                return None
            job = self.order[0]
            self.order.rotate(-1)
            if self.running[job] >= self.limits[job]:
                continue
            while self.queues[job]:
                future, function, prepare = self.queues[job][0]
                if future.cancelled():
                    self.queues[job].popleft()
                    continue
                try:
                    reservation = prepare() if prepare else None
                except Exception as exc:
                    self.queues[job].popleft()
                    if future.set_running_or_notify_cancel():
                        future.set_exception(exc)
                    continue
                if prepare and reservation is None:
                    break
                self.queues[job].popleft()
                if future.set_running_or_notify_cancel():
                    self.running[job] += 1
                    return job, future, function, reservation
                if reservation:
                    reservation.close()
            if not self.running[job] and not self.queues[job]:
                del self.queues[job], self.running[job], self.limits[job]
                self.order.remove(job)
        return None

    def _worker(self):
        while True:
            with self.condition:
                work = self._next()
                while work is None:
                    if self.closed and not any(self.queues.values()):
                        return
                    self.condition.wait(.1 if any(self.queues.values()) else None)
                    work = self._next()
            job, future, function, reservation = work
            try:
                future.set_result(function(reservation) if reservation else function())
            except BaseException as exc:
                future.set_exception(exc)
            finally:
                if reservation:
                    reservation.close()
                with self.condition:
                    self.running[job] -= 1
                    if not self.running[job] and not self.queues[job]:
                        del self.queues[job], self.running[job], self.limits[job]
                        self.order.remove(job)
                    self.condition.notify_all()

    def shutdown(self, wait=False, cancel_futures=True):
        with self.condition:
            self.closed = True
            if cancel_futures:
                for tasks in self.queues.values():
                    for future, *_ in tasks:
                        future.cancel()
                    tasks.clear()
            self.condition.notify_all()
        if wait:
            for worker in self.workers:
                worker.join()
