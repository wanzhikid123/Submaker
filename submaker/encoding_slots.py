"""One fair, cancellable encoding queue shared by all videos and GPUs."""
from collections import deque
import threading

from .media import check
from .config import MAX_VIDEO_ENCODERS, VIDEO_ENCODERS_PER_GPU


class EncodingLease:
    def __init__(self, pool, gpu, cpu, owner):
        self.pool, self.gpu, self.cpu = pool, gpu, cpu
        self.owner = owner
        self.closed = False

    def release(self):
        with self.pool.condition:
            if self.closed:
                return
            self.closed = True
            self.pool.active -= 1
            self.pool.owners[self.owner] -= 1
            if not self.pool.owners[self.owner]:
                self.pool.owners.pop(self.owner)
            if self.cpu:
                self.pool.cpu_active -= 1
            else:
                self.pool.gpus[self.gpu] -= 1
            self.pool.condition.notify_all()


class EncodingSlots:
    def __init__(self, limit=3):
        self.condition = threading.Condition()
        self.active = self.cpu_active = 0
        self.gpus = {}
        self.owners = {}
        self.waiters = deque()
        self.set_limit(limit)

    def set_limit(self, limit):
        if not 1 <= limit <= MAX_VIDEO_ENCODERS:
            raise ValueError("同时编码视频上限只能为 1 至 6 路")
        with self.condition:
            self.limit = limit
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return {"active": self.active, "limit": self.limit, "waiting": len(self.waiters), "gpus": dict(self.gpus)}

    def acquire(self, candidates, cancel=None, cpu=False, on_wait=None, owner=None):
        if not cpu and not candidates:
            raise ValueError("没有可用的视频编码 GPU")
        owner = owner if owner is not None else object()
        ticket = (owner, object(), tuple(dict.fromkeys(candidates)), cpu)
        with self.condition:
            self.waiters.append(ticket)
        try:
            while True:
                check(cancel)
                with self.condition:
                    ordered = sorted(self.waiters, key=lambda t: self.owners.get(t[0], 0))
                    # A waiter tied to a full GPU must not leave the other GPU
                    # idle. Preserve owner fairness/FIFO among runnable tickets.
                    runnable = [t for t in ordered if (not self.cpu_active if t[3] else
                                any(self.gpus.get(g, 0) < VIDEO_ENCODERS_PER_GPU for g in t[2]))]
                    if self.active < self.limit and runnable and runnable[0] is ticket:
                        available = [g for g in ticket[2] if self.gpus.get(g, 0) < VIDEO_ENCODERS_PER_GPU]
                        gpu = None if cpu else min(available, key=lambda g: self.gpus.get(g, 0))
                        self.active += 1
                        self.owners[owner] = self.owners.get(owner, 0)+1
                        if cpu:
                            self.cpu_active += 1
                        else:
                            self.gpus[gpu] = self.gpus.get(gpu, 0)+1
                        self.waiters.remove(ticket)
                        self.condition.notify_all()
                        return EncodingLease(self, gpu, cpu, owner)
                    status = {"active": self.active, "limit": self.limit, "position": ordered.index(ticket)+1}
                if on_wait:
                    on_wait(status)
                with self.condition:
                    self.condition.wait(.2)
        finally:
            with self.condition:
                if ticket in self.waiters:
                    self.waiters.remove(ticket)
                    self.condition.notify_all()
