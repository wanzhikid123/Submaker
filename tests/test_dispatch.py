import threading
import time

from submaker.dispatch import FairTranslationExecutor
from submaker.llm import Gate


def test_second_film_is_not_starved_by_first_films_large_backlog():
    executor = FairTranslationExecutor(max_workers=2)
    entered, release = threading.Event(), threading.Event()
    order = []
    def first():
        entered.set()
        release.wait(3)
        return "first"
    try:
        executor.submit("a", first, max_parallel=1)
        assert entered.wait(1)
        backlog = [executor.submit("a", lambda: order.append("a"), max_parallel=1) for _ in range(50)]
        second = executor.submit("b", lambda: order.append("b") or "second", max_parallel=1)
        assert second.result(timeout=1) == "second"
        assert order == ["b"]
        assert executor.stats("a")["queued"] == 50
    finally:
        release.set()
        executor.shutdown(wait=True)


def test_shared_api_gate_serves_waiters_in_fifo_order():
    gate = Gate(1)
    assert gate.acquire()
    order = []
    def request(name):
        gate.acquire(timeout=None)
        try:
            order.append(name)
        finally:
            gate.release()
    threads = []
    for name in ("a", "b", "c"):
        thread = threading.Thread(target=request, args=(name,))
        thread.start()
        threads.append(thread)
        deadline = time.monotonic()+1
        while len(gate.waiters) < len(threads) and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(gate.waiters) == len(threads)
    gate.release()
    for thread in threads:
        thread.join(timeout=2)
    assert order == ["a", "b", "c"]
