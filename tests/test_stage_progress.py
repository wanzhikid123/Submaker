from types import SimpleNamespace
from unittest.mock import Mock

from submaker.pipeline import Scheduler


def test_stage_percentage_updates_do_not_bypass_persistence_throttle(monkeypatch):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.pausing = set()
    scheduler.last_persist = {}
    scheduler.store = SimpleNamespace(update=Mock())
    scheduler.callback = Mock()
    now = [10.0]
    monkeypatch.setattr("submaker.pipeline.time.monotonic", lambda: now[0])
    scheduler.emit("job", "软字幕封装", .83, stage_progress=-.1)
    now[0] += .1
    scheduler.emit("job", "软字幕封装", .85, stage_progress=.12)
    assert scheduler.store.update.call_count == 1
    assert scheduler.callback.call_count == 2
    assert scheduler.callback.call_args_list[0].args[1]["stage_progress"] == 0
    assert scheduler.callback.call_args.args[1]["stage_progress"] == .12
    now[0] += 2
    scheduler.emit("job", "软字幕封装", .99, stage_progress=1.1)
    assert scheduler.store.update.call_count == 2
    assert scheduler.callback.call_args.args[1]["stage_progress"] == 1
