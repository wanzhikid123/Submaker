import json
import queue
import threading
import time

import pytest

from submaker.config import Profile, Settings
from submaker.pipeline import Scheduler
from submaker.store import Store
from submaker.telemetry import provider_detail_lines


def original_settings():
    return Settings(in_flight=1, full_context_enabled=True, review_enabled=True, review=Profile(name="ChatGPT", url="https://chat.example/v1", model="old-review"),
                    translate=Profile(name="Translator", url="https://translate.example/v1", model="old-translation"))


def changed_settings(settings):
    current = Settings.from_dict(settings.as_dict())
    current.review = Profile(name="OpenRouter", url="https://router.example/v1", model="new-review", effort="high", rpm=5)
    current.translate.model = "new-translation"
    current.source, current.target = "en", "fr"
    current.output_mode, current.output_container = "burn", "mp4"
    return current


@pytest.fixture
def work_queue(tmp_path, monkeypatch):
    source = tmp_path / "fixture.mp4"
    source.write_bytes(b"fixture")
    settings = original_settings()
    store = Store(tmp_path / "jobs.sqlite")
    scheduler = Scheduler(store, settings, client=object())
    submitted = []
    monkeypatch.setattr(scheduler.executor, "submit", lambda fn, row, cancel: submitted.append(row))
    monkeypatch.setattr(scheduler.prefetch_executor, "submit", lambda *args: None)
    try:
        yield source, store, scheduler, submitted
    finally:
        scheduler.active.clear()
        scheduler.shutdown()
        store.close()


@pytest.mark.parametrize("state", ["排队中", "已暂停", "等待恢复"])
def test_start_binds_all_waiting_jobs_to_current_settings(work_queue, state):
    source, store, scheduler, submitted = work_queue
    old = scheduler.settings
    old.repair_via_review = False
    first, second = [store.add(source, old) for _ in range(2)]
    store.update(first, state, "original status")
    store.put(first, "chunk:0", [{"text": "saved recognition"}])
    store.add_usage(first, {"total_tokens": 123})
    current = changed_settings(old)
    current.repair_via_review = True
    scheduler.configure(current)
    scheduler.start()

    rows = {row["id"]: row for row in store.jobs()}
    assert [row["id"] for row in submitted] == [first]
    for row in [*rows.values(), *submitted]:
        assert json.loads(row["config"]) == current.as_dict()
        assert row["state"] == "排队中"
    assert second not in scheduler.active  # Even tasks waiting for a slot use the Start snapshot.
    assert store.get(first, "chunk:0") == [{"text": "saved recognition"}]
    assert rows[first]["total_tokens"] == 123 and rows[first]["token_requests"] == 1
    assert any("OpenRouter（全片审校）" in line for line in provider_detail_lines(rows[first]))
    assert not any("ChatGPT（全片审校）" in line for line in provider_detail_lines(rows[first]))
    current.review.model = "edited-after-start"
    assert json.loads(rows[first]["config"])["review"]["model"] == "new-review"


def test_start_leaves_active_completed_failed_and_deleting_jobs_unchanged(work_queue):
    source, store, scheduler, submitted = work_queue
    old = scheduler.settings
    states = ["处理中", "完成", "完成（有提示）", "等待处理", "删除中"]
    jobs = [store.add(source, old) for _ in states]
    for job, state in zip(jobs, states):
        store.update(job, state, "keep status", "existing-output.mkv")
    scheduler.active[jobs[0]] = threading.Event()
    scheduler.recognizing.add(jobs[0])
    pending = store.add(source, old)
    current = changed_settings(old)
    scheduler.configure(current)
    scheduler.start()

    rows = {row["id"]: row for row in store.jobs()}
    for job, state in zip(jobs, states):
        assert rows[job]["state"] == state and rows[job]["detail"] == "keep status"
        assert rows[job]["output"] == "existing-output.mkv"
        assert json.loads(rows[job]["config"]) == old.as_dict()
    assert not scheduler.active[jobs[0]].is_set()
    assert not submitted
    assert json.loads(rows[pending]["config"]) == current.as_dict()
    # The next job receives the updated settings once the existing worker finishes.
    del scheduler.active[jobs[0]]
    scheduler.recognizing.discard(jobs[0])
    scheduler._pump()
    assert [row["id"] for row in submitted] == [pending]
    assert json.loads(submitted[0]["config"])["review"]["model"] == "new-review"


def test_running_resources_still_cannot_be_changed(work_queue):
    source, store, scheduler, _ = work_queue
    active = store.add(source, scheduler.settings)
    scheduler.active[active] = threading.Event()
    current = changed_settings(scheduler.settings)
    current.threads = 8
    with pytest.raises(RuntimeError, match="暂停"):
        scheduler.configure(current)
    assert scheduler.settings.threads == 16
    assert json.loads(store.jobs()[0]["config"])["review"]["name"] == "ChatGPT"


def test_pipeline_uses_start_settings_for_translation_review_repair_and_title(tmp_path, monkeypatch):
    source = tmp_path / "fixture.mp4"
    source.write_bytes(b"fixture")
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.bin").touch()
    video = {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 320,
             "height": 240, "pix_fmt": "yuv420p", "bit_rate": "1000000"}
    audio = {"index": 1, "codec_type": "audio", "codec_name": "aac"}
    info = {"video": video, "audio": audio, "streams": [video, audio], "duration": 1,
            "format": {"bit_rate": "1000000"}}
    calls = []

    class Pool:
        fatal = None

        def __init__(self, settings, *args):
            self.settings, self.events = settings, queue.Queue()

        def register(self, *args):
            return self.events

        def submit(self, task, cancel):
            assert task["language"] == "en"
            self.events.put({"type": "segment", "chunk": 0, "gpu": 0,
                             "segment": {"start": .1, "end": .8, "text": "source", "logprob": -.1, "no_speech": 0}})
            self.events.put({"type": "done", "chunk": 0})

        def unregister(self, *args):
            pass

        def shutdown(self):
            pass

    class Client:
        def translate(self, settings, rows, context, cancel=None, instruction=""):
            calls.append(("repair" if instruction else "translate", settings.translate.model, settings.target))
            return {row["id"]: "translated" for row in rows}

        def brief(self, settings, *args):
            calls.append(("brief", settings.review_profile().model, settings.target))
            return "brief"

        def review(self, settings, rows, *args):
            calls.append(("review", settings.review_profile().model, settings.target))
            return [{"ids": [rows[0]["id"]], "severity": "error", "reason": "repair fixture"}]

        def title(self, settings, *args):
            calls.append(("title", settings.review_profile().model, settings.target))
            return "result"

    monkeypatch.setattr("submaker.pipeline.ASRPool", Pool)
    monkeypatch.setattr("submaker.pipeline.probe", lambda *args: info)
    monkeypatch.setattr("submaker.pipeline.burn", lambda src, dst, *args: dst.write_bytes(b"fixture output"))
    old = original_settings()
    old.model_path, old.output_dir = str(model), str(tmp_path / "out")
    store = Store(tmp_path / "jobs.sqlite")
    job = store.add(source, old)
    current = changed_settings(old)
    current.encoding, current.decoding = "quality", "software"
    scheduler = Scheduler(store, current, client=Client())
    try:
        scheduler.start()
        deadline = time.monotonic() + 5
        while scheduler.active and time.monotonic() < deadline:
            time.sleep(.01)
        row = store.jobs()[0]
        assert row["state"] == "完成", row["detail"]
        assert row["output"].endswith(".mp4")
        assert calls == [("translate", "new-translation", "fr"), ("brief", "new-review", "fr"),
                         ("review", "new-review", "fr"), ("repair", "new-review", "fr"), ("title", "new-review", "fr")]
    finally:
        scheduler.shutdown()
        store.close()


def test_gui_start_saves_and_displays_current_review_settings(tmp_path, monkeypatch, qt_application):
    from submaker.gui import MainWindow

    old = original_settings()
    saved = []
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: old))
    monkeypatch.setattr(Settings, "save", lambda self: saved.append(self.as_dict()))
    monkeypatch.setattr("submaker.gui.Store", lambda: Store(tmp_path / "jobs.sqlite"))
    video = {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 320,
             "height": 240, "pix_fmt": "yuv420p", "bit_rate": "1000000"}
    monkeypatch.setattr("submaker.gui.probe", lambda *args: {
        "video": video, "streams": [video], "duration": 1, "format": {"bit_rate": "1000000"}})
    source = tmp_path / "fixture.mp4"
    source.write_bytes(b"fixture")
    window = MainWindow()
    submitted = []
    monkeypatch.setattr(window.scheduler.executor, "submit", lambda fn, row, cancel: submitted.append(row))
    try:
        window.add_paths([source])
        assert "开始队列时使用当前设置" in window.status.text()
        window.settings = changed_settings(old)
        window.start()
        assert len(submitted) == len(saved) == 1
        assert json.loads(submitted[0]["config"]) == saved[0]
        assert "按当前设置启动" in window.status.text()
        detail = window.queue.item(0, 5).text()
        assert "OpenRouter（全片审校）" in detail and "ChatGPT（全片审校）" not in detail
        # Editing the next queue's settings must not change an active task's estimate.
        window.media_info[submitted[0]["id"]] = {"info": {
            "video": video, "streams": [video], "duration": 1, "format": {"bit_rate": "1000000"}}}
        window.refresh()
        estimate = window.queue.item(0, 4).text()
        window.settings.video_quality = 10
        window.settings.video_codec = "h264"
        window.refresh()
        assert window.queue.item(0, 4).text() == estimate
        assert "任务启动时的设置" in window.queue.item(0, 4).toolTip()
    finally:
        window.scheduler.active.clear()
        window.close()
        window.store.close()
