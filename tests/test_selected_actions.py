import collections
import dataclasses
import json
import queue
import threading
import time

import pytest
from PySide6.QtCore import QItemSelectionModel

from submaker.config import Profile, Settings
from submaker.gui import MainWindow
from submaker.llm import Client, JOB_CONTEXT
from submaker.media import Cancelled, check
from submaker.pipeline import Scheduler
from submaker.store import Store


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.005)
    assert predicate(), "Timed out waiting for task control"


@pytest.fixture
def selected_queue(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    settings = Settings(in_flight=2, translate=Profile(name="Translator", url="https://translate.example/v1", model="old"))
    store = Store(tmp_path / "jobs.sqlite")
    jobs = [store.add(source, settings) for _ in range(5)]
    scheduler = Scheduler(store, settings, client=object())
    submitted = []
    monkeypatch.setattr(scheduler.executor, "submit", lambda fn, row, cancel: submitted.append(row))
    monkeypatch.setattr(scheduler.prefetch_executor, "submit", lambda *args: None)
    try:
        yield store, scheduler, jobs, settings, submitted
    finally:
        scheduler.active.clear()
        scheduler.shutdown()
        store.close()


def test_pause_only_selected_running_and_waiting_tasks_and_keep_queue_running(selected_queue):
    store, scheduler, jobs, settings, submitted = selected_queue
    for job in jobs[:2]:
        scheduler.active[job] = threading.Event()
        scheduler.recognizing.add(job)
        store.update(job, "处理中", "running")
    store.update(jobs[3], "已暂停", "unselected pause")
    store.update(jobs[4], "完成", "completed", "existing.mkv")
    store.put(jobs[0], "saved", {"done": 12})
    store.add_usage(jobs[0], {"total_tokens": 123})
    scheduler.accepting = True
    before = {row["id"]: row for row in store.jobs()}
    assert scheduler.pause([jobs[0], jobs[2]]) == 2
    rows = {row["id"]: row for row in store.jobs()}
    assert scheduler.accepting and scheduler.active[jobs[0]].is_set()
    assert not scheduler.active[jobs[1]].is_set()
    assert rows[jobs[0]]["state"] == "暂停中" and rows[jobs[2]]["state"] == "已暂停"
    assert all(rows[job] == before[job] for job in (jobs[1], jobs[3], jobs[4]))
    assert all(row["config"] == before[row["id"]]["config"] for row in rows.values())
    assert store.get(jobs[0], "saved") == {"done": 12} and rows[jobs[0]]["total_tokens"] == 123
    scheduler.emit(jobs[0], "late progress", .5, "must not undo pause")
    assert store.jobs()[0]["state"] == "暂停中" and not submitted


def test_retry_selected_tasks_does_not_start_other_queued_or_paused_tasks(selected_queue):
    store, scheduler, jobs, settings, submitted = selected_queue
    store.update(jobs[2], "等待处理", "unselected error")
    store.update(jobs[3], "已暂停", "selected pause")
    store.update(jobs[4], "完成", "completed", "existing.mkv")
    before = {row["id"]: row for row in store.jobs()}
    current = dataclasses.replace(settings, translate=dataclasses.replace(settings.translate, model="new"))
    assert scheduler.retry_many([jobs[1], jobs[3]], current) == 2
    assert not scheduler.accepting
    assert [row["id"] for row in submitted] == [jobs[1], jobs[3]]
    rows = {row["id"]: row for row in store.jobs()}
    assert all(rows[job] == before[job] for job in (jobs[0], jobs[2], jobs[4]))
    assert all(json.loads(rows[job]["config"])["translate"]["model"] == "new" for job in (jobs[1], jobs[3]))
    scheduler.active.clear()
    scheduler.recognizing.clear()
    scheduler._pump()
    assert len(submitted) == 2, "Finishing a selected retry woke the rest of the queue"


def test_pause_cancels_a_selected_retry_waiting_for_a_slot(selected_queue):
    store, scheduler, jobs, settings, submitted = selected_queue
    settings.in_flight = 1
    scheduler.active[jobs[0]] = threading.Event()
    scheduler.recognizing.add(jobs[0])
    assert scheduler.retry_many([jobs[1]], settings) == 1
    assert jobs[1] in scheduler.requested and not submitted
    scheduler.pause([jobs[1]])
    scheduler.active.clear()
    scheduler.recognizing.clear()
    scheduler._pump()
    assert jobs[1] not in scheduler.requested and not submitted
    assert store.jobs()[1]["state"] == "已暂停"


def test_empty_selection_does_not_change_queue_or_configuration(selected_queue):
    store, scheduler, _, settings, submitted = selected_queue
    before = store.jobs()
    assert scheduler.pause([]) == scheduler.retry_many([], settings) == 0
    assert store.jobs() == before and not scheduler.accepting and not submitted


@pytest.mark.parametrize("cancel_retry", [None, "pause", "delete", "all"])
def test_running_selected_retry_waits_for_old_requests_and_leaves_other_task_untouched(tmp_path, monkeypatch, cancel_retry):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.bin").touch()
    settings = Settings(in_flight=2, model_path=str(model), output_mode="burn", encoding="quality", decoding="software",
                        output_dir=str(tmp_path / "out"), translate=Profile(url="http://localhost/v1", model="old", rpm=0))
    store = Store(tmp_path / "jobs.sqlite")
    jobs = [store.add(source, settings) for _ in range(2)]
    allow_cancel = threading.Event()
    finish = threading.Event()
    calls, asr_calls = [], collections.Counter()

    class Pool:
        fatal = None

        def __init__(self, settings, *args):
            self.settings, self.queues = settings, {}

        def register(self, job):
            self.queues[job] = queue.Queue()
            return self.queues[job]

        def submit(self, task, cancel):
            asr_calls[task["job"]] += 1
            self.queues[task["job"]].put({"type": "segment", "chunk": 0, "gpu": 0,
                                          "segment": {"start": .1, "end": .8, "text": "source", "logprob": -.1, "no_speech": 0}})
            self.queues[task["job"]].put({"type": "done", "chunk": 0})

        def unregister(self, job):
            self.queues.pop(job, None)

        def shutdown(self):
            pass

    class FakeClient:
        def translate(self, settings, rows, context, cancel=None, **kwargs):
            job = JOB_CONTEXT.get()
            calls.append((job, settings.translate.model, cancel))
            while not finish.wait(.005):
                if cancel.is_set():
                    # Model a request that needs extra time to settle. Its
                    # checkpoint writes must never race the replacement worker.
                    assert allow_cancel.wait(5)
                    raise Cancelled()
            check(cancel)
            return {row["id"]: "translated" for row in rows}

        def title(self, *args):
            return JOB_CONTEXT.get()

    video = {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 320,
             "height": 240, "pix_fmt": "yuv420p", "bit_rate": "1000000"}
    audio = {"index": 1, "codec_type": "audio", "codec_name": "aac"}
    info = {"video": video, "audio": audio, "streams": [video, audio], "duration": 1, "format": {"bit_rate": "1000000"}}
    monkeypatch.setattr("submaker.pipeline.ASRPool", Pool)
    monkeypatch.setattr("submaker.pipeline.probe", lambda *args: info)
    monkeypatch.setattr("submaker.pipeline.burn", lambda src, dst, *args: dst.write_bytes(b"completed"))
    scheduler = Scheduler(store, settings, client=FakeClient())
    try:
        scheduler.start()
        wait_for(lambda: len(calls) == 2)
        current = dataclasses.replace(settings, translate=dataclasses.replace(settings.translate, model="new"))
        assert scheduler.retry_many([jobs[0]], current) == 1
        wait_for(lambda: calls[0 if calls[0][0] == jobs[0] else 1][2].is_set())
        assert jobs[0] in scheduler.pending_retries
        assert json.loads(store.jobs()[0]["config"])["translate"]["model"] == "old"
        assert not next(cancel for job, _, cancel in calls if job == jobs[1]).is_set()
        if cancel_retry == "pause":
            scheduler.pause([jobs[0]])
        elif cancel_retry == "delete":
            scheduler.delete([jobs[0]])
        elif cancel_retry == "all":
            scheduler.pause()
        allow_cancel.set()
        if cancel_retry is None:
            wait_for(lambda: any(job == jobs[0] and name == "new" for job, name, _ in calls))
            assert asr_calls[jobs[0]] == 1
            assert len([job for job, _, _ in calls if job == jobs[1]]) == 1
            assert json.loads(store.jobs()[1]["config"])["translate"]["model"] == "old"
        else:
            wait_for(lambda: jobs[0] not in scheduler.active)
            assert not any(name == "new" for _, name, _ in calls)
        if cancel_retry != "all":
            assert not next(cancel for job, _, cancel in calls if job == jobs[1]).is_set()
        finish.set()
        wait_for(lambda: not scheduler.active)
        rows = {row["id"]: row for row in store.jobs()}
        if cancel_retry is None:
            assert all(row["state"] == "完成" for row in rows.values())
        elif cancel_retry == "delete":
            assert jobs[0] not in rows and rows[jobs[1]]["state"] == "完成"
        else:
            assert rows[jobs[0]]["state"] == "已暂停"
        assert source.read_bytes() == b"source"
    finally:
        allow_cancel.set()
        finish.set()
        scheduler.pause()
        wait_for(lambda: not scheduler.active)
        scheduler.shutdown()
        store.close()


def test_gui_uses_selected_rows_instead_of_stale_current_row(tmp_path, monkeypatch, qt_application):
    settings = Settings(translate=Profile(url="https://translate.example/v1", model="old"))
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: settings))
    monkeypatch.setattr("submaker.gui.Store", lambda: Store(tmp_path / "jobs.sqlite"))
    window = MainWindow()
    submitted = []
    monkeypatch.setattr(window.scheduler.executor, "submit", lambda fn, row, cancel: submitted.append(row))
    monkeypatch.setattr(window.scheduler.prefetch_executor, "submit", lambda *args: None)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    jobs = [window.store.add(source, settings) for _ in range(5)]
    window.store.update(jobs[3], "等待处理", "unselected error")
    window.store.update(jobs[4], "完成", "unselected complete", "existing.mkv")
    window.media_info = {job: {"error": "fixture"} for job in jobs}

    def select(indices):
        window.queue.setCurrentCell(4, 0)
        window.queue.clearSelection()
        for index in indices:
            window.queue.selectionModel().select(window.queue.model().index(index, 0),
                QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)

    try:
        window.refresh()
        select([])
        before = window.store.jobs()
        window.pause_button.click()
        assert "请先选择" in window.status.text() and window.store.jobs() == before
        window.retry_button.click()
        assert "请先选择" in window.status.text() and not submitted and not window.scheduler.accepting
        select([0, 2])
        window.pause_button.click()
        rows = window.store.jobs()
        assert rows[0]["state"] == rows[2]["state"] == "已暂停"
        assert rows[1]["state"] == "排队中" and rows[3] == before[3] and rows[4] == before[4]
        select([0, 2])
        window.settings.translate.model = "new"
        window.refresh()  # Selection must survive periodic refresh.
        window.retry_button.click()
        assert [row["id"] for row in submitted] == [jobs[0], jobs[2]]
        assert not window.scheduler.accepting
        rows = window.store.jobs()
        assert json.loads(rows[1]["config"])["translate"]["model"] == "old"
        assert all(json.loads(rows[index]["config"])["translate"]["model"] == "new" for index in (0, 2))
        assert rows[3] == before[3] and rows[4] == before[4]
    finally:
        window.scheduler.active.clear()
        window.close()
        window.store.close()


def test_retry_only_clears_faults_for_its_profiles_and_keeps_shared_rate_budget():
    first = Profile(url="https://first.example/v1/", model="first", rpm=2)
    second = Profile(url="https://second.example/v1", model="second")
    client = Client()
    first_key = ("https://first.example/v1/chat/completions", first.model, first.effort, first.key_id)
    second_key = ("https://second.example/v1/chat/completions", second.model, second.effort, second.key_id)
    client.faults.update({first_key: "first failure", second_key: "second failure"})
    gate = client.limits.for_profile(first)
    gate.throttle(10)
    cooldown = gate.cooldown_until
    client.reset_faults([first])
    assert first_key not in client.faults and client.faults[second_key] == "second failure"
    assert gate.cooldown_until == cooldown and client.limits.for_profile(first) is gate


def test_interrupted_pause_recovers_as_resumable_task(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    path = tmp_path / "jobs.sqlite"
    store = Store(path)
    job = store.add(source, Settings())
    store.put(job, "saved", {"done": 12})
    store.update(job, "暂停中", "stopping")
    store.close()
    store = Store(path)
    try:
        assert store.jobs()[0]["state"] == "等待恢复" and store.get(job, "saved") == {"done": 12}
    finally:
        store.close()
