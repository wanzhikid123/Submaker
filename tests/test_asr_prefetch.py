import collections
import json
import queue
import threading
import time

import pytest

from submaker.config import Profile, Settings
from submaker.llm import JOB_CONTEXT
from submaker.media import check
from submaker.pipeline import Scheduler, recognition_key
from submaker.store import Store


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.005)
    assert predicate(), "Timed out waiting for the scheduler"


class QueueHarness:
    """Nine real scheduler jobs, controlled ASR/LLM/export, no GPU or remote API."""

    def __init__(self, root, monkeypatch, mode="burn", hold_asr=(), boundaries=False):
        self.root = root
        self.mode = mode
        self.store = Store(root / "jobs.sqlite")
        model = root / "model"
        model.mkdir()
        (model / "model.bin").write_bytes(b"model")
        self.settings = Settings(in_flight=6, model_path=str(model), output_dir=str(root / "out"),
                                 output_mode="soft" if mode == "soft" else "burn", video_segmented=False,
                                 translate=Profile(url="https://translate.example/v1", model="translation"),
                                 review=Profile(url="https://review.example/v1", model="review"))
        if boundaries:
            self.settings.chunk_seconds = 30
        self.jobs, self.sources = [], []
        for index in range(9):
            source = root / f"source-{index}.mp4"
            source.write_bytes(f"source {index}".encode())
            self.sources.append(source)
            self.jobs.append(self.store.add(source, self.settings))
        self.asr_release = {job: threading.Event() for job in self.jobs}
        self.llm_release = {job: threading.Event() for job in self.jobs}
        self.output_release = {job: threading.Event() for job in self.jobs}
        for index, job in enumerate(self.jobs):
            if index not in hold_asr:
                self.asr_release[job].set()
        self.asr_calls = collections.Counter()
        self.calls, self.exports, self.events = [], [], []
        self.translation_inputs = []
        self.pool_count = 0
        owner = self

        class Pool:
            fatal = None

            def __init__(self, settings, *args):
                self.settings, self.queues = settings, {}
                owner.pool_count += 1

            def register(self, job):
                self.queues[job] = queue.Queue()
                return self.queues[job]

            def submit(self, task, cancel):
                job = task["job"]
                owner.asr_calls[job] += 1
                channel = self.queues[job]

                def transcribe():
                    while not owner.asr_release[job].wait(.01):
                        if cancel.is_set():
                            channel.put({"type": "cancelled", "chunk": task["chunk"]})
                            return
                    start, end, text = .1, .8, "source"
                    if boundaries:
                        start, end, text = (26, 33, "joined") if task["chunk"] < 0 else (26, 29, "left") if task["chunk"] == 0 else (30.5, 33, "right")
                    channel.put({"type": "segment", "chunk": task["chunk"], "gpu": 0,
                                 "segment": {"start": start, "end": end, "text": text, "logprob": -.1, "no_speech": 0}})
                    channel.put({"type": "done", "chunk": task["chunk"]})

                threading.Thread(target=transcribe, daemon=True).start()

            def unregister(self, job):
                self.queues.pop(job, None)

            def shutdown(self):
                pass

        class Client:
            def translate(self, settings, rows, context, cancel=None, instruction=""):
                job = JOB_CONTEXT.get()
                owner.calls.append(("translate", job, settings.target))
                owner.translation_inputs.append((job, [row["text"] for row in rows]))
                while not owner.llm_release[job].wait(.01):
                    check(cancel)
                check(cancel)
                return {row["id"]: "translated" for row in rows}

            def brief(self, settings, *args):
                owner.calls.append(("brief", JOB_CONTEXT.get(), settings.target))
                return "brief"

            def review(self, settings, *args):
                owner.calls.append(("review", JOB_CONTEXT.get(), settings.target))
                return []

            def title(self, settings, stem, cancel):
                owner.calls.append(("title", JOB_CONTEXT.get(), settings.target))
                return stem + "-translated"

        video = {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 320,
                 "height": 240, "pix_fmt": "yuv420p", "bit_rate": "1000000"}
        audio = {"index": 1, "codec_type": "audio", "codec_name": "aac"}
        info = {"video": video, "audio": audio, "streams": [video, audio], "duration": 60 if boundaries else 1,
                "format": {"bit_rate": "1000000"}}
        monkeypatch.setattr("submaker.pipeline.ASRPool", Pool)
        monkeypatch.setattr("submaker.pipeline.probe", lambda *args: info)

        def export(source, output, subtitle, info, settings, cancel, *args):
            job = JOB_CONTEXT.get()
            owner.exports.append(job)
            if args:
                args[0](.12)
            while not owner.output_release[job].wait(.01):
                check(cancel)
            check(cancel)
            output.write_bytes(b"completed fixture")
            if args:
                args[0](1.0)

        monkeypatch.setattr("submaker.pipeline.burn", export)
        monkeypatch.setattr("submaker.remux.mux_subtitles", export)
        if mode == "segmented":
            monkeypatch.setattr("submaker.segmented.enabled", lambda *args: True)

            def segmented_export(source, output, subtitle, info, settings, cancel, slots, job, progress, worker, waiting):
                lease = slots.acquire(settings.gpus, cancel, on_wait=waiting)
                try:
                    worker("encoding_started", lease.gpu, 0, 2)
                    worker("encoding_started", lease.gpu, 1, 2)  # Release one film slot only once.
                    export(source, output, subtitle, info, settings, cancel, progress)
                finally:
                    worker("encoding_done", lease.gpu, 0, 2)
                    lease.release()

            monkeypatch.setattr("submaker.segmented.burn_segmented", segmented_export)
        self.client = Client()
        self.scheduler = Scheduler(self.store, self.settings, self.observe, self.client)

    def observe(self, kind, value):
        self.events.append((kind, value))

    def translated(self):
        return {job for kind, job, _ in self.calls if kind == "translate"}

    def prepared(self):
        return {job for job in self.jobs if self.store.get(job, "whisper_prepared")}

    def start(self):
        self.scheduler.start()

    def finish(self):
        for event in [*self.asr_release.values(), *self.llm_release.values(), *self.output_release.values()]:
            event.set()
        wait_for(lambda: all(row["state"] == "完成" for row in self.store.jobs()))
        wait_for(lambda: not self.scheduler.active)

    def close(self):
        self.scheduler.pause()
        for event in [*self.asr_release.values(), *self.llm_release.values(), *self.output_release.values()]:
            event.set()
        wait_for(lambda: not self.scheduler.active)
        self.scheduler.shutdown()
        self.store.close()


@pytest.fixture
def harness(tmp_path, monkeypatch):
    queues = []

    def create(**kwargs):
        queue = QueueHarness(tmp_path, monkeypatch, **kwargs)
        queues.append(queue)
        return queue

    yield create
    for queue in queues:
        queue.close()


@pytest.mark.parametrize("mode", ["burn", "soft", "segmented"])
def test_export_reports_current_stage_fraction_separate_from_overall(harness, mode):
    work = harness(mode=mode)
    work.start()
    job = work.jobs[0]
    work.llm_release[job].set()
    wait_for(lambda: any(kind == "progress" and event["job"] == job and event.get("stage_progress") == .12
                         for kind, event in work.events))
    stage = {"burn": "烧录字幕", "soft": "软字幕封装", "segmented": "分段编码"}[mode]
    progress = [event for kind, event in work.events if kind == "progress" and event["job"] == job
                and event.get("stage_progress") == .12]
    assert progress[-1]["stage"] == stage
    assert progress[-1]["progress"] == pytest.approx(.8492)
    assert any(event.get("stage_progress") == 0 for kind, event in work.events
               if kind == "progress" and event["job"] == job and event["stage"] == stage)
    work.finish()
    assert any(event.get("stage_progress") == 1 for kind, event in work.events
               if kind == "progress" and event["job"] == job and event["stage"] == stage)


@pytest.mark.parametrize("mode", ["burn", "soft", "segmented"])
def test_nine_jobs_prepare_last_three_without_llm_and_promote_when_export_starts(harness, mode):
    work = harness(mode=mode)
    work.start()
    wait_for(lambda: len(work.prepared()) == 9 and not work.scheduler.prefetching)
    assert work.translated() == set(work.jobs[:6])
    rows = {row["id"]: row for row in work.store.jobs()}
    for job in work.jobs[6:]:
        assert rows[job]["state"] == "排队中" and "Whisper 已完成" in rows[job]["detail"]
        assert not any(call[1] == job for call in work.calls)
        assert rows[job]["token_requests"] == 0
    assert not work.exports
    work.llm_release[work.jobs[0]].set()
    wait_for(lambda: work.jobs[0] in work.exports and work.jobs[6] in work.translated())
    assert not work.output_release[work.jobs[0]].is_set()  # The first export is still running.
    assert work.jobs[7] not in work.translated() and work.jobs[8] not in work.translated()
    assert len(work.scheduler.active.keys() - work.scheduler.prefetching - work.scheduler.exporting) == 6
    work.finish()
    assert work.asr_calls == {job: 1 for job in work.jobs}, "Prepared text was transcribed twice"
    assert not work.store.db.execute("SELECT 1 FROM checkpoints").fetchone()
    assert all(source.exists() for source in work.sources)
    assert not list(work.root.rglob("*.ass")) and not list(work.root.rglob("*.srt"))


@pytest.mark.parametrize("mode", ["burn", "soft", "segmented"])
def test_waiting_for_encoder_or_remux_does_not_release_translation_slot(harness, mode):
    work = harness(mode=mode)
    work.start()
    wait_for(lambda: len(work.prepared()) == 9 and not work.scheduler.prefetching)
    leases = []
    if mode == "soft":
        assert work.scheduler.remux_slot.acquire(blocking=False)
    else:
        leases = [work.scheduler.encoding_slots.acquire([0, 1], threading.Event()) for _ in range(6)]
    try:
        work.llm_release[work.jobs[0]].set()
        if mode == "soft":
            wait_for(lambda: any(kind == "progress" and value["job"] == work.jobs[0] and value["stage"] == "等待封装"
                                 for kind, value in work.events))
        else:
            wait_for(lambda: work.scheduler.encoding_slots.snapshot()["waiting"] == 1)
        assert not work.exports and not work.scheduler.exporting
        assert work.jobs[6] not in work.translated()
    finally:
        if mode == "soft":
            work.scheduler.remux_slot.release()
        else:
            for lease in leases:
                lease.release()
    wait_for(lambda: work.jobs[0] in work.exports and work.jobs[6] in work.translated())
    work.finish()


def test_foreground_whisper_has_priority_over_extra_preprocessing(harness):
    work = harness(hold_asr=range(6))
    work.start()
    wait_for(lambda: len(work.asr_calls) == 6)
    assert set(work.asr_calls) == set(work.jobs[:6]) and not work.scheduler.prefetching
    assert not work.calls
    for job in work.jobs[:6]:
        work.asr_release[job].set()
    wait_for(lambda: len(work.prepared()) == 9)
    work.finish()


def test_in_progress_preprocessing_keeps_its_fifo_place_when_a_slot_opens(harness):
    work = harness(hold_asr=[6])
    work.start()
    wait_for(lambda: work.jobs[6] in work.asr_calls)
    work.llm_release[work.jobs[0]].set()
    wait_for(lambda: work.jobs[0] in work.exports)
    assert work.jobs[6] not in work.translated() and work.jobs[7] not in work.asr_calls
    work.asr_release[work.jobs[6]].set()
    wait_for(lambda: work.jobs[6] in work.translated())
    assert work.asr_calls[work.jobs[6]] == 1
    work.finish()


@pytest.mark.parametrize("action", ["pause", "delete", "clear"])
def test_preprocessing_is_cancellable_and_deletable_without_touching_sources(harness, action):
    work = harness(hold_asr=[6])
    work.start()
    wait_for(lambda: work.jobs[6] in work.asr_calls)
    if action == "pause":
        work.scheduler.pause()
        wait_for(lambda: not work.scheduler.active)
        assert work.store.get(work.jobs[0], "chunk:0")
        work.asr_release[work.jobs[6]].set()
        work.scheduler.start()
        wait_for(lambda: len(work.prepared()) == 9)
        work.finish()
        assert all(work.asr_calls[job] == 1 for job in work.jobs[:6])
    elif action == "delete":
        work.scheduler.delete([work.jobs[6]])
        wait_for(lambda: not any(row["id"] == work.jobs[6] for row in work.store.jobs()))
        assert not work.store.db.execute("SELECT 1 FROM checkpoints WHERE job=?", (work.jobs[6],)).fetchone()
        assert not any(call[1] == work.jobs[6] for call in work.calls)
        work.finish()
    else:
        work.scheduler.delete()
        wait_for(lambda: not work.scheduler.active and not work.store.jobs())
        assert not work.store.db.execute("SELECT 1 FROM checkpoints").fetchone()
    assert all(source.exists() for source in work.sources)


def test_restart_reuses_prepared_text_after_api_target_and_output_changes(harness):
    work = harness()
    work.start()
    wait_for(lambda: len(work.prepared()) == 9 and not work.scheduler.prefetching)
    work.scheduler.pause()
    wait_for(lambda: not work.scheduler.active)
    work.scheduler.shutdown()
    current = Settings.from_dict(work.settings.as_dict())
    current.review.model = "new-review"
    current.translate.model = "new-translation"
    current.target, current.output_mode = "fr", "soft"
    work.calls.clear()
    work.scheduler = Scheduler(work.store, current, work.observe, work.client)
    work.start()
    work.finish()
    assert work.asr_calls == {job: 1 for job in work.jobs} and work.pool_count == 1
    assert all(target == "fr" for _, _, target in work.calls)
    assert all(json.loads(row["config"])["review"]["model"] == "new-review" for row in work.store.jobs())


def test_preprocessing_repairs_chunk_boundaries_before_admitting_translation(harness):
    work = harness(boundaries=True)
    work.start()
    wait_for(lambda: len(work.prepared()) == 9 and not work.scheduler.prefetching)
    for job in work.jobs[6:]:
        assert [row["text"] for row in work.store.get(job, "final_segments")] == ["joined"]
        assert work.asr_calls[job] == 3  # Two windows and one boundary reread.
    work.llm_release[work.jobs[0]].set()
    wait_for(lambda: work.jobs[6] in work.translated())
    assert [texts for job, texts in work.translation_inputs if job == work.jobs[6]] == [["joined"]]
    work.finish()
    assert all(work.asr_calls[job] == 3 for job in work.jobs)


@pytest.mark.parametrize("field,value", [("source", "en"), ("preset", "fast"), ("chunk_seconds", 30),
                                         ("overlap", 4), ("audio_stream", 2), ("glossary", "new term")])
def test_prepared_text_identity_tracks_recognition_inputs(tmp_path, field, value):
    settings = Settings()
    job = {"fingerprint": "fixture"}
    before = recognition_key(job, settings)
    setattr(settings, field, value)
    assert recognition_key(job, settings) != before
