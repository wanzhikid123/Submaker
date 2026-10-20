import concurrent.futures as cf
import dataclasses
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from submaker.config import Profile, Settings, PROVIDERS
from submaker.llm import APIError, Client, Gate, JOB_CONTEXT, translation_key, translation_unit_keys, profile_identity, profile_tag
from submaker.pipeline import Scheduler
from submaker.dispatch import FairTranslationExecutor
from submaker.media import Cancelled
from submaker.store import Store
from submaker.telemetry import api_notice, api_provider_records


def response(value):
    return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}})


def settings_pair():
    return Settings(translate=Profile(name="MiniServer", url="http://localhost/a", model="a", concurrency=2, rpm=0, key_id="key-a"), translators=[Profile(name="Second", url="http://localhost/b", model="b", concurrency=2, rpm=0, key_id="key-b")])


def test_migration_pool_capacity_and_title_use_only_review_profile():
    settings = settings_pair()
    old = settings.as_dict()
    old["title"] = {"url": "http://localhost/obsolete", "model": "obsolete"}
    settings = Settings.from_dict(old)
    assert "title" not in settings.as_dict()
    assert settings.translation_parallelism() == 4
    settings.translators.append(dataclasses.replace(settings.translate, concurrency=1))
    assert settings.translation_parallelism() == 3
    settings.translators[-1].enabled = False
    assert settings.translation_parallelism() == 4
    settings.review = Profile(name="Audit", url="http://localhost/review", model="review", rpm=9)
    calls = []
    client = Client(key_provider=lambda _: "")
    def request(profile, *args, **kwargs):
        calls.append(profile)
        return {"title": "Translated"}
    client.request = request
    assert client.title(settings, "source") == "Translated"
    assert calls == [settings.review]
    assert Profile.from_dict({"model": "old"}).rpm == 30


def test_two_real_http_endpoints_run_at_combined_capacity_without_crossing_limits():
    lock, release, all_entered = threading.Lock(), threading.Event(), threading.Event()
    active, peaks, keys, total_peak = {}, {}, {}, [0]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            model = body["model"]
            with lock:
                active[model] = active.get(model, 0)+1
                peaks[model] = max(peaks.get(model, 0), active[model])
                keys[model] = self.headers.get("Authorization")
                total_peak[0] = max(total_peak[0], sum(active.values()))
                if sum(active.values()) == 4:
                    all_entered.set()
            release.wait(4)
            content = {"translations": [{"id": r["id"], "text": "translated-"+model} for r in json.loads(body["messages"][1]["content"])["units"]]}
            payload = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content)}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            with lock:
                active[model] -= 1
        def log_message(self, *args):
            pass
    servers = [ThreadingHTTPServer(("127.0.0.1", 0), Handler) for _ in range(2)]
    for server in servers:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    settings = settings_pair()
    settings.translate.url = f"http://127.0.0.1:{servers[0].server_port}/v1"
    settings.translators[0].url = f"http://127.0.0.1:{servers[1].server_port}/v1"
    client = Client(key_provider=lambda p: p.key_id)
    pool = cf.ThreadPoolExecutor(6)
    try:
        futures = [pool.submit(client.translate, settings, [{"id": str(i), "text": "source"}], []) for i in range(6)]
        assert all_entered.wait(3), "Combined provider capacity was not used"
        assert peaks == {"a": 2, "b": 2} and total_peak[0] == 4
        release.set()
        answers = [f.result(4) for f in futures]
        assert keys == {"a": "Bearer key-a", "b": "Bearer key-b"}
        assert {p["model"] for a in answers for p in a.providers.values()} == {"a", "b"}
        assert all(g.active == 0 for g in client.limits.slots.values())
    finally:
        release.set()
        pool.shutdown(wait=True)
        for server in servers:
            server.shutdown()
            server.server_close()


@pytest.mark.parametrize("kind", ["429", "quota", "401", "503", "timeout", "invalid_response"])
def test_one_provider_failure_does_not_stop_other_provider_or_hide_its_warning(tmp_path, kind):
    settings = settings_pair()
    source = tmp_path/"source.mp4"
    source.write_bytes(b"fixture")
    store = Store(tmp_path/"jobs.sqlite")
    job = store.add(source, settings)
    events, calls = [], []
    def status(event):
        store.record_api(job, event)
        events.append(event)
    client = Client(key_provider=lambda p: p.key_id, status=status)
    def post(url, body, headers, timeout, cancel):
        calls.append(body["model"])
        if body["model"] == "a":
            if kind == "timeout":
                raise httpx.ReadTimeout("fixture")
            if kind == "invalid_response":
                return httpx.Response(200, content=b"invalid response envelope")
            return httpx.Response(int(kind) if kind.isdigit() else 429, headers={"Retry-After": "2"}, json={"error": {"code": "insufficient_quota" if kind == "quota" else "rate_limit"}})
        units = json.loads(body["messages"][1]["content"])["units"]
        return response({"translations": [{"id": r["id"], "text": "good"} for r in units]})
    client._post = post
    token = JOB_CONTEXT.set(job)
    try:
        result = client.translate(settings, [{"id": "one", "text": "source"}], [])
        assert result == {"one": "good"} and calls == ["a", "b"]
        assert result.providers["one"]["model"] == "b"
        records = api_provider_records(store.jobs()[0])
        assert len(records) == 2
        assert any(name == "MiniServer" and api_notice(row) for name, row in records)
        assert next(row for name, row in records if name == "Second")["api_state"] == "success"
        assert all(g.active == 0 for g in client.limits.slots.values())
        assert "key-a" not in json.dumps(events) and "key-b" not in json.dumps(events)
        if kind in ("quota", "401"):
            client.translate(settings, [{"id": "two", "text": "source"}], [])
            assert calls.count("a") == 1
        updated = store.jobs()[0]
        settings.translate.enabled = False
        settings.review = settings.translators[0]
        updated["config"] = json.dumps(settings.as_dict())
        assert [name for name, _ in api_provider_records(updated)] == ["Second"]
        store.delete([job])
        for event in events:
            store.record_api(job, event)
        assert store.db.execute("select count(*) from api_providers").fetchone()[0] == 0
    finally:
        JOB_CONTEXT.reset(token)
        store.close()


def test_rpm_is_paced_over_the_full_window_and_shared_across_stages(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("submaker.llm.time.monotonic", lambda: now[0])
    gate = Gate(16, 3)
    for instant in (100, 120, 140):
        now[0] = instant
        assert gate.try_acquire()
        gate.release()
        assert not gate.try_acquire()
    now[0] = 159.99
    assert not gate.try_acquire()
    now[0] = 160
    assert gate.try_acquire()
    gate.release()
    assert gate.snapshot()["rpm_used"] == 3
    client = Client(key_provider=lambda _: "")
    profile = Profile(url="http://localhost/v1", model="translate", concurrency=5, rpm=12)
    one = client.limits.for_profile(profile)
    two = client.limits.for_profile(dataclasses.replace(profile, model="review", rpm=6, concurrency=2))
    assert one is two and two.rpm == 6 and two.limit == 2


def test_rpm_counts_retries_and_cancellation_leaves_no_held_pool_slots():
    client = Client(key_provider=lambda _: "")
    times = []
    def post(*args):
        times.append(time.monotonic())
        return response({"ok": True}) if len(times) > 1 else httpx.Response(200, json={"choices": [{"message": {"content": "invalid"}}]})
    client._post = post
    client.request(Profile(url="http://localhost/v1", model="m", rpm=600), "test", {})
    assert len(times) == 2 and times[1]-times[0] >= .09
    settings = settings_pair()
    for p in settings.translation_profiles():
        client.limits.for_profile(p).throttle(60)
    cancelled = threading.Event()
    timer = threading.Timer(.05, cancelled.set)
    timer.start()
    try:
        with pytest.raises(Cancelled):
            client.translate(settings, [{"id": "x", "text": "x"}], [], cancelled)
        assert not client.activity_snapshot()
        assert all(g.active == 0 for g in client.limits.slots.values())
    finally:
        timer.cancel()


def test_adding_a_provider_reuses_existing_eligible_unit_keys_and_preserves_provenance():
    settings = settings_pair()
    one = dataclasses.replace(settings, translators=[])
    row = {"id": "a", "text": "source"}
    original = "unit:"+translation_key(one, [row], None)
    assert original in list(translation_unit_keys(settings, row))
    key = translation_key(settings, [row], None)
    settings.translate.rpm = 7
    settings.translators[0].concurrency = 12
    assert translation_key(settings, [row], None) == key
    settings.translate.enabled = False
    assert original not in list(translation_unit_keys(settings, row))


def test_provider_presets_and_separate_credentials_can_be_edited_and_removed(monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QTabWidget
    from submaker.gui import SettingsDialog
    app = QApplication.instance() or QApplication([])
    saved_keys = []
    monkeypatch.setattr(Settings, "save", lambda self: None)
    monkeypatch.setattr("submaker.gui.set_key", lambda profile, key: saved_keys.append((profile.key_id, key)))
    settings = settings_pair()
    settings.translators.clear()
    dialog = SettingsDialog(settings)
    try:
        added = dialog.pool.add_profile("google")
        assert added.url.text() == PROVIDERS["google"][1]
        assert not added.enabled.isChecked()
        added.model.setText("chosen-google-model")
        added.enabled.setChecked(True)
        added.concurrent.setValue(4)
        added.rpm.setValue(12)
        added.key.setText("fixture-google-key")
        key_id = added.value().key_id
        assert key_id != settings.translate.key_id
        assert added.value().key_id == key_id
        removed = dialog.pool.add_profile("openrouter")
        dialog.pool.remove_current()
        assert len(dialog.pool.editors) == 2
        captions = [dialog.findChild(QTabWidget).tabText(i) for i in range(dialog.findChild(QTabWidget).count())]
        assert "文件名翻译" not in captions
        dialog.accept()
        assert dialog.settings.translators[0].rpm == 12
        assert dialog.settings.translation_parallelism() == 6
        assert saved_keys == [(key_id, "fixture-google-key")]
        assert "fixture-google-key" not in json.dumps(dialog.settings.as_dict())
        assert not settings.translators
    finally:
        dialog.close()


@pytest.mark.parametrize("provider", ["openai", "openrouter", "google"])
def test_provider_wire_format_preserves_selected_model_effort_and_auth(provider):
    profile = Profile(provider=provider, url=PROVIDERS[provider][1], model="explicit-model-id", effort="high", json_mode=True, rpm=0)
    client = Client(key_provider=lambda _: "fixture-key")
    captured = []
    def post(url, body, headers, timeout, cancel):
        captured.append((url, body, headers))
        return response({"ok": True})
    client._post = post
    assert client.request(profile, "test", {}) == {"ok": True}
    url, body, headers = captured[0]
    assert url == PROVIDERS[provider][1]+"/chat/completions"
    assert headers["Authorization"] == "Bearer fixture-key"
    assert body["model"] == "explicit-model-id" and body["reasoning_effort"] == "high"
    assert body["response_format"] == {"type": "json_object"}
    assert ("x-goog-api-client" in headers) == (provider == "google")


def test_recent_rate_limit_restores_only_its_provider_and_not_completed_history(tmp_path):
    settings = settings_pair()
    source = tmp_path/"source.mp4"
    source.write_bytes(b"fixture")
    store = Store(tmp_path/"jobs.sqlite")
    job = store.add(source, settings)
    store.update(job, "等待处理", "HTTP 429")
    store.record_api(job, {"profile_id": profile_tag(settings.translators[0]), "profile_name": "Second", "event": "rate_limit", "code": 429, "effective_limit": 1, "configured_limit": 2, "retry_at": time.time()+10})
    scheduler = Scheduler(store, settings)
    try:
        assert scheduler.client.limits.for_profile(settings.translate).limit == 2
        assert scheduler.client.limits.for_profile(settings.translators[0]).limit == 1
    finally:
        scheduler.shutdown()
    store.update(job, "完成", "done")
    scheduler = Scheduler(store, settings)
    try:
        assert scheduler.client.limits.for_profile(settings.translators[0]).limit == 2
    finally:
        scheduler.shutdown()
        store.close()


@pytest.mark.parametrize("repair_via_review", [True, False])
def test_pipeline_keeps_completed_single_provider_units_when_pool_is_added(tmp_path, monkeypatch, repair_via_review):
    import queue
    source = tmp_path/"source.mp4"
    source.write_bytes(b"fixture")
    model = tmp_path/"model"
    model.mkdir()
    (model/"model.bin").touch()
    settings = settings_pair()
    settings.repair_via_review = repair_via_review
    settings.full_context_enabled = settings.review_enabled = True
    settings.output_mode = "burn"  # This fixture stubs the burn exporter, not real media.
    settings.review = Profile(name="Reviewer", url="http://localhost/review", model="review", effort="high", enabled=False, rpm=0, key_id="review-key")
    second = settings.translators.pop()
    settings.model_path, settings.output_dir = str(model), str(tmp_path/"out")
    video = {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720, "pix_fmt": "yuv420p", "bit_rate": "1000000"}
    audio = {"index": 1, "codec_type": "audio", "codec_name": "aac"}
    info = {"video": video, "audio": audio, "streams": [video, audio], "duration": 1, "format": {"bit_rate": "1000000"}}
    class Pool:
        fatal = None
        def __init__(self, settings, *args):
            self.settings, self.events = settings, queue.Queue()
        def register(self, *args): return self.events
        def submit(self, task, cancel):
            self.events.put({"type": "segment", "chunk": 0, "gpu": 0, "segment": {"start": .1, "end": .8, "text": "source", "logprob": -.1, "no_speech": 0}})
            self.events.put({"type": "done", "chunk": 0})
        def unregister(self, *args): pass
        def shutdown(self): pass
    class TestClient(Client):
        reviews = 0
        def brief(self, *args): return "brief"
        def review(self, *args):
            self.reviews += 1
            if self.reviews == 1:
                raise APIError("injected audit outage")
            return [{"ids": [args[1][0]["id"]], "severity": "error", "reason": "incorrect number"}]
        def title(self, *args): return "result"
    monkeypatch.setattr("submaker.pipeline.ASRPool", Pool)
    monkeypatch.setattr("submaker.pipeline.probe", lambda *args: info)
    monkeypatch.setattr("submaker.pipeline.burn", lambda src, dst, *args: dst.write_bytes(b"test output"))
    client = TestClient(key_provider=lambda p: p.key_id)
    calls = []
    def post(url, body, headers, *args):
        calls.append(body["model"])
        if body["model"] == "review":
            assert url.endswith("/review/chat/completions")
            assert body["reasoning_effort"] == "high" and headers["Authorization"] == "Bearer review-key"
        units = json.loads(body["messages"][1]["content"])["units"]
        return response({"translations": [{"id": r["id"], "text": "translated"} for r in units]})
    client._post = post
    store = Store(tmp_path/"jobs.sqlite")
    job = store.add(source, settings)
    progress_events = []
    scheduler = Scheduler(store, settings, callback=lambda kind, data: progress_events.append(data) if kind == "progress" else None, client=client)
    try:
        scheduler.start()
        deadline = time.monotonic()+4
        while scheduler.active and time.monotonic() < deadline:
            time.sleep(.01)
        assert store.jobs()[0]["state"] == "等待处理" and calls == ["a"]
        settings.translators.append(second)
        scheduler.retry(job, settings)
        while scheduler.active and time.monotonic() < deadline:
            time.sleep(.01)
        assert store.jobs()[0]["state"] == "完成", store.jobs()[0]["detail"]
        assert any(e["stage"] == "全片审校 0/1" for e in progress_events)
        assert any(e["stage"] == "全片审校 1/1" for e in progress_events)
        assert len(calls) == 2 and calls[0] == "a", "Adding capacity bought already translated units again"
        assert calls[-1] == "review" if repair_via_review else calls[-1] in ("a", "b")
        assert not store.db.execute("select 1 from checkpoints where job=?", (job,)).fetchone()
    finally:
        scheduler.shutdown()
        store.close()


def test_rpm_waiting_jobs_do_not_occupy_workers_needed_by_other_endpoints():
    settings = settings_pair()
    limited = dataclasses.replace(settings, translators=[])
    ready = dataclasses.replace(settings, translate=settings.translators[0], translators=[])
    client = Client(key_provider=lambda _: "")
    client.limits.for_profile(limited.translate).throttle(60)
    calls = []
    def post(url, body, *args):
        calls.append(body["model"])
        units = json.loads(body["messages"][1]["content"])["units"]
        return response({"translations": [{"id": r["id"], "text": "ok"} for r in units]})
    client._post = post
    executor = FairTranslationExecutor(max_workers=2)
    rows = [{"id": "x", "text": "source"}]
    try:
        for _ in range(20):
            executor.submit("limited", lambda reservation: client.translate(limited, rows, [], reservation=reservation), max_parallel=16, prepare=client.translation_reserver(limited))
        future = executor.submit("ready", lambda reservation: client.translate(ready, rows, [], reservation=reservation), prepare=client.translation_reserver(ready))
        assert future.result(1) == {"x": "ok"}
        assert calls == ["b"] and executor.stats("limited")["running"] == 0
        assert executor.stats("limited")["queued"] == 20
        assert len(client.limits.for_profile(ready.translate).request_times) == 1
        assert all(g.active == 0 for g in client.limits.slots.values())
    finally:
        executor.shutdown(wait=True)
