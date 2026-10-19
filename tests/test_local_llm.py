import concurrent.futures as cf
import dataclasses
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from PySide6.QtWidgets import QApplication

from submaker.config import LocalModel, LocalRole, Profile, Settings
from submaker.gui import MainWindow, SettingsDialog
from submaker.llm import APIError, APITruncated, Client, Limits, profile_identity
from submaker.local_llm import LocalRuntime, command, model_identity, verify_model
from submaker.media import Cancelled
from submaker.pipeline import Scheduler
from submaker.store import Store


def test_local_snapshot_routes_all_roles_and_reserves_one_gpu():
    settings = Settings(llm_mode="local", local_review=LocalRole("high", 2048, 2400))
    settings.translate = Profile(url="https://unused.example/v1", model="unused")
    settings.translators = [dataclasses.replace(settings.translate, name="unused second")]
    settings.review = dataclasses.replace(settings.translate, name="unused reviewer")
    saved = Settings.from_dict(settings.as_dict())
    effective = saved.execution_settings()
    assert effective.gpus == [0] and effective.video_gpu == 0
    assert effective.translate.provider == effective.review.provider == "local"
    assert len(effective.translation_profiles()) == 1 and effective.translators == []
    assert effective.translate.effort == "none" and effective.review.effort == "high"
    repair = effective.repair_settings()
    assert profile_identity(repair.translation_profiles()[0]) == profile_identity(effective.review_profile())
    assert repair.translation_profiles()[0].name == "本地审校 / 修正"
    assert saved.translate.model == "unused" and len(saved.translators) == 1
    assert saved.gpus == [0, 1]  # The user's API configuration is retained.
    saved.local_model.gpu = 0
    assert saved.execution_settings().gpus == [1]
    assert saved.execution_settings().video_gpu == 1


def test_local_mode_rejects_remote_pool_before_credentials_or_transport(monkeypatch):
    client = Client(local_only=True, key_provider=lambda _: pytest.fail("credential access"))
    monkeypatch.setattr(client, "_post", lambda *args: pytest.fail("remote request"))
    remote = Profile(url="https://unused.example/v1", model="unused")
    with pytest.raises(APIError, match="禁用网络 API"):
        client.request(remote, "test", {}, profiles=[Settings(llm_mode="local").local_profile("translate"), remote])


def test_local_failures_never_fall_back_and_are_reported(monkeypatch):
    class Runtime:
        def ensure(self, options, cancel):
            raise RuntimeError("GPU 显存不足")
    events = []
    client = Client(local_only=True, local_runtime=Runtime(), status=events.append,
                    key_provider=lambda _: pytest.fail("local inference must not read keys"))
    monkeypatch.setattr(client, "_post", lambda *args: pytest.fail("transport after failed load"))
    settings = Settings(llm_mode="local")
    with pytest.raises(APIError, match="显存不足"):
        client.translate(settings, [{"id": "s1", "text": "test"}], [])
    assert events[-1]["event"] == "paused" and "显存不足" in events[-1]["message"]
    assert not any(e["event"] == "request" for e in events)


def test_translation_and_review_share_one_local_inference_slot():
    settings = Settings(llm_mode="local", local_review=LocalRole("high"))
    limits = Limits()
    translate = limits.for_profile(settings.local_profile("translate"))
    review = limits.for_profile(settings.local_profile("review"))
    assert translate is review and translate.limit == 1 and translate.rpm == 0
    assert translate.try_acquire() and not review.try_acquire()
    translate.release()
    assert review.try_acquire()
    review.release()


def test_old_local_parallel_settings_are_ignored_without_losing_model_or_roles():
    saved = Settings(llm_mode="local").as_dict()
    saved["local_model"]["parallelism"] = 16
    saved["local_review"]["effort"] = "high"
    settings = Settings.from_dict(saved)
    assert "parallelism" not in settings.as_dict()["local_model"]
    assert settings.local_review.effort == "high"
    assert settings.translation_parallelism() == 1
    assert settings.local_profile("review").concurrency == 1
    effective = settings.execution_settings().as_dict()
    effective["translate"]["concurrency"] = 16
    assert Settings.from_dict(effective).translation_parallelism() == 1
    # Even an old or hand-edited profile cannot create extra local requests.
    profile = settings.local_profile("translate")
    profile.concurrency = 16
    gate = Limits().for_profile(profile)
    assert gate.limit == 1


def test_model_identity_changes_when_selected_weights_change(tmp_path):
    path = tmp_path/"model.gguf"
    path.write_bytes(b"GGUF1")
    settings = Settings(llm_mode="local", local_model=LocalModel(model_path=str(path)))
    before = profile_identity(settings.local_profile("translate"))
    path.write_bytes(b"GGUFdifferent")
    assert profile_identity(settings.local_profile("translate")) != before


def test_ollama_indexes_the_selected_gguf_without_an_extra_weight_copy(tmp_path, monkeypatch):
    source = tmp_path/"weights.gguf"
    source.write_bytes(b"GGUFsynthetic-weights")
    options = LocalModel(model_path=str(source))
    runtime = LocalRuntime()
    try:
        directory, digest = runtime._ollama_store(options, None)
        blob = directory/"blobs"/("sha256-"+digest)
        assert os.path.samefile(source, blob)
        assert runtime._ollama_store(options, None) == (directory, digest)
        assert source.read_bytes() == b"GGUFsynthetic-weights"
    finally:
        runtime.stop()


@pytest.mark.parametrize("replacement", [b"GGUF-original", b"GGUF-modified"])
def test_ollama_cache_identity_survives_blob_touch_but_tracks_replaced_weights(tmp_path, replacement):
    source = tmp_path/"weights.gguf"
    source.write_bytes(b"GGUF-original")
    options = LocalModel(model_path=str(source))
    runtime = LocalRuntime()
    try:
        directory, digest = runtime._ollama_store(options, None)
        before = model_identity(options)
        blob = directory/"blobs"/("sha256-"+digest)
        stamp = source.stat().st_mtime_ns
        os.utime(blob, ns=(stamp+10_000_000, stamp+10_000_000))
        assert model_identity(options) == before
        assert verify_model(options) == digest
        updated = tmp_path/"replacement.gguf"
        updated.write_bytes(replacement)
        updated.replace(source)
        assert model_identity(options) != before
        directory, updated_digest = runtime._ollama_store(options, None)
        assert os.path.samefile(source, directory/"blobs"/("sha256-"+updated_digest))
        assert (updated_digest == digest) == (replacement == b"GGUF-original")
    finally:
        runtime.stop()


def test_ollama_blob_touch_keeps_real_owned_service_running(tmp_path, monkeypatch):
    script = tmp_path/"health.py"
    script.write_text('''import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200)
  self.end_headers()
 def log_message(self,*args): pass
HTTPServer(("127.0.0.1",int(sys.argv[1])),Handler).serve_forever()
''')
    engine, weights = tmp_path/"ollama.exe", tmp_path/"model.gguf"
    engine.touch()
    weights.write_bytes(b"GGUF-synthetic")
    options = LocalModel(server_path=str(engine), model_path=str(weights))
    monkeypatch.setattr("submaker.local_llm.command", lambda options, port, token: [sys.executable, str(script), str(port)])
    envs, popen = [], subprocess.Popen
    def launch(*args, **kwargs):
        envs.append(kwargs["env"])
        return popen(*args, **kwargs)
    monkeypatch.setattr("submaker.local_llm.subprocess.Popen", launch)
    def prepare(runtime, options, base, digest, cancel):
        stamp = weights.stat().st_mtime_ns+10_000_000
        os.utime(weights.parent/".submaker-local-model"/"blobs"/("sha256-"+digest), ns=(stamp, stamp))
        return {}
    monkeypatch.setattr(LocalRuntime, "_prepare_ollama", prepare)
    runtime = LocalRuntime()
    child = None
    try:
        result = runtime.ensure(options)
        child = runtime.process
        assert runtime.ensure(options) == result and runtime.process is child
        assert len(envs) == 1 and envs[0]["OLLAMA_NUM_PARALLEL"] == "1"
        assert runtime.snapshot()["phase"] == "ready"
    finally:
        runtime.stop()
    assert child is not None and child.poll() is not None


@pytest.mark.parametrize("actual", [1, 2])
def test_local_http_is_fixed_single_request_even_if_backend_supports_more(actual):
    lock, peak, active = threading.Lock(), [0], [0]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.loads(body["messages"][1]["content"])
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(.15)
            payload = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(data)}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            with lock:
                active[0] -= 1
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    class Runtime:
        def ensure(self, options, cancel):
            return f"http://127.0.0.1:{server.server_port}/v1/chat/completions", "test-local"
        def snapshot(self):
            return {"parallelism": actual, "parallelism_verified": True}
    client = Client(local_only=True, local_runtime=Runtime(), key_provider=lambda _: pytest.fail("network credentials"))
    settings = Settings(llm_mode="local")
    profile = settings.local_profile("translate")
    try:
        assert client.request(profile, "echo JSON", {"warmup": True}) == {"warmup": True}
        with cf.ThreadPoolExecutor(max_workers=2) as executor:
            replies = list(executor.map(lambda i: client.request(profile, "echo JSON", {"number": i}), (17, 29)))
        assert replies == [{"number": 17}, {"number": 29}]
        assert peak == [1] and active == [0]
        assert client.limits.for_profile(settings.local_profile("review")).limit == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_ollama_rejects_cpu_spill_instead_of_reporting_gpu_ready(monkeypatch):
    runtime = LocalRuntime()
    calls = []
    def post(url, body, *args):
        calls.append((url, body))
        return httpx.Response(200, json={"done": True})
    monkeypatch.setattr(Client, "_post", staticmethod(post))
    original = httpx.Client
    def reply(request):
        return httpx.Response(200, json={"models": [{"name": "submaker-local:latest", "size": 1000, "size_vram": 600}]})
    monkeypatch.setattr("submaker.local_llm.httpx.Client", lambda **kwargs: original(transport=httpx.MockTransport(reply), **kwargs))
    try:
        with pytest.raises(RuntimeError, match="未完整加载到指定 GPU"):
            runtime._prepare_ollama(LocalModel(), "http://127.0.0.1:12345", "test-digest", None)
        assert calls[0][1]["parameters"]["num_gpu"] == 999
        assert calls[1][1]["keep_alive"] == -1
    finally:
        runtime.stop()


def test_runtime_command_is_gpu_only_and_loopback_only():
    args = command(LocalModel(gpu=1, server_path="llama-server.exe"), 4567, "test-ephemeral-token")
    assert args[args.index("--host")+1] == "127.0.0.1"
    assert args[args.index("--device")+1] == "CUDA0"  # Only GPU 1 is visible in the child environment.
    assert args[args.index("--parallel")+1] == "1"
    assert args[args.index("--gpu-layers")+1] == "999"
    assert "--offline" in args and "--no-context-shift" in args
    assert args[args.index("--ctx-size")+1] == "16384"
    assert "--no-slots" not in args  # Authenticated numeric counters for speed.


def test_runtime_missing_weights_or_engine_fails_without_any_network(tmp_path, monkeypatch):
    monkeypatch.setattr(httpx.Client, "get", lambda *args, **kwargs: pytest.fail("no request before valid files"))
    runtime = LocalRuntime()
    try:
        with pytest.raises(RuntimeError, match="未安装本地 CUDA"):
            runtime.ensure(LocalModel(server_path=str(tmp_path/"missing.exe")))
        assert runtime.snapshot()["phase"] == "error"
        engine = tmp_path/"engine.exe"
        engine.touch()
        with pytest.raises(RuntimeError, match="未找到本地 GGUF"):
            runtime.ensure(LocalModel(server_path=str(engine), model_path=str(tmp_path/"missing.gguf")))
        assert runtime.process is None
    finally:
        runtime.stop()


def test_windows_app_control_error_is_actionable():
    runtime = LocalRuntime()
    class Process:
        def poll(self):
            return -1058471934  # 0xc0e90002
    runtime.process = Process()
    try:
        assert "4551" in runtime._error() and "应用程序控制" in runtime._error()
    finally:
        runtime.process = None
        runtime.stop()


@pytest.mark.parametrize("cancel_load", [False, True])
def test_runtime_owns_real_child_reuses_and_stops_it(tmp_path, monkeypatch, cancel_load):
    script = tmp_path/"health_server.py"
    script.write_text('''import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(503 if sys.argv[2] == "wait" else 200)
  self.end_headers()
 def log_message(self,*args): pass
HTTPServer(("127.0.0.1",int(sys.argv[1])),Handler).serve_forever()
''')
    path = tmp_path/"model.gguf"
    path.write_bytes(b"GGUF")
    options = LocalModel(server_path=sys.executable, model_path=str(path))
    envs = []
    popen = subprocess.Popen
    def launch(*args, **kwargs):
        envs.append(kwargs["env"])
        return popen(*args, **kwargs)
    monkeypatch.setattr("submaker.local_llm.subprocess.Popen", launch)
    monkeypatch.setattr("submaker.local_llm.command", lambda options, port, token: [sys.executable, str(script), str(port), "wait" if cancel_load else "ready"])
    runtime = LocalRuntime()
    cancel = threading.Event()
    child = None
    try:
        if cancel_load:
            timer = threading.Timer(.4, cancel.set)
            timer.start()
            with pytest.raises(Cancelled):
                runtime.ensure(options, cancel)
            timer.join()
            assert runtime.process is None
        else:
            url, token = runtime.ensure(options)
            child = runtime.process
            assert runtime.ensure(options) == (url, token)
            assert runtime.process is child and len(envs) == 1
            assert envs[0]["CUDA_VISIBLE_DEVICES"] == "1"
            assert envs[0]["HF_HUB_OFFLINE"] == "1"
            assert runtime.snapshot()["phase"] == "ready"
    finally:
        runtime.stop()
    assert child is None or child.poll() is not None


def test_local_review_splits_over_capacity_without_skipping_any_units(monkeypatch):
    settings = Settings(llm_mode="local")
    client = Client(local_only=True)
    checked = []
    def request(profile, system, payload, *args, **kwargs):
        if len(payload["units"]) > 2:
            raise APITruncated("test context capacity")
        checked.extend(r["source"] for r in payload["units"])
        return {"issues": [{"ids": [r["id"]], "severity": "error", "reason": "test"} for r in payload["units"]]}
    monkeypatch.setattr(client, "request", request)
    rows = [{"id": f"original-{i}", "source": f"source-{i}", "translation": "translation"} for i in range(5)]
    issues = client.review(settings, rows, "brief")
    assert checked == [r["source"] for r in rows]
    assert {i for issue in issues for i in issue["ids"]} == {r["id"] for r in rows}


def test_llama_context_error_is_split_instead_of_disabling_local_engine(monkeypatch):
    settings = Settings(llm_mode="local")
    class Runtime:
        def ensure(self, *args):
            return "http://127.0.0.1:12345/v1/chat/completions", "test"
    client = Client(local_only=True, local_runtime=Runtime())
    monkeypatch.setattr(client, "_post", lambda *args: httpx.Response(400, json={"error": {"code": 400, "type": "exceed_context_size_error"}}))
    with pytest.raises(APITruncated):
        client.request(settings.local_profile("review"), "test", {})
    assert not client.faults


def test_switching_mode_stops_old_whisper_pool_before_local_model(tmp_path):
    store = Store(tmp_path/"jobs.sqlite")
    scheduler = Scheduler(store, Settings())
    stopped = []
    class Pool:
        settings = Settings()
        def shutdown(self):
            stopped.append(True)
    scheduler.pool = Pool()
    try:
        scheduler.configure(Settings(llm_mode="local"))
        assert stopped == [True] and scheduler.pool is None and scheduler.client.local_only
        scheduler.active["test"] = threading.Event()
        with pytest.raises(RuntimeError, match="暂停"):
            scheduler.configure(Settings())
    finally:
        scheduler.active.clear()
        scheduler.shutdown()
        store.close()


def test_local_settings_ui_disables_api_and_saves_roles_separately(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(Settings, "save", lambda self: None)
    engine, model = tmp_path/"llama-server.exe", tmp_path/"model.gguf"
    engine.touch()
    model.write_bytes(b"GGUF")
    settings = Settings(local_model=LocalModel(str(engine), str(model)))
    settings.translate = Profile(url="https://unused.example/v1", model="unused")
    dialog = SettingsDialog(settings)
    try:
        dialog.llm_mode.setCurrentIndex(dialog.llm_mode.findData("local"))
        assert not dialog.pool.isEnabled() and not dialog.review.isEnabled()
        assert dialog.translation_stack.currentWidget() is dialog.local_translate
        assert dialog.review_stack.currentWidget() is dialog.local_review
        dialog.local_review.effort.setCurrentIndex(dialog.local_review.effort.findData("high"))
        dialog.local_review.max_tokens.setValue(2048)
        assert not hasattr(dialog.local_model_editor, "parallelism")
        dialog.video_concurrency.setValue(3)
        dialog.video_decode_threads.setValue(2)
        dialog.video_filter_threads.setValue(3)
        dialog.accept()
        assert dialog.settings.llm_mode == "local"
        assert dialog.settings.local_review.effort == "high" and dialog.settings.local_translate.effort == "none"
        assert dialog.settings.translate.model == "unused"
        assert dialog.settings.local_profile("translate").concurrency == 1
        assert dialog.settings.video_concurrency == 3
        assert (dialog.settings.video_decode_threads, dialog.settings.video_filter_threads) == (2, 3)
    finally:
        dialog.close()
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: dialog.settings))
    monkeypatch.setattr("submaker.gui.Store", lambda: Store(tmp_path/"gui.sqlite"))
    window = MainWindow()
    try:
        assert not window.gpu1.isEnabled() and not window.gpu1.isChecked()
        assert window.gpu0.isEnabled() and window.gpu0.isChecked()
        assert window.current_settings().execution_settings().gpus == [0]
        assert "本地" in window.gpu_workers[1].role.text()
        monkeypatch.setattr(window.scheduler.client.runtime(), "snapshot", lambda: {
            "phase": "ready", "tokens_per_second": 33.0, "speed_phase": "generating",
            "message": "本地模型已就绪"})
        window.refresh_gpu_workers()
        assert "并发" not in window.gpu_workers[1].role.text()
        assert window.gpu_workers[1].speed.text() == "33.0 tokens/s"
        assert not window.gpu_workers[1].speed.isHidden()
        assert window.gpu_workers[0].speed.isHidden()
        window.settings.local_model.gpu = 0
        window.refresh_gpu_workers()
        assert not window.gpu_workers[0].speed.isHidden() and window.gpu_workers[1].speed.isHidden()
        window.settings.llm_mode = "api"
        window.refresh_gpu_workers()
        assert all(worker.speed.isHidden() for worker in window.gpu_workers.values())
    finally:
        window.close()
        window.store.close()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_local_pipeline_real_loopback_transport_and_video_export(tmp_path, monkeypatch):
    source = tmp_path/"original.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=navy:s=320x240:r=25:d=2",
                    "-f", "lavfi", "-i", "sine=frequency=400:duration=2", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source)], check=True)
    captured, loads, sequence = [], [], []
    translated = threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            captured.append(body)
            system = body["messages"][0]["content"]
            data = json.loads(body["messages"][1]["content"])
            if system.startswith("Translate subtitles"):
                result = {"translations": [{"id": r["id"], "text": "你好。"} for r in data["units"]]}
                sequence.append("repair" if data["repair_instruction"] else "translate")
                translated.set()
            elif system.startswith("Audit every"):
                result = {"issues": [{"ids": [data["units"][0]["id"]], "reason": "test repair", "severity": "error"}]}
                sequence.append("review")
            elif system.startswith("Create a concise"):
                result = {"brief": "A greeting."}
            else:
                result = {"title": "translated-title"}
            payload = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}], "usage": {"total_tokens": 20}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    class Runtime:
        def ensure(self, options, cancel):
            loads.append(options.gpu)
            return f"http://127.0.0.1:{server.server_port}/v1/chat/completions", "test-local-token"
        def stop(self):
            pass
    class Pool:
        fatal = None
        def __init__(self, settings, *args):
            self.settings = settings
            assert settings.gpus == [0] and settings.video_gpu == 0
            self.events = queue.Queue()
        def register(self, ident):
            return self.events
        def submit(self, task, cancel):
            def work():
                self.events.put({"type": "segment", "chunk": 0, "segment": {"start": .1, "end": 1.8, "text": "こんにちは。"}})
                assert translated.wait(8)
                sequence.append("asr_done")
                self.events.put({"type": "done", "chunk": 0})
            threading.Thread(target=work, daemon=True).start()
        def unregister(self, ident):
            pass
        def shutdown(self):
            pass
    monkeypatch.setattr("submaker.pipeline.ASRPool", Pool)
    whisper = tmp_path/"whisper"
    whisper.mkdir()
    (whisper/"model.bin").touch()
    settings = Settings(llm_mode="local", model_path=str(whisper), output_dir=str(tmp_path/"out"),
                        local_review=LocalRole("high", 2048), encoding="quality", decoding="software")
    local_weights = tmp_path/"local.gguf"
    local_weights.write_bytes(b"GGUFsynthetic-model")
    settings.local_model.model_path = str(local_weights)
    settings.translate = Profile(url="https://must-never-call.example/v1", model="unused")
    store = Store(tmp_path/"jobs.sqlite")
    job = store.add(source, settings)
    scheduler = Scheduler(store, settings)
    scheduler.client.local_runtime = Runtime()
    scheduler.client.key_provider = lambda _: pytest.fail("local inference accessed network credentials")
    try:
        scheduler.start()
        deadline = time.monotonic()+20
        while scheduler.active and time.monotonic() < deadline:
            time.sleep(.05)
        row = store.jobs()[0]
        assert row["state"] == "完成", row["detail"]
        assert sequence == ["translate", "asr_done", "review", "repair"]
        assert loads == [1]*5
        assert all(b["model"] == "submaker-local" for b in captured)
        assert captured[0]["max_tokens"] == 4096 and captured[0]["chat_template_kwargs"]["enable_thinking"] is False
        assert all(b["max_tokens"] == 2048 and b["reasoning_effort"] == "high" for b in captured[1:])
        assert row["total_tokens"] == 100 and row["token_requests"] == 5
        assert store.get(job, "chunk:0") is None
        assert not list((tmp_path/"out").glob("*.ass")) and not list((tmp_path/"out").glob("*.srt"))
    finally:
        scheduler.shutdown()
        store.close()
        server.shutdown()
        server.server_close()
