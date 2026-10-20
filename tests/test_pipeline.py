import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from submaker.config import Settings
from submaker.pipeline import Scheduler
from submaker.store import Store, fingerprint


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg not available")
@pytest.mark.parametrize("failure", ["none", "soft", "burn_mp4", "title", "translate", "repair", "review_batch", "segment", "segment_fallback"])
def test_streaming_translation_review_repair_and_real_burn(tmp_path, monkeypatch, failure):
    source = tmp_path / "original.mp4"
    subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i","color=c=navy:s=640x360:r=25:d=3","-f","lavfi","-i","sine=frequency=400:duration=3","-c:v","libx264","-c:a","aac","-shortest",str(source)],check=True)
    before = fingerprint(source)
    translated = threading.Event()
    sequence = []
    class FakePool:
        fatal = None
        def __init__(self, *args):
            self.q = queue.Queue()
            self.settings = args[0]
        def register(self, ident):
            return self.q
        def submit(self, task, cancel):
            def work():
                for i in range(45 if failure == "review_batch" else 1):
                    segment = {"start":0.2+i*.05,"end":0.24+i*.05,"text":f"こんにちは{i}。","logprob":-0.1,"no_speech":0.01} if failure == "review_batch" else {"start":0.2,"end":2.7,"text":"こんにちは。","logprob":-0.1,"no_speech":0.01}
                    self.q.put({"type":"segment","chunk":0,"gpu":0,"segment":segment})
                assert translated.wait(10), "Translation did not start before ASR chunk completed"
                sequence.append("asr_done")
                self.q.put({"type":"done","chunk":0})
            threading.Thread(target=work, daemon=True).start()
        def unregister(self, ident):
            pass
        def shutdown(self):
            pass
    class FakeClient:
        reviewed = 0
        title_calls = 0
        translate_calls = 0
        repair_failed = False
        def translate(self, settings, rows, context, cancel=None, instruction=""):
            sequence.append("repair" if instruction else "translate")
            translated.set()
            self.translate_calls += 1
            if failure == "repair" and instruction and not self.repair_failed:
                self.repair_failed = True
                raise RuntimeError("injected repair outage")
            if failure == "translate" and self.translate_calls == 1:
                raise RuntimeError("injected translation outage")
            return {r["id"]:"你好。" for r in rows}
        def brief(self, *args):
            return "日常问候"
        def review(self, settings, rows, *args):
            self.reviewed += 1
            sequence.append("review")
            if failure == "review_batch" and self.reviewed == 2:
                raise RuntimeError("injected audit response outage")
            if failure == "review_batch":
                return []
            assert self.reviewed == 1, "An additional audit was run after repair"
            return [{"ids":[rows[0]["id"]],"severity":"error","reason":"test repair"}] if self.reviewed == 1 else []
        def title(self, *args):
            self.title_calls += 1
            if failure == "title" and self.title_calls == 1:
                raise RuntimeError("injected title failure")
            return "问候测试"
    monkeypatch.setattr("submaker.pipeline.ASRPool",FakePool)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.bin").touch()
    settings = Settings(model_path=str(model), output_dir=str(tmp_path/"out"), gpus=[0], encoding="quality", decoding="software",
                        full_context_enabled=True, review_enabled=True,
                        output_mode="soft" if failure == "soft" else "burn", output_container="mp4" if failure == "burn_mp4" else "mkv")
    if failure.startswith("segment"):
        if not shutil.which("nvidia-smi"):
            pytest.skip("GPU export required")
        settings.encoding = "hardware"
        monkeypatch.setattr("submaker.segmented.enabled", lambda *args: True)
        if failure == "segment_fallback":
            from submaker.segmented import TimelineError
            def fallback(*args, **kwargs):
                sequence.append("segment_fallback")
                raise TimelineError("injected unsupported timeline")
            monkeypatch.setattr("submaker.segmented.burn_segmented", fallback)
    settings.translate.url, settings.translate.model = "http://127.0.0.1/v1", "mock"
    store = Store(tmp_path/"jobs.sqlite")
    ident = store.add(source,settings)
    scheduler = Scheduler(store,settings,client=FakeClient())
    scheduler.start()
    deadline = time.monotonic()+40
    while scheduler.active and time.monotonic()<deadline:
        time.sleep(0.1)
    if failure not in ("none", "soft", "burn_mp4", "segment", "segment_fallback"):
        assert store.jobs()[0]["state"] == "等待处理"
        assert store.get(ident,"chunk:0") is not None  # Translation outage must not abort local ASR.
        translations_before = sequence.count("translate") + sequence.count("repair")
        reviews_before = sequence.count("review")
        scheduler.retry(ident)
        while scheduler.active and time.monotonic()<deadline:
            time.sleep(0.1)
        if failure in ("title", "review_batch"):
            assert sequence.count("translate") + sequence.count("repair") == translations_before
        if failure == "title":
            assert sequence.count("review") == reviews_before
        if failure == "repair":
            assert sequence.count("review") == reviews_before, "Audit repeated after a repair outage"
        if failure == "review_batch":
            assert sequence.count("review") == reviews_before + 1, "Previously completed audit batch was sent again"
    row = store.jobs()[0]
    assert row["state"] == "完成", row["detail"]
    if failure == "segment_fallback":
        assert sequence.count("segment_fallback") == 1
    assert Path(row["output"]).exists()
    assert Path(row["output"]).suffix == (".mp4" if failure == "burn_mp4" else ".mkv")
    from submaker.media import probe
    exported = probe(row["output"], settings)
    assert sum(s["codec_type"] == "subtitle" for s in exported["streams"]) == (1 if failure == "soft" else 0)
    if failure == "soft":
        assert exported["video"]["codec_name"] == "h264"
    assert not list((tmp_path/"out").rglob("*.srt"))
    assert not list((tmp_path/"out").rglob("*.ass"))
    assert not list((tmp_path/"out").glob(".submaker-*"))
    assert sequence.index("translate") < sequence.index("asr_done") < sequence.index("review")
    if failure != "review_batch":
        assert sequence.index("review") < sequence.index("repair")
    assert store.get(ident,"chunk:0") is None
    assert fingerprint(source) == before
    assert not list((tmp_path/"out").glob("*.partial*"))
    scheduler.shutdown()
    store.close()
