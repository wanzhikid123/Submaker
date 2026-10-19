import queue
import shutil
import subprocess
import threading
import time

import pytest

from submaker.config import Settings
from submaker.pipeline import Scheduler
from submaker.store import Store


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_slow_first_reply_does_not_hide_completed_batches_or_progress(tmp_path, monkeypatch):
    source = tmp_path/"source.mp4"
    subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i","color=c=navy:s=320x240:r=25:d=3","-f","lavfi","-i","sine=frequency=400:duration=3","-c:v","libx264","-c:a","aac","-shortest",str(source)], check=True)
    first_started, asr_done, release, visible_progress = (threading.Event() for _ in range(4))
    requests = []
    class Pool:
        fatal = None
        def __init__(self, settings, *args):
            self.settings = settings
            self.events = queue.Queue()
        def register(self, ident):
            return self.events
        def submit(self, task, cancel):
            def produce():
                for i in range(40):
                    self.events.put({"type":"segment","chunk":0,"gpu":0,"segment":{"start":i*.06,"end":i*.06+.04,"text":f"source-{i}","logprob":-.1,"no_speech":0.0}})
                    if i == 0:
                        assert first_started.wait(2), "First stable sentence was buffered until ASR ended"
                self.events.put({"type":"done","chunk":0})
                asr_done.set()
            threading.Thread(target=produce,daemon=True).start()
        def unregister(self, ident):
            pass
        def shutdown(self):
            pass
    class Client:
        def translate(self, settings, rows, context, cancel=None, instruction=""):
            requests.append(len(rows))
            if rows[0]["text"] == "source-0":
                first_started.set()
                assert release.wait(8)
            else:
                assert asr_done.wait(2)
            return {r["id"]:"译文" for r in rows}
        def brief(self, *args):
            return "brief"
        def review(self, *args):
            return []
        def title(self, *args):
            return "test"
    def observe(kind, data):
        if kind == "preview" and asr_done.is_set() and data["translated"] >= 12:
            visible_progress.set()
    monkeypatch.setattr("submaker.pipeline.ASRPool", Pool)
    model=tmp_path/"model"
    model.mkdir()
    (model/"model.bin").touch()
    settings=Settings(model_path=str(model),output_dir=str(tmp_path/"out"),gpus=[0],encoding="quality",decoding="software")
    settings.translate.url,settings.translate.model="http://localhost/v1","mock"
    store=Store(tmp_path/"jobs.sqlite")
    store.add(source,settings)
    scheduler=Scheduler(store,settings,observe,Client())
    try:
        scheduler.start()
        assert visible_progress.wait(4), "UI waited for slow first future and hid later completed batches"
        assert not release.is_set()
        assert "已译" in store.jobs()[0]["detail"]
        release.set()
        deadline=time.monotonic()+10
        while scheduler.active and time.monotonic()<deadline:
            time.sleep(.05)
        assert store.jobs()[0]["state"] == "完成"
        assert requests[0] == 1
        assert max(requests) == 12
        assert sum(requests) == 40
        assert len(requests) <= 5
    finally:
        release.set()
        scheduler.shutdown()
        store.close()
