import queue
import threading
import time

from submaker.config import Settings
from submaker.media import Cancelled, estimate_output, reserve_output, temporary_subtitles, video_plan
from submaker.pipeline import Scheduler
from submaker.store import Store
import pytest


def media_info():
    video={"index":0,"codec_type":"video","codec_name":"h264","width":1280,"height":720,"pix_fmt":"yuv420p","bit_rate":"2000000"}
    audio={"index":1,"codec_type":"audio","codec_name":"aac","bit_rate":"128000"}
    return {"video":video,"audio":audio,"streams":[video,audio],"duration":3600,"format":{"bit_rate":"2128000"}}


def test_task_delete_cancels_active_work_and_never_deletes_media(tmp_path,monkeypatch):
    source=tmp_path/"source.mp4"
    source.write_bytes(b"protected source")
    output=tmp_path/"completed.mkv"
    output.write_bytes(b"protected completed output")
    model=tmp_path/"model"
    model.mkdir()
    (model/"model.bin").touch()
    info=media_info()
    info["duration"]=3
    monkeypatch.setattr("submaker.pipeline.probe",lambda *args:info)
    entered=threading.Event()
    class Pool:
        fatal=None
        def __init__(self,settings,*args):
            self.settings=settings
            self.q=queue.Queue()
        def register(self,*args): return self.q
        def submit(self,task,cancel):
            self.q.put({"type":"segment","chunk":0,"gpu":0,"segment":{"start":.1,"end":2,"text":"source","logprob":-.1,"no_speech":0}})
            self.q.put({"type":"done","chunk":0})
        def unregister(self,*args): pass
        def shutdown(self): pass
    class Client:
        def translate(self,settings,rows,context,cancel=None,instruction=""):
            entered.set()
            assert cancel.wait(5)
            raise Cancelled()
    monkeypatch.setattr("submaker.pipeline.ASRPool",Pool)
    settings=Settings(model_path=str(model),output_dir=str(tmp_path),in_flight=1)
    settings.translate.url,settings.translate.model="http://localhost/v1","test"
    store=Store(tmp_path/"jobs.sqlite")
    jobs=[store.add(source,settings) for _ in range(3)]
    store.update(jobs[2],"完成","",output)
    scheduler=Scheduler(store,settings,client=Client())
    try:
        scheduler.start()
        assert entered.wait(3)
        scheduler.delete([jobs[0]])
        deadline=time.monotonic()+5
        while any(r["id"]==jobs[0] for r in store.jobs()) and time.monotonic()<deadline:
            time.sleep(.02)
        assert {r["id"] for r in store.jobs()}==set(jobs[1:])
        scheduler.delete()
        while (scheduler.active or store.jobs()) and time.monotonic()<deadline:
            time.sleep(.02)
        assert not scheduler.active and not store.jobs()
        store.put(jobs[0],"late-result","ignored")
        store.add_usage(jobs[0],{"total_tokens":10})
        assert store.db.execute("SELECT count(*) FROM checkpoints").fetchone()[0]==0
        assert store.db.execute("SELECT count(*) FROM usage").fetchone()[0]==0
        assert source.read_bytes()==b"protected source"
        assert output.read_bytes()==b"protected completed output"
    finally:
        scheduler.shutdown()
        store.close()


def test_estimate_tracks_codec_quality_and_bitrate_without_claiming_exact_size():
    info=media_info()
    compact=estimate_output(info,Settings())
    avc=estimate_output(info,Settings(video_codec="h264"))
    sharp=estimate_output(info,Settings(video_quality=20))
    assert compact["bytes"] < avc["bytes"]
    assert compact["bytes"] < sharp["bytes"]
    for value in (compact,avc,sharp):
        assert value["low"] < value["bytes"] <= value["high"]
    fixed=estimate_output(info,Settings(size_mode="bitrate",video_bitrate_kbps=1000))
    assert abs(fixed["bytes"]-3600*(1000000+128000)/8*1.02)<1
    assert compact["high"] < 3600*2128000/8


def test_output_reservation_creates_no_standalone_subtitles(tmp_path):
    video,lock=reserve_output(tmp_path,"测试译名")
    assert video.name=="测试译名.mkv"
    assert not video.with_suffix('.srt').exists()
    assert not (tmp_path/"独立字幕").exists()
    next_video,_=reserve_output(tmp_path,"测试译名")
    assert next_video.name=="测试译名.v002.mkv"


@pytest.mark.parametrize("fail",[False,True])
def test_temporary_ass_is_removed_after_success_or_failure(tmp_path,fail):
    try:
        with temporary_subtitles(tmp_path,[{"start":1.25,"end":2.5,"text":"测试\n第二行"}]) as subtitle:
            assert subtitle.suffix==".ass" and subtitle.exists()
            text=subtitle.read_text(encoding="utf-8-sig")
            assert "0:00:01.25,0:00:02.50" in text and "\\N" in text
            if fail:
                raise RuntimeError("injected burn failure")
    except RuntimeError:
        assert fail
    assert not list(tmp_path.iterdir())


def test_gpu_mode_never_silently_falls_back_to_cpu_for_hdr():
    info=media_info()
    info['video'].update(codec_name='hevc',pix_fmt='yuv420p10le',color_transfer='smpte2084')
    with pytest.raises(ValueError,match='不会自动切换CPU'):
        video_plan(info,Settings())
