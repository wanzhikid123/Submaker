import dataclasses
import concurrent.futures as cf
import hashlib
import re
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from submaker import media, segmented
from submaker.config import Settings
from submaker.encoding_slots import EncodingSlots


def fixture(root, ten=False, attach=True):
    source = root/"source.mkv"
    original = root/"original.srt"
    media.write_srt(original, [{"start": 0, "end": 7, "text": "old subtitle"}])
    metadata = root/"chapters.txt"
    metadata.write_text(";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=3000\ntitle=First\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=3000\nEND=7400\ntitle=Second\n")
    attachment = root/"attachment.txt"
    attachment.write_text("attachment fixture")
    args = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=30000/1001:d=7",
            "-f", "lavfi", "-i", "sine=frequency=400:duration=7.5", "-itsoffset", "0.37", "-f", "lavfi", "-i", "sine=frequency=800:duration=7",
            "-i", str(original), "-f", "ffmetadata", "-i", str(metadata), "-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:s",
            "-map_metadata", "4", "-map_chapters", "4", "-vf", "drawbox=x=0:y=165:w=iw:h=75:c=black:t=fill,setpts='PTS+if(gte(N,90),0.3/TB,0)'", "-fps_mode", "passthrough",
            "-c:v", "libx265" if ten else "libx264", "-pix_fmt", "yuv420p10le" if ten else "yuv420p", "-threads", "2"]
    if ten:
        args += ["-x265-params", "pools=2:frame-threads=1"]
    args += ["-c:a", "aac", "-c:s", "srt"]
    if attach:
        args += ["-attach", str(attachment), "-metadata:s:t:0", "mimetype=text/plain"]
    args += [str(source)]
    subprocess.run(args, check=True, capture_output=True)
    return source


def test_segmentation_is_optional_gpu_only_and_skips_short_or_single_slot_jobs():
    video = {"codec_name": "h264", "pix_fmt": "yuv420p"}
    settings = Settings(output_mode="burn")
    info = {"duration": 60, "video": video}
    assert segmented.enabled(info, settings)
    for changed in (dataclasses.replace(settings, video_segmented=False), dataclasses.replace(settings, video_concurrency=1), dataclasses.replace(settings, encoding="quality")):
        assert not segmented.enabled(info, changed)
    assert not segmented.enabled({**info, "duration": 5}, settings)


@pytest.mark.skipif(not shutil.which("nvidia-smi"), reason="V100 GPU required")
@pytest.mark.parametrize("codec,ten,container", [("hevc", False, "mkv"), ("h264", False, "mkv"), ("hevc", True, "mkv"), ("hevc", False, "mp4"), ("h264", False, "mp4")])
def test_real_segmented_gpu_vfr_subtitle_cuts_audio_chapters_and_attachments(tmp_path, codec, ten, container):
    import numpy as np
    source = fixture(tmp_path, ten, attach=container=="mkv")
    settings = Settings(output_mode="burn", output_container=container, gpus=[0], video_gpu=0, video_codec=codec, video_quality=24, size_mode="quality")
    info = media.probe(source, settings)
    pool = EncodingSlots(3)
    events = []
    output = tmp_path/("result."+container)
    with media.temporary_subtitles(tmp_path, [{"start": .2, "end": 6.8, "text": "跨分段边界字幕"}]) as subtitle:
        segmented.burn_segmented(source, output, subtitle, info, settings, threading.Event(), pool, "test",
                                 on_worker=lambda kind, gpu, index, total: events.append((kind, gpu, index, pool.snapshot()["active"])))
        baseline = tmp_path/("single."+container)
        media.burn(source, baseline, subtitle, info, settings, threading.Event(), lambda _: None)
    # A fine transport clock must never silently change NVENC's bitrate or
    # quality. Compare the same output frames with the original single path.
    comparison = subprocess.run(["ffmpeg", "-v", "info", "-i", str(baseline), "-i", str(output),
        "-filter_complex", "[0:v]setpts=N/(30*TB)[a];[1:v]setpts=N/(30*TB)[b];[a][b]ssim", "-an", "-f", "null", "-"], capture_output=True, check=True)
    score = float(re.search(r"All:([0-9.]+)", comparison.stderr.decode("utf-8", "replace"))[1])
    assert score > .97
    assert .5 < output.stat().st_size/baseline.stat().st_size < 2
    assert max(e[3] for e in events) == 3
    assert pool.snapshot()["active"] == pool.snapshot()["waiting"] == 0
    result = media.probe(output, settings)
    assert result["video"]["codec_name"] == codec
    assert result["video"]["pix_fmt"] == ("yuv420p10le" if ten else "yuv420p")
    assert len(result["chapters"]) == 2
    origin = float(info["format"].get("start_time") or 0)
    for old, new in zip(info["chapters"], result["chapters"]):
        assert old["tags"]["title"] == new["tags"]["title"]
        for key in ("start_time", "end_time"):
            # MP4 chapter tracks cover a leading gap from zero; subsequent
            # boundaries still retain the source chapter times plus offset.
            if container == "mp4" and old is info["chapters"][0] and key == "start_time":
                assert float(new[key]) == 0
            else:
                assert abs(float(new[key])-float(old[key])+origin) < .002
    assert not any(s["codec_type"] == "subtitle" for s in result["streams"])
    assert sum(s["codec_type"] == "attachment" for s in result["streams"]) == (1 if container=="mkv" else 0)
    original_times = segmented.frame_times(source, info["video"]["index"], settings, None, float(info["format"].get("start_time") or 0))
    output_times = segmented.frame_times(output, 0, settings, None)
    assert len(original_times) == len(output_times) == 210
    assert max(abs(a-b) for a,b in zip(original_times, output_times)) <= .002
    for index in range(2):
        hashes = []
        for path in (source, output):
            value = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:a:{index}", "-c", "copy", "-f", "hash", "pipe:1"], capture_output=True, check=True).stdout
            hashes.append(value)
        assert hashes[0] == hashes[1]
    for moment in (2.4, 2.6, 4.9, 5.1):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(moment), "-i", str(output), "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"], check=True, capture_output=True).stdout
        pixels = np.frombuffer(raw, np.uint8).reshape(240, 320, 3)
        # Anti-aliased small glyphs over the source's black caption area.
        assert np.count_nonzero(np.all(pixels[190:225] > 140, axis=2)) > 70
    assert not list(tmp_path.glob(".submaker-*")) and not list(tmp_path.glob("*.partial*"))


@pytest.mark.skipif(not shutil.which("nvidia-smi"), reason="V100 GPU required")
def test_segment_cancellation_cleans_processes_buffers_and_all_encoding_slots(tmp_path):
    source = fixture(tmp_path)
    settings = Settings(output_mode="burn", gpus=[0], video_gpu=0)
    info = media.probe(source, settings)
    pool, cancel = EncodingSlots(3), threading.Event()
    started = []
    def event(kind, gpu, index, total):
        if kind == "encoding_started":
            started.append(index)
            if len(started) == 3:
                cancel.set()
    with media.temporary_subtitles(tmp_path, []) as subtitle:
        with pytest.raises(media.Cancelled):
            segmented.burn_segmented(source, tmp_path/"result.mkv", subtitle, info, settings, cancel, pool, "cancel", on_worker=event)
    assert pool.snapshot()["active"] == pool.snapshot()["waiting"] == 0
    assert not (tmp_path/"result.mkv").exists() and not list(tmp_path.glob("*.partial*"))


def test_failed_segment_encoder_releases_every_slot_and_removes_partial(tmp_path, monkeypatch):
    source = fixture(tmp_path)
    settings = Settings(output_mode="burn", gpus=[0], video_gpu=0)
    info = media.probe(source, settings)
    pool = EncodingSlots(3)
    original = segmented.burn_command
    def fail(*args, **kwargs):
        command = original(*args, **kwargs)
        command[command.index("-c:v:0")+1] = "missing-test-encoder"
        return command
    monkeypatch.setattr(segmented, "burn_command", fail)
    with media.temporary_subtitles(tmp_path, []) as subtitle:
        with pytest.raises(RuntimeError, match="分段GPU编码失败"):
            segmented.burn_segmented(source, tmp_path/"result.mkv", subtitle, info, settings, threading.Event(), pool, "failure")
    assert pool.snapshot()["active"] == pool.snapshot()["waiting"] == 0
    assert not (tmp_path/"result.mkv").exists() and not list(tmp_path.glob("*.partial*"))


@pytest.mark.skipif(not shutil.which("nvidia-smi"), reason="V100 GPU required")
@pytest.mark.parametrize("devices,limit", [([0], 3), ([0, 1], 6)])
def test_two_movies_share_slots_and_out_of_order_segments_do_not_deadlock(tmp_path, monkeypatch, devices, limit):
    source = fixture(tmp_path)
    settings = Settings(output_mode="burn", gpus=devices, video_gpu=-1, video_concurrency=limit)
    info = media.probe(source, settings)
    pool, lock, peaks = EncodingSlots(limit), threading.Lock(), []
    original = segmented.burn_command
    def delay_first(*args, **kwargs):
        command = original(*args, **kwargs)
        if kwargs["segment"][0] == 0:
            command[command.index("-i"):command.index("-i")] = ["-re"]
        return command
    monkeypatch.setattr(segmented, "burn_command", delay_first)
    def event(*args):
        with lock:
            snapshot = pool.snapshot()
            peaks.append(snapshot["active"])
            assert all(active <= 3 for active in snapshot["gpus"].values())
    with media.temporary_subtitles(tmp_path, []) as subtitle:
        def run(index):
            output = tmp_path/f"movie-{index}.mkv"
            segmented.burn_segmented(source, output, subtitle, info, settings, threading.Event(), pool, f"movie-{index}", on_worker=event)
            return output
        with cf.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run, i) for i in range(2)]
            assert all(f.result(timeout=30).exists() for f in futures)
    assert max(peaks) == limit
    assert pool.snapshot()["active"] == pool.snapshot()["waiting"] == 0
    assert not list(tmp_path.glob("*.partial*"))


@pytest.mark.skipif(not shutil.which("nvidia-smi"), reason="Two V100 GPUs required")
@pytest.mark.parametrize("codec", ["hevc", "h264"])
def test_single_movie_uses_six_segments_on_both_gpus_and_preserves_timeline(tmp_path, codec):
    source = fixture(tmp_path)
    settings = Settings(output_mode="burn", gpus=[0, 1], video_concurrency=6, video_codec=codec)
    info = media.probe(source, settings)
    pool = EncodingSlots(6)
    wave = threading.Barrier(6)
    events, lock = [], threading.Lock()
    def event(kind, gpu, index, total):
        if kind == "encoding_started":
            with lock:
                events.append((gpu, pool.snapshot()))
            # Hold the first six admitted chunks before launching the encoders
            # so the test cannot pass with a silently three-entry ready queue.
            if index < 6:
                wave.wait(timeout=10)
    output = tmp_path/"dual.mkv"
    with media.temporary_subtitles(tmp_path, [{"start": .1, "end": 7, "text": "双卡六路边界字幕"}]) as subtitle:
        segmented.burn_segmented(source, output, subtitle, info, settings, threading.Event(), pool, "dual", on_worker=event)
    assert {gpu for gpu, _ in events[:6]} == {0, 1}
    assert events[5][1]["gpus"] == {0: 3, 1: 3}
    assert all(s["active"] <= 6 and max(s["gpus"].values()) <= 3 for _, s in events)
    original = segmented.frame_times(source, info["video"]["index"], settings, None, float(info["format"].get("start_time") or 0))
    actual = segmented.frame_times(output, 0, settings, None)
    assert len(actual) == len(original) == 210
    assert all(abs(a-b) <= .002 for a, b in zip(original, actual))
    result = media.probe(output, settings)
    assert result["video"]["codec_name"] == codec
    assert sum(s["codec_type"] == "audio" for s in result["streams"]) == 2
    assert pool.snapshot()["active"] == pool.snapshot()["waiting"] == 0
    assert not list(tmp_path.glob("*.partial*"))
