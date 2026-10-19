import dataclasses
import hashlib
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from submaker.config import Settings
from submaker import media


@pytest.mark.parametrize("gpu,codec,ten", [(0, "hevc", False), (1, "h264", False), (1, "hevc", True)])
def test_parallel_decode_preserves_every_decoded_frame_and_video_parameters(tmp_path, monkeypatch, gpu, codec, ten):
    if not shutil.which("nvidia-smi"):
        pytest.skip("NVIDIA GPU not available")
    source = tmp_path/"motion.mkv"
    source_command = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=2.4", "-f", "lavfi", "-i", "sine=frequency=440:duration=2.4", "-threads", "2"]
    if ten:
        source_command += ["-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "pools=2:frame-threads=1"]
    else:
        source_command += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if codec == "h264":
        source_command += ["-vf", "setpts='PTS+if(gte(N,30),0.5/TB,0)'", "-fps_mode", "passthrough"]
    subprocess.run([*source_command, "-c:a", "aac", str(source)], check=True, capture_output=True)
    settings = Settings(gpus=[0, 1], threads=16, video_gpu=gpu, video_codec=codec, encoder_preset="p7", video_quality=24, video_concurrency=1)
    info = media.probe(source, settings)
    baseline, improved = tmp_path/"baseline.mkv", tmp_path/"improved.mkv"
    default_budget = media.video_thread_budget
    commands = []
    commits = []
    rename = media.os.rename
    def commit(source_path, destination_path):
        before = Path(source_path).stat()
        rename(source_path, destination_path)
        after = Path(destination_path).stat()
        # Publishing the encoded file must rename it, not copy/rewrite the video.
        assert (before.st_ino, before.st_size) == (after.st_ino, after.st_size)
        commits.append((Path(source_path), Path(destination_path)))
    monkeypatch.setattr(media.os, "rename", commit)
    popen = media.subprocess.Popen
    def capture(command, *args, **kwargs):
        commands.append(command)
        return popen(command, *args, **kwargs)
    monkeypatch.setattr(media.subprocess, "Popen", capture)
    with media.temporary_subtitles(tmp_path, [{"start": .1, "end": 1.1, "text": "并行解码测试"}, {"start": 1.1, "end": 2.3, "text": "字幕时间与画面不变"}]) as subtitle:
        monkeypatch.setattr(media, "video_thread_budget", lambda *args: {"decode": 1, "filter": 1})
        media.burn(source, baseline, subtitle, info, settings, threading.Event(), lambda _: None)
        monkeypatch.setattr(media, "video_thread_budget", default_budget)
        media.burn(source, improved, subtitle, info, settings, threading.Event(), lambda _: None)
    hashes = []
    frame_counts = []
    for path in (baseline, improved):
        result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-pix_fmt", "yuv420p10le" if ten else "yuv420p", "-f", "framemd5", "pipe:1"], check=True, capture_output=True)
        frames = [line for line in result.stdout.splitlines() if line and not line.startswith(b"#")]
        hashes.append(hashlib.sha256(b"\n".join(frames)).digest())
        frame_counts.append(len(frames))
        result_info = media.probe(path, settings)
        assert result_info["video"]["codec_name"] == codec
        assert (result_info["video"]["width"], result_info["video"]["height"]) == (640, 360)
        assert not any(s["codec_type"] == "subtitle" for s in result_info["streams"])
        if ten:
            assert result_info["video"]["pix_fmt"] == "yuv420p10le"
    assert frame_counts == [60, 60]
    assert hashes[0] == hashes[1], "Parallelism changed pixels or frame timing"
    encodes = [c for c in commands if "-filter:v:0" in c]
    assert len(encodes) == 2
    before, after = encodes
    assert before[before.index("-threads")+1] == "1"
    assert after[after.index("-threads")+1] == "4"
    assert after[after.index("-filter_threads")+1] == "4"
    for name in ("-c:v:0", "-gpu", "-preset", "-cq", "-rc-lookahead", "-multipass", "-maxrate", "-pix_fmt:v:0"):
        assert before[before.index(name)+1] == after[after.index(name)+1]
    assert not list(tmp_path.glob(".submaker-*"))
    assert commits == [(p.with_suffix(".partial.mkv"), p) for p in (baseline, improved)]
    assert {p.name for p in tmp_path.iterdir()} == {source.name, baseline.name, improved.name}
