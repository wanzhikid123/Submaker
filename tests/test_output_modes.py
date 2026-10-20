import dataclasses
import hashlib
import json
import shutil
import subprocess
import threading

import pytest

from submaker.config import Settings
from submaker.media import Cancelled, burn, estimate_output, output_plan, probe, reserve_output, temporary_subtitles, write_srt
from submaker.remux import mux_subtitles
from submaker.store import fingerprint


def fixture(root, container="mkv", attachment=True, ten=False, hdr=False):
    source = root/("source."+container)
    old = root/"old.srt"
    write_srt(old, [{"start": .1, "end": 3.7, "text": "OLD ORIGINAL SUBTITLE"}])
    chapters = root/"chapters.txt"
    chapters.write_text(";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=2000\ntitle=First\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=2000\nEND=4000\ntitle=Last\n")
    command = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=4",
               "-f", "lavfi", "-i", "sine=frequency=400:duration=4", "-itsoffset", "0.3", "-f", "lavfi", "-i", "sine=frequency=800:duration=3.5",
               "-i", str(old), "-f", "ffmetadata", "-i", str(chapters), "-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:s",
               "-map_metadata", "4", "-map_chapters", "4", "-threads", "2", "-c:v", "libx265" if ten else "libx264",
               "-pix_fmt", "yuv420p10le" if ten else "yuv420p", "-c:a", "aac", "-c:s", "mov_text" if container == "mp4" else "srt"]
    if ten:
        parameters = "pools=2:frame-threads=1"
        if hdr:
            parameters += ":master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=1000,400:hdr10=1"
            command += ["-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc"]
        command += ["-x265-params", parameters]
    if attachment:
        attached = root/"attachment.txt"
        attached.write_text("fixture attachment")
        command += ["-attach", str(attached), "-metadata:s:t:0", "mimetype=text/plain"]
    subprocess.run([*command, str(source)], check=True, capture_output=True)
    return source


def stream_hash(path, kind, index):
    return subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:{kind}:{index}",
                           "-c", "copy", "-f", "hash", "pipe:1"], capture_output=True, check=True).stdout


def test_new_defaults_and_old_task_snapshots_keep_different_output_modes(tmp_path, monkeypatch):
    assert Settings().output_mode == "soft" and Settings().output_extension() == "mkv"
    assert Settings.from_dict({"source": "ja"}).output_mode == "burn"
    assert Settings.from_dict(Settings().as_dict()).output_mode == "soft"
    monkeypatch.setattr("submaker.config.DATA", tmp_path)
    (tmp_path/"settings.json").write_text(json.dumps({"video_quality": 25, "encoding": "hardware"}))
    assert Settings.load().output_mode == "soft" and Settings.load().video_quality == 25
    assert Settings(output_container="mp4").output_extension() == "mkv"
    assert Settings(output_mode="burn", output_container="mp4").output_extension() == "mp4"


def test_soft_size_estimate_tracks_input_size_instead_of_encoder_parameters():
    info = {"duration": 3600, "source_bytes": 1024**3}
    settings = Settings()
    estimate = estimate_output(info, settings)
    assert .99 < estimate["bytes"]/info["source_bytes"] < 1.02
    assert estimate == estimate_output(info, dataclasses.replace(settings, video_quality=15, video_codec="h264", size_mode="bitrate", video_bitrate_kbps=100))


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
@pytest.mark.parametrize("container,ten,hdr", [("mkv", False, False), ("mp4", False, False), ("mkv", True, False), ("mkv", True, True)])
def test_soft_mux_copies_media_and_replaces_subtitle_without_encoding(tmp_path, monkeypatch, container, ten, hdr):
    source = fixture(tmp_path, container, attachment=container=="mkv", ten=ten, hdr=hdr)
    before = fingerprint(source)
    settings = Settings(font="Arial", font_size=18, video_codec="h264")
    info = probe(source, settings)
    assert output_plan(info, settings) == "mkv"  # Even 10-bit/HDR bypasses the burn encoder.
    from submaker import media
    popen, commands = media.subprocess.Popen, []
    def capture(command, *args, **kwargs):
        commands.append(command)
        return popen(command, *args, **kwargs)
    monkeypatch.setattr(media.subprocess, "Popen", capture)
    output = tmp_path/"translated.mkv"
    with temporary_subtitles(tmp_path, [{"start": .2, "end": 3.8, "text": "新的中文字幕"}], settings) as subtitle:
        mux_subtitles(source, output, subtitle, info, settings, threading.Event())
    result = probe(output, settings)
    assert result["video"]["codec_name"] == info["video"]["codec_name"]
    assert result["video"]["pix_fmt"] == info["video"]["pix_fmt"]
    assert len(result["chapters"]) == 2
    assert sum(s["codec_type"] == "attachment" for s in result["streams"]) == (1 if container=="mkv" else 0)
    subs = [s for s in result["streams"] if s["codec_type"] == "subtitle"]
    assert len(subs) == 1 and subs[0]["codec_name"] == "ass" and subs[0]["disposition"]["default"] == 1
    assert subs[0]["tags"]["language"] == "chi" or subs[0]["tags"]["language"] == "zho"
    export = next(c for c in commands if "-c" in c and str(output.with_suffix(".partial.mkv")) in c)
    assert export[export.index("-c")+1] == "copy"
    assert not any(flag in export for flag in ("-hwaccel", "-filter:v:0", "-c:v:0", "-preset", "-cq"))
    for kind, count in (("v", 1), ("a", 2)):
        for index in range(count):
            assert stream_hash(source, kind, index) == stream_hash(output, kind, index)
    text = subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-map", "0:s:0", "-c", "copy", "-f", "ass", "pipe:1"], capture_output=True, check=True).stdout.decode("utf-8-sig")
    assert "新的中文字幕" in text and "OLD ORIGINAL SUBTITLE" not in text and "Arial,18" in text
    assert fingerprint(source) == before
    assert not list(tmp_path.glob("*.partial*")) and not list(tmp_path.glob(".submaker-*"))


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
@pytest.mark.parametrize("codec", ["h264", "hevc"])
def test_real_cpu_burn_can_export_mp4_without_audio_loss(tmp_path, codec):
    source = fixture(tmp_path, attachment=False)
    settings = Settings(output_mode="burn", output_container="mp4", encoding="quality", decoding="software", video_codec=codec, threads=4)
    info = probe(source, settings)
    assert output_plan(info, settings) == "mp4"
    output = tmp_path/"translated.mp4"
    with temporary_subtitles(tmp_path, [{"start": .1, "end": 3.8, "text": "MP4烧录字幕"}], settings) as subtitle:
        burn(source, output, subtitle, info, settings, threading.Event(), lambda _: None)
    result = probe(output, settings)
    assert "mp4" in result["format"]["format_name"] and result["video"]["codec_name"] == codec
    assert not any(s["codec_type"] == "subtitle" for s in result["streams"])
    assert len(result["chapters"]) == 2
    for index in range(2):
        assert stream_hash(source, "a", index) == stream_hash(output, "a", index)
    assert not list(tmp_path.glob("*.partial*"))


def test_mp4_incompatible_tracks_are_rejected_before_processing():
    info = {"video": {"index": 0, "codec_name": "h264", "pix_fmt": "yuv420p"}, "streams": [{"index": 1, "codec_type": "audio", "codec_name": "vorbis"}]}
    settings = Settings(output_mode="burn", output_container="mp4")
    with pytest.raises(ValueError, match="MKV"):
        output_plan(info, settings)
    info["streams"] = [{"index": 2, "codec_type": "attachment"}]
    with pytest.raises(ValueError, match="MKV"):
        output_plan(info, settings)
    settings.output_mode = "soft"
    assert output_plan(info, settings) == "mkv"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_remux_cancellation_and_failed_validation_never_publish_partial(tmp_path, monkeypatch):
    source = fixture(tmp_path)
    settings = Settings()
    info = probe(source, settings)
    from submaker import remux
    with temporary_subtitles(tmp_path, [{"start": .1, "end": 3.8, "text": "取消测试"}]) as subtitle:
        cancel = threading.Event()
        cancel.set()
        output = tmp_path/"cancelled.mkv"
        with pytest.raises(Cancelled):
            mux_subtitles(source, output, subtitle, info, settings, cancel)
        def fail(*args):
            raise RuntimeError("injected verification failure")
        monkeypatch.setattr(remux, "validate_copy", fail)
        with pytest.raises(RuntimeError, match="injected"):
            mux_subtitles(source, output, subtitle, info, settings, threading.Event())
        assert not output.exists() and not list(tmp_path.glob("*.partial*"))


def test_file_reservation_keeps_both_container_extensions_safe(tmp_path):
    mkv, mkv_lock = reserve_output(tmp_path, "same", "mkv")
    mp4, mp4_lock = reserve_output(tmp_path, "same", "mp4")
    duplicate, _ = reserve_output(tmp_path, "same", "mp4")
    assert mkv.suffix == ".mkv" and mp4.suffix == ".mp4" and duplicate.name == "same.v002.mp4"
    assert mkv_lock != mp4_lock
