import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from submaker.config import Settings
from submaker.media import burn, probe, temporary_subtitles, write_srt
from submaker.asr import decode_window

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")


def test_audio_and_chapters_survive_burn_without_duplicate_soft_subtitles(tmp_path):
    import numpy as np
    source=tmp_path/"source.mp4"
    old_sub=tmp_path/"old.srt"
    write_srt(old_sub,[{"start":0,"end":3,"text":"old subtitle"}])
    metadata=tmp_path/"chapters.txt"
    metadata.write_text(";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=2000\ntitle=Part 1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=2000\nEND=4000\ntitle=Part 2\n")
    command=["ffmpeg","-v","error","-f","lavfi","-i","color=c=navy:s=320x240:r=25:d=4","-f","lavfi","-i","sine=frequency=400:duration=4","-itsoffset","0.4","-f","lavfi","-i","sine=frequency=800:duration=4","-i",str(old_sub),"-f","ffmetadata","-i",str(metadata),"-map","0:v","-map","1:a","-map","2:a","-map","3:s","-map_metadata","4","-map_chapters","4","-c:v","libx264","-c:a","aac","-c:s","mov_text",str(source)]
    subprocess.run(command,check=True,capture_output=True)
    settings=Settings(gpus=[0],audio_stream=2,encoding="quality",decoding="software")
    info=probe(source,settings)
    task={"job":"test","ffmpeg":settings.ffmpeg,"source":str(source),"audio_index":2,"decode_start":0,"decode_end":1}
    samples=decode_window(task,threading.Semaphore(1),{})
    assert np.max(np.abs(samples[:int(.25*16000)])) < 0.001
    assert np.max(np.abs(samples[int(.6*16000):])) > 0.01
    target=tmp_path/"target.srt"
    write_srt(target,[{"start":0.5,"end":3.5,"text":"测试字幕"}])
    output=tmp_path/"result.mkv"
    burn(source,output,target,info,settings,threading.Event(),lambda _:None)
    result=probe(output,settings)
    assert len([s for s in result["streams"] if s["codec_type"]=="audio"])==2
    assert [s["codec_name"] for s in result["streams"] if s["codec_type"]=="subtitle"]==[]
    assert len(result["chapters"])==2
    old_audio=next(s for s in info["streams"] if s["index"]==2)
    new_audio=[s for s in result["streams"] if s["codec_type"]=="audio"][1]
    assert abs(float(old_audio["start_time"])-float(new_audio["start_time"]))<0.15


def test_static_hdr10_and_mastering_metadata_are_preserved(tmp_path):
    source=tmp_path/"hdr.mkv"
    parameters="pools=2:frame-threads=1:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=1000,400:hdr10=1"
    subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i","color=c=navy:s=320x240:r=25:d=2","-f","lavfi","-i","sine=frequency=400:duration=2","-c:v","libx265","-pix_fmt","yuv420p10le","-color_primaries","bt2020","-color_trc","smpte2084","-colorspace","bt2020nc","-x265-params",parameters,"-c:a","aac",str(source)],check=True,capture_output=True)
    settings=Settings(gpus=[0],encoding="quality",decoding="hardware" if shutil.which("nvidia-smi") else "software")
    info=probe(source,settings)
    assert any(s.get("side_data_type")=="Mastering display metadata" for s in info["video"]["side_data_list"])
    target=tmp_path/"target.srt"
    write_srt(target,[{"start":0.1,"end":1.9,"text":"HDR 字幕测试"}])
    output=tmp_path/"result.mkv"
    burn(source,output,target,info,settings,threading.Event(),lambda _:None)
    result=probe(output,settings)
    assert result["video"]["pix_fmt"]=="yuv420p10le"
    assert result["video"]["color_transfer"]=="smpte2084"
    assert any(s.get("side_data_type")=="Content light level metadata" for s in result["video"]["side_data_list"])


@pytest.mark.parametrize("gpu,codec,preset", [(0,"h264","p3"),(1,"h264","p7"),(0,"hevc","p7"),(1,"hevc","p3")])
def test_real_gpu_decode_subtitle_render_and_nvenc_output(tmp_path, monkeypatch, gpu, codec, preset):
    import numpy as np
    if not shutil.which("nvidia-smi"):
        pytest.skip("NVIDIA GPU not available")
    source = tmp_path / "gpu-source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=navy:s=640x360:r=25:d=2", "-f", "lavfi", "-i", "sine=frequency=400:duration=2", "-c:v", "libx264", "-c:a", "aac", str(source)], check=True, capture_output=True)
    settings = Settings(gpus=[gpu], video_gpu=gpu, video_codec=codec, encoder_preset=preset)
    info = probe(source, settings)
    output = tmp_path / "result.mkv"
    from submaker import media
    original_popen, commands = media.subprocess.Popen, []
    def capture(command, *args, **kwargs):
        commands.append(command)
        return original_popen(command, *args, **kwargs)
    monkeypatch.setattr(media.subprocess, "Popen", capture)
    with temporary_subtitles(tmp_path, [{"start": .1, "end": 1.9, "text": "V100 字幕测试"}]) as subtitle:
        burn(source, output, subtitle, info, settings, threading.Event(), lambda _: None)
    assert not list(tmp_path.rglob("*.ass")) and not list(tmp_path.rglob("*.srt"))
    command = next(c for c in commands if "-hwaccel" in c)
    assert command[command.index("-hwaccel")+1] == "cuda"
    assert command[command.index("-gpu")+1] == str(gpu)
    assert command[command.index("-c:v:0")+1] == codec + "_nvenc"
    assert command[command.index("-preset")+1] == preset
    assert "-maxrate" in command
    assert probe(output, settings)["video"]["codec_name"] == codec
    pixels = subprocess.run(["ffmpeg", "-v", "error", "-ss", "1", "-i", str(output), "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"], check=True, capture_output=True).stdout
    rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(360, 640, 3)
    assert np.count_nonzero(np.all(rgb > 210, axis=2)) > 50, "Burned subtitle did not produce visible white glyphs"
