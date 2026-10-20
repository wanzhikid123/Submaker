"""Copy original media into Matroska with one default ASS subtitle track."""
from array import array
from collections import deque
import math
import subprocess
import threading

from .config import LANGUAGES
from .media import CREATE_NO_WINDOW, check, partial_output, probe, run_export

LANGUAGE_CODES = {"ja":"jpn", "zh":"zho", "en":"eng", "ko":"kor", "de":"deu", "fr":"fra",
                  "es":"spa", "ru":"rus", "it":"ita", "pt":"por", "ar":"ara", "hi":"hin", "th":"tha", "vi":"vie"}


def preserved_streams(info):
    primary = info["video"]
    return [primary]+[s for kind in ("video", "audio", "attachment") for s in info["streams"]
                       if s["codec_type"] == kind and s["index"] != primary["index"]]


def mux_command(source, output, subtitle, info, settings):
    offset = float(info.get("format", {}).get("start_time") or 0)
    args = [settings.ffmpeg, "-hide_banner", "-nostdin", "-y", "-copyts", "-itsoffset", str(-offset),
            "-i", str(source), "-i", str(subtitle)]
    for index, stream in enumerate(preserved_streams(info)):
        args += ["-map", f"0:{stream['index']}", f"-map_metadata:s:{index}", f"0:s:{stream['index']}"]
    label = next((name for name, code in LANGUAGES.items() if code == settings.target), settings.target)
    language = LANGUAGE_CODES.get(settings.target.lower().split("-")[0], "und")
    args += ["-map", "1:0", "-map_metadata", "0", "-map_chapters", "0", "-c", "copy",
             "-disposition:s:0", "default", "-metadata:s:s:0", "language="+language,
             "-metadata:s:s:0", "title="+label, "-avoid_negative_ts", "disabled",
             "-max_muxing_queue_size", "4096", "-progress", "pipe:1", "-nostats", str(partial_output(output))]
    return args


def packet_times(path, indices, settings, cancel, offset=0):
    """One packet-only scan per file; small numeric arrays, never decoded media."""
    values = {index: array("d") for index in indices}
    args = [settings.ffprobe, "-v", "error", "-show_packets", "-show_entries", "packet=stream_index,pts_time",
            "-of", "compact=p=0:nk=0", str(path)]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=CREATE_NO_WINDOW,
                               text=True, encoding="utf-8", errors="replace")
    done, errors = threading.Event(), deque(maxlen=30)
    def read_errors():
        for line in process.stderr:
            errors.append(line)
    def guard():
        while not done.wait(.1):
            if cancel.is_set() and process.poll() is None:
                process.kill()
                return
    reader, watcher = threading.Thread(target=read_errors, daemon=True), threading.Thread(target=guard, daemon=True)
    reader.start()
    watcher.start()
    count = 0
    try:
        for line in process.stdout:
            check(cancel)
            fields = dict(part.split("=", 1) for part in line.strip().split("|") if "=" in part)
            index = int(fields.get("stream_index", -1))
            if index not in values:
                continue
            try:
                pts = float(fields.get("pts_time", "nan"))-offset
            except ValueError:
                pts = math.nan
            if not math.isfinite(pts):
                raise RuntimeError("源片或成品存在缺失时间戳，无法确认软字幕封装的音画同步")
            values[index].append(pts)
            count += 1
            if count > 16_000_000:
                raise RuntimeError("音视频时间轴超过内存验证上限，未提交成品")
        process.wait()
        reader.join(timeout=2)
        check(cancel)
        if process.returncode or errors:
            raise RuntimeError("音视频时间轴读取失败："+"".join(errors)[-1500:])
        return values
    finally:
        done.set()
        if process.poll() is None:
            process.kill()
        process.wait()
        reader.join(timeout=2)
        watcher.join(timeout=2)
        process.stdout.close()
        process.stderr.close()


def validate_copy(source, partial, info, settings, cancel):
    result = probe(partial, settings)
    before = preserved_streams(info)
    after = [s for s in result["streams"] if s["codec_type"] in ("video", "audio", "attachment")]
    if len(before) != len(after):
        raise RuntimeError("软字幕封装的原有轨道数量不一致，未提交成品")
    for old, new in zip(before, after):
        for key in ("codec_type", "codec_name", "width", "height", "pix_fmt", "sample_rate", "channels"):
            if old.get(key) != new.get(key):
                raise RuntimeError("软字幕封装改变了原有音视频参数，未提交成品")
    subtitles = [s for s in result["streams"] if s["codec_type"] == "subtitle"]
    if len(subtitles) != 1 or subtitles[0]["codec_name"] != "ass" or not subtitles[0].get("disposition", {}).get("default"):
        raise RuntimeError("成品应包含一条默认显示的 ASS 字幕轨")
    if abs(info["duration"]-result["duration"]) > 1 or len(info.get("chapters", [])) != len(result.get("chapters", [])):
        raise RuntimeError("软字幕封装的时长或章节校验失败")
    pairs = [(old, new) for old, new in zip(before, after) if old["codec_type"] in ("video", "audio") and not old.get("disposition", {}).get("attached_pic")]
    original = packet_times(source, [a["index"] for a, _ in pairs], settings, cancel, float(info.get("format", {}).get("start_time") or 0))
    finished = packet_times(partial, [b["index"] for _, b in pairs], settings, cancel)
    for old, new in pairs:
        a, b = original[old["index"]], finished[new["index"]]
        if not a or len(a) != len(b) or any(abs(x-y) > .002 for x, y in zip(a, b)):
            raise RuntimeError("软字幕封装的音视频包数或时间戳不一致，未提交成品")


def mux_subtitles(source, output, subtitle, info, settings, cancel, on_progress=lambda value: None):
    command = mux_command(source, output, subtitle, info, settings)
    run_export(command, output, info, cancel, on_progress,
               lambda partial: validate_copy(source, partial, info, settings, cancel), "软字幕封装")
