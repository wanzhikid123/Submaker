from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import tempfile
import unicodedata
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Cancelled(Exception):
    pass


def check(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled("任务已暂停")


def probe(path, settings):
    p = subprocess.run([settings.ffprobe, "-v", "error", "-show_streams", "-show_format", "-show_chapters", "-of", "json", str(path)], capture_output=True, creationflags=CREATE_NO_WINDOW, timeout=45)
    if p.returncode:
        raise RuntimeError("媒体探测失败：" + p.stderr.decode("utf-8", "replace")[-1500:])
    info = json.loads(p.stdout)
    videos = [s for s in info["streams"] if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")]
    audios = [s for s in info["streams"] if s["codec_type"] == "audio"]
    if not videos or not audios:
        raise RuntimeError("需要有效的主视频与音轨")
    aliases = {"ja": "jpn", "zh": "zho", "en": "eng", "ko": "kor", "de": "deu", "fr": "fra"}
    source = settings.source.split("-")[0]
    matches = [a for a in audios if a.get("tags", {}).get("language") in (source, aliases.get(source))]
    audio = next((a for a in audios if a["index"] == settings.audio_stream), None)
    if settings.audio_stream >= 0 and audio is None:
        raise RuntimeError("指定的音轨不存在，请修改任务音轨设置")
    audio = audio or max(matches or audios, key=lambda a: a.get("disposition", {}).get("default", 0))
    if settings.output_mode == "burn" and videos[0].get("color_transfer") in ("smpte2084", "arib-std-b67"):
        first = subprocess.run([settings.ffprobe,"-v","error","-select_streams",str(videos[0]["index"]),"-read_intervals","%+#1","-show_frames","-show_entries","frame=side_data_list","-of","json",str(path)],capture_output=True,creationflags=CREATE_NO_WINDOW,timeout=45)
        if first.returncode == 0:
            frames = json.loads(first.stdout).get("frames", [])
            if frames:
                videos[0]["side_data_list"] = videos[0].get("side_data_list", []) + frames[0].get("side_data_list", [])
    duration = float(info.get("format", {}).get("duration", 0))
    if not 0 < duration < 24 * 3600:
        raise RuntimeError("无法确定有效时长（首版支持24小时以内素材）")
    info.update(video=videos[0], audio=audio, duration=duration, source_bytes=Path(path).stat().st_size)
    return info


def legal_stem(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    value = re.sub(r"\s+", " ", value)[:140].rstrip(" .")
    if not value:
        raise ValueError("标题为空")
    if value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
        value = "_" + value
    return value


def preserve_filename_tokens(original, translated):
    # Machine-readable identifiers survive even when a model drops one.
    pattern = r"(?i)\b(?:[a-z]{2,8}-\d{2,8}|S\d{1,3}E\d{1,3}|(?:19|20)\d{2}|\d{3,4}p|[xh][ .]?26[45]|HEVC|AVC|BluRay|WEB-DL|WEBRip)\b"
    tokens = re.findall(pattern, original)
    missing = [t for t in dict.fromkeys(tokens) if t.casefold() not in translated.casefold()]
    return (translated + (" [" + " ".join(missing) + "]" if missing else "")).strip()


def srt_time(t):
    ms = max(0, round(float(t) * 1000))
    return f"{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}"


def format_cues(segments, translations, bilingual=False, target="zh-CN"):
    cues = []
    # Subtitle readability split is independent of translation units.
    limit = 22 if target.startswith(("zh", "ja", "ko")) else 44
    for s in sorted(segments, key=lambda x: (x["start"], x["end"])):
        text = translations[s["id"]].strip()
        if not text:
            raise ValueError("存在空译文，不能导出")
        text = re.sub(r"<[^>]*>", "", text).replace("{", "（").replace("}", "）")
        lines = []
        for line in text.splitlines():
            while len(line) > limit:
                cut = max([line.rfind(c, max(1, limit // 2), limit + 1) + 1 for c in " ，。！？、,.!?;"])
                cut = cut if cut > 0 else limit
                lines.append(line[:cut].strip())
                line = line[cut:].strip()
            if line:
                lines.append(line)
        groups = ["\n".join(lines[i:i + 2]) for i in range(0, len(lines), 2)] or [text]
        duration = max(0.1, s["end"] - s["start"])
        total = sum(len(g) for g in groups)
        start = s["start"]
        for i, group in enumerate(groups):
            end = s["end"] if i == len(groups) - 1 else start + duration * len(group) / max(1, total)
            if bilingual:
                group = s["text"].strip() + "\n" + group.replace("\n", " ")
            cues.append({"start": start, "end": max(start + 0.05, end), "text": group})
            start = end
    # Same-speaker boundaries never manufacture time beyond the next cue.
    for a, b in zip(cues, cues[1:]):
        if a["end"] > b["start"] and b["start"] > a["start"]:
            a["end"] = b["start"]
    return cues


def write_srt(path, cues):
    text = "\n\n".join(f"{i}\n{srt_time(c['start'])} --> {srt_time(c['end'])}\n{c['text']}" for i, c in enumerate(cues, 1)) + "\n"
    Path(path).write_text(text, encoding="utf-8-sig")


def write_ass(path, cues, font="Microsoft YaHei", font_size=24):
    header = "[Script Info]\nScriptType: v4.00+\nPlayResX: 384\nPlayResY: 288\nWrapStyle: 0\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Microsoft YaHei,24,&H00FFFFFF,&H000000FF,&H00111111,&H00000000,0,0,0,0,100,100,0,0,1,1.6,0.5,2,10,10,24,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    font = re.sub(r"[^\w \-]", "", font) or "Microsoft YaHei"
    header = header.replace("Style: Default,Microsoft YaHei,24,", f"Style: Default,{font},{int(font_size)},")
    def clock(value):
        cs = max(0, round(value*100))
        return f"{cs//360000}:{cs//6000%60:02}:{cs//100%60:02}.{cs%100:02}"
    lines = []
    for cue in cues:
        text = cue["text"].replace("\\", "＼").replace("{", "（").replace("}", "）").replace("\r", "").replace("\n", "\\N")
        lines.append(f"Dialogue: 0,{clock(cue['start'])},{clock(cue['end'])},Default,,0,0,0,,{text}\n")
    Path(path).write_text(header+"".join(lines), encoding="utf-8-sig")


@contextmanager
def temporary_subtitles(directory, cues, settings=None):
    directory = Path(directory).resolve()
    temporary = tempfile.TemporaryDirectory(prefix=".submaker-", dir=directory)
    location = Path(temporary.name).resolve()
    try:
        if location.parent != directory or not location.name.startswith(".submaker-"):
            raise RuntimeError("字幕临时目录超出输出目录")
        subtitle = location / "captions.ass"
        write_ass(subtitle, cues, settings.font, settings.font_size) if settings else write_ass(subtitle, cues)
        yield subtitle
    finally:
        if location.resolve().parent == directory and location.name.startswith(".submaker-"):
            temporary.cleanup()


_output_lock = threading.Lock()


def reserve_output(directory, title, extension="mkv"):
    if extension not in ("mkv", "mp4"):
        raise ValueError("不支持的输出容器")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = legal_stem(title)
    with _output_lock:
        for index in range(1, 10000):
            suffix = "" if index == 1 else f".v{index:03}"
            video = directory / f"{stem}{suffix}.{extension}"
            lock = video.with_name(video.name+".reserve")
            if video.exists():
                continue
            try:
                with lock.open("x", encoding="utf-8") as f:
                    f.write(str(os.getpid()))
                return video, lock
            except FileExistsError:
                continue
    raise RuntimeError("无法分配不冲突的输出文件名")


def _positive(value, fallback=0):
    try:
        result = float(value)
        return result if 0 < result < 1e12 else fallback
    except (TypeError, ValueError):
        return fallback


def bitrate_plan(info, settings):
    video = info["video"]
    others = [s for s in info.get("streams", []) if s.get("index") != video.get("index") and s.get("codec_type") in ("audio", "video")]
    copied = sum(_positive(s.get("bit_rate") or s.get("tags", {}).get("BPS"), 192000 if s.get("codec_type") == "audio" else 0) for s in others)
    total = _positive(info.get("format", {}).get("bit_rate"), _positive(info.get("source_bytes")) * 8 / info["duration"])
    source = _positive(video.get("bit_rate") or video.get("tags", {}).get("BPS"), max(150000, total-copied))
    factor = .60 if settings.video_codec == "hevc" else .85
    reference = source * factor * 2 ** ((26-settings.video_quality)/6)
    if settings.size_mode == "bitrate":
        target = max(64000, settings.video_bitrate_kbps * 1000)
        cap = round(target * 1.15)
    elif settings.size_mode == "compact":
        target = max(64000, min(source * .9, reference))
        cap = max(64000, min(round(source), round(target * 1.2)))
    elif settings.size_mode == "quality":
        target, cap = max(64000, reference), 0
    else:
        raise ValueError("未知的输出体积模式")
    return {"target": round(target), "cap": cap, "copied": copied}


def estimate_output(info, settings):
    if settings.output_mode == "soft":
        size = _positive(info.get("source_bytes"), _positive(info.get("format", {}).get("bit_rate"))*info["duration"]/8)
        subtitle_bytes = min(2*1024**2, max(2048, info["duration"]*32))
        return {"bytes": round(size*1.002+subtitle_bytes), "low": round(size*.98), "high": round(size*1.03+1024**2)}
    rates = bitrate_plan(info, settings)
    seconds = info["duration"]
    overhead = 1.02
    predicted = (rates["target"] + rates["copied"]) * seconds / 8 * overhead
    low = (rates["target"] * (.55 if not rates["cap"] else .75) + rates["copied"]) * seconds / 8 * overhead
    high = ((rates["cap"] or rates["target"]*1.7) + rates["copied"]) * seconds / 8 * overhead
    return {"bytes": round(predicted), "low": round(low), "high": round(high)}


def video_plan(info, settings):
    video = info["video"]
    hdr = video.get("color_transfer") in ("smpte2084", "arib-std-b67")
    ten = "10" in video.get("pix_fmt", "") or int(video.get("bits_per_raw_sample", "0") or 0) > 8
    if settings.video_codec not in ("h264", "hevc") or settings.encoder_preset not in {f"p{i}" for i in range(1,8)} or not 0 <= settings.video_quality <= 51:
        raise ValueError("视频编码、GPU速度预设或质量参数无效")
    if settings.video_codec == "h264" and (ten or hdr):
        raise ValueError("10-bit/HDR 素材请使用 H.265，以保留原始位深与色彩")
    if hdr and settings.encoding == "hardware":
        raise ValueError("当前GPU烧录路径不能完整保留静态HDR元数据，已停止；可显式选择CPU兼容编码，不会自动切换CPU")
    decode = settings.decoding == "hardware" and video.get("pix_fmt") in ("yuv420p", "yuv420p10le", "nv12", "p010le") and (video.get("codec_name") == "hevc" or video.get("codec_name") == "h264" and not ten)
    return {"decode": decode, "encode": settings.encoding == "hardware", "hdr": hdr, "ten": ten, "gpu": settings.video_gpu if settings.video_gpu >= 0 else settings.gpus[0]}


def output_plan(info, settings):
    extension = settings.output_extension()
    if settings.output_mode == "burn":
        video_plan(info, settings)
        if extension == "mp4":
            if any(s["codec_type"] == "attachment" for s in info["streams"]):
                raise ValueError("MP4 无法保留原片附件，请选择 MKV 烧录，以保留原有字体等附件")
            supported_audio = {"aac", "mp3", "ac3", "eac3", "alac", "flac", "opus"}
            unsupported = [s.get("codec_name", "unknown") for s in info["streams"] if s["codec_type"] == "audio" and s.get("codec_name") not in supported_audio]
            extra_video = [s.get("codec_name", "unknown") for s in info["streams"] if s["codec_type"] == "video" and s["index"] != info["video"]["index"] and s.get("codec_name") not in {"h264", "hevc", "av1", "mpeg4", "mjpeg", "png"}]
            if unsupported or extra_video:
                raise ValueError("MP4 无法直接保留这些原片轨道："+", ".join(unsupported+extra_video)+"；请选择 MKV，避免额外转码")
    return extension


def partial_output(output):
    return output.with_name(output.stem+".partial"+output.suffix)


def container_options(settings):
    return ["-tag:v:0", "hvc1"] if settings.output_extension() == "mp4" and settings.video_codec == "hevc" else []


def video_thread_budget(settings, plan):
    if not plan["encode"]:
        return {"decode": settings.video_decode_threads or 1, "filter": settings.video_filter_threads or 1}
    # Reserve CPU work for Whisper/audio and the optional local LLM, then divide
    # the remaining budget across the configured global number of encoders.
    reserved = 2*len(settings.gpus)+1+(settings.local_model.threads if settings.llm_mode == "local" else 0)
    available = max(2, settings.threads-reserved)
    per_video = max(2, available//max(1, min(6, settings.video_concurrency)))
    decode = min(4, max(1, per_video//2))
    filters = min(4, max(1, per_video-decode))
    return {"decode": settings.video_decode_threads or decode, "filter": settings.video_filter_threads or filters}


def burn_command(source, output, subtitle, info, settings, segment=None):
    partial = partial_output(output)
    # CWD + a local relative path avoids Windows drive/colon filter escaping.
    relative = Path(os.path.relpath(subtitle, output.parent)).as_posix().replace("'", "'\\''").replace(":", "\\:").replace(",", "\\,").replace("[", "\\[").replace("]", "\\]")
    font = re.sub(r"[^\w \-]", "", settings.font)
    vf = f"subtitles=filename='{relative}':force_style='FontName={font},FontSize={settings.font_size},PrimaryColour=&H00FFFFFF,OutlineColour=&H00111111,BorderStyle=1,Outline=1.6,Shadow=0.5,MarginV=24'"
    v = info["video"]
    plan = video_plan(info, settings)
    thread_budget = video_thread_budget(settings, plan)
    hdr, ten, gpu = plan["hdr"], plan["ten"], plan["gpu"]
    if hdr and any(x.get("side_data_type") in ("DOVI configuration record", "HDR Dynamic Metadata SMPTE2094-40 (HDR10+)") for x in v.get("side_data_list", [])):
        raise RuntimeError("该视频含动态HDR元数据，当前烧录路径不能完整保留；请先提供可保留色彩的静态HDR版本")
    if plan["decode"]:
        vf = "hwdownload,format=" + ("p010le" if ten else "nv12") + "," + vf
    if segment is not None:
        start, end = segment
        seek = max(0, int(start)-1)
        rate = Fraction(v.get("r_frame_rate") or v.get("avg_frame_rate") or "25")
        if rate <= 0:
            raise ValueError("无法确定分段编码帧率")
        # Seek before the boundary, restore the global subtitle clock, then
        # select a half-open interval. This also retains variable frame gaps.
        vf = f"setpts=PTS+round({seek}/TB),trim=start={start:.9f}:end={end:.9f},"+vf
        # Caption rendering uses global time. A temporary frame-number clock
        # then gives NVENC the same nominal FPS and rate control as single mode.
        # The streaming remuxer restores each original PTS before final muxing.
        vf += f",setpts=N/({rate.numerator}/{rate.denominator}*TB)"
    crf = str(settings.video_quality)
    software_preset = {"quality":"medium","balanced":"fast","fast":"veryfast"}.get(settings.preset,"medium")
    rates = bitrate_plan(info, settings)
    if plan["encode"]:
        codec = ["-c:v:0", "hevc_nvenc" if settings.video_codec == "hevc" else "h264_nvenc", "-gpu", str(gpu), "-preset", settings.encoder_preset, "-tune", "hq", "-rc", "vbr", "-cq", crf, "-b:v", str(rates["target"] if rates["cap"] else 0), "-spatial-aq", "1", "-rc-lookahead", "32"]
        if rates["cap"]:
            codec += ["-maxrate", str(rates["cap"]), "-bufsize", str(rates["cap"]*2)]
        if settings.encoder_preset in ("p5", "p6", "p7"):
            codec += ["-multipass", "fullres" if settings.encoder_preset == "p7" else "qres"]
    else:
        codec = ["-c:v:0", "libx265" if settings.video_codec == "hevc" else "libx264", "-preset", software_preset, "-crf", crf]
    args = [settings.ffmpeg, "-hide_banner", "-nostdin", "-y", "-threads", str(thread_budget["decode"])]
    if plan["decode"]:
        args += ["-hwaccel", "cuda", "-hwaccel_device", str(gpu), "-hwaccel_output_format", "cuda"]
    if segment is not None:
        args += ["-ss", str(seek), "-t", str(end-seek+2)]
    args += ["-i", str(source), "-map", f"0:{v['index']}"]
    for stream in info["streams"] if segment is None else []:
        if stream["codec_type"] == "video" and stream["index"] != v["index"]:
            args += ["-map", f"0:{stream['index']}"]
    # Burned subtitles are the sole subtitle presentation in the new video.
    # Retaining selectable tracks lets language-preference players add a second set.
    if segment is None:
        args += ["-map", "0:a?", "-map", "0:t?", "-map_metadata", "0", "-map_chapters", "0"]
    args += ["-c", "copy", *codec, "-threads:v:0", str(max(1, settings.threads - len(settings.gpus) * 2 - 3)), "-filter_threads", str(thread_budget["filter"]), "-filter:v:0", vf, "-pix_fmt:v:0", "yuv420p10le" if ten or hdr else "yuv420p", "-fps_mode:v:0", "passthrough"]
    if settings.video_codec == "hevc" or ten or hdr:
        if not plan["encode"]:
            parameters = f"pools={max(1,settings.threads-len(settings.gpus)*2-3)}:frame-threads=1" + (":hdr10-opt=1:repeat-headers=1" if hdr else "")
            for side in v.get("side_data_list", []):
                if side.get("side_data_type") == "Mastering display metadata":
                    def scaled(key, factor):
                        return round(Fraction(str(side[key])) * factor)
                    try:
                        display = "".join(f"{name}({scaled(color+'_x',50000)},{scaled(color+'_y',50000)})" for name,color in (("G","green"),("B","blue"),("R","red")))
                        display += f"WP({scaled('white_point_x',50000)},{scaled('white_point_y',50000)})L({scaled('max_luminance',10000)},{scaled('min_luminance',10000)})"
                        parameters += ":master-display=" + display
                    except (KeyError, ValueError):
                        raise RuntimeError("HDR 母版元数据不完整，拒绝静默丢弃")
                elif side.get("side_data_type") == "Content light level metadata":
                    parameters += f":max-cll={int(side['max_content'])},{int(side['max_average'])}"
            args += ["-x265-params", parameters]
        for tag, flag in (("color_primaries", "-color_primaries:v:0"), ("color_transfer", "-color_trc:v:0"), ("color_space", "-colorspace:v:0")):
            if v.get(tag) not in (None, "unknown", "unspecified"):
                args += [flag, v[tag]]
    if segment is not None:
        args += ["-enc_time_base:v:0", f"{rate.denominator}/{rate.numerator}", "-mpegts_copyts", "1", "-mpegts_flags", "+resend_headers+initial_discontinuity",
                 "-muxdelay", "0", "-muxpreload", "0", "-avoid_negative_ts", "disabled", "-progress", "pipe:2", "-nostats", "-f", "mpegts", "pipe:1"]
    else:
        args += [*container_options(settings), "-max_muxing_queue_size", "4096", "-progress", "pipe:1", "-nostats", str(partial)]
    return args


def validate_burn(partial, info, settings):
    final_info = probe(partial, settings)
    v = info["video"]
    hdr = video_plan(info, settings)["hdr"]
    rotation = next((abs(int(float(s.get("rotation",0)))) % 180 for s in v.get("side_data_list",[]) if s.get("side_data_type")=="Display Matrix"),0)
    expected_width, expected_height = (v["height"],v["width"]) if rotation == 90 else (v["width"],v["height"])
    if abs(final_info["duration"] - info["duration"]) > 1.0 or final_info["video"]["width"] != expected_width or final_info["video"]["height"] != expected_height:
        raise RuntimeError("成品时长或尺寸校验失败，未提交输出")
    old_audio = [a["codec_name"] for a in info["streams"] if a["codec_type"] == "audio"]
    new_audio = [a["codec_name"] for a in final_info["streams"] if a["codec_type"] == "audio"]
    if old_audio != new_audio:
        raise RuntimeError("音轨保留校验失败")
    if any(s["codec_type"] == "subtitle" for s in final_info["streams"]):
        raise RuntimeError("成品不应包含会造成重复显示的软字幕轨")
    if sum(s["codec_type"] == "attachment" for s in info["streams"]) != sum(s["codec_type"] == "attachment" for s in final_info["streams"]):
        raise RuntimeError("原有字幕或附件数量校验失败")
    if len(info.get("chapters", [])) != len(final_info.get("chapters", [])):
        raise RuntimeError("章节保留校验失败")
    if hdr and final_info["video"].get("color_transfer") != v.get("color_transfer"):
        raise RuntimeError("HDR 色彩标志校验失败")
    for side in v.get("side_data_list", []):
        if side.get("side_data_type") in ("Mastering display metadata", "Content light level metadata") and not any(s.get("side_data_type") == side["side_data_type"] for s in final_info["video"].get("side_data_list", [])):
            raise RuntimeError("HDR 静态元数据保留校验失败")


def burn(source, output, subtitle, info, settings, cancel, on_progress):
    args = burn_command(source, output, subtitle, info, settings)
    run_export(args, output, info, cancel, on_progress, lambda partial: validate_burn(partial, info, settings), "视频烧录")


def run_export(args, output, info, cancel, on_progress, validate, label):
    check(cancel)
    partial = partial_output(output)
    # FFmpeg may emit a lot of errors: keep a bounded tail in RAM, never an intermediate video copy.
    p = subprocess.Popen(args, cwd=output.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=CREATE_NO_WINDOW, text=True, encoding="utf-8", errors="replace")
    errors = []
    abort_reason = []
    def read_errors():
        for line in p.stderr:
            errors.append(line)
            if len(errors) > 80:
                del errors[:20]
    threading.Thread(target=read_errors, daemon=True).start()
    def guard():
        while p.poll() is None:
            if cancel.wait(0.25):
                p.terminate()
                return
            if shutil.disk_usage(output.parent).free < 512 * 1024**2:
                abort_reason.append("输出磁盘剩余空间低于512MiB，已停止编码并保留文字恢复点")
                p.terminate()
                return
    threading.Thread(target=guard, daemon=True).start()
    try:
        for line in p.stdout:
            check(cancel)
            if line.startswith("out_time_us="):
                try:
                    on_progress(float(line.split("=", 1)[1]) / 1e6 / info["duration"])
                except ValueError:
                    pass
        p.wait()
        check(cancel)
        if p.returncode:
            if abort_reason:
                raise RuntimeError(abort_reason[0])
            raise RuntimeError(label+"失败：" + "".join(errors)[-3000:])
        validate(partial)
        check(cancel)
        if output.exists():
            raise RuntimeError("输出路径被其他程序占用，拒绝覆盖")
        os.rename(partial, output)
        on_progress(1.0)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
        if partial.exists():
            partial.unlink()
