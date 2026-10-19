"""Local codec quality/size study. Video samples stay in RAM; no LLM calls.

The report contains settings and aggregate metrics, never source paths or titles.
Small ASS/VMAF text files are deleted after each worker finishes.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
import subprocess
import tempfile
import time

from submaker.config import DATA, Settings
from submaker.media import CREATE_NO_WINDOW, bitrate_plan, probe, write_ass


def variants(suite):
    if suite == "speed":
        return [dict(name=f"{p}_compact_24_repeat_{i}", preset=p, mode="compact", cq=24)
                for i, p in enumerate(("p7", "p6", "p6", "p7", "p7", "p6"))]
    rows = [dict(name=f"p7_{mode}_{cq}", preset="p7", mode=mode, cq=cq)
            for mode in ("compact", "quality") for cq in (22, 24, 26, 28)]
    rows += [dict(name=f"p6_{mode}_24", preset="p6", mode=mode, cq=24) for mode in ("compact", "quality")]
    rows += [dict(name=f"p7_quality_{cq}", preset="p7", mode="quality", cq=cq) for cq in (30, 32)]
    rows += [dict(name="h264_p7_compact_24", preset="p7", mode="compact", cq=24, codec="h264")]
    if suite == "smoke":
        return [dict(name="lossless_control", lossless=True), rows[1]]
    if suite == "confirm":
        return [row for row in rows if row["name"] in ("p7_compact_24", "p7_quality_32", "p7_compact_26", "p6_compact_24")] + [
            dict(name="p7_compact_25", preset="p7", mode="compact", cq=25),
            dict(name="p7_quality_31", preset="p7", mode="quality", cq=31)]
    if suite == "extra":
        return [dict(name="p7_quality_32", preset="p7", mode="quality", cq=32),
                dict(name="p7_quality_32_main10", preset="p7", mode="quality", cq=32, ten=True),
                dict(name="p7_quality_30_main10", preset="p7", mode="quality", cq=30, ten=True),
                dict(name="p7_compact_25_main10", preset="p7", mode="compact", cq=25, ten=True),
                dict(name="p7_quality_32_temporal_aq", preset="p7", mode="quality", cq=32, temporal=True),
                dict(name="p7_quality_32_weighted", preset="p7", mode="quality", cq=32, weighted=True)]
    return rows


def active_jobs():
    with sqlite3.connect((DATA/"jobs.sqlite").resolve().as_uri()+"?mode=ro", uri=True) as con:
        return con.execute("SELECT count(*) FROM jobs WHERE state IN ('处理中','排队中')").fetchone()[0]


def invoke(settings, command, location, source, payload=None, timeout=180):
    result = subprocess.run(command, input=payload, capture_output=True,
                            cwd=location, creationflags=CREATE_NO_WINDOW, timeout=timeout)
    if result.returncode:
        error = result.stderr.decode("utf-8", "replace")
        for private in (str(source), source.name, str(location)):
            error = error.replace(private, "[local file]")
        raise RuntimeError(error[-1400:])
    return result


def percentile(values, p):
    values = sorted(values)
    position = (len(values)-1)*p
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def study(settings, source, info, offset, seconds, gpu, choices, measure=True):
    rows = []
    with tempfile.TemporaryDirectory(prefix="quality-study-", dir=DATA) as tmp:
        location = Path(tmp).resolve()
        assert location.parent == DATA.resolve()
        write_ass(location/"captions.ass", [{"start": 0, "end": seconds, "text": "Subtitle quality comparison"}])
        font = re.sub(r"[^\w \-]", "", settings.font)
        overlay = f"subtitles=filename='captions.ass':force_style='FontName={font},FontSize={settings.font_size},PrimaryColour=&H00FFFFFF,OutlineColour=&H00111111,BorderStyle=1,Outline=1.6,Shadow=0.5,MarginV=24'"
        # Use an absolute end: ffprobe may seek to an earlier keyframe, and a
        # relative duration can otherwise stop before our comparison window ends.
        packets = invoke(settings, [settings.ffprobe, "-v", "error", "-read_intervals", f"{offset}%{offset+seconds+1}",
            "-select_streams", str(info["video"]["index"]), "-show_entries", "packet=pts_time,size", "-of", "json", str(source)], location, source)
        source_bytes = sum(int(p["size"]) for p in json.loads(packets.stdout)["packets"]
                           if offset <= float(p.get("pts_time", -1)) < offset+seconds)
        fps = info["video"]["avg_frame_rate"]
        clock = f"settb=AVTB,setpts=N/({fps}*TB)"
        for choice in choices:
            if active_jobs():
                raise RuntimeError("A production task started; quality study stopped before the next encode")
            current = dataclasses.replace(settings, video_codec=choice.get("codec", "hevc"),
                encoder_preset=choice.get("preset", "p7"), video_quality=choice.get("cq", 24),
                size_mode=choice.get("mode", "quality"), video_gpu=gpu)
            rates = bitrate_plan(info, current)
            command = [settings.ffmpeg, "-hide_banner", "-nostdin", "-threads", "4", "-hwaccel", "cuda",
                       "-hwaccel_device", str(gpu), "-hwaccel_output_format", "cuda", "-ss", str(offset),
                       "-i", str(source), "-t", str(seconds), "-map", f"0:{info['video']['index']}", "-an", "-sn", "-dn",
                       # Preserve the production timestamps through encoding. On
                       # this NVENC path, rebasing with setpts changes rate control.
                       "-filter_threads", "4", "-vf", "hwdownload,format=nv12,"+overlay]
            if choice.get("lossless"):
                command += ["-c:v", "ffv1", "-level", "3", "-threads:v", "4", "-pix_fmt", "yuv420p"]
            else:
                command += ["-c:v", "hevc_nvenc" if current.video_codec == "hevc" else "h264_nvenc", "-gpu", str(gpu),
                    "-preset", current.encoder_preset, "-tune", "hq", "-rc", "vbr", "-cq", str(current.video_quality),
                    "-b:v", str(rates["target"] if rates["cap"] else 0), "-spatial-aq", "0" if choice.get("temporal") else "1",
                    "-rc-lookahead", "32", "-multipass", "fullres" if current.encoder_preset == "p7" else "qres",
                    "-pix_fmt", "p010le" if choice.get("ten") else "yuv420p"]
                if choice.get("temporal"):
                    command += ["-temporal-aq", "1"]
                if choice.get("weighted"):
                    command += ["-weighted_pred", "1"]
                if rates["cap"]:
                    command += ["-maxrate", str(rates["cap"]), "-bufsize", str(rates["cap"]*2)]
            command += ["-fps_mode", "passthrough", "-f", "matroska", "pipe:1"]
            started = time.perf_counter()
            try:
                result = invoke(settings, command, location, source)
                elapsed = time.perf_counter()-started
                payload = result.stdout
                if not measure:
                    row = dict(choice, offset_seconds=offset, duration_seconds=seconds, gpu=gpu,
                               encoded_bytes=len(payload), encode_seconds=round(elapsed, 3), speed=round(seconds/elapsed, 3))
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=True), flush=True)
                    del payload, result
                    continue
                metric_filter = f"[0:v]format=yuv420p,{clock},split=3[dv][ds][dp];[1:v]{overlay},format=yuv420p,{clock},split=3[rv][rs][rp];[dv][rv]libvmaf=log_fmt=json:log_path=vmaf.json:n_threads=4:shortest=1[vm];[ds][rs]ssim=shortest=1[ss];[dp][rp]psnr=shortest=1[ps]"
                measured = invoke(settings, [settings.ffmpeg, "-hide_banner", "-nostdin", "-threads", "4", "-i", "pipe:0",
                    "-threads", "4", "-ss", str(offset), "-i", str(source), "-t", str(seconds),
                    "-filter_complex_threads", "4", "-filter_complex", metric_filter, "-map", "[vm]", "-map", "[ss]", "-map", "[ps]",
                    "-an", "-sn", "-f", "null", "NUL"], location, source, payload)
                metrics = json.loads((location/"vmaf.json").read_text(encoding="utf-8"))
                frames = [f["metrics"]["vmaf"] for f in metrics["frames"]]
                log = measured.stderr.decode("utf-8", "replace")
                ssim = re.search(r"SSIM .*All:([0-9.]+)", log)
                psnr = re.search(r"PSNR .*average:([0-9.inf]+)", log)
                assert ssim and psnr and frames
                expected_frames = seconds*float(Fraction(fps))
                assert abs(len(frames)-expected_frames) <= 2, "Frame alignment/count mismatch"
                row = dict(choice, offset_seconds=offset, duration_seconds=seconds, gpu=gpu,
                    encoded_bytes=len(payload), source_video_packet_bytes=source_bytes,
                    size_ratio_video=round(len(payload)/source_bytes, 5), video_kbps=round(len(payload)*8/seconds/1000, 2),
                    encode_seconds=round(elapsed, 3), speed=round(seconds/elapsed, 3),
                    vmaf=round(statistics.mean(frames), 4), vmaf_p5=round(percentile(frames, .05), 4),
                    ssim=float(ssim[1]), psnr=psnr[1], frames=len(frames),
                    target_bps=rates["target"] if rates["cap"] else 0, max_bps=rates["cap"])
                if choice.get("lossless"):
                    assert row["ssim"] > .99999 and row["vmaf"] > 99.9, "Lossless metric control failed"
                del payload, result
            except Exception as exc:
                row = dict(choice, offset_seconds=offset, gpu=gpu, failed=True, error=str(exc))
                if choice.get("lossless"):
                    raise
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("smoke", "sweep", "confirm", "extra", "speed"), default="sweep")
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--offsets", default="")
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    assert 0 < args.seconds <= 30
    settings = Settings.load()
    with sqlite3.connect((DATA/"jobs.sqlite").resolve().as_uri()+"?mode=ro", uri=True) as con:
        source = Path(con.execute("SELECT source FROM jobs ORDER BY created DESC LIMIT 1").fetchone()[0])
    info = probe(source, settings)
    assert info["video"].get("pix_fmt") in ("yuv420p", "nv12") and info["video"].get("color_transfer") not in ("smpte2084", "arib-std-b67"), "Study requires 8-bit 4:2:0 SDR; never downconvert HDR"
    offsets = [float(s) for s in args.offsets.split(",")] if args.offsets else [round(info["duration"]*f) for f in (.2, .48, .76)]
    assert all(0 <= offset and offset+args.seconds <= info["duration"] for offset in offsets)
    gpus = [int(s) for s in args.gpus.split(",")]
    assert len(set(gpus)) == len(gpus) and set(gpus) <= set(settings.gpus)
    choices = variants(args.suite)
    def worker(index):
        return [row for offset in offsets[index::len(gpus)] for row in study(settings, source, info, offset, args.seconds, gpus[index], choices, measure=args.suite != "speed")]
    with concurrent.futures.ThreadPoolExecutor(len(gpus)) as executor:
        rows = [row for batch in executor.map(worker, range(len(gpus))) for row in batch]
    report = dict(suite=args.suite, source_geometry=[info["video"]["width"], info["video"]["height"]],
                  source_codec=info["video"]["codec_name"], source_duration_seconds=info["duration"],
                  source_video_bps=info["video"].get("bit_rate"), source_bytes=info.get("source_bytes"),
                  audio_copied_bps=bitrate_plan(info, settings)["copied"],
                  video_cache="RAM only", reference="source decoded with identical neutral subtitle overlay",
                  metrics="encode timing only" if args.suite == "speed" else "VMAF v0.6.1 at native source resolution, all frames; SSIM; PSNR",
                  small_text_files_removed=True, results=rows)
    destination = DATA/f"h265-quality-{args.suite}.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": destination.name, "results": len(rows), "failed": sum(bool(r.get("failed")) for r in rows)}), flush=True)


if __name__ == "__main__":
    main()
