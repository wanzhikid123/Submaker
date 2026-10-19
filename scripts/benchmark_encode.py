"""Local encoder benchmark. Encoded samples stay in RAM; no API or film upload."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path

from submaker.config import DATA, ROOT, Settings
from submaker.media import CREATE_NO_WINDOW, bitrate_plan, probe, write_ass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--offset", type=float, default=300)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--phase", choices=("initial", "tuning", "validation", "cuda"), default="initial")
    args = parser.parse_args()
    settings = Settings.load()
    with sqlite3.connect(f"file:{(DATA/'jobs.sqlite').as_posix()}?mode=ro", uri=True) as con:
        source = con.execute("SELECT source FROM jobs ORDER BY created DESC LIMIT 1").fetchone()[0]
    info = probe(source, settings)
    if info["video"].get("pix_fmt") not in ("yuv420p", "nv12"):
        raise ValueError("This benchmark matrix is for 8-bit 4:2:0 SDR; do not downconvert 10-bit/HDR for benchmarking")
    rates = bitrate_plan(info, settings)
    duration = min(args.seconds, info["duration"]-args.offset)
    assert 0 < duration <= 120
    report = {"phase": args.phase, "source_geometry": [info["video"]["width"], info["video"]["height"]], "source_codec": info["video"]["codec_name"], "source_fps": info["video"].get("avg_frame_rate"), "offset_seconds": args.offset, "sample_seconds": duration, "cq": settings.video_quality, "codec": settings.video_codec, "gpu": args.gpu, "encoded_media_cache": "RAM only", "results": []}
    samples = {}
    with tempfile.TemporaryDirectory(prefix="encode-bench-", dir=DATA) as temporary:
        location = Path(temporary).resolve()
        assert location.parent == DATA.resolve()
        cues = [{"start": i, "end": min(duration, i+4), "text": "字幕性能测试：画质、时间与内容保持一致。"} for i in range(0, int(duration), 6)]
        write_ass(location/"captions.ass", cues)
        font = re.sub(r"[^\w \-]", "", settings.font)
        overlay = f"subtitles=filename='captions.ass':force_style='FontName={font},FontSize={settings.font_size},PrimaryColour=&H00FFFFFF,OutlineColour=&H00111111,BorderStyle=1,Outline=1.6,Shadow=0.5,MarginV=24'"
        variants = [
            ("baseline", 1, "yuv420p", True, settings.encoder_preset, "fullres"),
            ("filter4", 4, "yuv420p", True, settings.encoder_preset, "fullres"),
            ("filter8", 8, "yuv420p", True, settings.encoder_preset, "fullres"),
            ("native_nv12", 4, "nv12", True, settings.encoder_preset, "fullres"),
            ("software_decode", 4, "nv12", False, settings.encoder_preset, "fullres"),
            ("encoder_only", 1, None, True, settings.encoder_preset, "fullres"),
            ("p6_comparison", 4, "nv12", True, "p6", "qres"),
        ]
        if args.phase == "tuning":
            variants = [(name, 4, "nv12", True, settings.encoder_preset, "fullres") for name in ("baseline", "decode_threads4", "decode_threads8", "extra_frames32", "cuvid", "primary_context", "primary_threads4")]
        if args.phase == "validation":
            variants = [(name, threads, "yuv420p", True, settings.encoder_preset, "fullres") for name, threads in (("baseline", 1), ("decode_threads2", 1), ("decode_threads4", 4), ("baseline_repeat", 1))]
        if args.phase == "cuda":
            variants = [(name, 4, "cuda" if name == "explicit_cuda_upload" else "yuv420p", True, settings.encoder_preset, "fullres") for name in ("baseline", "explicit_cuda_upload", "baseline_repeat")]
        for name, threads, pixel, hardware_decode, preset, multipass in variants:
            decode_threads = "8" if name == "decode_threads8" or not hardware_decode else "4" if name in ("decode_threads4", "primary_threads4") else "2" if name == "decode_threads2" else "1"
            if args.phase == "cuda":
                decode_threads = "4"
            cmd = [settings.ffmpeg, "-hide_banner", "-nostdin", "-benchmark", "-threads", decode_threads]
            primary = name in ("primary_context", "primary_threads4", "explicit_cuda_upload")
            if primary:
                cmd += ["-init_hw_device", f"cuda=encode:{args.gpu},primary_ctx=1", "-filter_hw_device", "encode"]
            if hardware_decode:
                cmd += ["-hwaccel", "cuda", "-hwaccel_device", "encode" if primary else str(args.gpu), "-hwaccel_output_format", "cuda"]
            if name == "extra_frames32":
                cmd += ["-extra_hw_frames", "32"]
            if name == "cuvid":
                cmd += ["-c:v", "hevc_cuvid" if info["video"]["codec_name"] == "hevc" else "h264_cuvid"]
            cmd += ["-ss", str(args.offset), "-i", source, "-t", str(duration), "-map", f"0:{info['video']['index']}", "-an", "-sn", "-dn"]
            if pixel:
                vf = ("hwdownload,format=nv12," if hardware_decode else "")+overlay
                if name == "explicit_cuda_upload":
                    vf += ",format=nv12,hwupload"
                cmd += ["-filter_threads", str(threads), "-vf", vf, "-pix_fmt", pixel]
            cmd += ["-c:v", "hevc_nvenc" if settings.video_codec == "hevc" else "h264_nvenc", "-gpu", str(args.gpu), "-preset", preset, "-tune", "hq", "-rc", "vbr", "-cq", str(settings.video_quality), "-b:v", str(rates["target"] if rates["cap"] else 0), "-spatial-aq", "1", "-rc-lookahead", "32", "-multipass", multipass]
            if rates["cap"]:
                cmd += ["-maxrate", str(rates["cap"]), "-bufsize", str(rates["cap"]*2)]
            cmd += ["-fps_mode", "passthrough", "-f", "matroska", "pipe:1"]
            started = time.perf_counter()
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=location, creationflags=CREATE_NO_WINDOW, timeout=180)
            elapsed = time.perf_counter()-started
            stderr = result.stderr.decode("utf-8", "replace")
            if result.returncode:
                row = {"variant": name, "failed": True, "error": stderr[-1200:]}
            else:
                samples[name] = result.stdout
                decoded = subprocess.run([settings.ffmpeg, "-v", "error", "-i", "pipe:0", "-map", "0:v:0", "-pix_fmt", "yuv420p", "-f", "framemd5", "pipe:1"], input=result.stdout, capture_output=True, creationflags=CREATE_NO_WINDOW, timeout=90, check=True)
                frame_hashes = [line.rsplit(b",", 1)[-1].strip() for line in decoded.stdout.splitlines() if line and not line.startswith(b"#")]
                row = {"variant": name, "elapsed_seconds": round(elapsed, 3), "realtime_speed": round(duration/elapsed, 3), "encoded_bytes": len(result.stdout), "frames": len(frame_hashes), "decoded_frame_hash": hashlib.sha256(b"".join(frame_hashes)).hexdigest(), "decode_threads": int(decode_threads), "filter_threads": threads, "encoder_preset": preset, "multipass": multipass, "pixel_input": pixel, "hardware_decode": hardware_decode}
                bench = re.search(r"bench: utime=([\d.]+)s stime=([\d.]+)s rtime=([\d.]+)s", stderr)
                if bench:
                    row["cpu_user_seconds"], row["cpu_system_seconds"], row["ffmpeg_wall_seconds"] = map(float, bench.groups())
            report["results"].append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
        baseline = report["results"][0]
        for row in report["results"]:
            if not row.get("failed"):
                row["speedup_vs_baseline"] = round(baseline["elapsed_seconds"]/row["elapsed_seconds"], 3)
                row["decoded_frames_identical_to_baseline"] = row["decoded_frame_hash"] == baseline["decoded_frame_hash"]
        report["temporary_files_removed"] = True
    destination = ROOT/"docs"/f"encode-benchmark-{args.phase}-gpu{args.gpu}-{int(args.offset)}.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RAM samples released; benchmark report saved", flush=True)


if __name__ == "__main__":
    main()
