"""Opt-in synthetic single/parallel export comparison; no LLM requests."""
import argparse
import dataclasses
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import threading
import time

from submaker.config import DATA, Settings
from submaker.encoding_slots import EncodingSlots
from submaker.media import burn, probe, temporary_subtitles
from submaker.segmented import burn_segmented


def idle():
    database = DATA/"jobs.sqlite"
    if database.exists():
        with sqlite3.connect(database.resolve().as_uri()+"?mode=ro", uri=True) as connection:
            if connection.execute("SELECT 1 FROM jobs WHERE state IN ('处理中','排队中') LIMIT 1").fetchone():
                raise RuntimeError("请在生产队列空闲时运行分段编码对照")


def quality_score(source, output, subtitle, root):
    # Timestamp preservation is checked separately. Pair decoded frames by
    # index so millisecond mux rounding cannot skew the quality comparison.
    filters = f"[0:v]settb=1/30,setpts=N,subtitles={subtitle.relative_to(root).as_posix()}[ref];[1:v]settb=1/30,setpts=N[candidate];[candidate][ref]ssim"
    result = subprocess.run(["ffmpeg", "-v", "info", "-threads", "2", "-i", str(source), "-threads", "2", "-i", str(output),
                             "-filter_complex_threads", "2", "-lavfi", filters, "-an", "-f", "null", "-"], cwd=root, capture_output=True, check=True)
    score = re.search(r"All:([0-9.]+)", result.stderr.decode("utf-8", "replace"))
    return float(score[1]) if score else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-gpu", action="store_true", help="Compare one lane, GPU0 three lanes, and dual-GPU six lanes")
    options = parser.parse_args()
    idle()
    root = DATA/("segment-validation-dual" if options.dual_gpu else "segment-validation")
    root.mkdir(exist_ok=True)
    source = root/"synthetic.mp4"
    if not source.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=120",
                        "-f", "lavfi", "-i", "sine=frequency=500:duration=120", "-c:v", "h264_nvenc", "-gpu", "0", "-preset", "p3", "-cq", "20",
                        "-c:a", "aac", str(source)], check=True)
    settings = Settings(output_mode="burn", gpus=[0, 1] if options.dual_gpu else [0], video_gpu=-1 if options.dual_gpu else 0,
                        video_concurrency=6 if options.dual_gpu else 3, video_quality=24,
                        video_decode_threads=1, video_filter_threads=1 if options.dual_gpu else 2)
    info = probe(source, settings)
    report = {"fixture": "synthetic 120s 720p30", "gpus": settings.gpus, "codec": "HEVC NVENC P6 CQ24 compact",
              "decode_threads": settings.video_decode_threads, "filter_threads": settings.video_filter_threads, "results": []}
    with temporary_subtitles(root, [{"start": .1, "end": 119.8, "text": "分段并行测试 · 字幕跨越编码边界"}]) as subtitle:
        names = ("single-before", "single-gpu-segmented", "segmented", "single-after") if options.dual_gpu else ("single-before", "segmented", "single-after")
        for name in names:
            idle()
            output = root/(name+".mkv")
            output.unlink(missing_ok=True)
            start = time.monotonic()
            peaks, lock = [], threading.Lock()
            if "segmented" in name:
                actual = dataclasses.replace(settings, gpus=[0], video_gpu=0, video_concurrency=3) if name == "single-gpu-segmented" else settings
                slots = EncodingSlots(actual.video_concurrency)
                def observe(*args):
                    with lock:
                        peaks.append(slots.snapshot())
                burn_segmented(source, output, subtitle, info, actual, threading.Event(), slots, "validation", on_worker=observe)
            else:
                burn(source, output, subtitle, info, dataclasses.replace(settings, video_gpu=0, video_concurrency=1), threading.Event(), lambda _: None)
            result = {"kind": name, "seconds": round(time.monotonic()-start, 3), "bytes": output.stat().st_size}
            if peaks:
                result["max_active"] = max(p["active"] for p in peaks)
                result["peak_per_gpu"] = {str(g): max(p["gpus"].get(g, 0) for p in peaks) for g in actual.gpus}
            report["results"].append(result)
            print(json.dumps(result), flush=True)
        # Compare both exports against identically captioned source frames.
        for name in names[:-1]:
            idle()
            report[name+"_ssim"] = quality_score(source, root/(name+".mkv"), subtitle, root)
    results = {r["kind"]: r for r in report["results"]}
    before, parallel, after = (results[name] for name in ("single-before", "segmented", "single-after"))
    report["speedup"] = round((before["seconds"]+after["seconds"])/2/parallel["seconds"], 3)
    report["size_ratio"] = round(parallel["bytes"]/before["bytes"], 4)
    if options.dual_gpu:
        report["dual_vs_single_gpu_segmented"] = round(results["single-gpu-segmented"]["seconds"]/parallel["seconds"], 3)
    (root/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
