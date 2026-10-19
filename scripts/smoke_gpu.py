"""Real GPU inference smoke, JSON report; no external API calls."""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import queue
import sys
import threading
import time
from pathlib import Path

from submaker.asr import ASRPool
from submaker.config import ROOT, Settings
from submaker.media import probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--start", type=int, default=60)
    parser.add_argument("--report", default="data/gpu-smoke.json")
    args = parser.parse_args()
    source = next((ROOT/"input").glob("*.mp4"))
    settings = Settings()
    info = probe(source, settings)
    report = {"test":"real-dual-v100-asr", "seconds_per_gpu":args.seconds,"events":[],"segments":[],"started":time.time()}
    pool = ASRPool(settings, lambda e: (report["events"].append(e), print(json.dumps(e),flush=True)))
    cancel = threading.Event()
    events = pool.register("smoke")
    for i in range(2):
        start = args.start+i*args.seconds
        pool.submit({"job":"smoke","chunk":i,"source":str(source),"audio_index":info["audio"]["index"],"ffmpeg":settings.ffmpeg,"language":"ja","glossary":"","core_start":start,"core_end":start+args.seconds,"decode_start":max(0,start-2),"decode_end":start+args.seconds+2},cancel)
    done = set()
    try:
        deadline=time.monotonic()+1200
        while len(done)<2 and time.monotonic()<deadline:
            if pool.fatal:
                raise RuntimeError(pool.fatal)
            try:
                event=events.get(timeout=1)
            except queue.Empty:
                continue
            if event["type"] == "segment":
                report["segments"].append(event)
                print(f"GPU{event['gpu']} chunk{event['chunk']} segment {len(report['segments'])}",flush=True)
            elif event["type"]=="done":
                done.add(event["chunk"])
            elif event["type"]=="error":
                raise RuntimeError(event["error"])
        if len(done)!=2:
            raise TimeoutError("GPU smoke timed out")
        report["passed"] = bool(report["segments"])
        report["elapsed"] = time.time()-report["started"]
    except Exception as exc:
        report["passed"]=False
        report["error"]=str(exc)
        print("FAILED",str(exc),flush=True)
    finally:
        pool.shutdown()
        raw_segments=report.pop("segments",[])
        report["segment_count"]=len(raw_segments)
        report["segments_per_gpu"]={str(gpu):sum(s["gpu"]==gpu for s in raw_segments) for gpu in settings.gpus}
        Path(args.report).parent.mkdir(exist_ok=True,parents=True)
        Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("segments","events")}),flush=True)
    return 0 if report.get("passed") else 1


if __name__=="__main__":
    mp.freeze_support()
    sys.exit(main())
