"""Full-length local ASR/resource validation; no external API or film upload."""
from __future__ import annotations
import json
import multiprocessing as mp
import threading
import time
from pathlib import Path

import psutil
from submaker.config import ROOT, Settings
from submaker.pipeline import Scheduler
from submaker.store import Store, fingerprint


def main():
    source=next((ROOT/"input").glob("*.mp4"))
    directory=ROOT/"data"/"validation"
    directory.mkdir(parents=True,exist_ok=True)
    settings=Settings()
    store=Store(directory/"long-asr.sqlite")
    previous=store.jobs()
    previous_report_path=ROOT/"docs"/"long-asr-report.json"
    previous_report=json.loads(previous_report_path.read_text(encoding="utf-8")) if previous_report_path.exists() else {}
    reused=False
    if previous:
        row=previous[-1]
        ident=row["id"]
        reused=store.get(ident,"chunk:0") is not None
        store.update(ident,"排队中","恢复长片ASR测试")
    else:
        ident=store.add(source,settings)
    last=[0.0]
    stages=[]
    def callback(kind,data):
        if kind=="progress":
            stages.append(data["stage"])
            if time.monotonic()-last[0]>15:
                print(data["stage"],data.get("detail",""),flush=True)
                last[0]=time.monotonic()
        elif kind in ("gpu","error"):
            print(kind,json.dumps(data,ensure_ascii=False),flush=True)
    scheduler=Scheduler(store,settings,callback)
    before=fingerprint(source)
    started=time.monotonic()
    max_rss=0
    min_available=psutil.virtual_memory().available
    scheduler.start()
    while scheduler.active:
        processes=[psutil.Process()]+psutil.Process().children(recursive=True)
        rss=0
        for p in processes:
            try:
                rss+=p.memory_info().rss
            except psutil.Error:
                pass
        max_rss=max(max_rss,rss)
        min_available=min(min_available,psutil.virtual_memory().available)
        time.sleep(1)
    row=next(r for r in store.jobs() if r["id"]==ident)
    segments=store.get(ident,"final_segments",[])
    media=store.get(ident,"media",{})
    valid=bool(segments) and all(s["end"]>s["start"]>=0 and s["end"]<=media.get("duration",0)+1 for s in segments)
    ordered=all(a["start"]<=b["start"] for a,b in zip(segments,segments[1:]))
    report={"test":"full-length real dual-V100 ASR, no API", "passed":valid and ordered and "本地识别已完成" in row["detail"], "duration_seconds":media.get("duration"), "elapsed_seconds":round(time.monotonic()-started,2), "segments":len(segments), "first_timestamp":segments[0]["start"] if segments else None, "last_timestamp":segments[-1]["end"] if segments else None,"max_process_tree_rss_gib":round(max_rss/1024**3,2),"minimum_available_ram_gib":round(min_available/1024**3,2),"source_unchanged":fingerprint(source)==before,"state":row["state"],"detail":row["detail"],"human_quality_reference":False}
    if reused:
        report["validation_reused_asr_checkpoints"]=True
        report["original_cold_asr_run_seconds"]=previous_report.get("original_cold_asr_run_seconds",previous_report.get("elapsed_seconds"))
        report["original_peak_process_rss_gib"]=previous_report.get("original_peak_process_rss_gib",previous_report.get("max_process_tree_rss_gib"))
    (ROOT/"docs"/"long-asr-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False),flush=True)
    scheduler.shutdown()
    if report["passed"]:
        store.clean(ident)
    store.close()
    return 0 if report["passed"] else 1


if __name__=="__main__":
    mp.freeze_support()
    raise SystemExit(main())
