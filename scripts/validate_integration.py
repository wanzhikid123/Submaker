"""Real dual-GPU ASR + local mock HTTP API + real video burn, three-job queue."""
from __future__ import annotations
import json
import multiprocessing as mp
import shutil
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from submaker.config import ROOT, Settings
from submaker.llm import Client, Limits
from submaker.pipeline import Scheduler
from submaker.store import Store, fingerprint


def main():
    state={"calls":[],"reviewed":set(),"active":0,"peak_api_concurrency":0,"rate_limit_injected":False}
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            system=body["messages"][0]["content"]
            data=json.loads(body["messages"][1]["content"])
            kind="translate" if "Translate subtitles" in system else "review" if "Audit every" in system else "brief" if "continuity brief" in system else "title"
            with lock:
                state["active"]+=1
                state["peak_api_concurrency"]=max(state["peak_api_concurrency"],state["active"])
                state["calls"].append({"kind":kind,"time":time.monotonic(),"repair":bool(data.get("repair_instruction"))})
            try:
                time.sleep(0.05)
                if kind=="translate" and not state["rate_limit_injected"]:
                    state["rate_limit_injected"]=True
                    self.send_response(429)
                    self.send_header("Retry-After","1")
                    self.end_headers()
                    return
                if kind=="translate":
                    answer={"translations":[{"id":r["id"],"text":"测试译文，仅用于流水线验收。"} for r in data["units"]]}
                elif kind=="review":
                    ident=data["units"][0]["id"]
                    answer={"issues":[]}
                    with lock:
                        if ident not in state["reviewed"]:
                            state["reviewed"].add(ident)
                            answer={"issues":[{"ids":[ident],"severity":"error","reason":"注入问题以验证局部修复"}]}
                elif kind=="brief":
                    answer={"brief":"模拟全片上下文；不代表真实语义审校。"}
                else:
                    answer={"title":"流水线验收-"+data["filename_stem"]}
                payload=json.dumps({"choices":[{"finish_reason":"stop","message":{"content":json.dumps(answer,ensure_ascii=False)}}],"usage":{"total_tokens":20}},ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type","application/json")
                self.send_header("Content-Length",str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            finally:
                with lock:
                    state["active"]-=1
        def log_message(self,*args):
            pass
    server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    build=ROOT/"build"
    build.mkdir(exist_ok=True)
    output=ROOT/"output"/"validation"
    output.mkdir(exist_ok=True,parents=True)
    source=next((ROOT/"input").glob("*.mp4"))
    original_fingerprint=fingerprint(source)
    timeline=[]
    def callback(kind,data):
        if kind=="gpu" or kind=="complete" or kind=="error":
            timeline.append({"kind":kind,"event":data,"time":time.monotonic()})
            print(kind,json.dumps(data,ensure_ascii=False),flush=True)
    started=time.monotonic()
    scheduler=None
    store=None
    try:
        with tempfile.TemporaryDirectory(prefix="integration-",dir=build) as temp:
            temporary=Path(temp)
            settings=Settings(output_dir=str(output),chunk_seconds=120,full_context_enabled=True,review_enabled=True)
            settings.translate.url=f"http://127.0.0.1:{server.server_port}/v1"
            settings.translate.model="local-mock-validation"
            settings.translate.key_id="validation-no-key"
            store=Store(temporary/"jobs.sqlite")
            for i in range(3):
                clip=temporary/f"sample-{i+1}.mp4"
                subprocess.run([settings.ffmpeg,"-v","error","-f","lavfi","-i","color=c=0x151e30:s=640x360:r=25:d=180","-ss",str(60+i*20),"-i",str(source),"-t","180","-map","0:v:0","-map","1:a:0","-c:v","libx264","-preset","ultrafast","-c:a","aac","-shortest",str(clip)],check=True)
                store.add(clip,settings)
            scheduler=Scheduler(store,settings,callback,Client(Limits(),key_provider=lambda _:""))
            scheduler.start()
            deadline=time.monotonic()+240
            while scheduler.active and time.monotonic()<deadline:
                time.sleep(0.2)
            jobs=store.jobs()
            calls=state["calls"]
            first_translation=min((x["time"] for x in calls if x["kind"]=="translate"),default=float("inf"))
            first_asr_done=min((x["time"] for x in timeline if x["kind"]=="gpu" and x["event"]["type"]=="done"),default=0)
            repair_calls=sum(x["repair"] for x in calls)
            checkpoints=store.db.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
            report={"test":"three-video real dual-GPU ASR + MOCK HTTP translation/review/title + real burn", "passed":all(j["state"]=="完成" for j in jobs) and len(jobs)==3 and repair_calls>0 and checkpoints==0 and first_translation<first_asr_done,"elapsed_seconds":round(time.monotonic()-started,2),"jobs":[{"state":j["state"],"detail":j["detail"],"output":j["output"]} for j in jobs],"api_calls":len(calls),"peak_api_concurrency":state["peak_api_concurrency"],"translation_started_before_first_chunk_done":first_translation<first_asr_done,"repair_calls":repair_calls,"rate_limit_recovered":state["rate_limit_injected"],"remaining_checkpoint_rows":checkpoints,"original_source_unchanged":original_fingerprint==fingerprint(source),"real_translation_quality_tested":False}
            (ROOT/"docs"/"integration-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
            print(json.dumps(report,ensure_ascii=False),flush=True)
            scheduler.shutdown()
            scheduler=None
            store.close()
            store=None
    finally:
        if scheduler:
            scheduler.shutdown()
        if store:
            store.close()
        server.shutdown()
        server.server_close()
    return 0 if report["passed"] else 1


if __name__=="__main__":
    mp.freeze_support()
    raise SystemExit(main())
