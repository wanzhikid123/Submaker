"""Opt-in real Whisper + local LLM + NVENC check using public JFK test audio."""
from pathlib import Path
import subprocess
import time
import json

import httpx

from submaker.config import DATA, LocalRole, Settings
from submaker.pipeline import Scheduler
from submaker.store import Store


def main():
    directory = DATA/"local-pipeline-validation"
    directory.mkdir(exist_ok=True)
    source = directory/"Public speech test.mp4"
    if not source.exists():
        response = httpx.get("https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac", follow_redirects=True, timeout=30)
        response.raise_for_status()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=navy:s=640x360:r=25",
                        "-i", "pipe:0", "-t", "12", "-shortest", "-c:v", "libx264", "-c:a", "aac", str(source)], input=response.content, check=True)
    settings = Settings(llm_mode="local", source="en", chunk_seconds=30, output_dir=str(directory/"output"),
                        local_translate=LocalRole("none", 2048), local_review=LocalRole("none", 2048))
    store = Store(directory/"jobs.sqlite")  # Isolated test database; never opens production tasks.
    store.delete([row["id"] for row in store.jobs()])
    ident = store.add(source, settings)
    events = []
    last_stage = [""]
    def observe(kind, value):
        if kind == "progress" and value["stage"] != last_stage[0]:
            last_stage[0] = value["stage"]
            print(value["stage"], flush=True)
        if kind in ("gpu", "api", "error", "complete"):
            events.append({"kind": kind, **value})
    scheduler = Scheduler(store, settings, observe)
    started = time.monotonic()
    try:
        scheduler.start()
        deadline = time.monotonic()+900
        while scheduler.active and time.monotonic() < deadline:
            time.sleep(.2)
        if scheduler.active:
            raise RuntimeError("Local pipeline smoke test exceeded 15 minutes")
        row = store.jobs()[0]
        assert row["state"] == "完成", row["detail"]
        assert Path(row["output"]).exists()
        assert all(e.get("gpu") == 0 for e in events if e["kind"] == "gpu" and "gpu" in e)
        allocation = scheduler.client.runtime().snapshot()
        assert allocation["gpu"] == 1 and allocation["phase"] == "ready"
        report = {"passed": True, "seconds": round(time.monotonic()-started, 2), "whisper_gpu": 0, "video_gpu": 0,
                  "local_llm_gpu": 1, "allocation": allocation, "total_tokens": row["total_tokens"], "requests": row["token_requests"],
                  "output": row["output"], "output_bytes": Path(row["output"]).stat().st_size,
                  "cache_cleared": store.get(ident, "chunk:0") is None, "fixture": "OpenAI Whisper public JFK test audio"}
        (directory/"result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        scheduler.shutdown()
        store.close()


if __name__ == "__main__":
    main()
