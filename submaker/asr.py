from __future__ import annotations

import ctypes
import math
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from .media import CREATE_NO_WINDOW, Cancelled, check

_dll_handles = []


def cuda_paths():
    """Keep DLL directory handles alive; never depend on another project's CUDA."""
    root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    paths = [p for p in root.glob("*/bin") if p.is_dir()]
    paths += [Path(sys.prefix) / "Library" / "bin"]
    for p in paths:
        if p.is_dir():
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                _dll_handles.append(os.add_dll_directory(str(p)))


def decode_window(task, extraction_slot, cancelled):
    import numpy as np
    import psutil
    while psutil.virtual_memory().available < 4 * 1024**3:
        if cancelled.get(task["job"], False):
            raise Cancelled()
        time.sleep(0.25)
    with extraction_slot:
        if cancelled.get(task["job"], False):
            raise Cancelled()
        args = [task["ffmpeg"], "-hide_banner", "-nostdin", "-v", "error", "-threads", "1", "-ss", str(task["decode_start"]), "-i", task["source"], "-t", str(task["decode_end"] - task["decode_start"]), "-map", f"0:{task['audio_index']}", "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000", "-af", "aresample=async=1:first_pts=0", "-f", "f32le", "pipe:1"]
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=CREATE_NO_WINDOW)
        try:
            data, errors = p.communicate(timeout=120)
            if p.returncode:
                raise RuntimeError("音轨解码失败：" + errors.decode("utf-8", "replace")[-1000:])
            expected = (task["decode_end"] - task["decode_start"]) * 16000 * 4
            if len(data) > expected + 16000 * 4 * 2:
                raise RuntimeError("解码音频超出窗口内存预算")
            return np.frombuffer(data, dtype=np.float32).copy()
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()


def _worker(gpu, model_path, cpu_threads, tasks, events, extraction_slot, cancelled, ram_slots):
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    cuda_paths()
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_path, device="cuda", device_index=gpu, compute_type="float16", cpu_threads=cpu_threads, num_workers=1, local_files_only=True)
        events.put({"type": "ready", "gpu": gpu})
    except BaseException as exc:
        events.put({"type": "fatal", "gpu": gpu, "error": f"{type(exc).__name__}: {exc}"})
        return
    while True:
        task = tasks.get()
        if task is None:
            return
        ident = task["job"]
        common = {"job": ident, "chunk": task["chunk"], "gpu": gpu}
        acquired = 0
        try:
            if cancelled.get(ident, False):
                raise Cancelled()
            # 64MiB accounting tokens include raw float input + copies/features reserve.
            needed = max(1, math.ceil((task["decode_end"] - task["decode_start"]) * 16000 * 4 * 4 / (64 * 1024**2)))
            for _ in range(needed):
                while not ram_slots.acquire(timeout=0.2):
                    if cancelled.get(ident, False):
                        raise Cancelled()
                acquired += 1
            events.put(common | {"type": "started"})
            events.put(common | {"type": "extracting"})
            audio = decode_window(task, extraction_slot, cancelled)
            events.put(common | {"type": "transcribing"})
            segments, info = model.transcribe(audio, language=None if task["language"] == "auto" else task["language"].split("-")[0], task="transcribe", beam_size=task.get("beam_size",5), word_timestamps=True, vad_filter=True, vad_parameters={"min_silence_duration_ms": 500}, condition_on_previous_text=True, initial_prompt=task.get("glossary") or None)
            events.put(common | {"type": "language", "language": info.language, "probability": info.language_probability})
            for segment in segments:
                if cancelled.get(ident, False):
                    raise Cancelled()
                offset = task["decode_start"]
                words = [w for w in (segment.words or []) if task["core_start"] <= offset + (w.start + w.end) / 2 < task["core_end"]]
                if words:
                    start, end = words[0].start + offset, words[-1].end + offset
                    text = "".join(w.word for w in words).strip()
                elif not segment.words and task["core_start"] <= offset + (segment.start + segment.end) / 2 < task["core_end"]:
                    start, end, text = segment.start + offset, segment.end + offset, segment.text.strip()
                else:
                    continue
                start, end = max(0, start), min(task["decode_end"], end)
                if text and end > start:
                    events.put(common | {"type": "segment", "segment": {"start": round(start, 3), "end": round(end, 3), "text": text, "logprob": segment.avg_logprob, "no_speech": segment.no_speech_prob}})
            del audio
            events.put(common | {"type": "done"})
        except Cancelled:
            events.put(common | {"type": "cancelled"})
        except BaseException as exc:
            events.put(common | {"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            for _ in range(acquired):
                ram_slots.release()


class ASRPool:
    def __init__(self, settings, on_status=lambda _: None):
        self.settings = settings
        self.on_status = on_status
        self.ctx = mp.get_context("spawn")
        self.manager = self.ctx.Manager()
        self.cancelled = self.manager.dict()
        self.tasks = self.ctx.Queue(maxsize=8)
        self.events = self.ctx.Queue(maxsize=128)
        self.extract = self.ctx.BoundedSemaphore(1)
        self.ram = self.ctx.BoundedSemaphore(max(8, settings.ram_gib * 16))
        self.queues = {}
        self.lock = threading.RLock()
        self.processes = []
        self.fatal = None
        self.closed = False
        for gpu in settings.gpus:
            p = self.ctx.Process(target=_worker, args=(gpu, settings.model_path, min(2, max(1, settings.threads // (len(settings.gpus) * 2))), self.tasks, self.events, self.extract, self.cancelled, self.ram), daemon=True)
            p.start()
            self.processes.append(p)
        threading.Thread(target=self._dispatch, daemon=True).start()

    def _dispatch(self):
        while not self.closed:
            try:
                event = self.events.get(timeout=0.5)
            except queue.Empty:
                if any(not p.is_alive() for p in self.processes):
                    self.fatal = self.fatal or "识别进程异常退出，请检查GPU运行环境并重新启动队列"
                continue
            if event["type"] == "fatal":
                self.fatal = event["error"]
            if "job" not in event:
                self.on_status(event)
            else:
                if event["type"] in ("started", "extracting", "transcribing", "done", "error", "cancelled"):
                    self.on_status(event)
                with self.lock:
                    q = self.queues.get(event["job"])
                if q is not None:
                    q.put(event)

    def register(self, ident):
        # Only at most two bounded audio windows are outstanding per job; their text
        # events are tiny. Do not let a rate-limited job block other jobs' dispatch.
        q = queue.Queue()
        with self.lock:
            self.queues[ident] = q
        self.cancelled[ident] = False
        return q

    def submit(self, task, cancel):
        while True:
            check(cancel)
            if self.fatal:
                raise RuntimeError(self.fatal)
            try:
                self.tasks.put(task, timeout=0.2)
                return
            except queue.Full:
                continue

    def unregister(self, ident):
        if not self.closed:
            self.cancelled[ident] = True
        with self.lock:
            self.queues.pop(ident, None)

    def shutdown(self):
        self.closed = True
        for ident in list(self.cancelled.keys()):
            self.cancelled[ident] = True
        for p in self.processes:
            if p.is_alive():
                p.terminate()
            p.join(timeout=3)
        self.manager.shutdown()
