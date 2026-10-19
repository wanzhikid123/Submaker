from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import hashlib
import json
import math
import queue
import shutil
import threading
import time
from pathlib import Path

from .asr import ASRPool
from .dispatch import FairTranslationExecutor
from .encoding_slots import EncodingSlots
from .config import MAX_VIDEO_ENCODERS, Settings
from .llm import APIError, Client, Limits, JOB_CONTEXT, profile_identity, profile_tag, translation_key, translation_unit_keys
from .media import Cancelled, burn, check, estimate_output, format_cues, legal_stem, preserve_filename_tokens, probe, reserve_output, temporary_subtitles, video_plan
from .store import fingerprint


def segment_id(segment):
    return hashlib.sha256(json.dumps({k: segment[k] for k in ("start", "end", "text")}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]


def split_rows(rows, max_chars=18000, max_count=40):
    batch, size = [], 0
    for row in rows:
        n = len(json.dumps(row, ensure_ascii=False))
        if batch and (size + n > max_chars or len(batch) >= max_count):
            yield batch
            batch, size = [], 0
        batch.append(row)
        size += n
    if batch:
        yield batch


class Scheduler:
    def __init__(self, store, settings, callback=lambda *args: None, client=None):
        self.store, self.settings, self.callback = store, settings, callback
        self.client = client or Client(Limits(), self._record_usage, status=self._record_api, local_only=settings.llm_mode == "local")
        if client is None:
            for row in store.jobs():
                if row["state"].startswith("完成"):
                    continue
                snapshot = Settings.from_dict(json.loads(row["config"]))
                if snapshot.llm_mode != settings.llm_mode:
                    continue
                providers = row.get("api_providers") or []
                if providers:
                    configured = {profile_tag(p): p for p in [*snapshot.translation_profiles(), snapshot.review_profile()] if p.url and p.model}
                    for status in providers:
                        profile = configured.get(status["profile"])
                        if profile and status["rate_limits"] and time.time()-status["updated"] < 120:
                            gate = self.client.limits.for_profile(profile)
                            previous_limit = gate.limit
                            gate.throttle(max(0, status["retry_at"]-time.time()))
                            with gate.condition:
                                gate.limit = min(previous_limit, max(1, status["effective_limit"]))
                elif row.get("api_state") == "paused" and row.get("api_code") == 429 and snapshot.translate.url:
                    profile = snapshot.translate
                    gate = self.client.limits.for_profile(profile)
                    gate.throttle(10)
                    with gate.condition:
                        gate.limit = min(gate.limit, 2)
        self.pool = None
        self.executor = cf.ThreadPoolExecutor(max_workers=max(1, settings.in_flight), thread_name_prefix="video")
        self.translator = FairTranslationExecutor(max_workers=max(16, settings.translation_parallelism()))
        self.encoding_slots = EncodingSlots(settings.encoding_limit())
        self.active = {}
        self.deleting = set()
        self.previews = {}
        self.lock = threading.RLock()
        self.accepting = False
        self.last_persist = {}

    def _record_usage(self, usage):
        job = JOB_CONTEXT.get()
        if job:
            self.store.add_usage(job, usage, request_recorded=True)
        self.callback("usage", {"job": job, "usage": usage})

    def _record_api(self, event):
        if event.get("job"):
            self.store.record_api(event["job"], event)
        self.callback("api", event)

    def _acquire_encoder(self, info, settings, cancel, on_wait=None):
        hardware = video_plan(info, settings)["encode"]
        candidates = [settings.video_gpu] if settings.video_gpu >= 0 else settings.gpus
        slot = self.encoding_slots.acquire(candidates, cancel, cpu=not hardware, on_wait=on_wait)
        return slot, dataclasses.replace(settings, video_gpu=slot.gpu if hardware else settings.video_gpu,
                                         video_concurrency=self.encoding_slots.limit)

    def emit(self, job, stage, progress=0, detail=""):
        now = time.monotonic()
        last_time, last_stage = self.last_persist.get(job, (0, ""))
        if now-last_time >= 2 or stage != last_stage:
            self.store.update(job, "处理中", detail or stage)
            self.last_persist[job] = (now, stage)
        self.callback("progress", {"job": job, "stage": stage, "progress": progress, "detail": detail})

    def start(self):
        if isinstance(self.client, Client):
            self.client.local_only = self.settings.llm_mode == "local"
        self.accepting = True
        for row in self.store.jobs():
            if row["state"] in ("等待恢复", "已暂停") and row["id"] not in self.active:
                self.store.update(row["id"], "排队中", "等待继续")
        self._pump()

    def _pump(self):
        with self.lock:
            for row in self.store.jobs():
                if not self.accepting or len(self.active) >= self.settings.in_flight:
                    break
                if row["state"] != "排队中" or row["id"] in self.active:
                    continue
                cancel = threading.Event()
                self.active[row["id"]] = cancel
                self.executor.submit(self._run, row, cancel)

    def retry(self, ident, settings=None, title=None):
        with self.lock:
            if ident in self.active:
                return
        if settings is not None:
            with self.store.lock, self.store.db:
                saved = self.store.db.execute("SELECT config FROM jobs WHERE id=?", (ident,)).fetchone()
                value = settings.as_dict()
                if saved:
                    value["repair_via_review"] = Settings.from_dict(json.loads(saved[0])).repair_via_review
                self.store.db.execute("UPDATE jobs SET config=? WHERE id=?", (json.dumps(value, ensure_ascii=False), ident))
        if title:
            self.store.put(ident, "manual_title", legal_stem(title))
        if hasattr(self.client, "reset_faults"):
            self.client.reset_faults()
        self.store.update(ident, "排队中", "等待恢复")
        self.accepting = True
        self._pump()

    def pause(self, ident=None):
        if ident is None:
            self.accepting = False
        with self.lock:
            for key, cancel in self.active.items():
                if ident is None or ident == key:
                    cancel.set()

    def delete(self, jobs=None):
        """Cancel active work before removing its records, never remove media files."""
        with self.lock:
            if jobs is None:
                self.accepting = False
                jobs = [row["id"] for row in self.store.jobs()]
            jobs = set(jobs)
            for ident in jobs:
                if ident in self.active:
                    self.deleting.add(ident)
                    self.store.update(ident, "删除中", "正在停止任务并清理运行缓存")
                    self.active[ident].set()
                else:
                    self.store.delete([ident])
                    self.previews.pop(ident, None)
                    self.last_persist.pop(ident, None)
        self.callback("refresh", {})
        return len(jobs)

    def _ensure_pool(self, settings):
        with self.lock:
            if self.pool is None:
                if not Path(settings.model_path, "model.bin").exists():
                    raise RuntimeError("未安装 large-v3 模型，请先点击“安装模型”")
                self.pool = ASRPool(settings, lambda e: self.callback("gpu", e))
            current = self.pool.settings
            if any(getattr(current, key) != getattr(settings, key) for key in ("gpus", "threads", "ram_gib", "model_path")):
                raise RuntimeError("识别资源配置已更改，请暂停并重启工作台，再使用当前设置重试；不会静默使用其他GPU")
            if self.pool.fatal:
                raise RuntimeError(self.pool.fatal)
            return self.pool

    def configure(self, settings):
        with self.lock:
            self._configure(settings)

    def _configure(self, settings):
        if not 1 <= settings.video_concurrency <= MAX_VIDEO_ENCODERS:
            raise ValueError("编码并发应为 1 至 6")
        if self.active:
            resources = ("llm_mode", "local_model", "gpus", "threads", "ram_gib", "model_path", "in_flight", "video_concurrency", "video_segmented", "video_decode_threads", "video_filter_threads")
            if any(getattr(settings, key) != getattr(self.settings, key) for key in resources):
                raise RuntimeError("请先暂停并等待在途任务停止，再更改 GPU、CPU 线程或模型/编码并发")
            return
        effective = settings.execution_settings()
        if self.pool and any(getattr(self.pool.settings, key) != getattr(effective, key) for key in ("gpus", "threads", "ram_gib", "model_path")):
            self.pool.shutdown()
            self.pool = None
        if isinstance(self.client, Client):
            self.client.local_only = settings.llm_mode == "local"
            if settings.local_model != self.settings.local_model:
                self.client.limits.discard(self.settings.local_profile("translate"))
            if self.client.local_runtime and (settings.llm_mode != "local" or settings.local_model != self.settings.local_model):
                self.client.local_runtime.stop()
        if settings.in_flight != self.settings.in_flight:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = cf.ThreadPoolExecutor(max_workers=max(1, settings.in_flight), thread_name_prefix="video")
        self.encoding_slots.set_limit(effective.video_concurrency)
        self.settings = settings

    def _run(self, job, cancel):
        ident = job["id"]
        job_context = JOB_CONTEXT.set(ident)
        settings = Settings.from_dict(json.loads(job["config"])).execution_settings()
        pending = []
        reserved = None
        try:
            if settings.llm_mode != self.settings.llm_mode:
                raise APIError("任务保存的模型模式与当前模式不同；请使用当前设置重试，避免调用未选择的接口")
            if settings.llm_mode == "local" and settings.local_model != self.settings.local_model:
                raise APIError("任务保存的本地模型资源与当前配置不同；请使用当前设置重试，避免切换在途模型或占用保留的 GPU")
            if settings.llm_mode == "local":
                from .local_llm import verify_model
                self.emit(ident, "检查本地模型", .005)
                verify_model(settings.local_model, cancel)
            if fingerprint(job["source"]) != job["fingerprint"]:
                raise RuntimeError("源文件已经改变，拒绝复用旧恢复点；请重新添加视频")
            self.emit(ident, "媒体预检", 0.01)
            info = probe(job["source"], settings)
            video_plan(info, settings)
            directory = Path(settings.output_dir) if settings.output_dir else Path(job["source"]).parent
            directory.mkdir(exist_ok=True, parents=True)
            if shutil.disk_usage(directory).free < max(1024**3, estimate_output(info, settings)["high"] * 1.1):
                raise RuntimeError("输出空间不足，请更换输出目录后重试")
            translations = {}
            segments = []
            translation_failure = [None]
            last_preview = [0.0]
            last_translation = [time.monotonic()]
            asr_finished = [False]
            # Text/futures are small. A six-request gate starves both GPUs on a slow
            # remote service; use a bounded text backlog independent of audio windows.
            pending_limit = max(128, min(8192, settings.ram_gib * 1024))

            def submit_translation(batch):
                if not batch:
                    return
                if not settings.translation_profiles():
                    return
                if translation_failure[0] is not None:
                    return
                rows = [{"id": s["id"], "text": s["text"], "start": s["start"], "end": s["end"]} for s in batch]
                # Completed units retain their validated request provenance; future context
                # is reconciled by the mandatory full-film audit, not by buying them again.
                missing = []
                for r in rows:
                    saved_unit = next((saved for key in translation_unit_keys(settings, r) if (saved := self.store.get(ident, key))), None)
                    if saved_unit:
                        translations[r["id"]] = saved_unit["text"]
                    else:
                        missing.append(r)
                rows = missing
                if not rows:
                    return
                preceding = [s for s in segments if s["end"] <= batch[0]["start"] and s["id"] not in {b["id"] for b in batch}][-4:]
                context = [{"source": s["text"], "translation": translations.get(s["id"], "")} for s in preceding]
                key = translation_key(settings, rows, context)
                saved = self.store.get(ident, key)
                if saved:
                    translations.update(saved)
                    return
                def translate(reservation=None):
                    context_token = JOB_CONTEXT.set(ident)
                    try:
                        result = self.client.translate(settings, rows, context, cancel, reservation=reservation) if reservation else self.client.translate(settings, rows, context, cancel)
                        self.store.put(ident, key, result)
                        for r in rows:
                            provider = getattr(result, "providers", {}).get(r["id"])
                            saved = {"text": result[r["id"]], "request": key}
                            self.store.put(ident, "unit:" + translation_key(settings, [r], None), saved)
                            if provider:
                                self.store.put(ident, "unit:" + translation_key(settings, [r], None, provider), {**saved, "profile": provider})
                        return result
                    finally:
                        JOB_CONTEXT.reset(context_token)
                self.translator.ensure_capacity(settings.translation_parallelism())
                prepare = self.client.translation_reserver(settings) if isinstance(self.client, Client) else None
                pending.append(self.translator.submit(ident, translate, max_parallel=settings.translation_parallelism(), prepare=prepare))

            def collect(block=False):
                while True:
                    check(cancel)
                    for f in [future for future in pending if future.done()]:
                        try:
                            translations.update(f.result())
                            last_translation[0] = time.monotonic()
                        except Cancelled:
                            raise
                        except cf.CancelledError:
                            pass
                        except Exception as exc:
                            translation_failure[0] = translation_failure[0] or exc
                            for future in pending:
                                if future is not f:
                                    future.cancel()
                        pending.remove(f)
                    now = time.monotonic()
                    if now-last_preview[0] >= 0.8 or (block and not pending):
                        valid_ids = {s["id"] for s in segments}
                        translated_count = len(valid_ids.intersection(translations))
                        self.previews[ident] = [{**s, "translation": translations.get(s["id"], "")} for s in segments]
                        self.callback("preview", {"job": ident, "recognized": len(segments), "translated": translated_count})
                        if asr_finished[0]:
                            stats = self.translator.stats(ident)
                            wait = int(now-last_translation[0])
                            detail = f"识别已完成 · 已译 {translated_count}/{len(valid_ids)} 段 · 请求执行/等待接口 {stats['running']}，排队 {stats['queued']}"
                            if pending:
                                detail += f" · 距上次返回 {wait}秒（超时 {settings.translate.timeout}秒）"
                            self.emit(ident, "翻译", 0.55 + 0.05 * translated_count/max(1,len(valid_ids)), detail)
                        last_preview[0] = now
                    if not block or not pending:
                        break
                    cancel.wait(0.1)

            # ASR cache identity includes language, model and segmentation settings.
            weights = Path(settings.model_path, "model.bin")
            model_identity = {"path": str(weights.resolve()), "size": weights.stat().st_size if weights.exists() else None, "mtime": weights.stat().st_mtime_ns if weights.exists() else None}
            asr_contract = {"source": settings.source, "model": model_identity, "preset": settings.preset, "chunk": settings.chunk_seconds, "overlap": settings.overlap, "audio": info["audio"]["index"], "glossary": settings.glossary}
            old_contract = self.store.get(ident, "asr_contract")
            if old_contract != asr_contract:
                self.store.clean(ident)
                self.store.put(ident, "asr_contract", asr_contract)
            self.store.put(ident, "media", {"duration": info["duration"], "audio_index": info["audio"]["index"], "streams": [{"index": s["index"], "type": s["codec_type"], "codec": s["codec_name"]} for s in info["streams"]]})
            pool = None
            events = None
            count = math.ceil(info["duration"] / settings.chunk_seconds)
            chunks = list(range(count))
            completed = set()
            for i in chunks:
                saved = self.store.get(ident, f"chunk:{i}")
                if saved is not None:
                    completed.add(i)
                    segments.extend(saved)
            segments.sort(key=lambda s: s["start"])
            for batch in split_rows(segments, max_chars=6000, max_count=20):
                submit_translation(batch)
                if len(pending) >= pending_limit:
                    collect(True)
            remaining = [i for i in chunks if i not in completed]
            in_progress = {}
            live_batches = {}
            batch_times = {}
            first_live_submission = [False]
            def flush_live(chunk=None, force=False):
                now = time.monotonic()
                for key in list(live_batches):
                    if chunk is not None and key != chunk:
                        continue
                    batch = live_batches[key]
                    if force or len(batch) >= 12 or sum(len(s['text']) for s in batch) >= 4000 or now-batch_times[key] >= 1.0:
                        submit_translation(batch)
                        del live_batches[key], batch_times[key]
            first_dispatch = True
            detected_language = self.store.get(ident, "asr_language") if settings.source == "auto" else settings.source
            if remaining:
                pool = self._ensure_pool(settings)
                events = pool.register(ident)
            def task(i, start=None, end=None):
                a = i * settings.chunk_seconds if start is None else start
                b = min(info["duration"], a + settings.chunk_seconds) if end is None else end
                return {"job": ident, "chunk": i, "source": job["source"], "audio_index": info["audio"]["index"], "ffmpeg": settings.ffmpeg, "language": detected_language or settings.source, "beam_size": {"quality":5,"balanced":3,"fast":1}.get(settings.preset,5), "glossary": settings.glossary, "core_start": a, "core_end": b, "decode_start": max(0, a - settings.overlap), "decode_end": min(info["duration"], b + settings.overlap)}
            while remaining or in_progress:
                check(cancel)
                if pool.fatal:
                    raise RuntimeError(pool.fatal)
                # Small submissions allow other videos to enter the shared GPU queue.
                parallel_windows = (1 if len(self.active) > 1 else len(settings.gpus)) if first_dispatch else (1 if len(getattr(pool, "queues", {ident: events})) > 1 else len(settings.gpus))
                if settings.source == "auto" and not detected_language:
                    parallel_windows = 1
                while remaining and len(in_progress) < parallel_windows:
                    i = remaining.pop(0)
                    in_progress[i] = []
                    pool.submit(task(i), cancel)
                first_dispatch = False
                try:
                    event = events.get(timeout=0.3)
                except queue.Empty:
                    flush_live()
                    collect()
                    continue
                kind, i = event["type"], event["chunk"]
                if kind == "segment":
                    s = event["segment"]
                    s["id"] = segment_id(s)
                    in_progress[i].append(s)
                    segments.append(s)
                    segments.sort(key=lambda s: s["start"])
                    # First stable sentence starts immediately. Following sentences form
                    # small batches with <=1s buffering, rather than one HTTP call each.
                    if not first_live_submission[0]:
                        submit_translation([s])
                        first_live_submission[0] = True
                    else:
                        live_batches.setdefault(i, []).append(s)
                        batch_times.setdefault(i, time.monotonic())
                        flush_live(i)
                    if len(pending) >= pending_limit:
                        collect(True)
                    else:
                        collect()
                    self.emit(ident, "识别 / 翻译并行", 0.05 + 0.50 * (len(completed) + min(1, s["end"] / settings.chunk_seconds % 1)) / count, f"已识别 {len(segments)} 段 / 已翻译 {len(translations)} 段")
                elif kind == "done":
                    flush_live(i, force=True)
                    self.store.put(ident, f"chunk:{i}", in_progress.pop(i))
                    completed.add(i)
                elif kind == "language":
                    if settings.source == "auto" and not detected_language:
                        detected_language = event["language"]
                        self.store.put(ident, "asr_language", detected_language)
                    self.callback("language", {"job": ident, "language": event["language"], "probability": event["probability"]})
                elif kind in ("error", "cancelled"):
                    if kind == "cancelled":
                        raise Cancelled()
                    raise RuntimeError(event["error"])
            flush_live(force=True)
            asr_finished[0] = True
            collect()
            if not segments:
                raise RuntimeError("未识别到有效对白，未生成带字幕成品")

            # Re-read small boundary regions with both sides present; replace provisional units.
            boundaries = [i * settings.chunk_seconds for i in range(1, count)]
            suspicious = [b for b in boundaries if any(abs(s["start"]-b)<2.5 or abs(s["end"]-b)<2.5 for s in segments)]
            for n, boundary in enumerate(suspicious):
                saved = self.store.get(ident, f"boundary:{boundary}")
                if saved is None:
                    if pool is None:
                        pool = self._ensure_pool(settings)
                        events = pool.register(ident)
                    index = -int(boundary)-1
                    pool.submit(task(index, max(0, boundary - 8), min(info["duration"], boundary + 8)), cancel)
                    saved = []
                    while True:
                        check(cancel)
                        if pool.fatal:
                            raise RuntimeError(pool.fatal)
                        try:
                            e = events.get(timeout=0.3)
                        except queue.Empty:
                            collect()
                            continue
                        if e["chunk"] != index:
                            continue
                        if e["type"] == "segment":
                            s = e["segment"]
                            s["id"] = segment_id(s)
                            saved.append(s)
                        elif e["type"] == "done":
                            break
                        elif e["type"] in ("error", "cancelled"):
                            raise RuntimeError(e.get("error", "边界识别已取消"))
                    self.store.put(ident, f"boundary:{boundary}", saved)
                if saved:
                    # Only replace units entirely inside the repair window; extend replacement
                    # to any original segment touched, so no half sentence silently disappears.
                    touching = [s for s in segments if s["start"] < boundary + 6 and s["end"] > boundary - 6]
                    if touching:
                        a, b = min(s["start"] for s in touching), max(s["end"] for s in touching)
                        # Keep old units if the narrow re-read cannot cover their extent.
                        if saved[0]["start"] <= a + 0.6 and saved[-1]["end"] >= b - 0.6:
                            old_ids = {s["id"] for s in touching}
                            segments = [s for s in segments if s["id"] not in old_ids] + saved
                            segments.sort(key=lambda s: s["start"])
                            for batch in split_rows(saved, max_count=8):
                                submit_translation(batch)
                self.emit(ident, "边界复核", 0.56, f"连接处 {n+1}/{len(suspicious)}")
            collect(True)
            self.store.put(ident, "final_segments", segments)
            self.store.put(ident, "final_translations", translations)
            if pool is not None:
                pool.unregister(ident)

            if not settings.translation_profiles():
                raise APIError("本地识别已完成并保存文字；请填写 API 地址和模型，保存后重试将复用识别结果")
            if translation_failure[0] is not None:
                raise APIError("本地识别已完成；翻译等待恢复：" + str(translation_failure[0]))

            rows = [{"id": s["id"], "source": s["text"], "translation": translations[s["id"]]} for s in segments]
            local = settings.llm_mode == "local"
            brief_chars = max(500, min(4000, (settings.local_model.context-settings.local_review.max_tokens-2500)//3)) if local else 18000
            audit_chars = max(1000, min(6000, (settings.local_model.context-settings.local_review.max_tokens-2500-brief_chars)//2)) if local else 18000
            def review_hash(content):
                return hashlib.sha256(json.dumps({"rows": content, "profile": profile_identity(settings.review or settings.translate), "glossary": settings.glossary, "target": settings.target}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            review_identity = review_hash(rows)
            cached_review = self.store.get(ident, "review-once-v1:" + review_identity)
            def audit_call(kind, content, call):
                key = "audit-v2:" + kind + ":" + review_hash(content)
                cached = self.store.get(ident, key)
                if cached is not None:
                    return cached["value"]
                value = call()
                self.store.put(ident, key, {"value": value})
                return value

            issues = []
            if cached_review:
                translations.update(cached_review["translations"])
                issues = cached_review["issues"]
            else:
                self.emit(ident, "全片审校", 0.60, "整理全片上下文，逐段覆盖检查")
                state_key = "audit-once-state-v1:" + review_identity
                audit_state = self.store.get(ident, state_key)
                legacy_review = self.store.get(ident, "review:" + review_identity)
                if audit_state is None and legacy_review:
                    audit_state = {"brief": "", "translations": legacy_review["translations"], "issues": legacy_review["issues"], "repaired": []}
                if audit_state:
                    translations.update(audit_state["translations"])
                    brief = audit_state["brief"]
                    issues = audit_state["issues"]
                else:
                    briefs = []
                    brief_batches = list(split_rows(rows, max_chars=audit_chars if local else 24000, max_count=20 if local else 100))
                    for i, batch in enumerate(brief_batches):
                        briefs.append(audit_call("brief", batch, lambda: self.client.brief(settings, batch, cancel))[:2000])
                        self.emit(ident, "整理全片上下文", 0.60, f"上下文 {i+1}/{len(brief_batches)}")
                    for reduction in range(8):
                        if len("\n".join(briefs)) <= brief_chars:
                            break
                        before = len("\n".join(briefs))
                        summary_cap = max(120, min(1000, brief_chars//2)) if local else 2000
                        briefs = [audit_call("brief", batch, lambda: self.client.brief(settings, batch, cancel))[:summary_cap] for batch in split_rows([{"brief": b} for b in briefs], max_chars=audit_chars if local else 18000, max_count=4 if local else 8)]
                        if len("\n".join(briefs)) >= before:
                            raise APIError("全片上下文未能压缩到模型容量内；已保留恢复点，请增加上下文长度")
                    if len("\n".join(briefs)) > brief_chars:
                        raise APIError("全片上下文超过模型容量；已保留恢复点，请增加上下文长度")
                    brief = "\n".join(briefs)
                    issues = []
                    batches = list(split_rows(rows, max_chars=audit_chars, max_count=12 if local else 40))
                    self.emit(ident, f"全片审校 0/{len(batches)}", 0.60, "逐段覆盖检查，等待首批审校结果")
                    for i, batch in enumerate(batches):
                        check(cancel)
                        # Include neighbors to audit sentence/terminology continuity across windows.
                        offset = rows.index(batch[0])
                        context_rows = rows[max(0, offset-3): min(len(rows), offset+len(batch)+3)]
                        issues.extend(audit_call("review", {"round": 0, "rows": context_rows, "brief": brief}, lambda: self.client.review(settings, context_rows, brief, cancel)))
                        self.emit(ident, f"全片审校 {i+1}/{len(batches)}", 0.60 + 0.12 * (i+1)/len(batches), f"一次审校 · 覆盖 {i+1}/{len(batches)}")
                    # Neighboring windows may report the same issue more than once.
                    issues = list({json.dumps(x, sort_keys=True, ensure_ascii=False): x for x in issues}.values())
                    audit_state = {"brief": brief, "translations": translations, "issues": issues, "repaired": []}
                self.store.put(ident, state_key, audit_state)
                affected = {j for x in issues for j in x["ids"]}
                repaired = set(audit_state["repaired"])
                repair_settings = settings.repair_settings()
                repair_batches = list(split_rows([s for s in segments if s["id"] in affected-repaired], max_count=8))
                for i, batch in enumerate(repair_batches):
                    batch_ids = {s["id"] for s in batch}
                    relevant = [x for x in issues if batch_ids.intersection(x["ids"])]
                    instruction = json.dumps(relevant, ensure_ascii=False)
                    begin = max(0, segments.index(batch[0])-4)
                    end = min(len(segments), segments.index(batch[-1])+5)
                    context = {"global_brief": brief, "neighbors": [{"source": s["text"], "translation": translations.get(s["id"], "")} for s in segments[begin:end]]}
                    repair_rows = [{"id": s["id"], "text": s["text"]} for s in batch]
                    fixed = audit_call("repair-once", {"translation_policy": translation_key(repair_settings, [], None), "source": settings.source, "prompt": settings.prompt, "rows": repair_rows, "context": context, "instruction": instruction}, lambda: self.client.translate(repair_settings, repair_rows, context, cancel, instruction))
                    translations.update(fixed)
                    repaired.update(batch_ids)
                    audit_state.update(translations=translations, repaired=sorted(repaired))
                    self.store.put(ident, state_key, audit_state)
                    self.emit(ident, "自动修正", 0.72 + 0.06 * (i+1)/len(repair_batches), f"修正 {i+1}/{len(repair_batches)} · 完成后直接导出")
                rows = [{"id": s["id"], "source": s["text"], "translation": translations[s["id"]]} for s in segments]
                self.store.put(ident, "review-once-v1:" + review_identity, {"translations": translations, "issues": issues})
                self.store.put(ident, "review-once-v1:" + review_hash(rows), {"translations": translations, "issues": issues})
            for s in segments:
                r = {"id": s["id"], "text": s["text"], "start": s["start"], "end": s["end"]}
                self.store.put(ident, "unit:" + translation_key(settings, [r], None), {"text": translations[s["id"]], "request": "full-film-reviewed"})
            self.store.put(ident, "final_translations", translations)
            self.store.put(ident, "issues", issues)
            self.previews[ident] = [{**s, "translation": translations.get(s["id"], "")} for s in segments]

            self.emit(ident, "翻译文件名", 0.80)
            title_key = "title:" + hashlib.sha256(json.dumps({"profile": profile_identity(settings.review_profile()), "target": settings.target, "stem": Path(job["source"]).stem}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            title = self.store.get(ident, "manual_title") or self.store.get(ident, title_key)
            if not title:
                title = preserve_filename_tokens(Path(job["source"]).stem, self.client.title(settings, Path(job["source"]).stem, cancel))
                self.store.put(ident, title_key, title)
            cues = format_cues(segments, translations, settings.bilingual, settings.target)
            output, lock = reserve_output(directory, title)
            reserved = (output, lock)
            self.emit(ident, "等待烧录", 0.82)
            from .segmented import TimelineError, burn_segmented, enabled as segment_enabled
            segmented_done = False
            if segment_enabled(info, settings):
                segment_progress = [0.0]
                def wait_segment(status):
                    self.emit(ident, "分段编码", .83+.16*segment_progress[0], f"分段等待编码槽 · 正在编码 {status['active']}/{status['limit']} 路")
                def progress_segment(value):
                    segment_progress[0] = max(segment_progress[0], value)
                    self.emit(ident, "分段编码", .83+.16*segment_progress[0], f"分段NVENC · 按序写入成品 {segment_progress[0]*100:.1f}%")
                def segment_worker(event, gpu, index, total):
                    self.callback("gpu", {"type": event, "gpu": gpu, "job": ident, "worker": index})
                self.emit(ident, "分段编码", .83, "检查完整视频时间轴 · 准备并行烧录")
                try:
                    with temporary_subtitles(directory, cues) as subtitle:
                        burn_segmented(job["source"], output, subtitle, info, settings, cancel, self.encoding_slots, ident,
                            progress_segment, segment_worker, wait_segment)
                    segmented_done = True
                except TimelineError:
                    self.emit(ident, "烧录字幕", .83, "分段时间轴不适合拼接，改用单路GPU保持同步")
            if not segmented_done:
                self._burn_single(job, info, settings, cancel, cues, output, directory)
            lock.unlink(missing_ok=True)
            self.store.update(ident, "完成", f"{len(cues)} 条字幕；一次审校，已自动处理 {len(issues)} 项修订；运行缓存已清理", output)
            self.store.clean(ident)
            self.previews.pop(ident, None)
            self.callback("complete", {"job": ident, "output": str(output), "issues": len(issues)})
        except Cancelled:
            cancel.set()
            if ident in self.deleting:
                self.store.update(ident, "删除中", "正在清理运行缓存")
            else:
                self.store.update(ident, "已暂停", "文字恢复点已保存；恢复时仅重解码未完成音频窗口")
        except Exception as exc:
            cancel.set()
            self.store.update(ident, "等待处理", str(exc))
            self.callback("error", {"job": ident, "error": str(exc)})
        finally:
            for f in pending:
                f.cancel()
            # Let in-flight calls finish before task checkpoint cleanup/retry can race them.
            for f in pending:
                try:
                    f.result(timeout=settings.translate.timeout + 5)
                except BaseException:
                    pass
            if self.pool is not None:
                self.pool.unregister(ident)
            if reserved:
                reserved[1].unlink(missing_ok=True)
            with self.lock:
                self.active.pop(ident, None)
                self.last_persist.pop(ident, None)
                if ident in self.deleting:
                    self.store.delete([ident])
                    self.deleting.discard(ident)
                    self.previews.pop(ident, None)
            self.callback("refresh", {})
            JOB_CONTEXT.reset(job_context)
            self._pump()

    def _burn_single(self, job, info, settings, cancel, cues, output, directory):
        ident = job["id"]
        encoder, video_settings = self._acquire_encoder(info, settings, cancel,
            lambda status: self.emit(ident, "等待烧录", .82, f"编码排队第 {status['position']} 个 · 正在编码 {status['active']}/{status['limit']} 路"))
        plan = None
        try:
            plan = video_plan(info, video_settings)
            if plan["encode"] or plan["decode"]:
                self.callback("gpu", {"type": "encoding_started", "gpu": plan["gpu"], "job": ident, "encoding": plan["encode"]})
            mode = "NVDEC 解码" if plan["decode"] else "CPU 解码"
            mode += f" / {video_settings.video_codec.upper()} NVENC" if plan["encode"] else " / CPU 编码" + ("（保留静态HDR）" if plan["hdr"] else "")
            self.emit(ident, "烧录字幕", 0.83)
            with temporary_subtitles(directory, cues) as subtitle:
                burn(job["source"], output, subtitle, info, video_settings, cancel, lambda p: self.emit(ident, "烧录字幕", 0.83 + 0.16 * min(1, max(0, p)), f"{mode} · 编码 {p*100:.1f}%"))
        finally:
            if plan and (plan["encode"] or plan["decode"]):
                self.callback("gpu", {"type": "encoding_done", "gpu": plan["gpu"], "job": ident})
            encoder.release()

    def shutdown(self):
        self.pause()
        if self.pool:
            self.pool.shutdown()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.translator.shutdown(wait=False, cancel_futures=True)
        if isinstance(self.client, Client) and self.client.local_runtime:
            self.client.local_runtime.stop()
