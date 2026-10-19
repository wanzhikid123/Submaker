from __future__ import annotations

import csv
import io
import json
import subprocess
import threading
import time

from .media import CREATE_NO_WINDOW
from .config import Settings
from .llm import profile_tag


def read_gpu_status():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,utilization.encoder,utilization.decoder", "--format=csv,noheader,nounits"], capture_output=True, text=True, creationflags=CREATE_NO_WINDOW, timeout=3)
    if result.returncode:
        raise RuntimeError("GPU状态读取失败")
    rows = {}
    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row) != 6:
            continue
        def number(index):
            try:
                return float(row[index].strip())
            except ValueError:
                return None
        gpu = int(row[0].strip())
        rows[gpu] = dict(zip(("utilization", "memory_used", "memory_total", "encoder", "decoder"), (number(i) for i in range(1, 6))))
        rows[gpu]["updated"] = time.time()
    return rows


class GpuMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.rows = {}
        self.stopped = threading.Event()
        threading.Thread(target=self._poll, name="gpu-status", daemon=True).start()

    def _poll(self):
        while not self.stopped.is_set():
            started = time.monotonic()
            try:
                rows = read_gpu_status()
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
                rows = {}
            with self.lock:
                self.rows = rows
            self.stopped.wait(max(0.1, 1.0-(time.monotonic()-started)))

    def snapshot(self):
        with self.lock:
            return {gpu: dict(row) for gpu, row in self.rows.items()}

    def stop(self):
        self.stopped.set()


def gpu_role(gpu, metrics, states, enabled, asr_jobs=None, encoding_jobs=None, job=None):
    asr_jobs, encoding_jobs = asr_jobs or {}, encoding_jobs or {}
    values, state = metrics.get(gpu, {}), states.get(gpu)
    if gpu not in enabled:
        return "未启用"
    if gpu in encoding_jobs:
        return "视频编解码" if job is None or encoding_jobs[gpu] == job else "其他任务编解码"
    if gpu in asr_jobs:
        role = "音轨准备" if state == "extracting" else "Whisper识别"
        return "其他任务"+role if job is not None and asr_jobs[gpu] != job else role
    if state == "error":
        return "异常"
    if values.get("encoder", 0) or values.get("decoder", 0):
        return "视频编解码"
    if values.get("utilization", 0):
        return "运行中"
    return "待命" if values else "读取状态中"


def gpu_details(metrics, states, enabled, asr_jobs=None, encoding_jobs=None, job=None, video_util=False):
    lines = []
    for gpu in (0, 1):
        values = metrics.get(gpu, {})
        role = gpu_role(gpu, metrics, states, enabled, asr_jobs, encoding_jobs, job)
        utilization = "—" if values.get("utilization") is None else f"{values['utilization']:.0f}%"
        memory = f" · 显存 {values['memory_used']/1024:.1f}/{values['memory_total']/1024:.0f} GiB" if values.get("memory_used") is not None and values.get("memory_total") else ""
        lines.append(f"GPU {gpu}：{role} · 利用率 {utilization}{memory}")
        if video_util:
            encoder = "—" if values.get("encoder") is None else f"{values['encoder']:.0f}%"
            decoder = "—" if values.get("decoder") is None else f"{values['decoder']:.0f}%"
            lines.append(f"  解码 {decoder} · 编码 {encoder}")
    return "\n".join(lines)


def api_notice(row, now=None):
    now = time.time() if now is None else now
    remaining = max(0, int((row.get("api_retry_at") or 0)-now+.999))
    limit, configured = row.get("api_effective_limit") or 0, row.get("api_configured_limit") or 0
    count = row.get("api_rate_limits") or 0
    parallel = f" · 实际并发上限 {limit}/{configured}" if configured else ""
    if row.get("api_state") in ("paused", "quota") and "额度不足" in (row.get("api_message") or ""):
        return row["api_message"] + parallel
    if row.get("state") == "等待处理" and "HTTP 429" in (row.get("detail") or ""):
        return row["detail"] + parallel
    if remaining:
        return f"HTTP 429 限流退避 · {remaining}秒后可重试{parallel}"
    if row.get("api_state") in ("paused", "quota"):
        return (row.get("api_message") or "接口限流，任务已暂停") + parallel
    if row.get("api_state") in ("timeout", "server_error"):
        return (row.get("api_message") or "接口请求异常，正在重试") + parallel
    rpm_remaining = max(0, int((row.get("api_rpm_ready_at") or 0)-now+.999))
    if row.get("api_state") == "queued" and rpm_remaining:
        return f"RPM节流 · {rpm_remaining}秒后发送 · 已用 {row.get('api_rpm_used', 0)}/{row.get('api_rpm_limit', 0)} RPM" + parallel
    if count or row.get("api_partial"):
        history = "已知曾限流（旧版未记录次数）" if row.get("api_partial") and not count else f"429已记录 {count}次" + ("，历史次数不完整" if row.get("api_partial") else "")
        return history + (" · 已恢复" if row.get("api_state") == "success" else " · 等待接口") + parallel
    return ""


def api_provider_records(row):
    providers = row.get("api_providers") or []
    if not providers:
        return [("", row)]
    if row.get("config"):
        settings = Settings.from_dict(json.loads(row["config"]))
        enabled = {profile_tag(p) for p in [*settings.translation_profiles(), settings.review_profile()] if p.url and p.model}
        providers = [p for p in providers if p["profile"] in enabled]
    return [(p["name"], {"api_"+key: value for key, value in p.items()}) for p in providers]
