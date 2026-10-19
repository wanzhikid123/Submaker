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
    if row.get("api_state") == "switching":
        return "已尝试切换其他翻译接口 · " + (row.get("api_message") or "原接口暂不可用") + parallel
    if row.get("api_state") in ("paused", "quota") and row.get("api_message"):
        return row["api_message"] + parallel
    if row.get("state") == "等待处理" and "HTTP 429" in (row.get("detail") or ""):
        return row["detail"] + parallel
    if remaining:
        return f"HTTP 429 限流退避 · {remaining}秒后可重试{parallel}"
    if row.get("api_state") in ("paused", "quota"):
        return (row.get("api_message") or "接口异常，任务等待处理") + parallel
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


def provider_detail_lines(row, activity=(), quotas=(), queued=0, now=None):
    """One stable row per enabled task profile; retain retired profiles' 429 counts."""
    now = time.time() if now is None else now
    settings = Settings.from_dict(json.loads(row.get("config") or "{}"))
    profiles = [(p, p.name) for p in settings.translation_profiles()]
    current = {profile_tag(p) for p, _ in profiles}
    review = settings.review_profile()
    if review.url and review.model and profile_tag(review) not in current:
        profiles.append((review, review.name+"（全片审校）"))
        current.add(profile_tag(review))
    records = {p["profile"]: p for p in row.get("api_providers", [])}
    lines = []
    for profile, name in profiles:
        tag = profile_tag(profile)
        record = records.get(tag, {})
        status = {"api_"+key: value for key, value in record.items()}
        quota = next((q for q in quotas if q.get("profile") == tag or not q.get("profile") and q.get("name") == profile.name), {})
        active = [r for r in activity if r.get("profile_id") == tag or not r.get("profile_id") and r.get("profile_name") == profile.name and r.get("model") == profile.model]
        running = sum(r.get("phase", "request") == "request" for r in active)
        waiting = len(active)-running
        retry = max(0, int((quota["retry_seconds"] if "retry_seconds" in quota else record.get("retry_at", 0)-now)+.999))
        rpm_wait = max(0, int((quota["rpm_wait_seconds"] if "rpm_wait_seconds" in quota else record.get("rpm_ready_at", 0)-now)+.999))
        limit = quota.get("effective_limit", record.get("effective_limit") or profile.concurrency)
        rpm = quota.get("rpm_limit", profile.rpm)
        used = quota.get("rpm_used", record.get("rpm_used", 0) if now-record.get("updated", 0) < 60 else None)
        fault = quota.get("fault") or (record.get("message") if record.get("state") in ("paused", "quota", "timeout", "server_error", "switching") else "")
        if row.get("state", "").startswith("完成"):
            state = "已完成"
        elif record.get("state") == "switching":
            state = "已切换其他接口"
        elif fault:
            state = "内容拦截" if "内容策略拦截" in fault else "拒绝请求" if "模型拒绝" in fault else "额度不足" if "额度不足" in fault else "接口异常"
        elif retry:
            state = f"429退避 {retry}秒"
        elif running:
            state = "请求中"
        elif (waiting or queued) and rpm_wait:
            state = f"RPM节流 {rpm_wait}秒"
        elif waiting or queued and quota.get("active", 0) >= limit:
            state = "等待并发名额"
        elif row.get("state") in ("已暂停", "等待恢复", "等待处理"):
            state = "任务暂停" if row["state"] != "等待处理" else "任务待处理"
        else:
            state = "待命"
        rpm_text = f"RPM {used if used is not None else '—'}/{rpm}" if rpm else "RPM 不限"
        count = record.get("rate_limits", 0)
        line = f"{name}：{state} · 请求 {running} / 等待 {waiting} · 并发 {limit}/{profile.concurrency} · {rpm_text} · 429 {count}次"
        if active:
            longest = max(0, int(time.monotonic()-min(r.get("started", time.monotonic()) for r in active)))
            line += f" · {longest}秒"
        if fault and not row.get("state", "").startswith("完成"):
            line += "\n  "+fault
        lines.append(line)
    historical = {}
    for tag, record in records.items():
        if tag not in current and record.get("rate_limits"):
            name = record.get("name") or "接口"
            historical[name] = historical.get(name, 0)+record["rate_limits"]
    for name, count in sorted(historical.items()):
        lines.append(f"{name}（历史配置）：429 {count}次 · 当前任务未使用此配置")
    if not records and (row.get("api_rate_limits") or row.get("api_partial")):
        lines.append("历史接口记录："+api_notice(row, now))
    return lines
