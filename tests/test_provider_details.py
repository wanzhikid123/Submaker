import dataclasses
import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QProgressBar
from PySide6.QtCore import QCoreApplication, QEvent

from submaker.config import Profile, Settings
from submaker.gui import MainWindow
from submaker.llm import Client, profile_tag
from submaker.store import Store
from submaker.telemetry import provider_detail_lines


def settings_and_row():
    settings = Settings(translate=Profile(name="ChatGPT", url="http://localhost/chat", model="chat", concurrency=1, rpm=3), translators=[Profile(name="Gemini", url="http://localhost/gemini", model="gemini", concurrency=5, rpm=10, key_id="gemini"), Profile(name="OpenRouter", url="http://localhost/router", model="router", concurrency=2, rpm=10, key_id="router")])
    return settings, {"id": "fixture", "state": "处理中", "config": json.dumps(settings.as_dict()), "api_providers": []}


def test_enabled_providers_stay_in_place_through_request_idle_and_rpm_wait():
    settings, row = settings_and_row()
    client = Client(key_provider=lambda _: "")
    names = [p.name for p in settings.translation_profiles()]
    idle = provider_detail_lines(row, quotas=client.quota_snapshot(settings))
    assert [line.split("：")[0] for line in idle] == names
    assert all("待命" in line and "429 0次" in line for line in idle)
    active = [{"profile_id": profile_tag(settings.translate), "profile_name": "ChatGPT", "model": "chat", "phase": "request", "started": time.monotonic()-8}]
    running = provider_detail_lines(row, active, client.quota_snapshot(settings))
    assert [line.split("：")[0] for line in running] == names
    assert "请求中" in running[0] and "8秒" in running[0]
    quotas = client.quota_snapshot(settings)
    quotas[1].update(rpm_used=10, rpm_wait_seconds=6)
    paced = provider_detail_lines(row, quotas=quotas, queued=2)
    assert [line.split("：")[0] for line in paced] == names
    assert "RPM节流 6秒" in paced[1] and "RPM 10/10" in paced[1]
    assert [line.split("：")[0] for line in provider_detail_lines(row)] == names


def test_old_gemini_429_is_retained_without_marking_current_config_as_limited():
    settings, row = settings_and_row()
    current = settings.translators[0]
    older = dataclasses.replace(current, model="older-model")
    row["api_providers"] = [{"profile": profile_tag(current), "name": "Gemini", "state": "success", "rate_limits": 0}, {"profile": profile_tag(older), "name": "Gemini", "state": "rate_limit", "rate_limits": 3, "retry_at": time.time()+30}]
    lines = provider_detail_lines(row)
    assert "Gemini：待命" in lines[1] and "429 0次" in lines[1]
    assert lines[-1].startswith("Gemini（历史配置）：429 3次")
    assert "当前任务未使用此配置" in lines[-1]
    assert all("429退避" not in line for line in lines)


def test_translation_and_review_profiles_with_same_name_have_separate_activity():
    settings, row = settings_and_row()
    settings.review = dataclasses.replace(settings.translators[0], model="review")
    row["config"] = json.dumps(settings.as_dict())
    client = Client(key_provider=lambda _: "")
    activity = [{"profile_id": profile_tag(settings.review), "profile_name": "Gemini", "model": "review", "phase": "request", "started": time.monotonic()}]
    lines = provider_detail_lines(row, activity, client.quota_snapshot(settings))
    assert "Gemini：待命 · 请求 0" in lines[1]
    assert "Gemini（全片审校）：请求中 · 请求 1" in lines[-1]


def test_quota_inspection_does_not_resolve_dns_create_gates_or_change_limits(monkeypatch):
    settings, row = settings_and_row()
    settings.translate.url = "http://private.invalid/v1"
    settings.translate.allow_lan_http = True
    client = Client(key_provider=lambda _: "")
    monkeypatch.setattr("socket.getaddrinfo", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("UI inspection must not resolve DNS")))
    quotas = client.quota_snapshot(settings)
    assert len(quotas) == 3 and not client.limits.slots
    assert quotas[0]["rpm_used"] == 0 and quotas[0]["effective_limit"] == 1


def test_task_details_keep_all_enabled_names_after_active_request_disappears(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    settings, _ = settings_and_row()
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: settings))
    monkeypatch.setattr("submaker.gui.Store", lambda: Store(tmp_path/"jobs.sqlite"))
    window = MainWindow()
    try:
        source = tmp_path/"fixture.mp4"
        source.write_bytes(b"fixture")
        job = window.store.add(source, settings)
        video = {"index": 0, "codec_name": "h264", "codec_type": "video", "pix_fmt": "yuv420p", "bit_rate": "2000000", "width": 1280, "height": 720}
        window.media_info[job] = {"info": {"video": video, "streams": [video], "duration": 600, "format": {"bit_rate": "2000000"}}}
        window.scheduler.client.activity_snapshot = lambda: [{"job": job, "profile_id": profile_tag(settings.translate), "profile_name": "ChatGPT", "model": "chat", "phase": "request", "started": time.monotonic()}]
        window.refresh()
        assert "ChatGPT：请求中" in window.queue.item(0, 5).text()
        window.scheduler.client.activity_snapshot = lambda: []
        window.refresh()
        text = window.queue.item(0, 5).text()
        for name in ("ChatGPT", "Gemini", "OpenRouter"):
            assert text.count(name+"：") == 1
        assert "ChatGPT：待命" in text
    finally:
        window.close()
        window.store.close()
        # Destroy the widget cycle on the GUI thread before later HTTP test threads
        # can trigger Python GC and dispose of Qt objects from a worker thread.
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_review_count_and_terminal_failure_are_visible_without_other_provider_erasing_them(tmp_path, monkeypatch):
    settings, _ = settings_and_row()
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: settings))
    monkeypatch.setattr("submaker.gui.Store", lambda: Store(tmp_path/"jobs.sqlite"))
    window = MainWindow()
    try:
        source = tmp_path/"fixture.mp4"
        source.write_bytes(b"fixture")
        job = window.store.add(source, settings)
        window.media_info[job] = {"error": "fixture"}
        window.store.update(job, "处理中", "一次审校 · 覆盖 12/17")
        window.on_event("progress", {"job": job, "stage": "全片审校 12/17", "progress": .68})
        window.refresh()
        bar = window.queue.cellWidget(0, 2).findChild(QProgressBar)
        assert bar.format() == "全片审校 12/17" and bar.value() == 68
        error = "Gemini · 翻译：内容策略拦截（content_filter: PROHIBITED_CONTENT）；已保留恢复点"
        window.store.record_api(job, {"event": "paused", "profile_id": profile_tag(settings.translators[0]), "profile_name": "Gemini", "code": 200, "message": "内容策略拦截（content_filter: PROHIBITED_CONTENT）"})
        window.store.update(job, "等待处理", error)
        window.on_event("error", {"job": job, "error": error})
        assert not window.api_banner.isHidden()
        assert "Gemini" in window.api_banner.text() and "PROHIBITED_CONTENT" in window.api_banner.text()
        assert window.api_banner.text().count("PROHIBITED_CONTENT") == 1
        assert window.queue.cellWidget(0, 2).findChild(QProgressBar).format() == "等待处理"
        assert "Gemini：内容拦截" in window.queue.item(0, 5).text()
        window.store.record_api(job, {"event": "success", "profile_id": profile_tag(settings.translate), "profile_name": "ChatGPT", "code": 200})
        window.refresh()
        assert not window.api_banner.isHidden() and "PROHIBITED_CONTENT" in window.api_banner.text()
        # Even a local failure without an API status event needs an immediate banner.
        with window.store.db:
            window.store.db.execute("DELETE FROM api_providers WHERE job=?", (job,))
        window.store.update(job, "等待处理", "输出目录不可写；恢复点已保存")
        window.on_event("error", {"job": job, "error": "输出目录不可写；恢复点已保存"})
        assert "输出目录不可写" in window.api_banner.text()
    finally:
        window.close()
        window.store.close()


def test_pool_switching_retains_details_and_429_counts_but_only_final_failure_raises_banner(tmp_path, monkeypatch):
    settings, _ = settings_and_row()
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: settings))
    monkeypatch.setattr("submaker.gui.Store", lambda: Store(tmp_path/"jobs.sqlite"))
    window = MainWindow()
    try:
        source = tmp_path/"fixture.mp4"
        source.write_bytes(b"fixture")
        job = window.store.add(source, settings)
        window.media_info[job] = {"error": "fixture"}
        window.store.update(job, "处理中", "即时翻译")
        profile = settings.translators[0]
        window.store.record_api(job, {"event": "rate_limit", "profile_id": profile_tag(profile), "profile_name": profile.name,
                                     "pool_failover": True, "code": 429, "message": "HTTP 429：接口限流", "retry_at": time.time()+60})
        window.refresh()
        assert window.api_banner.isHidden()
        text = window.queue.item(0, 5).text()
        assert "Gemini：已切换其他接口" in text and "HTTP 429" in text and "429 1次" in text
        error = "所有启用的翻译接口均不可用：ChatGPT；Gemini；OpenRouter"
        window.store.update(job, "等待处理", error)
        window.on_event("error", {"job": job, "error": error})
        assert not window.api_banner.isHidden()
        assert all(p.name in window.api_banner.text() for p in settings.translation_profiles())
    finally:
        window.close()
        window.store.close()
