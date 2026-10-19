"""Render demo windows with isolated storage, never loading production jobs."""
import os
import tempfile
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtGui import QFontDatabase
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QTabWidget
from submaker import gui
from submaker.config import Settings, Profile
from submaker.store import Store
from submaker.llm import profile_tag

app = QApplication([])
# The Qt offscreen plugin does not enumerate Windows system fonts.
for name in ("msyh.ttc", "msyhbd.ttc", "segoeui.ttf"):
    font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name
    if font.exists():
        QFontDatabase.addApplicationFont(str(font))
gui.apply_theme(app)
destination = Path(__file__).resolve().parents[1] / "docs"
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    store = Store(root / "preview.sqlite")
    settings = Settings()
    settings.translate.name = "MiniServer"
    settings.translate.url, settings.translate.model = "https://example.com/v1", "configured-model"
    settings.translators = [Profile(name="OpenAI", url="https://api.openai.com/v1", model="your-model-id", provider="openai", concurrency=4, rpm=60, key_id="demo-openai"), Profile(name="Google AI", url="https://generativelanguage.googleapis.com/v1beta/openai", model="your-model-id", provider="google", concurrency=3, rpm=30, key_id="demo-google")]
    jobs = []
    for name in ("京都散步 · Kyoto Walk.mp4", "海洋故事 · Ocean Stories.mp4", "晨光 · Morning Light.mp4"):
        source = root / name
        source.write_bytes(b"design fixture")
        jobs.append(store.add(source, settings))
    store.update(jobs[0], "处理中", "NVDEC 解码 / HEVC NVENC · 编码 62.4%")
    store.update(jobs[1], "处理中", "边识别边翻译 · 已译 356/412 段")
    store.record_api(jobs[1], {"profile_id": profile_tag(settings.translate), "profile_name": "MiniServer", "event": "rate_limit", "code": 429, "message": "接口限流", "retry_at": time.time()+20, "effective_limit": 1, "configured_limit": 2})
    store.record_api(jobs[1], {"profile_id": profile_tag(settings.translators[0]), "profile_name": "OpenAI", "event": "success", "code": 200, "effective_limit": 4, "configured_limit": 4, "rpm_limit": 60, "rpm_used": 16})
    store.add_usage(jobs[0], {"prompt_tokens": 43120, "completion_tokens": 18470, "total_tokens": 61590})
    store.add_usage(jobs[1], {"prompt_tokens": 28410, "completion_tokens": 12340, "total_tokens": 40750})
    # Sparse design fixture: show a completed output size without generating media.
    finished = root/"晨光.mkv"
    with finished.open("wb") as stream:
        stream.truncate(492*1024**2)
    store.update(jobs[2], "完成", "一次审校与自动修订完成；运行缓存已清理", finished)
    store.add_usage(jobs[2], {"prompt_tokens": 22150, "completion_tokens": 11840, "total_tokens": 33990})
    with store.db:
        for ident, count in zip(jobs, (42, 31, 27)):
            store.db.execute("UPDATE usage SET requests=?,usage_reports=?,output_reports=? WHERE job=?", (count, count, count, ident))
    gui.Store = lambda: store
    Settings.load = classmethod(lambda cls: settings)
    class DemoGpuMonitor:
        def snapshot(self):
            return {0: {"utilization": 44, "memory_used": 5376, "memory_total": 16384, "encoder": 41, "decoder": 24}, 1: {"utilization": 92, "memory_used": 4992, "memory_total": 16384, "encoder": 0, "decoder": 0}}
        def stop(self):
            pass
    gui.GpuMonitor = DemoGpuMonitor
    demo_video={"index":0,"codec_name":"h264","codec_type":"video","pix_fmt":"yuv420p","width":1280,"height":720,"bit_rate":"2000000"}
    demo_info={"video":demo_video,"streams":[demo_video],"duration":3600,"format":{"bit_rate":"2000000"}}
    gui.probe = lambda *args, **kwargs: demo_info
    window = gui.MainWindow()
    window.media_info={ident:{"job":ident,"info":{"video":demo_video,"streams":[demo_video],"duration":3600,"format":{"bit_rate":"2000000"}}} for ident in jobs}
    window.resize(1500, 930)
    window.progress = {jobs[0]: {"stage": "烧录字幕", "progress": .93}, jobs[1]: {"stage": "即时翻译", "progress": .48}}
    window.gpu_states = {0: "ready", 1: "transcribing"}
    window.encoding_jobs = {0: jobs[0]}
    window.asr_jobs = {1: jobs[1]}
    window.scheduler.client.activity_snapshot = lambda: [
        {"job": jobs[1], "profile_name": "MiniServer", "model": "configured-model", "phase": "queued", "started": time.monotonic()-8, "effective_limit": 1, "configured_limit": 2, "rpm_limit": 30, "rpm_used": 12},
        *[{"job": jobs[1], "profile_name": "OpenAI", "model": "your-model-id", "phase": "request", "started": time.monotonic()-5, "effective_limit": 4, "configured_limit": 4, "rpm_limit": 60, "rpm_used": 16} for _ in range(2)],
    ]
    window.refresh()
    window.show()
    for _ in range(4):
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    window.grab().save(str(destination / "workbench-light.png"))
    dialog = gui.SettingsDialog(settings)
    dialog.findChild(QTabWidget).setCurrentIndex(0)
    dialog.show()
    app.processEvents()
    dialog.grab().save(str(destination / "providers-light.png"))
    dialog.findChild(QTabWidget).setCurrentIndex(3)
    app.processEvents()
    dialog.grab().save(str(destination / "settings-light.png"))
    dialog.findChild(QTabWidget).setCurrentIndex(2)
    app.processEvents()
    dialog.grab().save(str(destination / "encoding-light.png"))
    dialog.close()
    window.close()
    store.close()
print("Demo workbench and settings rendered; production database untouched")
