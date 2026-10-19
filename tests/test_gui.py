import os
import time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QFontComboBox, QPushButton

from submaker.config import Settings
from submaker.gui import MainWindow, SettingsDialog, STYLE, language_combo, language_value
from submaker.gpu_widgets import GpuUtilizationRing
from submaker.store import Store


def test_settings_language_and_role_overrides_preserve_shared_credentials(monkeypatch):
    app = QApplication.instance() or QApplication([])
    settings = Settings()
    settings.translate.url = "https://example.com/v1"
    settings.translate.key_id = "shared-credential"
    dialog = SettingsDialog(settings)
    assert dialog.review.inherit.isChecked()
    dialog.review.inherit.setChecked(False)
    dialog.review.model.setText("review-model")
    assert dialog.review.value().key_id == "shared-credential"
    dialog.review.url.setText("https://another.example.com/v1")
    assert dialog.review.value().key_id != "shared-credential"
    target = language_combo("zh-CN")
    assert language_value(target) == "zh-CN"
    target.setEditText("Indonesian")
    assert language_value(target) == "Indonesian"
    assert settings.translate.model == ""  # Editing dialog does not mutate task snapshots.
    dialog.close()


def test_preview_defaults_off_font_dropdown_and_automatic_export_ui(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLE)
    settings = Settings()
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: settings))
    monkeypatch.setattr(Settings, "save", lambda self: None)
    monkeypatch.setattr("submaker.gui.Store", lambda: Store(tmp_path / "jobs.sqlite"))
    dialog = SettingsDialog(settings)
    assert isinstance(dialog.font, QFontComboBox)
    assert not dialog.show_preview.isChecked()
    assert dialog.encode.currentData() == dialog.decode.currentData() == "hardware"
    assert dialog.codec.currentData() == "hevc"
    dialog.encoder_speed.setCurrentIndex(dialog.encoder_speed.findData("p7"))
    dialog.font.setCurrentFont(QFont("Segoe UI"))
    dialog.show_preview.setChecked(True)
    dialog.accept()
    assert dialog.settings.show_preview
    assert dialog.settings.font == dialog.font.currentFont().family()
    assert dialog.settings.encoder_preset == "p7"
    window = MainWindow()
    try:
        assert window.preview_area.isHidden()
        assert window.queue.columnCount() == 6
        window.gpu_monitor.snapshot=lambda:{0:{"utilization":75,"memory_used":5120,"memory_total":16384},1:{"utilization":0,"memory_used":5000,"memory_total":16384}}
        captions = [button.text() for button in window.findChildren(QPushButton)]
        assert "查看质量问题" not in captions
        assert "带告警导出" not in captions
        assert "删除选中任务" in captions and "清空全部任务" in captions
        source=tmp_path/"input.mp4"
        source.write_bytes(b"source")
        video={"index":0,"codec_name":"h264","codec_type":"video","pix_fmt":"yuv420p","bit_rate":"2000000","width":1280,"height":720}
        monkeypatch.setattr("submaker.gui.probe",lambda *args:{"video":video,"duration":3600,"streams":[video],"format":{"bit_rate":"2000000"}})
        window.add_paths([source])
        deadline=time.monotonic()+2
        while not window.media_info and time.monotonic()<deadline:
            app.processEvents()
            time.sleep(.01)
        assert window.queue.item(0,4).text().startswith("约 ")
        assert window.queue.horizontalHeaderItem(3).text() == "总 tokens / requests"
        assert window.queue.horizontalHeaderItem(4).text() == "预计 / 实际大小"
        job = window.rows[0]["id"]
        window.store.add_usage(job, {"total_tokens": 61590})
        output = tmp_path/"result.mkv"
        output.write_bytes(b"finished fixture")
        window.store.update(job, "完成", "运行缓存已清理", output)
        window.refresh()
        assert window.queue.item(0,3).text() == "61,590 / 1"
        assert window.queue.item(0,4).text().endswith("实际 16 B")
        assert "16 字节" in window.queue.item(0,4).toolTip()
        completed_estimate = window.queue.item(0,4).text().splitlines()[0]
        window.settings.video_quality = 10
        window.refresh()
        assert window.queue.item(0,4).text().splitlines()[0] == completed_estimate
        output.unlink()
        window.refresh()
        assert window.queue.item(0,4).text().endswith("实际 文件不可用")
        detail=window.queue.item(0,5).text()
        assert 'GPU 0' not in detail and 'GPU 1' not in detail
        assert 'GPU 0' in window.stat_values[1].toolTip() and 'GPU 1' in window.stat_values[1].toolTip()
        assert window.gpu_workers[0].ring.utilization == 75
        assert '显存' in window.gpu_workers[0].memory.text()
        window.store.update(job, "处理中", "模拟执行")
        window.store.record_api(window.rows[0]['id'],{'event':'rate_limit','code':429,'retry_at':time.time()+10,'effective_limit':2,'configured_limit':10})
        window.refresh()
        assert not window.api_banner.isHidden() and '429' in window.api_banner.text()
        window.queue.selectRow(0)
        window.refresh()
        assert len(window.queue.selectionModel().selectedRows())==1
        window.delete_button.click()
        assert not window.store.jobs()
        window.add_paths([source])
        window.clear_button.click()
        assert not window.store.jobs() and source.exists()
        window.settings.show_preview = True
        window.preview_area.setVisible(True)
        assert not window.preview_area.isHidden()
    finally:
        window.close()
        window.store.close()


def test_gpu_ring_renders_threshold_colors_and_handles_missing_samples():
    app = QApplication.instance() or QApplication([])
    ring = GpuUtilizationRing(0)
    try:
        for value, color in ((0, "#4e906b"), (45, "#4e906b"), (70, "#4e906b"), (70.1, "#d0a132"), (90, "#d0a132"), (90.1, "#cf6557"), (100, "#cf6557")):
            ring.set_utilization(value)
            assert ring.ring_color.name() == color
            pixels = ring.grab().toImage()
            assert any(pixels.pixelColor(x, y).name() == color for x in range(pixels.width()) for y in range(pixels.height()))
        for value in (None, float("nan"), float("inf")):
            ring.set_utilization(value)
            assert ring.utilization is None
            assert ring.accessibleDescription() == "—"
        ring.set_utilization(0)
        assert ring.accessibleDescription() == "0%"
    finally:
        ring.close()


def test_gpu_panel_timer_updates_without_reading_jobs_or_rebuilding_queue(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: Settings()))
    monkeypatch.setattr("submaker.gui.Store", lambda: Store(tmp_path/"jobs.sqlite"))
    window = MainWindow()
    try:
        window.timer.stop()
        readings = {0: {"utilization": 55, "memory_used": 1024, "memory_total": 16384, "encoder": 22, "decoder": 11}, 1: {"utilization": 92}}
        window.gpu_monitor.snapshot = lambda: readings
        window.store.jobs = lambda: (_ for _ in ()).throw(AssertionError("GPU repaint must not read task DB"))
        deadline = time.monotonic()+.8
        while window.gpu_workers[0].ring.utilization != 55 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert window.gpu_workers[0].ring.utilization == 55
        assert window.gpu_workers[1].ring.utilization == 92
        assert window.gpu_workers[0].video.text() == "解码 11%\n编码 22%"
        window.on_event("gpu", {"gpu": 0, "job": "fixture", "type": "transcribing"})
        assert window.gpu_workers[0].role.text() == "Whisper识别"
        window.on_event("gpu", {"gpu": 0, "job": "fixture", "type": "encoding_started"})
        assert window.gpu_workers[0].role.text() == "视频编解码"
    finally:
        window.close()
        assert not window.gpu_timer.isActive()
        window.store.close()
