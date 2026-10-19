from __future__ import annotations

import dataclasses
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import psutil
from PySide6.QtCore import QObject, Qt, QTimer, Signal, QUrl, QTranslator, QLibraryInfo, QItemSelectionModel
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QPalette
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFontComboBox, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QStackedWidget, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from . import __version__
from .config import DATA, LANGUAGES, ROOT, MAX_TRANSLATORS, PROVIDERS, Profile, Settings, get_key, set_key
from .llm import Client, endpoint
from .media import CREATE_NO_WINDOW, estimate_output, probe
from .pipeline import Scheduler
from .store import Store
from .telemetry import GpuMonitor, api_notice, api_provider_records, gpu_details, gpu_role, provider_detail_lines
from .gpu_widgets import GpuWorkerStatus

STYLE = """
QWidget { background:#faf9f5; color:#34332f; font-family:'Microsoft YaHei UI','Segoe UI'; font-size:13px; }
QLabel { background:transparent; }
QMainWindow { background:#faf9f5; }
QFrame#sidebar { background:#f0eee6; border-right:1px solid #e2dfd5; }
QFrame#card { background:#fffefa; border:1px solid #e5e1d8; border-radius:14px; }
QFrame#card QLabel { background:transparent; }
QLabel#brand { color:#38362f; font-size:23px; font-weight:800; }
QLabel#eyebrow { color:#a95439; font-size:11px; font-weight:700; }
QLabel#title { font-size:27px; font-weight:650; color:#302f2b; }
QLabel#muted { color:#77746b; }
QLabel#stat { font-size:26px; color:#39372f; font-weight:650; }
QLabel#section { font-size:15px; font-weight:650; color:#3c3931; }
QLabel#apiWarning { background:#fff0df; color:#995126; border:1px solid #e2b58b; border-radius:8px; padding:10px 13px; font-weight:600; }
QPushButton { background:#fffefa; border:1px solid #d9d4c8; border-radius:8px; padding:9px 15px; font-weight:600; }
QPushButton:hover { background:#ece8de; border-color:#b9b0a0; }
QPushButton:disabled { color:#aaa69c; background:#efede6; border-color:#e1ddd3; }
QPushButton#primary { background:#c66d4e; color:#ffffff; border-color:#c66d4e; }
QPushButton#primary:hover { background:#b75e41; }
QPushButton#quiet { background:transparent; color:#555146; border:1px solid #d7d1c5; }
QLineEdit,QComboBox,QSpinBox,QPlainTextEdit { background:#fffefa; border:1px solid #d9d4c8; border-radius:7px; padding:7px; selection-background-color:#ead1c4; selection-color:#34332f; }
QLineEdit:focus,QComboBox:focus,QSpinBox:focus { border-color:#bf7256; }
QComboBox QAbstractItemView { background:#fffefa; color:#34332f; selection-background-color:#ede2d7; selection-color:#34332f; }
QListWidget { background:#f2eee5; border:1px solid #ded7ca; border-radius:8px; padding:5px; }
QListWidget::item { padding:11px 8px; border-radius:6px; }
QListWidget::item:selected { background:#e8d6c7; color:#5f3e2b; }
QCheckBox { spacing:8px; background:transparent; }
QCheckBox::indicator { width:17px; height:17px; border-radius:4px; border:1px solid #bdb5a5; background:#fffefa; }
QCheckBox::indicator:checked { image:url("__CHECK_ICON__"); background:transparent; border:0; }
QTableWidget { background:#fffefa; alternate-background-color:#f6f4ed; border:1px solid #e0dbd0; border-radius:10px; gridline-color:#e9e4da; }
QTableWidget::item { padding:8px; border-bottom:1px solid #ece7dd; }
QTableWidget::item:selected { background:#eee0d4; color:#37332c; }
QHeaderView::section { background:#eeeae0; color:#746b5e; border:0; padding:10px; font-weight:600; }
QProgressBar { border:0; background:#eae4d8; border-radius:4px; height:10px; text-align:center; color:#554236; }
QProgressBar::chunk { background:#d99578; border-radius:4px; }
QTabWidget::pane { border:1px solid #ddd7cb; border-radius:8px; }
QTabBar::tab { background:#f1eee5; padding:10px 20px; border-bottom:2px solid transparent; }
QTabBar::tab:selected { color:#a95439; border-bottom:2px solid #bd7155; background:#fffefa; }
QGroupBox { border:1px solid #ddd6ca; border-radius:8px; margin-top:14px; padding:18px 10px 10px; }
QGroupBox::title { subcontrol-origin:margin; left:12px; color:#74695a; }
QScrollArea { border:0; background:transparent; }
QScrollBar:vertical { width:9px; background:#f3f0e8; }
QScrollBar::handle:vertical { background:#ccc4b5; border-radius:4px; min-height:30px; }
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical { height:0; }
QToolTip { background:#fffefa; color:#484136; border:1px solid #d3c7b7; padding:7px; }
"""
STYLE = STYLE.replace("__CHECK_ICON__", (ROOT / "assets" / "check.svg").as_posix())


def label(text, name=None):
    q = QLabel(text)
    if name:
        q.setObjectName(name)
    return q


def button(text, slot, primary=False):
    b = QPushButton(text)
    b.setObjectName("primary" if primary else "quiet")
    b.clicked.connect(slot)
    return b


def spin(value, minimum, maximum, suffix=""):
    q = QSpinBox()
    q.setRange(minimum, maximum)
    q.setValue(value)
    q.setSuffix(suffix)
    return q


def language_combo(value, source=False):
    q = QComboBox()
    q.setEditable(True)
    if source:
        q.setToolTip("选择常用语言，或输入 Whisper 支持的语言代码（如 sv、nl）。自动检测确定主语言后按单语处理。")
    for name, code in LANGUAGES.items():
        if code != "auto" or source:
            q.addItem(name, code)
    index = q.findData(value)
    if index >= 0:
        q.setCurrentIndex(index)
    else:
        q.setEditText(value)
    return q


def language_value(combo):
    return LANGUAGES.get(combo.currentText(), combo.currentText().strip())


def token_summary(row):
    requests = row.get("token_requests") or 0
    reports = row.get("usage_reports") or 0
    partial = row.get("usage_partial") or reports < requests
    total = ("≥" if partial else "") + f"{row.get('total_tokens', 0):,}" if reports else "—"
    count = ("≥" if row.get("requests_partial") else "") + f"{requests:,}"
    detail = f"累计总计 {row.get('total_tokens', 0):,} tokens / {requests:,} requests；含翻译、上下文、审校、修正及文件名。tokens 仅累计接口实际返回的 usage；requests 统计已发起的 HTTP 尝试，包含限流、超时和重试。"
    if row.get("usage_partial"):
        detail += " 本任务旧版用量未保存，此处仅含新版运行期间记录。"
    if reports < requests:
        detail += " 部分请求未返回总 token 用量，≥表示已知下限。"
    if row.get("requests_partial"):
        detail += " 旧版未统计全部请求尝试，requests 的 ≥ 表示已记录次数的下限。"
    return f"{total} / {count}", detail


def file_size(value):
    if value >= 1024**3:
        return f"{value/1024**3:.2f} GiB"
    if value >= 1024**2:
        return f"{value/1024**2:.0f} MiB"
    return f"{value/1024:.1f} KiB" if value >= 1024 else f"{value} B"


class Bridge(QObject):
    event = Signal(str, object)
    task_done = Signal(str, object)


class ProfileEditor(QWidget):
    def __init__(self, profile, role="translate", inherited=False):
        super().__init__()
        self.profile = dataclasses.replace(profile, verified_efforts=list(profile.verified_efforts))
        self.role = role
        self.changed_key_ids = {}
        self.bridge = Bridge()
        self.bridge.task_done.connect(self.test_done)
        outer = QVBoxLayout(self)
        archive_row = QHBoxLayout()
        self.saved_profiles = QComboBox()
        self.saved_profiles.addItem("选择已保存的接口配置…", None)
        self.profile_file = DATA / "profiles.json"
        self.archive = json.loads(self.profile_file.read_text(encoding="utf-8")) if self.profile_file.exists() else {}
        for name in self.archive:
            self.saved_profiles.addItem(name, name)
        archive_row.addWidget(self.saved_profiles, 1)
        archive_row.addWidget(button("加载", self.load_profile))
        archive_row.addWidget(button("另存配置", self.archive_profile))
        outer.addLayout(archive_row)
        self.inherit = QCheckBox("继承主翻译接口、模型与 effort")
        self.inherit.setChecked(inherited)
        if role != "translate":
            outer.addWidget(self.inherit)
        self.fields = QWidget()
        form = QFormLayout(self.fields)
        form.setSpacing(12)
        self.display_name = QLineEdit(profile.name)
        self.provider = QComboBox()
        for code, (caption, _) in PROVIDERS.items():
            self.provider.addItem(caption, code)
        self.provider.setCurrentIndex(max(0, self.provider.findData(profile.provider)))
        self.enabled = QCheckBox("启用此翻译接口，参与并行翻译")
        self.enabled.setChecked(profile.enabled)
        self.url = QLineEdit(profile.url)
        self.url.setPlaceholderText("https://example.com/v1 或完整 chat/completions 地址")
        self.model = QLineEdit(profile.model)
        self.model.setPlaceholderText("填写服务提供的准确模型 ID")
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("已保存的密钥不显示；留空保持原值")
        self.effort = QComboBox()
        self.effort.setEditable(True)
        for value in profile.verified_efforts:
            self.effort.addItem(value)
        self.effort.setCurrentText(profile.effort)
        self.effort.setToolTip("default 不发送 effort 参数；可输入接口支持的档位并测试。下拉列表只保存已验证档位。")
        self.timeout = spin(profile.timeout, 15, 1800, " 秒")
        self.concurrent = spin(profile.concurrency, 1, 16)
        self.concurrent.setToolTip("这是并发上限。HTTP 429限流时会统一退避并临时降低实际并发，恢复稳定后逐步提高；不会改写你设置的上限。")
        self.rpm = spin(profile.rpm, 0, 100000, " RPM")
        self.rpm.setSpecialValueText("不限 RPM")
        self.rpm.setToolTip("每分钟最多发出的请求数，按间隔均匀发送，重试也计入。0为不限制。请按服务配额设置；token/账号共享额度仍可能触发429。")
        self.json = QCheckBox("发送 response_format=json_object（需接口支持）")
        self.json.setChecked(profile.json_mode)
        self.lan = QCheckBox("信任此私有局域网 HTTP 服务")
        self.lan.setChecked(profile.allow_lan_http)
        for caption, widget in (("接口名称", self.display_name), ("服务类型", self.provider), ("API 地址", self.url), ("模型", self.model), ("API Key", self.key), ("Reasoning effort", self.effort), ("请求超时", self.timeout), ("并发上限", self.concurrent), ("每分钟请求上限", self.rpm), ("", self.json), ("", self.lan)):
            form.addRow(caption, widget)
        if role == "translate":
            form.addRow("", self.enabled)
        self.provider.currentIndexChanged.connect(self.provider_changed)
        outer.addWidget(self.fields)
        row = QHBoxLayout()
        self.test_button = button("测试当前配置", self.test)
        row.addWidget(self.test_button)
        self.result = label("密钥保存在 Windows 凭据管理器", "muted")
        self.result.setWordWrap(True)
        row.addWidget(self.result, 1)
        outer.addLayout(row)
        outer.addStretch()
        self.inherit.toggled.connect(lambda checked: self.fields.setDisabled(checked))
        self.fields.setDisabled(inherited)

    def provider_changed(self):
        base = PROVIDERS[self.provider.currentData()][1]
        if base:
            self.url.setText(base)
            self.model.clear()
            self.key.clear()
            self.lan.setChecked(False)
            self.result.setText("请填写此服务的模型ID与Key，并测试连接；地址可以自定义。")

    def load_profile(self):
        name = self.saved_profiles.currentData()
        if not name:
            return
        self.profile = Profile.from_dict(self.archive[name])
        self.changed_key_ids.clear()
        p = self.profile
        self.display_name.setText(p.name)
        self.provider.blockSignals(True)
        self.provider.setCurrentIndex(max(0, self.provider.findData(p.provider)))
        self.provider.blockSignals(False)
        self.enabled.setChecked(p.enabled)
        self.url.setText(p.url)
        self.model.setText(p.model)
        self.key.clear()
        self.effort.clear()
        self.effort.addItems(p.verified_efforts)
        self.effort.setCurrentText(p.effort)
        self.timeout.setValue(p.timeout)
        self.concurrent.setValue(p.concurrency)
        self.rpm.setValue(p.rpm)
        self.json.setChecked(p.json_mode)
        self.lan.setChecked(p.allow_lan_http)
        if self.role != "translate":
            self.inherit.setChecked(False)
        self.result.setText("配置已加载；密钥通过凭据引用读取，未显示明文")

    def archive_profile(self):
        name, ok = QInputDialog.getText(self, "保存接口配置", "配置名称：", text=self.profile.name)
        if not ok or not name.strip():
            return
        try:
            p = self.value()
            p.name = name.strip()
            if self.key.text():
                set_key(p, self.key.text())
            self.profile = p
            self.display_name.setText(p.name)
            self.archive[p.name] = dataclasses.asdict(p)
            temporary = self.profile_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.archive, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.profile_file)
            if self.saved_profiles.findData(p.name) < 0:
                self.saved_profiles.addItem(p.name, p.name)
            self.saved_profiles.setCurrentIndex(self.saved_profiles.findData(p.name))
            self.result.setText("接口配置已保存，API Key 仍仅保存在 Windows 凭据管理器")
        except Exception as exc:
            self.result.setText("保存配置失败：" + str(exc))

    def value(self):
        p = dataclasses.replace(self.profile)
        p.name = self.display_name.text().strip() or "未命名接口"
        p.provider, p.enabled, p.rpm = self.provider.currentData(), self.enabled.isChecked(), self.rpm.value()
        p.url, p.model, p.effort = self.url.text().strip(), self.model.text().strip(), self.effort.currentText().strip() or "default"
        p.timeout, p.concurrency = self.timeout.value(), self.concurrent.value()
        p.json_mode, p.allow_lan_http = self.json.isChecked(), self.lan.isChecked()
        if p.url != self.profile.url or (self.role != "translate" and self.key.text()):
            identity = (p.url, bool(self.key.text()) if self.role != "translate" else False)
            p.key_id = self.changed_key_ids.setdefault(identity, self.role + "-" + uuid.uuid4().hex)
        return p

    def save_key(self):
        if self.key.text() and not (self.role != "translate" and self.inherit.isChecked()):
            set_key(self.value(), self.key.text())

    def test(self):
        if self.inherit.isChecked() and self.role != "translate":
            self.result.setText("当前继承翻译配置，请到翻译接口页测试")
            return
        p = self.value()
        typed_key = self.key.text()
        self.test_button.setDisabled(True)
        self.result.setText("正在测试实际接口和所选 effort…")
        def run():
            try:
                client = Client(key_provider=lambda _: typed_key or get_key(p))
                result = client.request(p, 'Connection test. Output {"ok":true}.', {"test": True})
                if result.get("ok") is not True:
                    raise RuntimeError("接口可达，但测试JSON校验未通过")
                self.bridge.task_done.emit("ok", p.effort)
            except Exception as exc:
                self.bridge.task_done.emit("error", str(exc))
        threading.Thread(target=run, daemon=True).start()

    def test_done(self, kind, result):
        self.test_button.setEnabled(True)
        if kind == "ok":
            if result not in self.profile.verified_efforts:
                self.profile.verified_efforts.append(result)
                self.effort.addItem(result)
            self.result.setText("连接、JSON 和 effort 参数验证通过")
        else:
            self.result.setText(str(result))


class TranslationPoolEditor(QWidget):
    def __init__(self, settings):
        super().__init__()
        outer = QVBoxLayout(self)
        intro = label("启用的接口按空闲并发与RPM领取字幕批次。一个接口退避时，其余接口继续翻译；同端点和凭据引用共享额度。", "muted")
        intro.setWordWrap(True)
        outer.addWidget(intro)
        row = QHBoxLayout()
        self.list = QListWidget()
        self.list.setFixedWidth(180)
        self.stack = QStackedWidget()
        self.editors = []
        row.addWidget(self.list)
        row.addWidget(self.stack, 1)
        outer.addLayout(row, 1)
        actions = QHBoxLayout()
        self.provider_choice = QComboBox()
        for code, (name, _) in PROVIDERS.items():
            self.provider_choice.addItem(name, code)
        actions.addWidget(self.provider_choice)
        actions.addWidget(button("＋ 添加接口", lambda: self.add_profile(self.provider_choice.currentData())))
        self.remove_button = button("移除此接口", self.remove_current)
        actions.addWidget(self.remove_button)
        actions.addStretch()
        outer.addLayout(actions)
        self.summary = label("", "muted")
        self.summary.setWordWrap(True)
        outer.addWidget(self.summary)
        for profile in [settings.translate, *settings.translators]:
            self.append_editor(profile)
        self.list.currentRowChanged.connect(self.select_editor)
        self.list.setCurrentRow(0)
        self.refresh_summary()

    def append_editor(self, profile):
        editor = ProfileEditor(profile)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(editor)
        self.editors.append(editor)
        self.stack.addWidget(scroll)
        self.list.addItem(profile.name)
        for signal in (editor.display_name.textChanged, editor.enabled.toggled, editor.concurrent.valueChanged, editor.rpm.valueChanged, editor.url.textChanged, editor.model.textChanged):
            signal.connect(self.refresh_summary)
        self.refresh_summary()
        return editor

    def add_profile(self, provider="compatible"):
        if len(self.editors) >= MAX_TRANSLATORS:
            QMessageBox.information(self, "接口数量", f"最多添加{MAX_TRANSLATORS}个翻译接口。")
            return
        name, url = PROVIDERS[provider]
        editor = self.append_editor(Profile(name=name, url=url, provider=provider, enabled=False, key_id="translate-"+uuid.uuid4().hex))
        self.list.setCurrentRow(len(self.editors)-1)
        return editor

    def select_editor(self, index):
        self.stack.setCurrentIndex(index)
        self.remove_button.setEnabled(index > 0)
        self.remove_button.setToolTip("主接口可以停用；其他接口可以移除。更改在保存设置后生效。")

    def remove_current(self):
        index = self.list.currentRow()
        if index <= 0:
            return
        page = self.stack.widget(index)
        self.stack.removeWidget(page)
        self.list.takeItem(index)
        self.editors.pop(index)
        page.deleteLater()
        self.list.setCurrentRow(min(index, len(self.editors)-1))
        self.refresh_summary()

    def refresh_summary(self, *_):
        profiles = []
        for index, editor in enumerate(self.editors):
            p = editor.value()
            status = ("主接口 · " if index == 0 else "") + ("已启用" if p.enabled else "已停用")
            self.list.item(index).setText(p.name+"\n"+status)
            if p.enabled and p.url and p.model:
                profiles.append(p)
        capacity = Settings(translate=profiles[0], translators=profiles[1:]).translation_parallelism() if profiles else 0
        self.summary.setText(f"已就绪 {len(profiles)} 个接口 · 配置总并发上限 {capacity} · 各接口分别执行RPM；默认30 RPM，可按配额调整。")

    def values(self):
        profiles = [editor.value() for editor in self.editors]
        for index, p in enumerate(profiles):
            if p.enabled and (index > 0 or p.url or p.model):
                if not p.url or not p.model:
                    raise ValueError(f"{p.name}：请填写地址和模型，或停用此接口。")
                endpoint(p)
        return profiles[0], profiles[1:]

    def save_keys(self):
        for editor in self.editors:
            editor.save_key()


class SettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Submaker · 工作台设置")
        self.resize(1040, 860)
        self.settings = Settings.from_dict(settings.as_dict())
        outer = QVBoxLayout(self)
        tabs = QTabWidget()
        self.pool = TranslationPoolEditor(settings)
        self.translate = self.pool.editors[0]
        self.review = ProfileEditor(settings.review or settings.translate, "review", settings.review is None)
        tabs.addTab(self.pool, "翻译接口池")
        review_page = QWidget()
        review_layout = QVBoxLayout(review_page)
        review_note = label("新任务的全片上下文、审校、自动修正和文件名翻译统一使用此接口及模型。已有任务保留其保存的修正方式。", "muted")
        review_note.setWordWrap(True)
        review_layout.addWidget(review_note)
        review_scroll = QScrollArea()
        review_scroll.setWidgetResizable(True)
        review_scroll.setWidget(self.review)
        review_layout.addWidget(review_scroll)
        tabs.addTab(review_page, "全片审校 / 文件名")
        video = QWidget()
        video_form = QFormLayout(video)
        video_form.setSpacing(16)
        self.codec = QComboBox()
        self.codec.addItem("H.265 / HEVC · 紧凑体积（默认）", "hevc")
        self.codec.addItem("H.264 / AVC · 广泛兼容", "h264")
        self.codec.setCurrentIndex(max(0, self.codec.findData(settings.video_codec)))
        self.encoder_speed = QComboBox()
        for caption, value in (("very fast · P2", "p2"), ("fast · P3", "p3"), ("medium · P4", "p4"), ("slow · P5", "p5"), ("slower · P6（默认）", "p6"), ("very slow · P7", "p7")):
            self.encoder_speed.addItem(caption, value)
        self.encoder_speed.setCurrentIndex(max(0, self.encoder_speed.findData(settings.encoder_preset)))
        self.video_quality = spin(settings.video_quality, 14, 40, " CQ")
        self.video_quality.setToolTip("数值越小越清晰、文件通常越大；数值越大文件通常越小。默认26。")
        self.size_mode = QComboBox()
        for caption, value in (("紧凑体积 · 参考源视频限制码率（默认）", "compact"), ("优先质量 · 文件大小波动较大", "quality"), ("指定视频码率", "bitrate")):
            self.size_mode.addItem(caption, value)
        self.size_mode.setCurrentIndex(max(0, self.size_mode.findData(settings.size_mode)))
        self.video_bitrate = spin(settings.video_bitrate_kbps, 100, 200000, " kbps")
        self.video_bitrate.setEnabled(settings.size_mode == "bitrate")
        self.size_mode.currentIndexChanged.connect(lambda: self.video_bitrate.setEnabled(self.size_mode.currentData() == "bitrate"))
        for caption, field in (("输出编码", self.codec), ("GPU 速度预设", self.encoder_speed), ("画面质量", self.video_quality), ("体积策略", self.size_mode), ("目标视频码率", self.video_bitrate)):
            video_form.addRow(caption, field)
        encoding_note = label("全部速度档位使用 V100 NVENC。解码与字幕滤镜按线程预算并行，保留所选画质参数。\nP7 / very slow 使用全分辨率双遍；预计大小包含音轨，实际大小受画面复杂度影响。", "muted")
        encoding_note.setWordWrap(True)
        video_form.addRow(encoding_note)
        tabs.addTab(video, "视频编码")
        general = QWidget()
        form = QFormLayout(general)
        form.setSpacing(12)
        self.model_path = QLineEdit(settings.model_path)
        self.output = QLineEdit(settings.output_dir)
        self.output.setPlaceholderText("留空：原视频所在目录")
        self.ffmpeg = QLineEdit(settings.ffmpeg)
        self.ffprobe = QLineEdit(settings.ffprobe)
        self.chunk = spin(settings.chunk_seconds, 30, 600, " 秒")
        self.chunk.setToolTip("单窗口最大10分钟；识别句子会在窗口完成前送去翻译")
        self.ram = spin(settings.ram_gib, 1, 24, " GiB")
        self.inflight = spin(settings.in_flight, 1, 6)
        self.audio = spin(settings.audio_stream, -1, 128)
        self.audio.setSpecialValueText("自动选择匹配源语言的默认音轨")
        self.encode = QComboBox()
        self.encode.addItem("V100 GPU 硬件编码 · 默认", "hardware")
        self.encode.addItem("CPU 兼容编码 · 仅手动选择", "quality")
        self.encode.setCurrentIndex(max(0, self.encode.findData(settings.encoding)))
        self.decode = QComboBox()
        self.decode.addItem("V100 GPU 解码 · 默认", "hardware")
        self.decode.addItem("CPU 解码", "software")
        self.decode.setCurrentIndex(max(0, self.decode.findData(settings.decoding)))
        self.video_gpu = QComboBox()
        for caption, value in (("自动分配已选择的 GPU", -1), ("GPU 0", 0), ("GPU 1", 1)):
            self.video_gpu.addItem(caption, value)
        self.video_gpu.setCurrentIndex(max(0, self.video_gpu.findData(settings.video_gpu)))
        self.show_preview = QCheckBox("显示字幕预览（默认关闭）")
        self.show_preview.setChecked(settings.show_preview)
        self.bilingual = QCheckBox("烧录原文 + 译文双语字幕")
        self.bilingual.setChecked(settings.bilingual)
        self.font = QFontComboBox()
        self.font.setCurrentFont(QFont(settings.font))
        self.font_size = spin(settings.font_size, 12, 48)
        for caption, field in (("模型目录", self.model_path), ("输出目录", self.output), ("FFmpeg", self.ffmpeg), ("FFprobe", self.ffprobe), ("识别窗口", self.chunk), ("RAM 缓存上限", self.ram), ("在途视频上限", self.inflight), ("音轨流编号", self.audio), ("视频解码", self.decode), ("视频编码", self.encode), ("视频 GPU", self.video_gpu), ("字幕字体", self.font), ("字幕字号", self.font_size), ("", self.bilingual), ("", self.show_preview)):
            form.addRow(caption, field)
        general_scroll = QScrollArea()
        general_scroll.setWidgetResizable(True)
        general_scroll.setWidget(general)
        tabs.addTab(general_scroll, "资源与输出")
        prompts = QWidget()
        pv = QVBoxLayout(prompts)
        pv.addWidget(label("翻译要求"))
        self.prompt = QPlainTextEdit(settings.prompt)
        pv.addWidget(self.prompt)
        pv.addWidget(label("术语表 / 人名表（例如：原文 → 译名）"))
        self.glossary = QPlainTextEdit(settings.glossary)
        pv.addWidget(self.glossary)
        tabs.addTab(prompts, "翻译与术语")
        outer.addWidget(tabs)
        notice = label("保存后，新任务使用新配置；已有任务可选择“使用当前设置重试”。运行中的资源设置在重新启动工作台后生效。", "muted")
        notice.setWordWrap(True)
        outer.addWidget(notice)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("保存")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def accept(self):
        try:
            s = self.settings
            s.translate, s.translators = self.pool.values()
            review_profile = s.translate if self.review.inherit.isChecked() else self.review.value()
            if s.translation_profiles() and (not review_profile.url or not review_profile.model):
                raise ValueError("请为全片审校配置地址和模型，或让主翻译接口提供审校服务。文件名翻译也使用此接口。")
            self.pool.save_keys()
            self.review.save_key()
            s.review = None if self.review.inherit.isChecked() else self.review.value()
            s.model_path, s.output_dir = self.model_path.text().strip(), self.output.text().strip()
            s.ffmpeg, s.ffprobe = self.ffmpeg.text().strip(), self.ffprobe.text().strip()
            s.chunk_seconds, s.ram_gib, s.in_flight = self.chunk.value(), self.ram.value(), self.inflight.value()
            s.audio_stream, s.encoding = self.audio.value(), self.encode.currentData()
            s.video_codec, s.encoder_preset = self.codec.currentData(), self.encoder_speed.currentData()
            s.video_quality, s.size_mode = self.video_quality.value(), self.size_mode.currentData()
            s.video_bitrate_kbps = self.video_bitrate.value()
            s.decoding, s.video_gpu = self.decode.currentData(), self.video_gpu.currentData()
            s.show_preview = self.show_preview.isChecked()
            s.bilingual, s.font, s.font_size = self.bilingual.isChecked(), self.font.currentFont().family(), self.font_size.value()
            s.prompt, s.glossary = self.prompt.toPlainText(), self.glossary.toPlainText()
            s.save()
            super().accept()
        except Exception as exc:
            QMessageBox.warning(self, "设置未保存", str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Submaker Studio · 本地字幕工作台")
        self.setWindowIcon(QIcon(str(ROOT / "assets" / "app-light.ico")))
        self.resize(1380, 920)
        self.setMinimumSize(1120, 760)
        self.setAcceptDrops(True)
        self.settings = Settings.load()
        self.store = Store()
        self.bridge = Bridge()
        self.bridge.event.connect(self.on_event)
        self.bridge.task_done.connect(self.background_done)
        self.scheduler = Scheduler(self.store, self.settings, lambda kind, data: self.bridge.event.emit(kind, data))
        self.progress = {}
        self.gpu_states = {}
        self.asr_jobs = {}
        self.encoding_jobs = {}
        self.gpu_monitor = GpuMonitor()
        self.rows = []
        self.active_preview = None
        self.media_info = {}
        self.inspections = {}
        self.inspector = cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="media-info")
        self.closing = False
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(260)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(22, 28, 22, 22)
        side.setSpacing(13)
        side.addWidget(label("SUBMAKER", "brand"))
        side.addWidget(label("LOCAL SUBTITLE STUDIO", "eyebrow"))
        side.addSpacing(22)
        side.addWidget(label("语言与资源", "section"))
        self.preset = QComboBox()
        for text, code in (("高质量 · 默认", "quality"), ("均衡", "balanced"), ("快速", "fast")):
            self.preset.addItem(text,code)
        self.preset.setCurrentIndex(max(0,self.preset.findData(self.settings.preset)))
        self.preset.setToolTip("高质量采用更细致的识别和更高画质；快速缩短处理时间。所有预设均进行全片审校。")
        side.addWidget(self.preset)
        side.addWidget(label("识别语言", "muted"))
        self.source = language_combo(self.settings.source, True)
        side.addWidget(self.source)
        side.addWidget(label("翻译目标语言", "muted"))
        self.target = language_combo(self.settings.target)
        side.addWidget(self.target)
        side.addSpacing(10)
        side.addWidget(label("识别设备", "muted"))
        devices = QHBoxLayout()
        self.gpu0, self.gpu1 = QCheckBox("GPU 0"), QCheckBox("GPU 1")
        self.gpu0.setChecked(0 in self.settings.gpus)
        self.gpu1.setChecked(1 in self.settings.gpus)
        devices.addWidget(self.gpu0)
        devices.addWidget(self.gpu1)
        side.addLayout(devices)
        side.addWidget(label("CPU 全局线程预算", "muted"))
        self.threads = spin(self.settings.threads, 4, os.cpu_count() or 16, " 线程")
        side.addWidget(self.threads)
        side.addSpacing(14)
        self.model_status = label("", "muted")
        self.model_status.setWordWrap(True)
        side.addWidget(self.model_status)
        self.install_button = button("↓  安装 / 检查模型", self.install_model)
        side.addWidget(self.install_button)
        side.addWidget(button("API 与高级设置", self.open_settings))
        side.addStretch()
        side.addWidget(label("质量优先 · 内存音频缓存", "eyebrow"))
        small = label("本地识别 → 即时翻译\n全片审校 → 字幕烧录\n完成即清理运行缓存", "muted")
        small.setStyleSheet("line-height:160%;")
        side.addWidget(small)
        side.addSpacing(10)
        side.addWidget(label(f"v{__version__}  ·  Windows / V100", "muted"))
        layout.addWidget(sidebar)
        content = QWidget()
        body = QVBoxLayout(content)
        body.setContentsMargins(28, 26, 28, 20)
        body.setSpacing(17)
        top = QHBoxLayout()
        titles = QVBoxLayout()
        titles.addWidget(label("把每一句，准确带到另一种语言。", "title"))
        titles.addWidget(label("双 GPU 本地识别  /  翻译流水线  /  全片质量审校", "muted"))
        top.addLayout(titles, 1)
        badge = label("●  LOCAL WORKSPACE", "eyebrow")
        badge.setStyleSheet("background:#f1e7db;color:#986043;border:1px solid #e4d4c1;border-radius:10px;padding:10px;")
        top.addWidget(badge)
        body.addLayout(top)
        cards = QHBoxLayout()
        self.stat_values = []
        for heading, value, description in (("任务队列", "0", "就绪 / 处理中 / 已完成"), ("GPU WORKERS", "0 / 2", "Tesla V100 · FP16 / NVENC"), ("可用内存", "—", "有界 RAM 缓冲 · 完成即清理")):
            card = QFrame()
            card.setObjectName("card")
            cv = QVBoxLayout(card)
            cv.setContentsMargins(17, 13, 17, 13)
            value_label = label(value, "stat")
            if heading == "GPU WORKERS":
                header = QHBoxLayout()
                header.addWidget(label(heading, "muted"))
                header.addStretch()
                value_label.setStyleSheet("font-size:14px;font-weight:600;")
                header.addWidget(value_label)
                cv.addLayout(header)
                devices = QHBoxLayout()
                devices.setSpacing(14)
                self.gpu_workers = {gpu: GpuWorkerStatus(gpu) for gpu in (0, 1)}
                for worker in self.gpu_workers.values():
                    devices.addWidget(worker, 1)
                cv.addLayout(devices)
            else:
                cv.addWidget(label(heading, "muted"))
                cv.addWidget(value_label)
                description_label = label(description, "muted")
                description_label.setWordWrap(True)
                cv.addWidget(description_label)
            self.stat_values.append(value_label)
            cards.addWidget(card, 2 if heading == "GPU WORKERS" else 1)
        body.addLayout(cards)
        self.api_banner = label("", "apiWarning")
        self.api_banner.setWordWrap(True)
        self.api_banner.hide()
        body.addWidget(self.api_banner)
        toolbar = QHBoxLayout()
        toolbar.addWidget(label("处理队列", "section"))
        toolbar.addStretch()
        toolbar.addWidget(button("＋ 添加视频", self.add_files))
        toolbar.addWidget(button("添加文件夹", self.add_folder))
        self.start_button = button("开始队列", self.start, True)
        toolbar.addWidget(self.start_button)
        toolbar.addWidget(button("暂停", self.pause))
        body.addLayout(toolbar)
        splitter = QSplitter(Qt.Vertical)
        queue_area = QWidget()
        qv = QVBoxLayout(queue_area)
        qv.setContentsMargins(0, 0, 0, 0)
        self.queue = QTableWidget(0, 6)
        self.queue.setHorizontalHeaderLabels(["视频 / 输出", "状态", "阶段进度", "总 tokens / requests", "预计 / 实际大小", "详细信息"])
        self.queue.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.queue.setColumnWidth(0, 220)
        self.queue.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.queue.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.queue.setColumnWidth(2, 135)
        self.queue.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.queue.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.queue.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.queue.setSelectionBehavior(QTableWidget.SelectRows)
        self.queue.setSelectionMode(QTableWidget.ExtendedSelection)
        self.queue.setEditTriggers(QTableWidget.NoEditTriggers)
        self.queue.verticalHeader().hide()
        self.queue.setAlternatingRowColors(True)
        self.queue.itemSelectionChanged.connect(self.show_preview)
        qv.addWidget(self.queue)
        actions = QHBoxLayout()
        actions.addWidget(button("使用当前设置重试", self.retry))
        actions.addWidget(button("填写输出译名", self.manual_title))
        self.delete_button = button("删除选中任务", self.delete_selected)
        self.clear_button = button("清空全部任务", self.clear_tasks)
        for action in (self.delete_button, self.clear_button):
            action.setToolTip("删除任务记录和运行缓存；正在运行的任务会先停止。保留源视频和已完成的视频。")
            actions.addWidget(action)
        actions.addStretch()
        actions.addWidget(button("打开输出", self.open_output))
        qv.addLayout(actions)
        splitter.addWidget(queue_area)
        preview_area = QWidget()
        self.preview_area = preview_area
        pv = QVBoxLayout(preview_area)
        pv.setContentsMargins(0, 8, 0, 0)
        preview_header = QHBoxLayout()
        preview_header.addWidget(label("字幕预览", "section"))
        self.preview_summary = label("选择任务查看原文与译文", "muted")
        preview_header.addStretch()
        preview_header.addWidget(self.preview_summary)
        pv.addLayout(preview_header)
        self.preview = QTableWidget(0, 3)
        self.preview.setHorizontalHeaderLabels(["时间", "识别原文", "目标语言译文"])
        self.preview.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.preview.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.preview.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.preview.verticalHeader().hide()
        self.preview.setEditTriggers(QTableWidget.NoEditTriggers)
        self.preview.setAlternatingRowColors(True)
        pv.addWidget(self.preview)
        splitter.addWidget(preview_area)
        preview_area.setVisible(self.settings.show_preview)
        splitter.setSizes([330, 240])
        body.addWidget(splitter, 1)
        self.status = label("就绪 · 拖入视频即可添加任务。API 可稍后配置，本地识别结果会保留供继续处理。", "muted")
        self.status.setWordWrap(True)
        body.addWidget(self.status)
        layout.addWidget(content, 1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(1500)
        self.gpu_timer = QTimer(self)
        self.gpu_timer.timeout.connect(self.refresh_gpu_workers)
        self.gpu_timer.start(250)
        self.refresh()

    def current_settings(self):
        s = Settings.from_dict(self.settings.as_dict())
        s.source, s.target = language_value(self.source), language_value(self.target)
        s.preset = self.preset.currentData()
        s.gpus = [i for i, w in enumerate((self.gpu0, self.gpu1)) if w.isChecked()]
        s.threads = self.threads.value()
        if not s.gpus:
            raise ValueError("请至少选择一张 GPU")
        if not s.target or s.target == "auto":
            raise ValueError("请指定翻译目标语言")
        return s

    def add_paths(self, paths):
        try:
            s = self.current_settings()
            records = self.store.jobs()
            existing = {r["source"] for r in records if r["state"] not in ("完成", "完成（有提示）")}
            outputs = {str(Path(r["output"]).resolve()) for r in records if r["output"]}
            count = 0
            for path in paths:
                p = Path(path).resolve()
                if p.is_file() and p.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts", ".m2ts", ".wmv"} and str(p) not in existing and str(p) not in outputs and ".subbed" not in p.stem:
                    self.store.add(str(p), s)
                    existing.add(str(p))
                    count += 1
            self.status.setText(f"已添加 {count} 个任务 · 语言和输出参数已保存为各任务快照")
            self.refresh()
            if self.scheduler.accepting:
                self.scheduler._pump()
        except Exception as exc:
            QMessageBox.warning(self, "未添加任务", str(exc))

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "添加视频", str(ROOT / "input"), "视频 (*.mp4 *.mkv *.mov *.avi *.webm *.ts *.m2ts *.wmv)")
        self.add_paths(files)

    def add_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择视频文件夹", str(ROOT))
        if path:
            self.add_paths([p for p in Path(path).rglob("*") if p.is_file()])

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = []
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            paths.extend(p.rglob("*") if p.is_dir() else [p])
        self.add_paths(paths)

    def start(self):
        try:
            s = self.current_settings()
            s.save()
            self.settings = s
            if not self.scheduler.active:
                self.scheduler.settings = s
            self.scheduler.start()
            self.status.setText("队列已启动 · 稳定字幕段产出后立即翻译，全部译完进行全片审校")
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "无法开始", str(exc))

    def pause(self):
        self.scheduler.pause()
        self.status.setText("正在暂停队列，保留文字恢复点；在途接口请求可能仍需等待返回")

    def selected(self):
        i = self.queue.currentRow()
        return self.rows[i] if 0 <= i < len(self.rows) else None

    def delete_selected(self):
        ids = [self.rows[index.row()]["id"] for index in self.queue.selectionModel().selectedRows()]
        if not ids:
            self.status.setText("请先选择要删除的任务，可使用 Ctrl / Shift 多选")
            return
        self.scheduler.delete(ids)
        self.status.setText(f"正在删除 {len(ids)} 个任务记录与缓存；源视频和已完成成品保留")
        self.refresh()

    def clear_tasks(self):
        count = self.scheduler.delete()
        self.status.setText(f"正在清空 {count} 个任务记录与缓存；源视频和已完成成品保留")
        self.refresh()

    def retry(self):
        row = self.selected()
        if row:
            try:
                self.scheduler.retry(row["id"], self.current_settings())
                self.refresh()
            except Exception as exc:
                QMessageBox.warning(self, "无法重试", str(exc))

    def manual_title(self):
        row = self.selected()
        if row:
            title, ok = QInputDialog.getText(self, "填写目标语言标题", "输入不带扩展名的输出译名：")
            if ok and title.strip():
                self.scheduler.retry(row["id"], title=title)

    def open_output(self):
        row = self.selected()
        path = Path(row["output"]).parent if row and row["output"] else Path(self.settings.output_dir or (Path(row["source"]).parent if row else ROOT))
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def open_settings(self):
        dialog = SettingsDialog(self.current_settings(), self)
        if dialog.exec() == QDialog.Accepted:
            self.settings = dialog.settings
            self.preview_area.setVisible(self.settings.show_preview)
            if self.settings.show_preview:
                self.show_preview()
            else:
                self.preview.setRowCount(0)
            self.status.setText("设置已保存 · 使用当前设置重试可更新已有任务；资源设置变更后请重启工作台")
            self.refresh()

    def install_model(self):
        self.install_button.setDisabled(True)
        self.status.setText("正在安装/检查 large-v3 模型（约3GB），模型文件会保留供后续任务复用…")
        path = self.settings.model_path
        def run():
            try:
                from .models import install_model
                install_model(path)
                self.bridge.task_done.emit("model", "large-v3 模型安装完成")
            except Exception as exc:
                self.bridge.task_done.emit("model-error", f"模型安装失败：{type(exc).__name__}: {exc}")
        threading.Thread(target=run, daemon=True).start()

    def background_done(self, kind, data):
        if self.closing:
            return
        if kind == "media-info":
            if any(row["id"] == data["job"] for row in self.rows):
                self.media_info[data["job"]] = data
            self.refresh()
            return
        self.install_button.setEnabled(True)
        self.status.setText(str(data))
        self.refresh()

    def on_event(self, kind, data):
        if kind == "progress":
            self.progress[data["job"]] = data
        elif kind == "gpu":
            gpu, event = data.get("gpu"), data["type"]
            if event in ("started", "extracting", "transcribing"):
                self.asr_jobs[gpu] = data.get("job")
                self.gpu_states[gpu] = event
            elif event == "encoding_started":
                self.encoding_jobs[gpu] = data.get("job")
            elif event == "encoding_done":
                if self.encoding_jobs.get(gpu) == data.get("job"):
                    self.encoding_jobs.pop(gpu, None)
            elif event in ("ready", "done", "cancelled", "error", "fatal"):
                if self.asr_jobs.get(gpu) == data.get("job"):
                    self.asr_jobs.pop(gpu, None)
                self.gpu_states[gpu] = "error" if event in ("error", "fatal") else "ready"
            self.refresh_gpu_workers()
        elif kind == "error":
            self.status.setText("任务等待处理：" + data["error"])
            self.refresh()
        elif kind == "complete":
            self.status.setText("任务完成，缓存已清理：" + Path(data["output"]).name)
        elif kind == "usage":
            self.refresh()
        elif kind == "api":
            if data.get("event") in ("rate_limit", "quota", "paused", "timeout", "server_error"):
                self.refresh()
        elif kind == "preview":
            self.show_preview()

    def refresh(self):
        selected_ids = {self.rows[index.row()]["id"] for index in self.queue.selectionModel().selectedRows() if index.row() < len(self.rows)}
        self.rows = self.store.jobs()
        valid = {row["id"] for row in self.rows}
        for ident in list(self.media_info):
            if ident not in valid:
                self.media_info.pop(ident, None)
        for ident in list(self.inspections):
            if ident not in valid:
                self.inspections.pop(ident).cancel()
        try:
            estimate_settings = self.current_settings()
        except ValueError:
            estimate_settings = self.settings
        self.queue.blockSignals(True)
        self.queue.setRowCount(len(self.rows))
        self.queue.clearSelection()
        api_activity = self.scheduler.client.activity_snapshot() if hasattr(self.scheduler.client, "activity_snapshot") else []
        warnings = []
        for i, row in enumerate(self.rows):
            self.queue.setRowHeight(i, 86)
            title = Path(row["source"]).name
            item = QTableWidgetItem(title)
            item.setToolTip(row["source"] + ("\n输出：" + row["output"] if row["output"] else ""))
            self.queue.setItem(i, 0, item)
            status = QTableWidgetItem(row["state"])
            status.setForeground(QColor("#36745a" if row["state"].startswith("完成") else "#ad6338" if row["state"] == "等待处理" else "#706454"))
            self.queue.setItem(i, 1, status)
            p = QProgressBar()
            p.setRange(0, 100)
            progress = self.progress.get(row["id"], {})
            p.setValue(100 if row["state"].startswith("完成") else int(progress.get("progress", 0)*100))
            inactive = row["state"] in ("等待处理", "已暂停", "等待恢复") or row["state"].startswith("完成")
            p.setFormat(row["state"] if inactive else progress.get("stage", row["state"]))
            p.setToolTip(row["detail"])
            p.setFixedHeight(22)
            progress_cell = QWidget()
            progress_layout = QVBoxLayout(progress_cell)
            progress_layout.setContentsMargins(8, 0, 8, 0)
            progress_layout.addStretch()
            progress_layout.addWidget(p)
            progress_layout.addStretch()
            self.queue.setCellWidget(i, 2, progress_cell)
            tokens, usage_detail = token_summary(row)
            token_item = QTableWidgetItem(tokens)
            token_item.setToolTip(usage_detail)
            self.queue.setItem(i, 3, token_item)
            metadata = self.media_info.get(row["id"])
            size_item = QTableWidgetItem("读取中…")
            if metadata and "info" in metadata:
                completed = row["state"].startswith("完成")
                size_settings = Settings.from_dict(json.loads(row["config"])) if completed else estimate_settings
                estimate = estimate_output(metadata["info"], size_settings)
                size_item.setText("约 " + file_size(estimate["bytes"]))
                basis = "任务保存的设置" if completed else "当前设置"
                size_item.setToolTip(f"{basis}估算：{file_size(estimate['low'])}–{file_size(estimate['high'])}；实际大小受画面复杂度影响。已有任务按保存的配置执行，使用当前设置重试可更新。")
            elif metadata:
                size_item.setText("无法估算")
                size_item.setToolTip(metadata["error"])
            elif row["id"] not in self.inspections and not self.closing:
                def inspect(ident=row["id"], source=row["source"]):
                    try:
                        result = {"job": ident, "info": probe(source, dataclasses.replace(estimate_settings, audio_stream=-1))}
                    except Exception as exc:
                        result = {"job": ident, "error": str(exc)}
                    if not self.closing:
                        self.bridge.task_done.emit("media-info", result)
                self.inspections[row["id"]] = self.inspector.submit(inspect)
            if row["state"].startswith("完成"):
                try:
                    if not row.get("output"):
                        raise OSError("未记录成品路径")
                    actual_bytes = Path(row["output"]).stat().st_size
                    actual = file_size(actual_bytes)
                    actual_detail = f"成品实际大小：{actual}（{actual_bytes:,} 字节），直接读取成品文件。"
                except OSError:
                    actual = "文件不可用"
                    actual_detail = "无法读取成品文件；文件可能已移动、删除或所在磁盘离线。"
                size_item.setText(size_item.text()+"\n实际 " + actual)
                size_item.setToolTip(size_item.toolTip()+"\n"+actual_detail)
            self.queue.setItem(i, 4, size_item)
            requests = [r for r in api_activity if r["job"] == row["id"]]
            lines = [row["detail"]]
            quotas = self.scheduler.client.quota_snapshot(Settings.from_dict(json.loads(row["config"]))) if hasattr(self.scheduler.client, "quota_snapshot") else []
            lines.extend(provider_detail_lines(row, requests, quotas, self.scheduler.translator.stats(row["id"])["queued"]))
            notices = []
            warning = False
            for name, status_row in api_provider_records(row):
                notice = api_notice(status_row)
                if notice:
                    notices.append((name+"：" if name else "")+notice)
                limited = status_row.get("api_state") in ("paused", "quota", "timeout", "server_error") or (status_row.get("api_retry_at") or 0) > time.time()
                if status_row.get("api_state") == "switching":
                    limited = False
                if limited and notice and not row["state"].startswith("完成"):
                    repeated = row["state"] == "等待处理" and status_row.get("api_message") and status_row["api_message"] in row["detail"]
                    if not repeated:
                        warnings.append((name+"：" if name else "")+notice)
                    warning = True
            if not notices and requests:
                last_error = next((r["last_error"] for r in reversed(requests) if r.get("last_error")), "")
                if last_error:
                    lines.append(last_error)
            if row["state"] == "等待处理" and row["detail"]:
                warnings.append(row["detail"])
                warning = True
            detail = QTableWidgetItem("\n".join(lines))
            detail.setToolTip("\n".join(lines)+"\n429次数按当前任务和配置累计；同端点/凭据的冷却、并发与RPM在任务之间共享。\n限流可能来自并发、请求频率或token配额。RPM为本进程最近60秒已占用的请求预算，含重试；其他应用的用量无法统计。"+("\n旧版历史限流次数未记录。" if row.get("api_partial") else ""))
            if warning:
                detail.setForeground(QColor("#995126"))
            self.queue.setItem(i, 5, detail)
            wrapped = self.queue.fontMetrics().boundingRect(0, 0, max(120, self.queue.columnWidth(5)-24), 10000, Qt.TextWordWrap, "\n".join(lines)).height()
            self.queue.setRowHeight(i, max(86, min(440, 24+wrapped)))
            if row["id"] in selected_ids:
                self.queue.selectionModel().select(self.queue.model().index(i, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        self.queue.blockSignals(False)
        self.api_banner.setVisible(bool(warnings))
        self.api_banner.setText("接口提示：" + " / ".join(dict.fromkeys(warnings)))
        completed = sum(r["state"].startswith("完成") for r in self.rows)
        self.stat_values[0].setText(f"{len(self.rows)}  /  {completed} 完成")
        self.refresh_gpu_workers()
        mem = psutil.virtual_memory()
        self.stat_values[2].setText(f"{mem.available/1024**3:.1f} GB 可用")
        model = Path(self.settings.model_path, "model.bin")
        self.model_status.setText("● large-v3 模型已安装" if model.exists() else "○ 尚未安装 large-v3 模型")
        self.start_button.setText("队列运行中" if self.scheduler.accepting and self.scheduler.active else "开始队列")

    def refresh_gpu_workers(self):
        metrics = self.gpu_monitor.snapshot()
        busy = len(set(self.asr_jobs) | set(self.encoding_jobs))
        ready = sum(gpu in metrics or self.gpu_states.get(gpu) == "ready" for gpu in self.settings.gpus)
        self.stat_values[1].setText(f"{busy} / {len(self.settings.gpus)} 工作" if busy else f"{ready} / {len(self.settings.gpus)} 就绪")
        gpu_text = gpu_details(metrics, self.gpu_states, self.settings.gpus, self.asr_jobs, self.encoding_jobs, video_util=True)
        tooltip = "Tesla V100 · FP16 / NVENC\n"+gpu_text+"\n每1秒后台采样，面板每250毫秒更新；待命时模型仍可能占用显存。"
        self.stat_values[1].setToolTip(tooltip)
        for gpu, worker in self.gpu_workers.items():
            worker.update_status(metrics.get(gpu, {}), gpu_role(gpu, metrics, self.gpu_states, self.settings.gpus, self.asr_jobs, self.encoding_jobs))
            worker.setToolTip(tooltip)

    def show_preview(self):
        if not self.settings.show_preview:
            return
        row = self.selected()
        if not row:
            return
        segments = self.scheduler.previews.get(row["id"])
        if segments is None:
            raw = self.store.get(row["id"], "final_segments", [])
            translations = self.store.get(row["id"], "final_translations", {})
            segments = [{**s, "translation": translations.get(s["id"], "")} for s in raw]
        if not segments and row["output"]:
            self.preview.setRowCount(0)
            self.preview_summary.setText("字幕已烧录至视频；任务运行缓存已清理")
            return
        # Keep GUI responsive for long films; the full text remains in recovery state.
        visible = segments[-300:]
        self.preview.setRowCount(len(visible))
        for i, s in enumerate(visible):
            seconds = int(s["start"])
            for j, value in enumerate((f"{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02}", s["text"], s.get("translation", ""))):
                self.preview.setItem(i, j, QTableWidgetItem(value))
        self.preview.resizeRowsToContents()
        self.preview_summary.setText(f"{len(segments)} 段 · 已译 {sum(bool(s.get('translation')) for s in segments)} · 显示最近300段")

    def closeEvent(self, event):
        if self.scheduler.active:
            response = QMessageBox.question(self, "暂停并退出", "退出将暂停在途任务并保存文字恢复点。确认退出？")
            if response != QMessageBox.Yes:
                event.ignore()
                return
        self.timer.stop()
        self.gpu_timer.stop()
        self.closing = True
        self.gpu_monitor.stop()
        self.inspector.shutdown(wait=False, cancel_futures=True)
        self.scheduler.shutdown()
        event.accept()


def apply_theme(app):
    app.setStyle("Fusion")
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    palette = QPalette()
    for role, color in ((QPalette.Window, "#faf9f5"), (QPalette.WindowText, "#34332f"), (QPalette.Base, "#fffefa"), (QPalette.Text, "#34332f"), (QPalette.Button, "#f0eee6"), (QPalette.ButtonText, "#34332f"), (QPalette.Highlight, "#ead1c4"), (QPalette.HighlightedText, "#34332f")):
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    app.setStyleSheet(STYLE)


def run_gui(resume=False):
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Submaker Studio")
    app.setWindowIcon(QIcon(str(ROOT / "assets" / "app-light.ico")))
    translator = QTranslator(app)
    if translator.load("qtbase_zh_CN", QLibraryInfo.path(QLibraryInfo.TranslationsPath)):
        app.installTranslator(translator)
    app._submaker_translator = translator
    apply_theme(app)
    window = MainWindow()
    window.show()
    if resume:
        QTimer.singleShot(0, window.start)
    return app.exec()
