from __future__ import annotations

import dataclasses
from pathlib import Path
import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QHBoxLayout, QGroupBox, QLabel,
                               QLineEdit, QPushButton, QFileDialog, QComboBox, QSpinBox)

from .config import LocalModel, LocalRole, Settings


def number(value, minimum, maximum):
    widget = QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    return widget


class LocalRoleEditor(QWidget):
    def __init__(self, options, review=False):
        super().__init__()
        outer = QVBoxLayout(self)
        note = QLabel("本地审校、上下文整理、自动修正和文件名翻译使用以下设置。" if review else "稳定字幕段产出后立即进入本地翻译队列。")
        note.setWordWrap(True)
        outer.addWidget(note)
        group = QGroupBox("本地审校参数" if review else "本地翻译参数")
        form = QFormLayout(group)
        form.setSpacing(16)
        self.effort = QComboBox()
        for text, value in (("关闭思考 · 直接生成", "none"), ("低 · low", "low"), ("中 · medium", "medium"), ("高 · high", "high")):
            self.effort.addItem(text, value)
        self.effort.setCurrentIndex(max(0, self.effort.findData(options.effort)))
        self.effort.setToolTip("关闭会禁用 thinking；其余档位传给模型模板，实际差异取决于 GGUF 模板支持。思考 token 计入输出上限。")
        self.max_tokens = number(options.max_tokens, 512, 8192)
        self.timeout = number(options.timeout, 60, 7200)
        self.timeout.setSuffix(" 秒")
        form.addRow("Reasoning effort", self.effort)
        form.addRow("单次最大输出 tokens", self.max_tokens)
        form.addRow("请求超时", self.timeout)
        outer.addWidget(group)
        caption = QLabel("GPU 与模型文件在“本地模型 / GPU”中设置。翻译与审校共用一个常驻模型，推理固定单路。网络接口池已停用。")
        caption.setWordWrap(True)
        caption.setObjectName("muted")
        outer.addWidget(caption)
        outer.addStretch()

    def value(self):
        return LocalRole(self.effort.currentData(), self.max_tokens.value(), self.timeout.value())


class LocalBridge(QObject):
    progress = Signal(str)
    done = Signal(bool, str)


class LocalModelEditor(QWidget):
    def __init__(self, options):
        super().__init__()
        self.options = dataclasses.replace(options)
        self.busy = False
        self.bridge = LocalBridge(self)
        self.bridge.progress.connect(self.show_status)
        self.bridge.done.connect(self.finished)
        outer = QVBoxLayout(self)
        title = QLabel("Huihui Qwen3.8-27B · IQ3_S · 约 12.0 GB")
        title.setObjectName("section")
        outer.addWidget(title)
        note = QLabel("模型专用 GPU 默认选 GPU 1，Whisper 和视频编解码使用另一张卡。处理任务期间仅访问本机；安装按钮需要联网下载官方权重与 CUDA 引擎。")
        note.setWordWrap(True)
        note.setObjectName("muted")
        outer.addWidget(note)
        group = QGroupBox("模型与资源")
        form = QFormLayout(group)
        form.setSpacing(14)
        self.model_path = QLineEdit(options.model_path)
        self.server_path = QLineEdit(options.server_path)
        for caption, field, pattern in (("GGUF 文件", self.model_path, "GGUF (*.gguf)"), ("本地引擎", self.server_path, "本地引擎 (ollama.exe llama-server.exe)")):
            row = QHBoxLayout()
            row.addWidget(field, 1)
            choose = QPushButton("选择…")
            choose.clicked.connect(lambda checked=False, target=field, filter=pattern: self.choose(target, filter))
            row.addWidget(choose)
            form.addRow(caption, row)
        self.gpu = QComboBox()
        self.gpu.addItem("GPU 1 · 本地模型专用（默认）", 1)
        self.gpu.addItem("GPU 0 · 本地模型专用", 0)
        self.gpu.setCurrentIndex(max(0, self.gpu.findData(options.gpu)))
        self.context = QComboBox()
        for value in (8192, 16384, 32768):
            self.context.addItem(f"{value:,} tokens" + (" · 默认" if value == 16384 else ""), value)
        self.context.setCurrentIndex(max(0, self.context.findData(options.context)))
        self.context.setToolTip("上下文越长越占显存；16 GB 显卡建议先用默认 16K，显存不足可改 8K。翻译与审校的输出也占上下文。")
        self.threads = number(options.threads, 1, 16)
        for caption, field in (("模型 GPU", self.gpu), ("上下文长度 / 每请求", self.context), ("本地引擎 CPU 线程", self.threads)):
            form.addRow(caption, field)
        outer.addWidget(group)
        actions = QHBoxLayout()
        self.install = QPushButton("↓ 安装预设模型与 CUDA 引擎 · 约 14 GB")
        self.install.clicked.connect(self.install_all)
        self.engine = QPushButton("仅安装 CUDA 引擎")
        self.engine.clicked.connect(self.install_engine)
        self.test = QPushButton("加载并测试本地模型")
        self.test.clicked.connect(self.test_model)
        for button in (self.install, self.engine, self.test):
            actions.addWidget(button)
        outer.addLayout(actions)
        self.status = QLabel("按需加载；退出软件自动释放显存。推理日志仅保留在内存。")
        self.status.setWordWrap(True)
        self.status.setObjectName("muted")
        outer.addWidget(self.status)
        self.paths = QLabel("")
        self.paths.setWordWrap(True)
        outer.addWidget(self.paths)
        outer.addStretch()

    def choose(self, target, filter):
        path, _ = QFileDialog.getOpenFileName(self, "选择本地模型文件", target.text(), filter)
        if path:
            target.setText(path)

    def value(self):
        return LocalModel(self.server_path.text().strip(), self.model_path.text().strip(), self.gpu.currentData(), self.context.currentData(), self.threads.value())

    def show_status(self, message):
        self.status.setText(message)

    def run(self, work):
        self.busy = True
        for button in (self.install, self.engine, self.test):
            button.setEnabled(False)
        bridge = self.bridge
        def progress(message):
            try:
                bridge.progress.emit(message)
            except RuntimeError:
                pass
        def run():
            try:
                message = work(progress)
                bridge.done.emit(True, message)
            except Exception as exc:
                try:
                    bridge.done.emit(False, str(exc))
                except RuntimeError:
                    pass
        threading.Thread(target=run, daemon=True, name="local-model-setup").start()

    def install_all(self):
        from .local_install import install_local
        defaults = LocalModel()
        self.model_path.setText(defaults.model_path)
        self.server_path.setText(defaults.server_path)
        self.status.setText("正在安装预设模型；下载支持断点续传，完成后校验 SHA256…")
        self.run(lambda progress: (install_local(defaults, progress), "模型与 CUDA 引擎已就绪，可以加载测试")[1])

    def install_engine(self):
        from .local_install import install_engine
        defaults = LocalModel()
        self.server_path.setText(defaults.server_path)
        self.status.setText("正在安装带数字签名的 Ollama 官方 CUDA 引擎…")
        self.run(lambda progress: (install_engine(defaults, progress), "CUDA 引擎已安装，请选择 GGUF 文件后测试")[1])

    def test_model(self):
        from .llm import Client
        options = self.value()
        self.status.setText("正在加载并测试本地模型；首次加载需要等待…")
        def work(progress):
            settings = Settings(llm_mode="local", local_model=options)
            client = Client(local_only=True)
            result = client.request(settings.local_profile("translate"), 'Connection test. Output {"ok":true}.', {"test": True})
            if result.get("ok") is not True:
                raise RuntimeError("模型可达，但 JSON 结果校验失败")
            return f"GPU {options.gpu} · 本地推理与 JSON 校验通过 · 固定单路推理"
        self.run(work)

    def finished(self, ok, message):
        self.busy = False
        for button in (self.install, self.engine, self.test):
            button.setEnabled(True)
        self.status.setText(message)
