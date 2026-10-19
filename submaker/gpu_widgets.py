from __future__ import annotations

import math

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget


class GpuUtilizationRing(QWidget):
    def __init__(self, gpu, parent=None):
        super().__init__(parent)
        self.gpu = gpu
        self.utilization = None
        self.setFixedSize(78, 78)
        self.setAccessibleName(f"GPU {gpu} 利用率")
        self.set_utilization(None)

    @property
    def ring_color(self):
        if self.utilization is None:
            return QColor("#bdb8ad")
        if self.utilization > 90:
            return QColor("#cf6557")
        if self.utilization > 70:
            return QColor("#d0a132")
        return QColor("#4e906b")

    def set_utilization(self, value):
        value = float(value) if value is not None else None
        if value is not None:
            value = max(0, min(100, value)) if math.isfinite(value) else None
        self.utilization = value
        percent = "—" if value is None else f"{value:.0f}%"
        self.setAccessibleDescription(percent)
        self.setToolTip(f"GPU {self.gpu} 利用率 {percent}\n≤70% 绿色 · >70% 黄色 · >90% 红色")
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        bounds = QRectF(6, 6, self.width()-12, self.height()-12)
        pen = QPen(QColor("#edeae2"), 6, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        painter.drawEllipse(bounds)
        if self.utilization is not None:
            pen.setColor(self.ring_color)
            painter.setPen(pen)
            painter.drawArc(bounds, 90*16, -max(1, round(self.utilization*3.6*16)))
        painter.setPen(QColor("#39372f"))
        font = QFont(self.font())
        font.setPixelSize(19)
        font.setWeight(QFont.DemiBold)
        painter.setFont(font)
        painter.drawText(QRectF(8, 20, self.width()-16, 25), Qt.AlignCenter, "—" if self.utilization is None else f"{self.utilization:.0f}%")
        font.setPixelSize(10)
        font.setWeight(QFont.Normal)
        painter.setFont(font)
        painter.setPen(QColor("#77746b"))
        painter.drawText(QRectF(8, 45, self.width()-16, 17), Qt.AlignCenter, f"GPU {self.gpu}")


class GpuWorkerStatus(QWidget):
    def __init__(self, gpu, parent=None):
        super().__init__(parent)
        self.ring = GpuUtilizationRing(gpu, self)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self.ring)
        details = QVBoxLayout()
        details.setSpacing(4)
        self.role, self.memory, self.video = QLabel("读取状态中"), QLabel("显存 —"), QLabel("解码 — · 编码 —")
        self.speed = QLabel()
        self.speed.setStyleSheet("font-size:12px;font-weight:600;color:#b87355;")
        self.speed.hide()
        for text in (self.role, self.memory, self.video):
            text.setWordWrap(True)
            text.setMinimumWidth(0)
            details.addWidget(text)
        details.insertWidget(1, self.speed)
        self.role.setStyleSheet("font-size:12px;font-weight:600;color:#484438;")
        self.memory.setStyleSheet("font-size:11px;color:#77746b;")
        self.video.setStyleSheet("font-size:11px;color:#77746b;")
        layout.addLayout(details, 1)

    def update_status(self, values, role, local=None):
        self.ring.set_utilization(values.get("utilization"))
        self.role.setText(role)
        self.speed.setVisible(local is not None)
        if local is not None:
            speed = local.get("tokens_per_second")
            measured = isinstance(speed, (int, float)) and math.isfinite(speed) and speed >= 0
            self.speed.setText((f"{speed:.1f}" if measured else "—")+" tokens/s")
            phase = {"generating": "生成中", "sampling": "正在采样", "prefill": "处理输入中", "idle": "待命"}.get(local.get("speed_phase"), "暂无速度数据")
            self.speed.setToolTip(f"{phase} · 当前输出生成速度\n最近约3秒平滑，不含输入处理时间；请求切换后重新采样。")
        if values.get("memory_used") is not None and values.get("memory_total"):
            self.memory.setText(f"显存 {values['memory_used']/1024:.1f}/{values['memory_total']/1024:.0f} GiB")
        else:
            self.memory.setText("显存 —")
        percent = lambda key: "—" if values.get(key) is None else f"{values[key]:.0f}%"
        self.video.setText(f"解码 {percent('decoder')}\n编码 {percent('encoder')}")
