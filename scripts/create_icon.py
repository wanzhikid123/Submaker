from pathlib import Path
from PySide6.QtGui import QImage, QPainter, QPainterPath, QColor, QPen
from PySide6.QtCore import Qt, QRectF
from PySide6.QtWidgets import QApplication

app=QApplication([])
root=Path(__file__).resolve().parents[1]
out=root/"assets"
out.mkdir(exist_ok=True)
image=QImage(256,256,QImage.Format_ARGB32)
image.fill(Qt.transparent)
p=QPainter(image)
p.setRenderHint(QPainter.Antialiasing)
p.setBrush(QColor("#faf9f5"))
p.setPen(QPen(QColor("#e4d4c1"),3))
p.drawRoundedRect(QRectF(4,4,248,248),52,52)
p.setPen(QPen(QColor("#c66d4e"),8))
p.setBrush(QColor("#fffefa"))
p.drawRoundedRect(QRectF(44,55,168,145),21,21)
p.setPen(Qt.NoPen)
p.setBrush(QColor("#c66d4e"))
play=QPainterPath()
play.moveTo(116,83)
play.lineTo(146,103)
play.lineTo(116,123)
play.closeSubpath()
p.drawPath(play)
p.setPen(QPen(QColor("#c66d4e"),8,Qt.SolidLine,Qt.RoundCap))
p.drawLine(76,150,180,150)
p.drawLine(76,174,151,174)
p.end()
image.save(str(out/"app.png"))
if not image.save(str(out/"app.ico"),"ICO"):
    raise RuntimeError("Icon export failed")
if not image.save(str(out/"app-light.ico"),"ICO"):
    raise RuntimeError("Light icon export failed")
print("ICON_CREATED")
