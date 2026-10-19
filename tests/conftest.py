import gc
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qt_application():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def dispose_test_widgets_on_gui_thread(qt_application):
    """Keep Qt alive and collect closed widget cycles before HTTP workers run GC."""
    yield
    from PySide6.QtCore import QCoreApplication, QEvent
    for widget in qt_application.topLevelWidgets():
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    gc.collect()
