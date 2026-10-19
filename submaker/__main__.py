from __future__ import annotations

import argparse
import multiprocessing
import sys


def main():
    multiprocessing.freeze_support()
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Submaker.Studio")
    parser = argparse.ArgumentParser(description="Submaker Studio")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted jobs after opening the workbench")
    args = parser.parse_args()
    if args.diagnose:
        import json
        import psutil
        import subprocess
        from .asr import cuda_paths
        from .config import Settings
        cuda_paths()
        import ctranslate2
        s = Settings.load()
        print(json.dumps({"threads": psutil.cpu_count(), "ram_gib": round(psutil.virtual_memory().total/1024**3,2), "ctranslate2": ctranslate2.__version__, "cuda_devices": ctranslate2.get_cuda_device_count(), "compute_types": list(ctranslate2.get_supported_compute_types("cuda",0)), "model_path": s.model_path}, ensure_ascii=False, indent=2))
        return 0
    from .gui import run_gui
    from .config import DATA
    import msvcrt
    lock = (DATA / "app.lock").open("a+b")
    lock.seek(0)
    if lock.read(1) == b"":
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.information(None, "Submaker Studio", "工作台已经运行，请切换到已有窗口。")
        return 0
    try:
        return run_gui(resume=args.resume)
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
