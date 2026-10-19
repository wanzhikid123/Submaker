"""Small native launcher for the installed, isolated local runtime."""
from pathlib import Path
import ctypes
import subprocess
import sys


def main():
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parents[2]
    else:
        root = Path(__file__).resolve().parents[1]
    python = root / ".venv" / "pythonw.exe"
    if not python.exists():
        ctypes.windll.user32.MessageBoxW(None, "找不到本机运行环境，请在项目目录运行 scripts/install.ps1。", "Submaker Studio", 0x10)
        return 1
    subprocess.Popen([str(python), "-s", "-m", "submaker"], cwd=root, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
