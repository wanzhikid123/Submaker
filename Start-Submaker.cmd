@echo off
cd /d "%~dp0"
if not exist ".venv\pythonw.exe" (
  echo Runtime missing. Run scripts\install.ps1 first.
  pause
  exit /b 1
)
start "" ".venv\pythonw.exe" -s -m submaker
