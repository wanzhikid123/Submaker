$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv\python.exe')) {
    conda create --prefix "$projectRoot\.venv" python=3.12 pip -y
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed' }
}
& '.\.venv\python.exe' -s scripts/download_cuda.py
if ($LASTEXITCODE -ne 0) { throw 'CUDA wheel download failed' }
& '.\.venv\python.exe' -s -m pip install --find-links build/wheels -r requirements.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Locked dependency installation failed' }
& '.\.venv\python.exe' -s -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& '.\.venv\python.exe' -s scripts/download_model.py
if ($LASTEXITCODE -ne 0) { throw 'Model download failed' }
& '.\.venv\python.exe' -s -m submaker --diagnose
if ($LASTEXITCODE -ne 0) { throw 'GPU diagnosis failed' }
$shortcutPath = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Submaker Studio.lnk'
$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($shortcutPath)
$shortcut.TargetPath = Join-Path $projectRoot '.venv\pythonw.exe'
$shortcut.Arguments = '-s -m submaker'
$shortcut.WorkingDirectory = $projectRoot
$shortcut.Description = 'Submaker Studio - local dual-GPU subtitle workbench'
$shortcut.Save()
Write-Host 'Installed. Open Submaker Studio from your desktop.'
