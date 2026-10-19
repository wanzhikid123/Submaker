$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$distTarget = [IO.Path]::GetFullPath((Join-Path $projectRoot 'dist\SubmakerStudio'))
$expectedPrefix = [IO.Path]::GetFullPath($projectRoot).TrimEnd('\') + '\'
if (-not $distTarget.StartsWith($expectedPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Build target outside project root' }
& '.\.venv\python.exe' -s -m PyInstaller --noconfirm --clean --onedir --windowed --name SubmakerStudio --icon assets/app-light.ico scripts/client_launcher.py
if ($LASTEXITCODE -ne 0) { throw 'Launcher build failed' }
$shortcutPath = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Submaker Studio.lnk'
$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($shortcutPath)
$shortcut.TargetPath = Join-Path $projectRoot 'dist\SubmakerStudio\SubmakerStudio.exe'
$shortcut.WorkingDirectory = $projectRoot
$shortcut.IconLocation = Join-Path $projectRoot 'assets\app-light.ico'
$shortcut.Description = 'Submaker Studio - local dual-GPU subtitle workbench'
$shortcut.Save()
Write-Host "Deployed: $projectRoot\dist\SubmakerStudio\SubmakerStudio.exe"
