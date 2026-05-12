# Titan one-shot installer for Windows PowerShell.
# Usage from a checkout: powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
# Usage from GitHub:
#   irm https://raw.githubusercontent.com/austindixson/Titan/main/scripts/install.ps1 | iex
# Override repo:
#   $env:TITAN_REPO_URL='https://github.com/OWNER/REPO.git'; .\scripts\install.ps1

$ErrorActionPreference = 'Stop'

$RepoUrl = if ($env:TITAN_REPO_URL) { $env:TITAN_REPO_URL } else { 'https://github.com/austindixson/Titan.git' }
$InstallRoot = if ($env:TITAN_INSTALL_ROOT) { $env:TITAN_INSTALL_ROOT } else { Join-Path $HOME '.titan' }
$VenvDir = Join-Path $InstallRoot 'venv'
$BinDir = Join-Path $HOME '.local\bin'

function Log($msg) { Write-Host "[titan-install] $msg" }
function Fail($msg) { Write-Error "[titan-install] ERROR: $msg"; exit 1 }

$PythonExe = $null
$PythonArgs = @()
if ($env:PYTHON) {
  $PythonExe = $env:PYTHON
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
  $PythonExe = 'py'
  $PythonArgs = @('-3')
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
  $PythonExe = 'python'
}
if (-not $PythonExe) { Fail 'Python 3.10+ is required. Install it from https://python.org/downloads/ and rerun.' }

& $PythonExe @PythonArgs -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) { Fail 'Python 3.10+ is required.' }

New-Item -ItemType Directory -Force -Path $InstallRoot, $BinDir | Out-Null
Log "creating virtual environment at $VenvDir"
& $PythonExe @PythonArgs -m venv $VenvDir
if ($LASTEXITCODE -ne 0) { Fail 'failed to create virtual environment' }

$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Fail 'failed to upgrade pip' }

if ((Test-Path 'pyproject.toml') -and (Test-Path 'src\titan')) {
  Log 'installing Titan from current checkout'
  & $VenvPython -m pip install -e .
} else {
  Log "installing Titan from $RepoUrl"
  & $VenvPython -m pip install "git+$RepoUrl"
}
if ($LASTEXITCODE -ne 0) { Fail 'failed to install Titan' }

$TitanExe = Join-Path $VenvDir 'Scripts\titan.exe'
$TitanTuiExe = Join-Path $VenvDir 'Scripts\titan-tui.exe'
$TitanCmd = Join-Path $BinDir 'titan.cmd'
$TitanTuiCmd = Join-Path $BinDir 'titan-tui.cmd'

Set-Content -Path $TitanCmd -Value "@echo off`r`n`"$TitanExe`" %*`r`n" -Encoding ASCII
Set-Content -Path $TitanTuiCmd -Value "@echo off`r`n`"$TitanTuiExe`" %*`r`n" -Encoding ASCII

& $TitanExe setup | Out-Null
& $TitanExe config set chat_recaps_enabled false | Out-Null

Log 'installed Titan'
Log "binaries: $TitanCmd and $TitanTuiCmd"
if (-not (($env:PATH -split ';') -contains $BinDir)) {
  Log "add this folder to PATH if needed: $BinDir"
}
Log 'next: titan doctor'
