$ErrorActionPreference = "Stop"

$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvDir = Join-Path $ProjectDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

function Invoke-Checked {
    param([Parameter(Mandatory = $true)][string]$CommandLine)
    Invoke-Expression $CommandLine
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $CommandLine"
    }
}

Write-Host "[setup] project: $ProjectDir"

if (Test-Path $VenvPython) {
    try {
        $prefix = & $VenvPython -c "import sys; print(sys.prefix)"
        $pyMm = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        $supported = @("3.10", "3.11") -contains $pyMm.Trim()
        if ($prefix.Trim() -ne $VenvDir -or -not $supported) {
            Write-Host "[setup] existing .venv points to old path, recreating..."
            if (-not $supported) {
                Write-Host "[setup] existing .venv python version ($pyMm) is not supported for Ray on Windows, recreating..."
            }
            Remove-Item -Recurse -Force $VenvDir
        }
    } catch {
        Write-Host "[setup] unable to validate existing venv, recreating..."
        if (Test-Path $VenvDir) { Remove-Item -Recurse -Force $VenvDir }
    }
}

if (-not (Test-Path $VenvPython)) {
    $created = $false
    foreach ($spec in @("-3.11", "-3.10")) {
        try {
            Invoke-Checked "py $spec -m venv `"$VenvDir`""
            $created = $true
            Write-Host "[setup] created venv with Python $spec"
            break
        } catch {
            if (Test-Path $VenvDir) { Remove-Item -Recurse -Force $VenvDir -ErrorAction SilentlyContinue }
        }
    }
    if (-not $created) {
        throw "[setup] Could not create venv with Python 3.11/3.10. Install Python 3.11 (recommended) and ensure py launcher is available."
    }
}

Invoke-Checked "`"$VenvPython`" -m pip install --upgrade pip"
Invoke-Checked "`"$VenvPython`" -m pip install -r `"$((Join-Path $ProjectDir "requirements.txt"))`""

& $VenvPython -c "import ray" 2>$null
if ($LASTEXITCODE -ne 0) {
    $pyVer = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    throw "[setup] Ray import failed after install. Python version: $pyVer. Ray wheels on Windows may not be available for this Python version. Use Python 3.10 or 3.11 and rerun setup."
}

Write-Host "[setup] done"
Write-Host "[setup] activate with: .\.venv\Scripts\Activate.ps1"
