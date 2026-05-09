$ErrorActionPreference = "Stop"

$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvDir = Join-Path $ProjectDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "[setup] project: $ProjectDir"

if (Test-Path $VenvPython) {
    try {
        $prefix = & $VenvPython -c "import sys; print(sys.prefix)"
        if ($prefix.Trim() -ne $VenvDir) {
            Write-Host "[setup] existing .venv points to old path, recreating..."
            Remove-Item -Recurse -Force $VenvDir
        }
    } catch {
        Write-Host "[setup] unable to validate existing venv, recreating..."
        if (Test-Path $VenvDir) { Remove-Item -Recurse -Force $VenvDir }
    }
}

if (-not (Test-Path $VenvPython)) {
    py -3 -m venv $VenvDir
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $ProjectDir "requirements.txt")

Write-Host "[setup] done"
Write-Host "[setup] activate with: .\.venv\Scripts\Activate.ps1"
