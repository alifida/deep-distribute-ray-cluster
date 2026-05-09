$ErrorActionPreference = "Stop"

$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvDir = Join-Path $ProjectDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[stop-head] .venv not found. Nothing to stop."
    exit 0
}

Write-Host "[stop-head] stopping Ray runtime"
& $VenvPython -m ray.scripts.scripts stop
Write-Host "[stop-head] done"
