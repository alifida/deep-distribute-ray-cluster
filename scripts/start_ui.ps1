$ErrorActionPreference = "Stop"

$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ConfigFile = Join-Path $PSScriptRoot "cluster_config.env"
if (Test-Path $ConfigFile) {
    Get-Content $ConfigFile | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) { return }
        if ($line -match "^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$") {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
}
$VenvDir = Join-Path $ProjectDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[ui] .venv missing, running setup_node.ps1..."
    & (Join-Path $PSScriptRoot "setup_node.ps1")
}

try {
    $prefix = & $VenvPython -c "import sys; print(sys.prefix)"
    if ($prefix.Trim() -ne $VenvDir) {
        Write-Host "[ui] stale .venv detected, running setup_node.ps1..."
        & (Join-Path $PSScriptRoot "setup_node.ps1")
    }
} catch {
    Write-Host "[ui] .venv validation failed, running setup_node.ps1..."
    & (Join-Path $PSScriptRoot "setup_node.ps1")
}

$HostIp = if ($env:HOST) { $env:HOST } else { "0.0.0.0" }
$Port = if ($env:PORT) { $env:PORT } else { "8080" }

Set-Location $ProjectDir
Write-Host "[ui] starting FastAPI UI on http://$HostIp`:$Port"
& $VenvPython -m uvicorn api:app --host $HostIp --port $Port
