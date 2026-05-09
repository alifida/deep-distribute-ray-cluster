param(
    [Parameter(Mandatory = $false)][string]$HeadIp,
    [Parameter(Mandatory = $false)][string]$HeadPort = "6379"
)

$ErrorActionPreference = "Stop"

function Test-RayImport {
    param([Parameter(Mandatory = $true)][string]$PythonExe)
    try {
        & $PythonExe -c "import ray" *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

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
    Write-Host "[worker] .venv missing, running setup_node.ps1..."
    & (Join-Path $PSScriptRoot "setup_node.ps1")
}

try {
    $prefix = & $VenvPython -c "import sys; print(sys.prefix)"
    $pyMm = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    $supported = @("3.10", "3.11") -contains $pyMm.Trim()
    if ($prefix.Trim() -ne $VenvDir -or -not $supported) {
        Write-Host "[worker] stale/unsupported .venv detected, running setup_node.ps1..."
        & (Join-Path $PSScriptRoot "setup_node.ps1")
    }
} catch {
    Write-Host "[worker] .venv validation failed, running setup_node.ps1..."
    & (Join-Path $PSScriptRoot "setup_node.ps1")
}

if (-not (Test-RayImport -PythonExe $VenvPython)) {
    Write-Host "[worker] ray not found in venv, installing ray..."
    & $VenvPython -m pip install ray
    if ($LASTEXITCODE -ne 0) {
        $pyVer = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
        throw "[worker] Failed to install ray. Python version: $pyVer. Use Python 3.10 or 3.11 on Windows and rerun setup_node.ps1."
    }
    if (-not (Test-RayImport -PythonExe $VenvPython)) {
        $pyVer = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
        throw "[worker] Ray import still failing. Python version: $pyVer. Use Python 3.10 or 3.11 and rerun setup."
    }
}

$HeadIp = if ($HeadIp) { $HeadIp } elseif ($env:HEAD_IP) { $env:HEAD_IP } else { "" }
$HeadPort = if ($HeadPort -and $HeadPort -ne "6379") { $HeadPort } elseif ($env:HEAD_PORT) { $env:HEAD_PORT } else { "6379" }
if (-not $HeadIp) {
    Write-Host "Usage: .\scripts\start_worker.ps1 <HEAD_IP> [HEAD_PORT]"
    Write-Host "Or set HEAD_IP in scripts\cluster_config.env"
    exit 1
}

$NumGpus = if ($env:NUM_GPUS) { $env:NUM_GPUS } else { "1" }
$NodeManagerPort = if ($env:WORKER_NODE_MANAGER_PORT) { $env:WORKER_NODE_MANAGER_PORT } elseif ($env:NODE_MANAGER_PORT) { $env:NODE_MANAGER_PORT } else { "10011" }
$ObjectManagerPort = if ($env:WORKER_OBJECT_MANAGER_PORT) { $env:WORKER_OBJECT_MANAGER_PORT } elseif ($env:OBJECT_MANAGER_PORT) { $env:OBJECT_MANAGER_PORT } else { "10012" }
$DashboardAgentPort = if ($env:WORKER_DASHBOARD_AGENT_LISTEN_PORT) { $env:WORKER_DASHBOARD_AGENT_LISTEN_PORT } elseif ($env:DASHBOARD_AGENT_LISTEN_PORT) { $env:DASHBOARD_AGENT_LISTEN_PORT } else { "10014" }
$MetricsPort = if ($env:WORKER_METRICS_EXPORT_PORT) { $env:WORKER_METRICS_EXPORT_PORT } elseif ($env:METRICS_EXPORT_PORT) { $env:METRICS_EXPORT_PORT } else { "10015" }
$MinWorkerPort = if ($env:WORKER_MIN_WORKER_PORT) { $env:WORKER_MIN_WORKER_PORT } elseif ($env:MIN_WORKER_PORT) { $env:MIN_WORKER_PORT } else { "12000" }
$MaxWorkerPort = if ($env:WORKER_MAX_WORKER_PORT) { $env:WORKER_MAX_WORKER_PORT } elseif ($env:MAX_WORKER_PORT) { $env:MAX_WORKER_PORT } else { "12999" }

Write-Host "[worker] stopping old Ray runtime"
& $VenvPython -m ray.scripts.scripts stop

$HeadAddress = "{0}:{1}" -f $HeadIp, $HeadPort
Write-Host ("[worker] connecting to {0}" -f $HeadAddress)
& $VenvPython -m ray.scripts.scripts start `
  --address="$HeadAddress" `
  --node-manager-port="$NodeManagerPort" `
  --object-manager-port="$ObjectManagerPort" `
  --dashboard-agent-listen-port="$DashboardAgentPort" `
  --metrics-export-port="$MetricsPort" `
  --min-worker-port="$MinWorkerPort" `
  --max-worker-port="$MaxWorkerPort" `
  --num-gpus="$NumGpus"

Write-Host "[worker] started and joined cluster"
