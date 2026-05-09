param(
    [Parameter(Mandatory = $true)][string]$HeadIp,
    [Parameter(Mandatory = $false)][string]$HeadPort = "6379"
)

$ErrorActionPreference = "Stop"

$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvDir = Join-Path $ProjectDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[worker] .venv missing, running setup_node.ps1..."
    & (Join-Path $PSScriptRoot "setup_node.ps1")
}

try {
    $prefix = & $VenvPython -c "import sys; print(sys.prefix)"
    if ($prefix.Trim() -ne $VenvDir) {
        Write-Host "[worker] stale .venv detected, running setup_node.ps1..."
        & (Join-Path $PSScriptRoot "setup_node.ps1")
    }
} catch {
    Write-Host "[worker] .venv validation failed, running setup_node.ps1..."
    & (Join-Path $PSScriptRoot "setup_node.ps1")
}

& $VenvPython -c "import ray" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[worker] ray not found in venv, installing ray..."
    & $VenvPython -m pip install ray
}

$NumGpus = if ($env:NUM_GPUS) { $env:NUM_GPUS } else { "1" }
$NodeManagerPort = if ($env:NODE_MANAGER_PORT) { $env:NODE_MANAGER_PORT } else { "10011" }
$ObjectManagerPort = if ($env:OBJECT_MANAGER_PORT) { $env:OBJECT_MANAGER_PORT } else { "10012" }
$DashboardAgentPort = if ($env:DASHBOARD_AGENT_LISTEN_PORT) { $env:DASHBOARD_AGENT_LISTEN_PORT } else { "10014" }
$MetricsPort = if ($env:METRICS_EXPORT_PORT) { $env:METRICS_EXPORT_PORT } else { "10015" }
$MinWorkerPort = if ($env:MIN_WORKER_PORT) { $env:MIN_WORKER_PORT } else { "12000" }
$MaxWorkerPort = if ($env:MAX_WORKER_PORT) { $env:MAX_WORKER_PORT } else { "12999" }

Write-Host "[worker] stopping old Ray runtime"
& $VenvPython -m ray.scripts.scripts stop

Write-Host "[worker] connecting to $HeadIp:$HeadPort"
& $VenvPython -m ray.scripts.scripts start `
  --address="$HeadIp`:$HeadPort" `
  --node-manager-port="$NodeManagerPort" `
  --object-manager-port="$ObjectManagerPort" `
  --dashboard-agent-listen-port="$DashboardAgentPort" `
  --metrics-export-port="$MetricsPort" `
  --min-worker-port="$MinWorkerPort" `
  --max-worker-port="$MaxWorkerPort" `
  --num-gpus="$NumGpus"

Write-Host "[worker] started and joined cluster"
