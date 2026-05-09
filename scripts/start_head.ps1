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
$mixedOsCluster = if ($env:RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER) { $env:RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER } else { "0" }
[Environment]::SetEnvironmentVariable("RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER", $mixedOsCluster, "Process")
$VenvDir = Join-Path $ProjectDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[head] .venv missing, running setup_node.ps1..."
    & (Join-Path $PSScriptRoot "setup_node.ps1")
}

try {
    $prefix = & $VenvPython -c "import sys; print(sys.prefix)"
    if ($prefix.Trim() -ne $VenvDir) {
        Write-Host "[head] stale .venv detected, running setup_node.ps1..."
        & (Join-Path $PSScriptRoot "setup_node.ps1")
    }
} catch {
    Write-Host "[head] .venv validation failed, running setup_node.ps1..."
    & (Join-Path $PSScriptRoot "setup_node.ps1")
}

& $VenvPython -c "import ray" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[head] ray not found in venv, installing ray..."
    & $VenvPython -m pip install ray
}

$RayPort = if ($env:RAY_PORT) { $env:RAY_PORT } else { "6379" }
$DashboardPort = if ($env:DASHBOARD_PORT) { $env:DASHBOARD_PORT } else { "8265" }
$NumGpus = if ($env:NUM_GPUS) { $env:NUM_GPUS } else { "1" }
$RestartRay = if ($env:RESTART_RAY) { $env:RESTART_RAY } else { "0" }
$NodeManagerPort = if ($env:NODE_MANAGER_PORT) { $env:NODE_MANAGER_PORT } else { "10001" }
$ObjectManagerPort = if ($env:OBJECT_MANAGER_PORT) { $env:OBJECT_MANAGER_PORT } else { "10002" }
$RayClientPort = if ($env:RAY_CLIENT_SERVER_PORT) { $env:RAY_CLIENT_SERVER_PORT } else { "10003" }
$DashboardAgentPort = if ($env:DASHBOARD_AGENT_LISTEN_PORT) { $env:DASHBOARD_AGENT_LISTEN_PORT } else { "10004" }
$MetricsPort = if ($env:METRICS_EXPORT_PORT) { $env:METRICS_EXPORT_PORT } else { "10005" }
$MinWorkerPort = if ($env:MIN_WORKER_PORT) { $env:MIN_WORKER_PORT } else { "11000" }
$MaxWorkerPort = if ($env:MAX_WORKER_PORT) { $env:MAX_WORKER_PORT } else { "11999" }
$RayTmpDir = if ($env:RAY_TMPDIR) { $env:RAY_TMPDIR } else { (Join-Path $ProjectDir ".ray_tmp") }
if (-not (Test-Path $RayTmpDir)) { New-Item -ItemType Directory -Path $RayTmpDir -Force | Out-Null }
[Environment]::SetEnvironmentVariable("RAY_TMPDIR", $RayTmpDir, "Process")

if ($RestartRay -eq "1") {
    Write-Host "[head] RESTART_RAY=1 -> stopping old Ray runtime"
    & $VenvPython -m ray.scripts.scripts stop
} else {
    & $VenvPython -m ray.scripts.scripts status 1>$null 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[head] Ray cluster already running. Skipping restart."
        Write-Host "[head] Use `$env:RESTART_RAY=1; .\scripts\start_head.ps1 to force restart."
        Write-Host "[head] dashboard: http://127.0.0.1:$DashboardPort"
        exit 0
    }
}

Write-Host "[head] starting Ray head on port $RayPort"
Write-Host "[head] ray temp dir: $RayTmpDir"
& $VenvPython -m ray.scripts.scripts start `
  --head `
  --port="$RayPort" `
  --dashboard-host=0.0.0.0 `
  --dashboard-port="$DashboardPort" `
  --node-manager-port="$NodeManagerPort" `
  --object-manager-port="$ObjectManagerPort" `
  --ray-client-server-port="$RayClientPort" `
  --dashboard-agent-listen-port="$DashboardAgentPort" `
  --metrics-export-port="$MetricsPort" `
  --min-worker-port="$MinWorkerPort" `
  --max-worker-port="$MaxWorkerPort" `
  --num-gpus="$NumGpus"

Write-Host "[head] started"
Write-Host "[head] dashboard: http://127.0.0.1:$DashboardPort"
