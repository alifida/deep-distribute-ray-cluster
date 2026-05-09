param(
    [Parameter(Mandatory = $false)][string]$TargetIp,
    [Parameter(Mandatory = $false)][ValidateSet("quick","full")][string]$Mode = "full"
)

$ErrorActionPreference = "Stop"

if (Test-Path (Join-Path $PSScriptRoot "cluster_config.env")) {
    Get-Content (Join-Path $PSScriptRoot "cluster_config.env") | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) { return }
        if ($line -match "^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$") {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
}

$TargetIp = if ($TargetIp) { $TargetIp } elseif ($env:HEAD_IP) { $env:HEAD_IP } else { "" }
if (-not $TargetIp) {
    Write-Host "Usage: .\scripts\check_ports.ps1 <TARGET_IP> [quick|full]"
    Write-Host "Or set HEAD_IP in scripts\cluster_config.env"
    exit 1
}

function Test-Port {
    param([string]$Host, [int]$Port)
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $iar = $client.BeginConnect($Host, $Port, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(1000, $false)
        if (-not $ok) {
            $client.Close()
            Write-Host ("FAIL {0}:{1}" -f $Host, $Port)
            return $false
        }
        $client.EndConnect($iar)
        $client.Close()
        Write-Host ("OK   {0}:{1}" -f $Host, $Port)
        return $true
    } catch {
        Write-Host ("FAIL {0}:{1}" -f $Host, $Port)
        return $false
    }
}

$controlPorts = @(6379,8265,10001,10002,10003,10004,10005,10011,10012,10014,10015)
$workerRangeHead = 11000..11999
$workerRangeWorker = 12000..12999

Write-Host "[check] target=$TargetIp mode=$Mode"
Write-Host ""

$failCount = 0
Write-Host "[check] control ports"
foreach ($p in $controlPorts) {
    if (-not (Test-Port -Host $TargetIp -Port $p)) { $failCount++ }
}

if ($Mode -eq "full") {
    Write-Host ""
    Write-Host "[check] worker port range 11000-11999"
    foreach ($p in $workerRangeHead) {
        if (-not (Test-Port -Host $TargetIp -Port $p)) { $failCount++ }
    }
    Write-Host ""
    Write-Host "[check] worker port range 12000-12999"
    foreach ($p in $workerRangeWorker) {
        if (-not (Test-Port -Host $TargetIp -Port $p)) { $failCount++ }
    }
}

Write-Host ""
if ($failCount -eq 0) {
    Write-Host "[check] PASS: all tested ports are reachable."
} else {
    Write-Host "[check] FAIL: $failCount port checks failed."
    exit 2
}
