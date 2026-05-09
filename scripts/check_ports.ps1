param(
    [Parameter(Mandatory = $true)][string]$TargetIp,
    [Parameter(Mandatory = $false)][ValidateSet("quick","full")][string]$Mode = "full"
)

$ErrorActionPreference = "Stop"

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
