# Watch the embedding_sensitivity orchestrator and stop it once kimi x mpnet map finishes,
# before score starts. Safe to re-run: exits if already done / orchestrator gone.
$ErrorActionPreference = "Continue"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$logRoot = Join-Path $root "outputs\experiments\embedding_sensitivity"
if (-not (Test-Path $logRoot)) {
  $logRoot = Join-Path $root "outputs\embedding_sensitivity"
}
$master = Join-Path $logRoot "run_all.log"
$kimiMaps = Join-Path $logRoot "all-mpnet-base-v2\kimi\maps"
$orchPid = 42640
$expected = 104

function Write-StopLog([string]$msg) {
    $line = "$(Get-Date -Format o) STOP: $msg"
    Add-Content -Path $master -Value $line -ErrorAction SilentlyContinue
    Write-Host $line
}

function Get-MapCount {
    if (-not (Test-Path $kimiMaps)) { return 0 }
    return @(Get-ChildItem $kimiMaps -Filter "*.json" -ErrorAction SilentlyContinue).Count
}

function Get-MapPython {
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -eq "python.exe" -and
            $_.CommandLine -like "*run_main.py*" -and
            $_.CommandLine -like "*--phase map*" -and
            $_.CommandLine -like "*--selector kimi*" -and
            $_.CommandLine -like "*all-mpnet-base-v2*"
        }
}

function Get-ScorePython {
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -eq "python.exe" -and
            $_.CommandLine -like "*run_main.py*" -and
            $_.CommandLine -like "*--phase score*" -and
            $_.CommandLine -like "*--selector kimi*" -and
            $_.CommandLine -like "*all-mpnet-base-v2*"
        }
}

function Stop-Orchestrator {
    param([string]$reason)
    Write-StopLog $reason

    foreach ($p in @(Get-ScorePython)) {
        Write-StopLog "killing score pid=$($p.ProcessId)"
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        $partial = Join-Path $logRoot "all-mpnet-base-v2\kimi\scores_kimi.csv"
        if (Test-Path $partial) {
            $len = (Get-Item $partial).Length
            if ($len -lt 2000) {
                Remove-Item $partial -Force -ErrorAction SilentlyContinue
                Write-StopLog "removed partial scores_kimi.csv bytes=$len"
            }
        }
    }

    if (Get-Process -Id $orchPid -ErrorAction SilentlyContinue) {
        Write-StopLog "killing orchestrator pid=$orchPid"
        Stop-Process -Id $orchPid -Force -ErrorAction SilentlyContinue
    } else {
        Write-StopLog "orchestrator pid=$orchPid already gone"
    }
    Write-StopLog "halted after kimi map - resume tomorrow with scripts/run_embedding_sensitivity_parallel.ps1"
}

$prior = Get-Content $master -ErrorAction SilentlyContinue | Select-String -SimpleMatch "STOP: halted after kimi map"
if ($prior) {
    Write-Host "Already halted previously."
    exit 0
}

Write-StopLog "watcher started expect=$expected orch=$orchPid"

while ($true) {
    $n = Get-MapCount
    $mapPy = @(Get-MapPython)
    $doneLine = Get-Content $master -ErrorAction SilentlyContinue | Select-String -SimpleMatch "DONE map kimi all-mpnet-base-v2"
    $orchAlive = [bool](Get-Process -Id $orchPid -ErrorAction SilentlyContinue)

    if (-not $orchAlive -and $mapPy.Count -eq 0) {
        if ($n -ge $expected) {
            Stop-Orchestrator "orchestrator already exited with $n maps - ensuring score did not start"
            exit 0
        }
        Write-StopLog "orchestrator gone early with only $n of $expected maps"
        exit 1
    }

    if ($n -ge $expected -and ($mapPy.Count -eq 0 -or $doneLine)) {
        Start-Sleep -Seconds 2
        Stop-Orchestrator "kimi map complete files=$n - stopping before score"
        exit 0
    }

    if (@(Get-ScorePython).Count -gt 0) {
        Stop-Orchestrator "score already started - aborting score, keeping maps"
        exit 0
    }

    Write-Host ("{0} kimi maps={1}/{2} map_py={3} orch={4}" -f (Get-Date -Format "HH:mm:ss"), $n, $expected, $mapPy.Count, $orchAlive)
    if ($n -ge ($expected - 5)) { Start-Sleep -Seconds 3 } else { Start-Sleep -Seconds 15 }
}
