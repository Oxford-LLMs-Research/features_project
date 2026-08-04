<#
.SYNOPSIS
  Full kimi ensemble mapping experiment (v1 run-tag): map → score → analysis.

.DESCRIPTION
  - Writes under outputs/experiments/ensemble_mapping/runs/v1/ (never smoke / main / format_pilot)
  - Tees all stages to outputs/experiments/ensemble_mapping/run_v1.log
  - Side watcher appends maps=N/104 progress every 45s during map
  - Ends with ALL COMPLETE

.EXAMPLE
  powershell -File scripts/run_ensemble_mapping_v1.ps1
#>
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$runTag = "v1"
$expectedMaps = 104
$fusionSlug = "union_max_sim_minilm_mpnet"
$selector = "kimi"
$disambiguator = "nemotron"
$mapWorkers = 2

$logRoot = Join-Path $root "outputs\experiments\ensemble_mapping"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$master = Join-Path $logRoot "run_v1.log"
$mapsDir = Join-Path $logRoot "runs\$runTag\$fusionSlug\$selector\maps"
$scoresCsv = Join-Path $logRoot "runs\$runTag\$fusionSlug\$selector\scores_$selector.csv"
$reportDir = Join-Path $logRoot "runs\$runTag"

function Log([string]$msg) {
    $line = "$(Get-Date -Format o) $msg"
    Add-Content -Path $master -Value $line -Encoding utf8
    Write-Host $line
}

function Invoke-PythonTee {
    param([string[]]$ArgumentList)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $py @ArgumentList *>&1 | ForEach-Object {
        $s = "$_"
        Add-Content -Path $master -Value $s -Encoding utf8
        Write-Host $s
    }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($code -ne 0) { throw "python failed exit=$code args=$($ArgumentList -join ' ')" }
}

function Count-Maps {
    if (-not (Test-Path $mapsDir)) { return 0 }
    return @(Get-ChildItem -Path $mapsDir -Filter "*.json" -File -ErrorAction SilentlyContinue).Count
}

function Start-MapProgressWatcher {
    param([System.Diagnostics.Process]$MapProcess)
    $watcherScript = {
        param($Master, $MapsDir, $Expected, $MapPid, $IntervalSec)
        $t0 = Get-Date
        $lastN = -1
        while ($true) {
            try {
                $proc = Get-Process -Id $MapPid -ErrorAction SilentlyContinue
                if (-not $proc) { break }
            } catch { break }
            $n = 0
            if (Test-Path $MapsDir) {
                $n = @(Get-ChildItem -Path $MapsDir -Filter "*.json" -File -ErrorAction SilentlyContinue).Count
            }
            $pct = if ($Expected -gt 0) { [math]::Round(100.0 * $n / $Expected, 1) } else { 0 }
            $elapsedMin = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
            $eta = "?"
            if ($n -gt $lastN -and $n -gt 0 -and $n -lt $Expected) {
                $rate = $n / [math]::Max(((Get-Date) - $t0).TotalSeconds, 1)
                $remain = ($Expected - $n) / [math]::Max($rate, 1e-9)
                $etaMin = [math]::Round($remain / 60.0, 1)
                $eta = "${etaMin}m"
            } elseif ($n -ge $Expected) {
                $eta = "0m"
            }
            $line = "$(Get-Date -Format o) [progress] maps=$n/$Expected (${pct}%) elapsed=${elapsedMin}m eta~$eta"
            Add-Content -Path $Master -Value $line -Encoding utf8
            Write-Host $line
            $lastN = $n
            if ($n -ge $Expected) { break }
            Start-Sleep -Seconds $IntervalSec
        }
    }
    return Start-Job -ScriptBlock $watcherScript -ArgumentList @(
        $master, $mapsDir, $expectedMaps, $MapProcess.Id, 45
    )
}

$py = (Get-Command python).Source
$env:PYTHONUNBUFFERED = "1"
if (-not $env:SCORE_N_DRAWS) { $env:SCORE_N_DRAWS = "10" }

Log "=== ensemble_mapping full kimi v1 ==="
Log "python=$py PYTHONUNBUFFERED=1 SCORE_N_DRAWS=$($env:SCORE_N_DRAWS) map_workers=$mapWorkers run_tag=$runTag"
Log "maps_dir=$mapsDir"
Log "scores_csv=$scoresCsv"
Log "report_dir=$reportDir"
Log "expected_maps=$expectedMaps (52 cells x 2 conditions)"

# ── Map ───────────────────────────────────────────────────────────────────────
# Skip map when already complete (resume after orchestrator died post-map).
$n0 = Count-Maps
if ($n0 -ge $expectedMaps) {
    Log "SKIP map maps already complete: $n0/$expectedMaps"
    $nDone = $n0
} else {
    Log "START map"
    New-Item -ItemType Directory -Force -Path $mapsDir | Out-Null
    Log "maps already present: $n0/$expectedMaps (resume skips existing)"

    $mapLog = Join-Path $logRoot "log_map_v1.txt"
    $mapArgs = @(
        "scripts/run_ensemble_mapping.py",
        "--phase", "map",
        "--selector", $selector,
        "--disambiguator", $disambiguator,
        "--arms", "C",
        "--run-tag", $runTag,
        "--map-workers", "$mapWorkers"
    )

    # Start map as a child process so the watcher can poll by PID while we tee stdout.
    $mapProc = Start-Process -FilePath $py -ArgumentList $mapArgs `
        -RedirectStandardOutput $mapLog -RedirectStandardError "$mapLog.err" `
        -PassThru -NoNewWindow -WorkingDirectory $root
    Log "map pid=$($mapProc.Id) (child log: $mapLog)"

    $watcher = Start-MapProgressWatcher -MapProcess $mapProc

    # Tail map log into master until process exits
    $tailPos = 0
    while (-not $mapProc.HasExited) {
        if (Test-Path $mapLog) {
            $content = Get-Content -Path $mapLog -Raw -ErrorAction SilentlyContinue
            if ($null -ne $content -and $content.Length -gt $tailPos) {
                $chunk = $content.Substring($tailPos)
                $tailPos = $content.Length
                foreach ($line in ($chunk -split "`r?`n")) {
                    if ($line.Length -gt 0) {
                        Add-Content -Path $master -Value $line -Encoding utf8
                        Write-Host $line
                    }
                }
            }
        }
        Start-Sleep -Seconds 5
    }
    # Final flush of unread stdout + any stderr
    Start-Sleep -Milliseconds 500
    if (Test-Path $mapLog) {
        $content = Get-Content -Path $mapLog -Raw -ErrorAction SilentlyContinue
        if ($null -ne $content -and $content.Length -gt $tailPos) {
            $chunk = $content.Substring($tailPos)
            foreach ($line in ($chunk -split "`r?`n")) {
                if ($line.Length -gt 0) {
                    Add-Content -Path $master -Value $line -Encoding utf8
                    Write-Host $line
                }
            }
        }
    }
    $errLog = "$mapLog.err"
    if (Test-Path $errLog) {
        Get-Content -Path $errLog -ErrorAction SilentlyContinue | ForEach-Object {
            $line = "[stderr] $_"
            Add-Content -Path $master -Value $line -Encoding utf8
            Write-Host $line
        }
    }
    Receive-Job $watcher -ErrorAction SilentlyContinue | Out-Null
    Stop-Job $watcher -ErrorAction SilentlyContinue
    Remove-Job $watcher -Force -ErrorAction SilentlyContinue

    # Start-Process ExitCode can be $null briefly / after race; treat null/empty as
    # success when maps dir already has the expected count (map completed cleanly).
    $nDone = Count-Maps
    $exitCode = $mapProc.ExitCode
    $exitEmpty = ($null -eq $exitCode) -or ("$exitCode".Trim() -eq "")
    if ($exitEmpty -and $nDone -ge $expectedMaps) {
        Log "WARN map ExitCode empty/null but maps=$nDone/$expectedMaps - treating as success"
    } elseif ($exitEmpty -or ($exitCode -ne 0)) {
        Log "FAIL map exit=$exitCode maps=$nDone/$expectedMaps"
        throw "map failed exit=$exitCode maps=$nDone/$expectedMaps"
    }
    Log "DONE map maps=$nDone/$expectedMaps"
}

# ── Score ─────────────────────────────────────────────────────────────────────
try {
    Log "START score"
    Invoke-PythonTee -ArgumentList @(
        "scripts/run_ensemble_mapping.py",
        "--phase", "score",
        "--selector", $selector,
        "--run-tag", $runTag
    )
    Log "DONE score -> $scoresCsv"
} catch {
    Log "FAIL score $_"
    throw
}

# ── Analysis ──────────────────────────────────────────────────────────────────
try {
    Log "START analysis"
    Invoke-PythonTee -ArgumentList @(
        "analysis/ensemble_mapping.py",
        "--selector", $selector,
        "--run-tag", $runTag
    )
    Log "DONE analysis -> $reportDir"
} catch {
    Log "FAIL analysis $_"
    throw
}
Log "ALL COMPLETE"
