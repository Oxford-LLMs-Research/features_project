# Big project update — all-surveys preliminary run.
#
# Sequence: wipe outputs -> rebuild manifest -> oracle smoke (WVS, 25 cells)
#           -> LLM smoke (1 cell, full pipeline) -> [gate CONFIRM_FULL_RUN=1]
#           -> full 6-survey loop -> aggregate.
#
# Log: outputs/_run_big_update.log
#
# Usage (smoke only — default):
#   .\run_big_update.ps1
#
# Smoke + full multi-survey:
#   $env:CONFIRM_FULL_RUN = "1"; .\run_big_update.ps1
#
# Tune concurrency (default 3, cpu_count=20 -> xgb_nthread=6 per cell):
#   $env:GRID_WORKERS = "4"; .\run_big_update.ps1
#
# Tighter category cap on "large" target bucket (default 15):
#   $env:LARGE_CAP = "12"; .\run_big_update.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$manifest = Join-Path "prelim" "prelim_manifest.yaml"
$log = Join-Path "outputs" "_run_big_update.log"
$gw = if ($env:GRID_WORKERS) { $env:GRID_WORKERS } else { "3" }

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"

New-Item -ItemType Directory -Path "outputs" -Force | Out-Null

function Invoke-PythonStep {
    param([string]$Name, [string[]]$PyArgs)
    "`n========== $Name $(Get-Date -Format o) ==========`n" | Tee-Object -FilePath $log -Append
    # Python warnings on stderr must not trigger terminating NativeCommandError
    # under $ErrorActionPreference Stop.
    $prevEa = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python @PyArgs 2>&1 | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) {
            throw "Step failed: $Name (exit $LASTEXITCODE)"
        }
    } finally {
        $ErrorActionPreference = $prevEa
    }
}

"Big update run start $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
"Workers: $gw  |  cpu_count: $([Environment]::ProcessorCount)" | Tee-Object -FilePath $log -Append

# 1. Wipe outputs (archive existing prelim log; preserve this run's log + previous archives).
if (Test-Path "outputs\_run_prelim_full.log") {
    Move-Item -Force "outputs\_run_prelim_full.log" "outputs\_run_prelim_full.previous.log"
    "Archived outputs\_run_prelim_full.log -> _run_prelim_full.previous.log" | Tee-Object -FilePath $log -Append
}

$logName = Split-Path $log -Leaf
$keep = @($logName, "_run_prelim_full.previous.log")
Get-ChildItem -Path "outputs" -Force |
    Where-Object { $keep -notcontains $_.Name } |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
"Wiped outputs/ (kept: $($keep -join ', '))" | Tee-Object -FilePath $log -Append

# 2. Rebuild manifest
Invoke-PythonStep "introspect_metadata" @("prelim/introspect_metadata.py")
Invoke-PythonStep "build_prelim_manifest" @("prelim/build_prelim_manifest.py")

# 3. Oracle smoke test — WVS only, oracle stage only
"`n[smoke] WVS oracle-only with grid_workers=$gw" | Tee-Object -FilePath $log -Append
$smokeStart = Get-Date
Invoke-PythonStep "smoke_oracle_wvs" @(
    "run_grid.py",
    "--survey", "wvs",
    "--from-manifest", $manifest,
    "--stop-after", "oracle",
    "--grid-workers", $gw
)
$smokeDur = (Get-Date) - $smokeStart
"[smoke] oracle wall time: $([math]::Round($smokeDur.TotalMinutes, 2)) min" | Tee-Object -FilePath $log -Append

# 4. LLM smoke — 1 cell, full pipeline. Reuses the cached oracle from step 3.
$firstWvsTarget = (& python -c "import yaml; d=yaml.safe_load(open('prelim/prelim_manifest.yaml', encoding='utf-8')); print(d['surveys']['wvs']['targets'][0])").Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($firstWvsTarget)) {
    throw "Could not extract first WVS target from manifest."
}
"[smoke] LLM 1-cell full pipeline: target=$firstWvsTarget country=Germany" | Tee-Object -FilePath $log -Append

Invoke-PythonStep "smoke_llm_${firstWvsTarget}_Germany" @(
    "run_grid.py",
    "--survey", "wvs",
    "--targets", $firstWvsTarget,
    "--countries", "Germany",
    "--grid-workers", "1"
)

# 5. Confirmation gate
if ($env:CONFIRM_FULL_RUN -ne "1") {
    "`n[gate] CONFIRM_FULL_RUN != 1 — stopping after smoke." | Tee-Object -FilePath $log -Append
    "      Re-run with `$env:CONFIRM_FULL_RUN = '1' to launch the full multi-survey grid." | Tee-Object -FilePath $log -Append
    "Big update finished (smoke only) $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    return
}

# 6. Full multi-survey loop
"`n[full] CONFIRM_FULL_RUN=1 -> launching full multi-survey grid" | Tee-Object -FilePath $log -Append

$surveys = @(
    "wvs",
    "afrobarometer",
    "arabbarometer",
    "asianbarometer",
    "latinobarometer",
    "ess_wave_11"
)

$fullStart = Get-Date
foreach ($s in $surveys) {
    Invoke-PythonStep "full_pipeline_$s" @(
        "run_grid.py",
        "--survey", $s,
        "--from-manifest", $manifest,
        "--stop-after", "full",
        "--grid-workers", $gw
    )
}
$fullDur = (Get-Date) - $fullStart
"[full] full multi-survey wall time: $([math]::Round($fullDur.TotalMinutes, 2)) min" | Tee-Object -FilePath $log -Append

# 7. Aggregate
Invoke-PythonStep "aggregate_summary" @("analysis/prelim_aggregate.py")

"`nBig update finished $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
