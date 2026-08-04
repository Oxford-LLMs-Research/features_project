# Full preliminary grid for all surveys (manifest 5x5 per survey): oracle + LLM + eval.
# Log: outputs/_run_prelim_full_<timestamp>.log
#
# Usage:
#   .\run_prelim_full_all_surveys.ps1
#
# Optional: refresh manifest first
#   $env:REBUILD_MANIFEST = "1"; .\run_prelim_full_all_surveys.ps1
#
# Tune concurrency (default 3):
#   $env:GRID_WORKERS = "4"; .\run_prelim_full_all_surveys.ps1
$ErrorActionPreference = "Stop"
# PowerShell 7+: do not treat stderr from native commands (e.g. Python warnings) as terminating.
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}
Set-Location $PSScriptRoot
$manifest = Join-Path "prelim" "prelim_manifest.yaml"
$gw = if ($env:GRID_WORKERS) { $env:GRID_WORKERS } else { "3" }

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"

New-Item -ItemType Directory -Path "outputs" -Force | Out-Null
New-Item -ItemType Directory -Path "outputs\logs" -Force | Out-Null
$log = Join-Path "outputs\logs" ("run_prelim_full_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))

function Invoke-PythonStep {
    param([string]$Name, [string[]]$PyArgs)
    "`n========== $Name $(Get-Date -Format o) ==========`n" | Tee-Object -FilePath $log -Append
    # Python warnings on stderr must not trigger terminating NativeCommandError under $ErrorActionPreference Stop.
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

"Prelim full pipeline start $(Get-Date -Format o); log: $log" | Tee-Object -FilePath $log -Append

if ($env:REBUILD_MANIFEST -eq "1") {
    Invoke-PythonStep "build_prelim_manifest" @(
        "prelim/build_prelim_manifest.py"
    )
}

$surveys = @(
    "wvs",
    "afrobarometer",
    "arabbarometer",
    "asianbarometer",
    "latinobarometer",
    "ess_wave_11"
)

foreach ($s in $surveys) {
    Invoke-PythonStep "full_pipeline_$s" @(
        "archive/run_grid.py",
        "--survey", $s,
        "--from-manifest", $manifest,
        "--stop-after", "full",
        "--grid-workers", $gw
    )
}

Invoke-PythonStep "aggregate_summary" @(
    "archive/prelim_aggregate.py"
)

Invoke-PythonStep "build_paper_figures" @(
    "paper/scripts/build_paper_figures.py"
)

Invoke-PythonStep "build_current_state_report" @(
    "paper/scripts/build_current_state_report.py"
)

"`nPrelim full pipeline finished $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
