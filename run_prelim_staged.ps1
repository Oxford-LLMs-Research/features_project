# Staged preliminary grid: manifest build → oracle-only sweep → (optional) full LLM+eval.
# Log: outputs/_run_prelim_staged.log
#
# Oracle-only (default):
#   .\run_prelim_staged.ps1
#
# After validating oracles, run full pipeline for all surveys:
#   $env:RUN_PRELIM_FULL = "1"; .\run_prelim_staged.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$manifest = Join-Path "prelim" "prelim_manifest.yaml"
$log = Join-Path "outputs" "_run_prelim_staged.log"

# Encourage deterministic CPU layering when overlapping cells call XGBoost.
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"

New-Item -ItemType Directory -Path "outputs" -Force | Out-Null

function Invoke-PythonStep {
    param([string]$Name, [string[]]$PyArgs)
    "`n========== $Name $(Get-Date -Format o) ==========`n" | Tee-Object -FilePath $log -Append
    & python @PyArgs 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name (exit $LASTEXITCODE)"
    }
}

"Prelim staged run start $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append

Invoke-PythonStep "introspect_metadata" @(
    "prelim/introspect_metadata.py"
)
Invoke-PythonStep "build_prelim_manifest" @(
    "prelim/build_prelim_manifest.py"
)

$surveys = @(
    "wvs",
    "afrobarometer",
    "arabbarometer",
    "asianbarometer",
    "latinobarometer",
    "ess_wave_11"
)

foreach ($s in $surveys) {
    Invoke-PythonStep "oracle_only_$s" @(
        "run_grid.py",
        "--survey", $s,
        "--from-manifest", $manifest,
        "--stop-after", "oracle",
        "--grid-workers", "4"
    )
}

if ($env:RUN_PRELIM_FULL -eq "1") {
    foreach ($s in $surveys) {
        Invoke-PythonStep "full_pipeline_$s" @(
            "run_grid.py",
            "--survey", $s,
            "--from-manifest", $manifest,
            "--stop-after", "full",
            "--grid-workers", "3"
        )
    }
} else {
    "`nSkipping full LLM+eval (set RUN_PRELIM_FULL=1 to enable).`n" | Tee-Object -FilePath $log -Append
}

Invoke-PythonStep "aggregate_summary" @(
    "analysis/prelim_aggregate.py"
)

"`nPrelim staged run finished $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
