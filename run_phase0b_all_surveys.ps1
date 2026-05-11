# Phase-0b multi-survey grid (sequential): uses prelim_manifest.yaml for 5x5 targets/countries.
# Log: outputs/_run_phase0b_full.log
# Regenerate the manifest with: python prelim/build_prelim_manifest.py
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$log = Join-Path "outputs" "_run_phase0b_full.log"
$manifest = Join-Path "prelim" "prelim_manifest.yaml"

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"

function Invoke-GridStep {
    param([string]$Name, [string[]]$GridArgs)
    "`n========== $Name $(Get-Date -Format o) ==========`n" | Tee-Object -FilePath $log -Append
    & python run_grid.py @GridArgs 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Grid step failed: $Name (exit $LASTEXITCODE)"
    }
}

New-Item -ItemType Directory -Path "outputs" -Force | Out-Null
"Phase-0b manifest-based run start $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append

if (-not (Test-Path $manifest)) {
    throw "Missing $manifest — run python prelim/build_prelim_manifest.py first."
}

$surveys = @(
    "wvs",
    "afrobarometer",
    "arabbarometer",
    "asianbarometer",
    "latinobarometer",
    "ess_wave_11"
)

foreach ($sid in $surveys) {
    Invoke-GridStep $sid @(
        "--survey", $sid,
        "--from-manifest", $manifest,
        "--grid-workers", "3"
    )
}

"`nPhase-0b manifest run finished $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
