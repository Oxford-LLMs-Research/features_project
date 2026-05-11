# Kimi / LLM smoke: 2 targets x 2 countries per survey (from prelim_manifest.yaml order).
# Skips ess_wave_11 (run separately when ready).
#
# Usage (repo root):
#   .\run_kimi_smoke_2x2_non_ess.ps1
#
# Optional:
#   $env:SMOKE_RUN_TAG = "kimi_smoke_2x2"   # output tag (default: kimi_smoke_2x2)
#   $env:GRID_WORKERS = "2"
#
# Log: outputs/_run_kimi_smoke_non_ess_<timestamp>.log
#
# Re-using the same SMOKE_RUN_TAG resumes from cached llm__<tag>/ per cell. For a clean
# retry after code/token fixes, set SMOKE_RUN_TAG to a new name or delete those folders.
$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}
Set-Location $PSScriptRoot

$tag = if ($env:SMOKE_RUN_TAG) { $env:SMOKE_RUN_TAG } else { "kimi_smoke_2x2" }
$gw = if ($env:GRID_WORKERS) { $env:GRID_WORKERS } else { "1" }

New-Item -ItemType Directory -Path "outputs" -Force | Out-Null
$log = Join-Path "outputs" ("_run_kimi_smoke_non_ess_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))

"Kimi smoke 2x2 (non-ESS) start $(Get-Date -Format o); tag=$tag; log=$log" | Tee-Object -FilePath $log -Append

# First two targets and first two countries per survey — same order as prelim/prelim_manifest.yaml
$runs = @(
    @{
        survey    = "wvs"
        targets   = @("Q263", "Q141")
        countries = @("Andorra", "Germany")
    },
    @{
        survey    = "afrobarometer"
        targets   = @("Q43A", "Q67A")
        countries = @("Angola", "Gabon")
    },
    @{
        survey    = "arabbarometer"
        targets   = @("Q1005B", "Q104")
        countries = @("Iraq", "Kuwait")
    },
    @{
        survey    = "asianbarometer"
        targets   = @("level", "SE2")
        countries = @("Australia", "Indonesia")
    },
    @{
        survey    = "latinobarometer"
        targets   = @("SEXO", "S20.A")
        countries = @("Argentina", "Colombia")
    }
)

foreach ($r in $runs) {
    "`n========== $($r.survey) 2x2 $(Get-Date -Format o) ==========`n" | Tee-Object -FilePath $log -Append
    $pyArgs = @(
        "run_grid.py",
        "--survey", $r.survey,
        "--targets"
    ) + $r.targets + @(
        "--countries"
    ) + $r.countries + @(
        "--grid-workers", $gw,
        "--run-tag", $tag
    )
    $prevEa = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python @pyArgs 2>&1 | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) {
            throw "run_grid.py failed for $($r.survey) (exit $LASTEXITCODE)"
        }
    }
    finally {
        $ErrorActionPreference = $prevEa
    }
}

"`nKimi smoke 2x2 (non-ESS) finished $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
