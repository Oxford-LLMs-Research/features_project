# Run one machine's oracle partition of the registered grid, detached from the
# terminal, with native threads pinned per worker. Confirmatory cells first (the
# ones scoring waits on), then the oracle-only cells; both phases resume, so
# rerunning after a crash or a stop only recomputes what is missing.
#
# Usage (from the repo root, env activated):
#   powershell -File scripts/run_oracle_partition.ps1 -Surveys afrobarometer,arabbarometer,asianbarometer -Procs 3
#   powershell -File scripts/run_oracle_partition.ps1 -Surveys ess_wave_11,latinobarometer,wvs -Procs 2
# Progress: the log file printed at start (under the Dropbox outputs/logs).
# stderr is merged by cmd, not PowerShell: in Windows PowerShell 5.1 a native
# command's 2>&1 wraps every stderr line as an error record with an 'At line'
# banner, which floods the log with AutoGluon warnings dressed as failures.
# Stop: Stop-Process on the python processes, or close the spawned window.
param(
    [Parameter(Mandatory = $true)][string[]]$Surveys,
    [int]$Procs = 2,
    [string]$Python = "python",
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
# Accept both `-Surveys a,b,c` (one comma-joined token when invoked via -File) and a real array.
$Surveys = @($Surveys | ForEach-Object { $_ -split "," } | Where-Object { $_ -ne "" })
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo

$threads = [math]::Max(1, [math]::Floor([Environment]::ProcessorCount / $Procs))
$env:OMP_NUM_THREADS = "$threads"
$env:OPENBLAS_NUM_THREADS = "$threads"
$env:MKL_NUM_THREADS = "$threads"
$env:NUMEXPR_NUM_THREADS = "$threads"
$env:PYTHONUNBUFFERED = "1"

$outputs = & $Python -c "from survey_features.config import OUTPUTS_DIR; print(OUTPUTS_DIR)"
$logDir = Join-Path $outputs "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$log = Join-Path $logDir ("oracle_run_{0}_{1}.log" -f $env:COMPUTERNAME, $stamp)

$surveyArg = $Surveys -join " "
$common = "scripts/rerun_oracles.py --cells-csv data/confirmatory_grid_cells.csv --survey $surveyArg --runtime-mode quick --autogluon-time-limit 600 --processes $Procs"
$body = @"
`$env:OMP_NUM_THREADS='$threads'; `$env:OPENBLAS_NUM_THREADS='$threads'; `$env:MKL_NUM_THREADS='$threads'; `$env:NUMEXPR_NUM_THREADS='$threads'; `$env:PYTHONUNBUFFERED='1'
Set-Location '$repo'
"[partition] surveys=$surveyArg procs=$Procs threads/worker=$threads started $(Get-Date)" | Tee-Object -FilePath '$log' -Append
"[partition] phase 1: role=confirmatory" | Tee-Object -FilePath '$log' -Append
cmd /c """$Python"" $common --role confirmatory 2>&1" | Tee-Object -FilePath '$log' -Append
"[partition] phase 2: role=oracle_only" | Tee-Object -FilePath '$log' -Append
cmd /c """$Python"" $common --role oracle_only 2>&1" | Tee-Object -FilePath '$log' -Append
"[partition] finished $(Get-Date)" | Tee-Object -FilePath '$log' -Append
"[partition] next: python scripts/oracle_provenance_census.py" | Tee-Object -FilePath '$log' -Append
"@

Write-Output "log: $log"
Write-Output ("surveys: " + ($Surveys -join ", "))
if ($Foreground) {
    Invoke-Expression $body
} else {
    $tmp = Join-Path $env:TEMP ("oracle_partition_{0}.ps1" -f $stamp)
    Set-Content -Path $tmp -Value $body -Encoding UTF8
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $tmp -WindowStyle Minimized
    Write-Output "detached; follow with: Get-Content -Wait '$log'"
}
