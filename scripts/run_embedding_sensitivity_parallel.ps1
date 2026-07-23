<#
.SYNOPSIS
  Resume embedding_sensitivity with process-level parallelism.

.DESCRIPTION
  Remaining work after tonight's halt (kimi×mpnet map done; deepseek×mpnet map+score done):

    Wave 1 (parallel):
      - score kimi × all-mpnet-base-v2
      - map  deepseek × all-roberta-large-v1
      - map  kimi × all-roberta-large-v1

    Wave 2 (parallel, after both roberta maps finish):
      - score deepseek × all-roberta-large-v1
      - score kimi × all-roberta-large-v1

    Then: python analysis/embedding_sensitivity.py

  Map phases skip existing JSON (safe resume). Score always rewrites its CSV.

  Optional env:
    SCORE_N_DRAWS  (default 10) — lower to e.g. 5 to speed score waves

.EXAMPLE
  powershell -File scripts/run_embedding_sensitivity_parallel.ps1
#>
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$logRoot = Join-Path $root "outputs\embedding_sensitivity"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$master = Join-Path $logRoot "run_parallel.log"

function Log([string]$msg) {
    $line = "$(Get-Date -Format o) $msg"
    Add-Content -Path $master -Value $line
    Write-Host $line
}

function Invoke-PythonLogged {
    # Python libraries often write warnings to stderr; do not treat that as failure.
    param(
        [string[]]$ArgumentList,
        [string]$LogFile
    )
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $py @ArgumentList *>&1 | Tee-Object -FilePath $LogFile
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($code -ne 0) { throw "python failed exit=$code args=$($ArgumentList -join ' ')" }
    return $code
}

function Start-JobLogged {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$LogFile
    )
    Log "START $Name"
    # Separate process so map (API/IO) and score (CPU) overlap cleanly.
    return Start-Process -FilePath $FilePath -ArgumentList $ArgumentList `
        -RedirectStandardOutput $LogFile -RedirectStandardError "$LogFile.err" `
        -PassThru -NoNewWindow -WorkingDirectory $root
}

function Wait-JobsOk {
    param([hashtable]$Procs)  # name -> Process
    $failed = @()
    foreach ($name in @($Procs.Keys)) {
        $p = $Procs[$name]
        $p.WaitForExit()
        if ($p.ExitCode -ne 0) {
            $failed += "$name (exit=$($p.ExitCode))"
            Log "FAIL $name exit=$($p.ExitCode)"
        } else {
            Log "DONE $name"
        }
    }
    if ($failed.Count) {
        throw "Failed jobs: $($failed -join ', ')"
    }
}

$py = (Get-Command python).Source
$env:PYTHONUNBUFFERED = "1"
# Keep default SCORE_N_DRAWS=10 unless caller set it — same eval protocol as deepseek×mpnet.
if (-not $env:SCORE_N_DRAWS) { $env:SCORE_N_DRAWS = "10" }
Log "=== embedding_sensitivity parallel resume ==="
Log "python=$py SCORE_N_DRAWS=$($env:SCORE_N_DRAWS) PYTHONUNBUFFERED=1"

# Pre-build roberta survey embedding caches BEFORE dual maps so two processes never
# race on the same survey_embeddings__*__all-roberta-large-v1.npz write.
Log "START prewarm roberta embeddings"
$prewarmLog = Join-Path $logRoot "log_prewarm_roberta.txt"
# Write script to a temp file — avoids PowerShell here-string / -c quoting issues.
$prewarmPy = Join-Path $logRoot "_prewarm_roberta.py"
@"
import os, sys
from pathlib import Path
ROOT = Path(r"$root")
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
from survey_features.config import OUTPUTS_DIR
from survey_features.layout import genuine_cells
from survey_features.retrieval import load_or_build_survey_embeddings
from survey_features.surveys import extract_survey_variables, load_survey
model = "all-roberta-large-v1"
surveys = sorted({s for s, _, _ in genuine_cells(OUTPUTS_DIR)})
print(f"prewarm surveys={surveys} model={model}", flush=True)
cfg = os.environ["DATA_CONFIG_PATH"]
for sid in surveys:
    _, meta = load_survey(sid, cfg)
    svars = extract_survey_variables(meta)
    load_or_build_survey_embeddings(svars, sid, model)
    print(f"  ok {sid}", flush=True)
print("prewarm done", flush=True)
"@ | Set-Content -Path $prewarmPy -Encoding UTF8
Invoke-PythonLogged -ArgumentList @($prewarmPy) -LogFile $prewarmLog
Remove-Item $prewarmPy -Force -ErrorAction SilentlyContinue
Log "DONE prewarm roberta embeddings"

# ── Wave 1 ────────────────────────────────────────────────────────────────────
# score (CPU) overlaps two maps (API). Outputs are isolated per selector×model.
$w1 = @{}

$scoreKimiMpnet = Join-Path $logRoot "log_score_kimi_all-mpnet-base-v2.txt"
$w1["score kimi mpnet"] = Start-JobLogged "score kimi mpnet" $py @(
    "scripts/run_main.py", "--phase", "score", "--selector", "kimi",
    "--embedding-model", "all-mpnet-base-v2"
) $scoreKimiMpnet

$mapDsRob = Join-Path $logRoot "log_map_deepseek_all-roberta-large-v1.txt"
$w1["map deepseek roberta"] = Start-JobLogged "map deepseek roberta" $py @(
    "scripts/run_main.py", "--phase", "map", "--selector", "deepseek",
    "--disambiguator", "nemotron", "--arms", "C",
    "--embedding-model", "all-roberta-large-v1"
) $mapDsRob

$mapKmRob = Join-Path $logRoot "log_map_kimi_all-roberta-large-v1.txt"
$w1["map kimi roberta"] = Start-JobLogged "map kimi roberta" $py @(
    "scripts/run_main.py", "--phase", "map", "--selector", "kimi",
    "--disambiguator", "nemotron", "--arms", "C",
    "--embedding-model", "all-roberta-large-v1"
) $mapKmRob

Log "Wave 1 running: $($w1.Keys -join ', ')"
Wait-JobsOk $w1

# ── Wave 2 ────────────────────────────────────────────────────────────────────
$w2 = @{}

$scoreDsRob = Join-Path $logRoot "log_score_deepseek_all-roberta-large-v1.txt"
$w2["score deepseek roberta"] = Start-JobLogged "score deepseek roberta" $py @(
    "scripts/run_main.py", "--phase", "score", "--selector", "deepseek",
    "--embedding-model", "all-roberta-large-v1"
) $scoreDsRob

$scoreKmRob = Join-Path $logRoot "log_score_kimi_all-roberta-large-v1.txt"
$w2["score kimi roberta"] = Start-JobLogged "score kimi roberta" $py @(
    "scripts/run_main.py", "--phase", "score", "--selector", "kimi",
    "--embedding-model", "all-roberta-large-v1"
) $scoreKmRob

Log "Wave 2 running: $($w2.Keys -join ', ')"
Wait-JobsOk $w2

# ── Analysis ──────────────────────────────────────────────────────────────────
Log "START analysis"
$anLog = Join-Path $logRoot "log_analysis.txt"
Invoke-PythonLogged -ArgumentList @("analysis/embedding_sensitivity.py") -LogFile $anLog
Log "DONE analysis"
Log "ALL COMPLETE"
