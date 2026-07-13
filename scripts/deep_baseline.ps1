# deep_baseline.ps1 — the full, believable baseline. Long-running; run it detached.
#
#   Start-Process pwsh -ArgumentList '-File','scripts/deep_baseline.ps1' -WindowStyle Hidden
#
# Writes everything to workspace/outputs/deep_baseline.log, and drops a DONE
# marker on the last line so a watcher can tell "finished" from "still going"
# without guessing from a timestamp.
#
# Three phases, because they answer three different questions:
#   models        did the reinstalled 7B come back with the tok/s it had before?
#   sprint x3     what does the hive actually score, and how much does it wobble?
#   contract x3   same, with the human's asserts instead of the model's
#
# x3 is the point. One run of this suite is a sample, not a measurement — see the
# docstring in bench.py.

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
$log = Join-Path $root "workspace\outputs\deep_baseline.log"

New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
Set-Content -Path $log -Value "deep baseline started $(Get-Date -Format o)" -Encoding utf8

function Phase($title, $benchArgs) {
    Add-Content $log ""
    Add-Content $log "############ $title ############"
    & $python scripts/bench.py @benchArgs 2>&1 | ForEach-Object { Add-Content $log $_ }
}

Phase "RAW MODELS (post-reinstall sanity check)" @("models")
Phase "PLAIN x3" @("sprint", "--repeat", "3")
Phase "CONTRACT x3" @("sprint", "--contract", "--repeat", "3")
Phase "HISTORY" @("history")

Add-Content $log ""
Add-Content $log "DONE $(Get-Date -Format o)"
