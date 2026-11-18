# Reproduce everything (Windows PowerShell 5.1):  powershell -ExecutionPolicy Bypass -File .\run_all.ps1
# NOTE: keep this file ASCII-only; PowerShell 5.1 reads BOM-less files as ANSI.
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    python -m venv .venv
    & $py -m pip install -r requirements.txt
}

& $py --version
& $py -m pip freeze

& $py validate.py                      # 1. validation first (non-zero exit on failure)
if ($LASTEXITCODE -ne 0) { throw "validate.py failed" }
& $py pilot.py load                    # 2. Legacy pilot: choose low/mid/high OBSS loads
& $py pilot.py timing                  # 3. pilot: warm-up, measurement length, seeds
& $py run_experiment.py main           # 4. main experiment (12 scenarios x 3 methods x 30 seeds)
& $py run_experiment.py sens           # 5. threshold / coefficient sensitivity
& $py run_experiment.py events         # 6. short event-log excerpt
& $py analyze.py                       # 7. summary CSV, paired differences, PNG, tables.md
& $py run_experiment.py improve        # 8. improvement variants (8 variants x 12 scenarios x 30 seeds)
& $py run_experiment.py rate           # 9. rate extension 10/15/20/25 Mbps (4 loads x 6 scenarios x 3 methods x 30 seeds)
& $py analyze_extra.py                 # 10. improvement tables, rate summary, change points, fig7
