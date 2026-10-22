# PowerShell Automation Wrapper for MicroLLM Benchmark Suites
# Usage:
#   .\scripts\run_all_benchmarks.ps1                 # Run all 10 benchmark suites (including 114,018 full splits & HF JEV corpora)
#   .\scripts\run_all_benchmarks.ps1 --fast          # Fast mode (skips 114,018-row full splits, runs 9 suites)
#   .\scripts\run_all_benchmarks.ps1 --quick         # Alias for --fast
#   .\scripts\run_all_benchmarks.ps1 --hf-jev-models # Run only the 12 Hugging Face JEV model datasets & export JSONL corpora

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

$PythonExe = "python"
if (Test-Path $VenvPython) {
    cmd /c "`"$VenvPython`" -c `"import numpy, scipy, pyarrow`" >nul 2>nul"
    if ($LASTEXITCODE -eq 0) {
        $PythonExe = $VenvPython
    }
}

$env:PYTHONPATH = Join-Path $RepoRoot "src"
$env:PYTHONWARNINGS = "ignore"
& $PythonExe -W "ignore" (Join-Path $RepoRoot "scripts\run_all_benchmarks.py") @args
exit $LASTEXITCODE

