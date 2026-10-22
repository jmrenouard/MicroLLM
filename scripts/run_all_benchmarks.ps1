# PowerShell Automation Wrapper for MicroLLM Benchmark Suites
# Usage:
#   .\scripts\run_all_benchmarks.ps1               # Run all 9 benchmark suites (including 114,018 full splits)
#   .\scripts\run_all_benchmarks.ps1 --fast        # Fast mode (skips 114,018-row full splits)

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

$PythonExe = "python"
if (Test-Path $VenvPython) {
    $ErrorActionPreference = "SilentlyContinue"
    & $VenvPython -c "import numpy, scipy, pyarrow" >$null 2>&1
    if ($LASTEXITCODE -eq 0) {
        $PythonExe = $VenvPython
    }
}

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $RepoRoot "src"
& $PythonExe (Join-Path $RepoRoot "scripts\run_all_benchmarks.py") @args
exit $LASTEXITCODE
