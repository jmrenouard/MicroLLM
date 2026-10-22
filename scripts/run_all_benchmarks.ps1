# PowerShell Automation Wrapper for MicroLLM Benchmark Suites
# Usage:
#   .\scripts\run_all_benchmarks.ps1               # Run all 9 benchmark suites (including 114,018 full splits)
#   .\scripts\run_all_benchmarks.ps1 --fast        # Fast mode (skips 114,018-row full splits)
#   .\scripts\run_all_benchmarks.ps1 --quick       # Alias for --fast

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

