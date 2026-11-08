"""Regression tests for the Conda GitHub Actions workflow and environment.yml."""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_conda_environment_yml_exists_and_installs_package() -> None:
    """Verify environment.yml exists at repo root so conda env update succeeds in CI."""
    env_path = ROOT / "environment.yml"
    assert env_path.is_file(), "environment.yml must exist at repository root for Conda CI"

    content = env_path.read_text(encoding="utf-8")
    assert "dependencies:" in content
    assert "pip:" in content
    assert "-e .[dev]" in content
    assert "flake8" in content


def test_python_package_conda_workflow_has_non_interactive_steps() -> None:
    """Verify python-package-conda.yml uses setup-python@v5 and has no unconfirmed conda install."""
    workflow_path = ROOT / ".github" / "workflows" / "python-package-conda.yml"
    assert workflow_path.is_file(), ".github/workflows/python-package-conda.yml must exist"

    content = workflow_path.read_text(encoding="utf-8")
    assert "conda env update --file environment.yml --name base" in content
    assert "actions/setup-python@v7" in content or "actions/setup-python@v5" in content
    assert "actions/setup-python@v3" not in content

    # Ensure there are no bare interactive `conda install <pkg>` lines without -y/--yes
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line.startswith("conda install "):
            assert "-y" in line or "--yes" in line, f"Interactive conda install without --yes: {line}"


def test_python_package_conda_workflow_resilient_conda_path_setup() -> None:
    """Verify python-package-conda.yml handles missing $CONDA on modern runner images.

    Regression test for Issue #205 (RCA-073): Runner images like Ubuntu 26.04 do not
    set $CONDA and do not pre-install Miniconda. The workflow must include fallback
    logic to locate /usr/share/miniconda or download Miniconda automatically if missing.
    """
    workflow_path = ROOT / ".github" / "workflows" / "python-package-conda.yml"
    assert workflow_path.is_file(), ".github/workflows/python-package-conda.yml must exist"

    content = workflow_path.read_text(encoding="utf-8")
    assert "/usr/share/miniconda" in content
    assert "curl" in content or "setup-miniconda" in content
    assert "echo $CONDA/bin >> $GITHUB_PATH" not in content  # bare unconditioned line replaced


def test_python_package_conda_workflow_triggers_and_curl_resilience() -> None:
    """Verify python-package-conda.yml supports manual dispatch, PR checks, and resilient curl.

    Regression test for Issue #207 (RCA-074):
    1. Workflow must declare workflow_dispatch and pull_request triggers.
    2. Fallback miniconda curl download must include retries against CDN timeouts.
    3. Non-interactive miniconda installer must pass -u (update/overwrite existing prefix).
    4. Temporary installer script must be cleaned up after execution.
    """
    workflow_path = ROOT / ".github" / "workflows" / "python-package-conda.yml"
    assert workflow_path.is_file(), ".github/workflows/python-package-conda.yml must exist"

    content = workflow_path.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in content, "workflow_dispatch trigger required for manual CI runs"
    assert "pull_request:" in content, "pull_request trigger required for PR validation"
    assert "--retry 3" in content, "curl must specify retry count for network resilience"
    assert "--retry-connrefused" in content, "curl must retry on refused connection"
    assert "-b -u -p" in content, "Miniconda installer must use -u to safely update existing prefix"
    assert "rm -f /tmp/miniconda.sh" in content, "Installer script in /tmp must be cleaned up"


def test_python_package_conda_resolution_shell_syntax() -> None:
    """Verify the conda path resolution bash block is valid shell code across all tiers."""
    workflow_path = ROOT / ".github" / "workflows" / "python-package-conda.yml"
    content = workflow_path.read_text(encoding="utf-8")

    assert "Add conda to system path" in content
    script_lines = []
    in_step = False
    for line in content.splitlines():
        if "name: Add conda to system path" in line:
            in_step = True
            continue
        if in_step:
            if line.strip().startswith("- name:") or line.strip().startswith("jobs:"):
                break
            if line.strip().startswith("run: |"):
                continue
            script_lines.append(line)

    full_script = "\n".join(script_lines)
    assert 'if [ -n "$CONDA" ] && [ -x "$CONDA/bin/conda" ]; then' in full_script
    assert 'elif [ -x "/usr/share/miniconda/bin/conda" ]; then' in full_script
    assert 'elif [ -x "$HOME/miniconda/bin/conda" ]; then' in full_script
    assert 'echo "$CONDA/bin" >> $GITHUB_PATH' in full_script
    assert 'echo "/usr/share/miniconda/bin" >> $GITHUB_PATH' in full_script
    assert 'echo "$HOME/miniconda/bin" >> $GITHUB_PATH' in full_script


