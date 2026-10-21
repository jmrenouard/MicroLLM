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
    assert "actions/setup-python@v5" in content
    assert "actions/setup-python@v3" not in content

    # Ensure there are no bare interactive `conda install <pkg>` lines without -y/--yes
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line.startswith("conda install "):
            assert "-y" in line or "--yes" in line, f"Interactive conda install without --yes: {line}"
