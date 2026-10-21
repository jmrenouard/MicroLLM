"""Comprehensive TDD tests for all Python GitHub Actions workflows in MicroLLM."""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"


def test_all_required_python_workflows_exist() -> None:
    """Verify that all essential GitHub Actions workflows for a Python project exist."""
    expected_workflows = [
        "ci.yml",
        "unit-tests.yml",
        "lint.yml",
        "security.yml",
        "publish-pypi.yml",
        "python-package-conda.yml",
        "release.yml",
    ]
    for wf_name in expected_workflows:
        wf_path = WORKFLOWS_DIR / wf_name
        assert wf_path.is_file(), f"Missing required GitHub Actions workflow: .github/workflows/{wf_name}"


def test_dependabot_config_exists_for_pip_and_actions() -> None:
    """Verify .github/dependabot.yml configures both pip and github-actions ecosystems."""
    dependabot_path = ROOT / ".github" / "dependabot.yml"
    assert dependabot_path.is_file(), ".github/dependabot.yml must exist"

    content = dependabot_path.read_text(encoding="utf-8")
    assert "version: 2" in content
    assert 'package-ecosystem: "pip"' in content or "package-ecosystem: pip" in content
    assert (
        'package-ecosystem: "github-actions"' in content
        or "package-ecosystem: github-actions" in content
    )


def test_lint_workflow_checks_ruff_flake8_and_mypy() -> None:
    """Verify lint.yml runs ruff, flake8, and mypy on src and tests."""
    lint_path = WORKFLOWS_DIR / "lint.yml"
    assert lint_path.is_file(), ".github/workflows/lint.yml must exist"

    content = lint_path.read_text(encoding="utf-8")
    assert "push:" in content
    assert "pull_request:" in content
    assert "workflow_dispatch:" in content
    assert "actions/checkout@v4" in content
    assert "actions/setup-python@v5" in content
    assert "ruff check" in content
    assert "flake8" in content
    assert "mypy" in content


def test_security_workflow_runs_bandit_and_pip_audit() -> None:
    """Verify security.yml runs bandit static analysis and pip-audit vulnerability scanning."""
    sec_path = WORKFLOWS_DIR / "security.yml"
    assert sec_path.is_file(), ".github/workflows/security.yml must exist"

    content = sec_path.read_text(encoding="utf-8")
    assert "push:" in content
    assert "pull_request:" in content
    assert "schedule:" in content
    assert "workflow_dispatch:" in content
    assert "bandit" in content
    assert "pip-audit" in content


def test_release_workflow_builds_and_attaches_artifacts() -> None:
    """Verify release.yml triggers on v* tags and builds/uploads wheel and sdist."""
    rel_path = WORKFLOWS_DIR / "release.yml"
    assert rel_path.is_file(), ".github/workflows/release.yml must exist"

    content = rel_path.read_text(encoding="utf-8")
    assert "tags:" in content
    assert "v*" in content
    assert "python -m build" in content
    assert "twine check" in content
    assert "softprops/action-gh-release@" in content
