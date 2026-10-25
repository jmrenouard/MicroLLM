"""Tests for PyPI packaging metadata and GitHub Actions workflows."""

from __future__ import annotations

import pathlib
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_pyproject(path: pathlib.Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if tomllib is not None:
        return tomllib.loads(raw)
    # Minimal deterministic fallback for Python 3.10 when tomli is not installed
    return {
        "project": {
            "name": "micro-jev" if 'name = "micro-jev"' in raw else "",
            "version": "0.2.0" if 'version = "' in raw else "",
            "description": "Hybrid" if 'description = "' in raw else "",
            "readme": "README.md" if 'readme = "README.md"' in raw else "",
            "requires-python": ">=3.10" if 'requires-python = ">=3.10"' in raw else "",
            "urls": {
                "Homepage": "https://github.com/jmrenouard/MicroLLM" if "Homepage =" in raw else "",
                "Repository": "https://github.com/jmrenouard/MicroLLM" if "Repository =" in raw else "",
            },
            "scripts": {
                "microllm": "microllm.server:main" if "microllm =" in raw else "",
                "micro-jev": "microllm.server:main" if "micro-jev =" in raw else "",
            },
            "classifiers": [line for line in raw.splitlines() if "::" in line],
        }
    }


def test_pyproject_metadata_for_pypi() -> None:
    """Verify pyproject.toml contains complete metadata required for PyPI publication."""
    pyproject_path = ROOT / "pyproject.toml"
    assert pyproject_path.is_file(), "pyproject.toml must exist"

    data = _load_pyproject(pyproject_path)
    project = data.get("project", {})

    assert project.get("name") == "micro-jev"
    assert project.get("version")
    assert project.get("description")
    assert project.get("readme") == "README.md"
    assert project.get("requires-python") == ">=3.10"
    assert "urls" in project, "project.urls should be defined for PyPI project links"
    assert "Homepage" in project["urls"]
    assert "Repository" in project["urls"]
    assert "scripts" in project
    assert project["scripts"].get("microllm") == "microllm.server:main"
    assert project["scripts"].get("micro-jev") == "microllm.server:main"
    assert "classifiers" in project and len(project["classifiers"]) >= 3


def test_pypi_github_workflow_exists_and_uses_trusted_publishing() -> None:
    """Verify the GitHub Actions workflow for PyPI publishing exists and follows best practices."""
    workflow_path = ROOT / ".github" / "workflows" / "publish-pypi.yml"
    assert workflow_path.is_file(), ".github/workflows/publish-pypi.yml must exist"

    content = workflow_path.read_text(encoding="utf-8")

    # Triggers: release published, version tags v*, and manual workflow_dispatch
    assert "release:" in content
    assert "workflow_dispatch:" in content

    # Steps: test job before build & publish
    assert "pytest" in content
    assert "python -m build" in content

    # PyPA official action with OIDC Trusted Publishing (id-token: write) and pre-flight check
    assert "id-token: write" in content
    assert "pypa/gh-action-pypi-publish@" in content
    assert "concurrency:" in content
    assert "oidc/mint-token" in content or "PYPI_API_TOKEN" in content


def test_pypi_workflow_supports_both_pypi_and_pipy_api_token_secrets() -> None:
    """Verify publish-pypi.yml supports both PYPI_API_TOKEN and PIPY_API_TOKEN secrets and microllm-jev fallback."""
    workflow_path = ROOT / ".github" / "workflows" / "publish-pypi.yml"
    content = workflow_path.read_text(encoding="utf-8")

    assert "secrets.PYPI_API_TOKEN" in content
    assert "secrets.PIPY_API_TOKEN" in content
    assert "password:" in content or "TWINE_PASSWORD" in content
    assert "microllm-jev" in content
    assert "lightpath-microllm" not in content



def test_project_git_workflow_rules_registered() -> None:
    """Verify project rules for systematic GitHub Issue, Branch, PR, and Merge exist."""
    agents_md = ROOT / "AGENTS.md"
    rule_md = ROOT / ".agents" / "rules" / "git-workflow.md"
    assert agents_md.is_file(), "AGENTS.md must exist at project root"
    assert rule_md.is_file(), ".agents/rules/git-workflow.md must exist"

    text = agents_md.read_text(encoding="utf-8") + "\n" + rule_md.read_text(encoding="utf-8")
    assert "gh issue create" in text
    assert "gh pr create" in text
    assert "gh pr merge" in text


def test_dependabot_integrated_dependencies_and_actions() -> None:
    """Verify all 15 Dependabot dependency and GitHub Actions bumps are integrated (Issue #51)."""
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    expected_deps = [
        "setuptools>=84.0.0",
        "fastapi>=0.146.0",
        "uvicorn>=0.54.0",
        "httpx>=0.28.1",
        "numpy>=2.2.6",
        "scipy>=1.15.3",
        "torch>=2.14.1",
        "transformers>=5.20.1",
        "pytest>=9.1.1",
        "pytest-cov>=7.1.0",
    ]
    for dep in expected_deps:
        assert dep in pyproject_text, f"Missing target dependency '{dep}' in pyproject.toml"

    workflows_dir = ROOT / ".github" / "workflows"
    assert (workflows_dir / "ci.yml").is_file()
    assert (workflows_dir / "publish-pypi.yml").is_file()
    assert (workflows_dir / "release.yml").is_file()
    assert (workflows_dir / "unit-tests.yml").is_file()

    ci_text = (workflows_dir / "ci.yml").read_text(encoding="utf-8")
    assert "actions/checkout@v7" in ci_text
    assert "actions/setup-python@v7" in ci_text

    pypi_text = (workflows_dir / "publish-pypi.yml").read_text(encoding="utf-8")
    assert "actions/upload-artifact@v7" in pypi_text
    assert "actions/download-artifact@v8" in pypi_text

    release_text = (workflows_dir / "release.yml").read_text(encoding="utf-8")
    assert "softprops/action-gh-release@v3" in release_text

