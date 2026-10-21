"""Tests for PyPI packaging metadata and GitHub Actions workflows."""

from __future__ import annotations

import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_pyproject_metadata_for_pypi() -> None:
    """Verify pyproject.toml contains complete metadata required for PyPI publication."""
    pyproject_path = ROOT / "pyproject.toml"
    assert pyproject_path.is_file(), "pyproject.toml must exist"

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})

    assert project.get("name") == "microllm"
    assert project.get("version")
    assert project.get("description")
    assert project.get("readme") == "README.md"
    assert project.get("requires-python") == ">=3.10"
    assert "urls" in project, "project.urls should be defined for PyPI project links"
    assert "Homepage" in project["urls"]
    assert "Repository" in project["urls"]
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

    # PyPA official action with OIDC Trusted Publishing (id-token: write)
    assert "id-token: write" in content
    assert "pypa/gh-action-pypi-publish@" in content


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

