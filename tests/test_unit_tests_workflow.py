"""Unit tests for the GitHub Actions Unit Tests workflow and project rules."""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_project_github_workflow_rules_exist() -> None:
    """Verify AGENTS.md and .agents/rules/github-workflow.md define the mandatory Issue->Branch->PR->Merge rule."""
    agents_md = ROOT / "AGENTS.md"
    rule_md = ROOT / ".agents" / "rules" / "github-workflow.md"

    assert agents_md.is_file(), "AGENTS.md must exist at repository root"
    assert rule_md.is_file(), ".agents/rules/github-workflow.md must exist"

    agents_text = agents_md.read_text(encoding="utf-8")
    rule_text = rule_md.read_text(encoding="utf-8")

    for doc in (agents_text, rule_text):
        assert "gh issue create" in doc
        assert "gh pr create" in doc
        assert "gh pr merge" in doc


def test_unit_tests_github_action_workflow() -> None:
    """Verify .github/workflows/unit-tests.yml exists and runs pytest with coverage across Python versions."""
    workflow_path = ROOT / ".github" / "workflows" / "unit-tests.yml"
    assert workflow_path.is_file(), ".github/workflows/unit-tests.yml must exist"

    content = workflow_path.read_text(encoding="utf-8")

    # Triggers on push, pull_request, and manual workflow_dispatch
    assert "push:" in content
    assert "pull_request:" in content
    assert "workflow_dispatch:" in content

    # Matrix includes Python 3.10, 3.11, and 3.12
    for py_ver in ("3.10", "3.11", "3.12"):
        assert py_ver in content

    # Uses actions/checkout@v4 and actions/setup-python@v5
    assert "actions/checkout@v4" in content
    assert "actions/setup-python@v5" in content

    # Installs dev dependencies and runs pytest with coverage
    assert 'pip install -e ".[dev]"' in content
    assert "pytest" in content
    assert "--cov=microllm" in content
