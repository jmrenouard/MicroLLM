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
    assert "actions/checkout@v7" in content or "actions/checkout@v4" in content
    assert "actions/setup-python@v7" in content or "actions/setup-python@v5" in content
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


def test_hf_and_onnx_scorers_pin_revision_and_server_narrows_choice_answer() -> None:
    """Regression test for Issue #48: Bandit B615 revision pinning & Mypy ChoiceAnswer narrowing."""
    import inspect
    from microllm.backends.base import create_hf_zero_shot_scorer, create_onnx_nli_scorer

    hf_sig = inspect.signature(create_hf_zero_shot_scorer)
    assert "revision" in hf_sig.parameters
    assert hf_sig.parameters["revision"].default == "main"

    onnx_sig = inspect.signature(create_onnx_nli_scorer)
    assert "revision" in onnx_sig.parameters
    assert onnx_sig.parameters["revision"].default == "main"

    base_src = (ROOT / "src" / "microllm" / "backends" / "base.py").read_text(encoding="utf-8")
    assert "from_pretrained(model_name, revision=revision)" in base_src
    assert "from_pretrained(tokenizer_name, revision=revision)" in base_src

    server_src = (ROOT / "src" / "microllm" / "server.py").read_text(encoding="utf-8")
    assert "isinstance(a2, ChoiceAnswer)" in server_src
    assert "isinstance(a2_rev, ChoiceAnswer)" in server_src


def test_bandit_b108_tempfile_and_mypy_web_benchmark_suite_type_annotations() -> None:
    """Regression test for Issue #79: Bandit B108 in rca_runner.py & Mypy assignment in web_benchmark_suite.py."""
    from microllm.rca_runner import _verify_rca_007

    rca_src = (ROOT / "src" / "microllm" / "rca_runner.py").read_text(encoding="utf-8")
    assert "import tempfile" in rca_src
    assert "tempfile.gettempdir()" in rca_src
    assert '"/tmp"' not in rca_src

    res_007 = _verify_rca_007()
    assert res_007["passed"] is True
    assert res_007["accuracy"] == 1.0

    web_bench_src = (ROOT / "src" / "microllm" / "web_benchmark_suite.py").read_text(encoding="utf-8")
    assert "predicted: Any = None" in web_bench_src

    reg_src = (ROOT / "src" / "microllm" / "registry.py").read_text(encoding="utf-8")
    assert "tokenizer_name_or_path" not in reg_src
    assert "tokenizer_name=" in reg_src
    assert 'revision="main"' not in reg_src.split("hf_hub_download", 1)[-1]
    assert "revision=revision" in reg_src



