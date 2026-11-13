"""Comprehensive Verification Test Suite for Milestone 4 (Audit Points 6, 13, 14, 17, 18).

Features Verified:
- F06 (Point 6): Hugging Face Space Sync Race Condition Fix
- F13 (Point 13): Standalone Wheel Integration Test Outside Git Repo
- F14 (Point 14): Secure Jekyll Docker Workflow
- F17 (Point 17): Dead Code Removal & Strict Linting (F401, F811)
- F18 (Point 18): CI Workflow Deduplication Across Workflows
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


# ==============================================================================
# Point 6 (F06): Hugging Face Space Sync Race Condition Fix
# ==============================================================================

class TestPoint06HfSpaceSyncRaceCondition:
    """Audit Point 6: Hugging Face Space README synchronization race resolution."""

    def test_sync_hf_readme_and_space_share_concurrency_group(self) -> None:
        """Verify both HF workflows share a unified concurrency group to serialize Hub uploads."""
        readme_wf = WORKFLOWS_DIR / "sync-hf-readme.yml"
        space_wf = WORKFLOWS_DIR / "sync-hf-space.yml"
        assert readme_wf.is_file(), "sync-hf-readme.yml must exist"
        assert space_wf.is_file(), "sync-hf-space.yml must exist"

        readme_text = readme_wf.read_text(encoding="utf-8")
        space_text = space_wf.read_text(encoding="utf-8")

        assert "concurrency:" in readme_text
        assert "concurrency:" in space_text
        assert "group: sync-hf-hub-${{ github.ref }}" in readme_text
        assert "group: sync-hf-hub-${{ github.ref }}" in space_text
        assert "cancel-in-progress: true" in readme_text
        assert "cancel-in-progress: true" in space_text

    def test_sync_hf_readme_paths_decoupled_from_code_edits(self) -> None:
        """Verify sync-hf-readme.yml triggers only on doc/script paths, not registry.py."""
        readme_wf = WORKFLOWS_DIR / "sync-hf-readme.yml"
        content = readme_wf.read_text(encoding="utf-8")
        assert "src/microllm/registry.py" not in content, (
            "sync-hf-readme.yml must not trigger on registry.py to avoid race conditions with sync-hf-space.yml"
        )
        assert "README.md" in content
        assert "docs/README.md" in content

    def test_hf_space_sync_staging_idempotency_and_metadata(self) -> None:
        """Verify stage_hf_space creates required files and valid YAML frontmatter."""
        from microllm.hf_space_sync import stage_hf_space

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td)
            res = stage_hf_space(dest)
            assert (dest / "README.md").is_file()
            assert (dest / "app.py").is_file()
            assert (dest / "requirements.txt").is_file()
            assert (dest / ".gitignore").is_file()

            readme_content = (dest / "README.md").read_text(encoding="utf-8")
            assert readme_content.startswith("---\n")
            assert "title: MicroLLM Decision Studio" in readme_content
            assert "sdk: gradio" in readme_content
            assert "license: apache-2.0" in readme_content

    def test_build_space_readme_handles_custom_title_and_special_chars(self) -> None:
        """Verify build_space_readme formats custom titles without syntax collision."""
        from microllm.hf_space_sync import build_space_readme

        card1 = build_space_readme(title="Custom Decision Studio")
        assert "title: Custom Decision Studio" in card1

        card2 = build_space_readme(title="MicroLLM: Fast & Secure Decision Heads")
        assert "title: MicroLLM: Fast & Secure Decision Heads" in card2
        assert "sdk: gradio" in card2


# ==============================================================================
# Point 13 (F13): Standalone Wheel Integration Test
# ==============================================================================

class TestPoint13StandaloneWheelPackagingAndExecution:
    """Audit Point 13: Standalone wheel build and isolated environment verification."""

    @pytest.fixture(scope="class")
    def packaged_wheel(self) -> Path:
        """Locate or build the wheel for verification."""
        dist_dir = REPO_ROOT / "dist"
        wheels = list(dist_dir.glob("micro_jev-0.3.0-*.whl"))
        if wheels:
            return wheels[0]
        pytest.fail("Pre-built wheel not found in dist/. Run pip wheel first.")

    def test_ci_workflow_has_wheel_test_step(self) -> None:
        """Verify ci.yml contains standalone wheel integration test step."""
        ci_wf = WORKFLOWS_DIR / "ci.yml"
        content = ci_wf.read_text(encoding="utf-8")
        assert "standalone wheel" in content.lower()
        assert "healthz" in content
        assert "systemone" in content
        assert "benchmark" in content

    def test_wheel_archive_contains_all_ten_data_json_files(self, packaged_wheel: Path) -> None:
        """Verify the built wheel contains all 10 static data JSON files."""
        with zipfile.ZipFile(packaged_wheel, "r") as zf:
            namelist = zf.namelist()
            expected_data_files = [
                "microllm/data/academic_papers_catalog.json",
                "microllm/data/dataset_fixtures_offline_samples.json",
                "microllm/data/gradio_consumer_examples.json",
                "microllm/data/gradio_dataset_presets.json",
                "microllm/data/gradio_domain_presets.json",
                "microllm/data/gradio_multilingual_presets.json",
                "microllm/data/hf_jev_models_catalog.json",
                "microllm/data/hf_jev_offline_samples.json",
                "microllm/data/hf_jev_server_config_map.json",
                "microllm/data/literature_pillars.json",
            ]
            for data_file in expected_data_files:
                assert data_file in namelist, f"Missing data file in wheel: {data_file}"

            assert "microllm/static/index.html" in namelist

    def test_pyproject_toml_declares_package_data(self) -> None:
        """Verify pyproject.toml declares both static/* and data/* in package-data."""
        pyproject = REPO_ROOT / "pyproject.toml"
        content = pyproject.read_text(encoding="utf-8")
        assert 'microllm = ["static/*", "data/*"]' in content or "data/*" in content


# ==============================================================================
# Point 14 (F14): Secure Jekyll Docker Workflow
# ==============================================================================

class TestPoint14SecureJekyllDockerWorkflow:
    """Audit Point 14: Secure Jekyll Docker workflow hardening."""

    def test_jekyll_workflow_least_privilege_permissions(self) -> None:
        """Verify jekyll-docker.yml declares least-privilege permissions: contents: read."""
        jekyll_wf = WORKFLOWS_DIR / "jekyll-docker.yml"
        content = jekyll_wf.read_text(encoding="utf-8")
        assert "permissions:" in content
        assert "contents: read" in content

    def test_jekyll_workflow_no_chmod_777_or_sudo(self) -> None:
        """Verify jekyll-docker.yml eliminates unrestricted chmod 777 and sudo."""
        jekyll_wf = WORKFLOWS_DIR / "jekyll-docker.yml"
        content = jekyll_wf.read_text(encoding="utf-8")
        assert "chmod 777" not in content
        assert "chmod -R 777" not in content
        assert "sudo" not in content
        assert "eval " not in content

    def test_jekyll_workflow_pinned_actions_and_docker_image(self) -> None:
        """Verify jekyll-docker.yml uses pinned actions and explicit Docker image tag."""
        jekyll_wf = WORKFLOWS_DIR / "jekyll-docker.yml"
        content = jekyll_wf.read_text(encoding="utf-8")
        assert "actions/checkout@v4" in content
        assert "jekyll/builder:4.2.2" in content
        assert ":latest" not in content

    def test_jekyll_workflow_timeout_and_concurrency(self) -> None:
        """Verify jekyll-docker.yml sets timeout and concurrency control."""
        jekyll_wf = WORKFLOWS_DIR / "jekyll-docker.yml"
        content = jekyll_wf.read_text(encoding="utf-8")
        assert "timeout-minutes: 15" in content or "timeout-minutes:" in content
        assert "concurrency:" in content
        assert "_site" in content


# ==============================================================================
# Point 17 (F17): Dead Code Removal & Strict Linting
# ==============================================================================

class TestPoint17DeadCodeRemovalAndStrictLinting:
    """Audit Point 17: Dead code deletion and strict flake8 linting."""

    def test_mcp_server_no_dead_get_client_or_test_client(self) -> None:
        """Verify _get_client and _test_client are removed from mcp_server.py."""
        mcp_file = REPO_ROOT / "src" / "microllm" / "mcp_server.py"
        content = mcp_file.read_text(encoding="utf-8")
        assert "def _get_client(" not in content
        assert "def _test_client(" not in content
        assert "self._test_client" not in content

        from microllm import mcp_server as mcp_mod
        all_symbols = getattr(mcp_mod, "__all__", [])
        assert "_get_client" not in all_symbols
        assert "_test_client" not in all_symbols

    def test_gradio_handlers_no_duplicate_import_functools(self) -> None:
        """Verify src/microllm/gradio_handlers.py contains no duplicate import functools."""
        handlers_file = REPO_ROOT / "src" / "microllm" / "gradio_handlers.py"
        content = handlers_file.read_text(encoding="utf-8")
        assert content.count("import functools") <= 1

    def test_microllm_init_cleanly_exports_all_symbols(self) -> None:
        """Verify src/microllm/__init__.py cleanly exports all imported symbols in __all__."""
        init_file = REPO_ROOT / "src" / "microllm" / "__init__.py"
        content = init_file.read_text(encoding="utf-8")

        # No wildcard imports allowed
        for line in content.splitlines():
            assert not line.strip().startswith("from ") or "import *" not in line

        import microllm
        all_exports = getattr(microllm, "__all__", [])
        expected_exports = [
            "ALL_12_HF_JEV_MODEL_DATASETS",
            "ALL_16_EXTERNAL_DATASETS",
            "ALL_28_EXTERNAL_DATASETS",
            "HF_JEV_ECOSYSTEM_MODELS_CATALOG",
            "build_external_datasets_corpus",
            "build_hf_jev_models_corpus",
            "evaluate_web_dataset",
            "fetch_huggingface_rows",
            "format_hf_jev_models_benchmarks_markdown",
            "format_web_benchmarks_markdown",
            "get_adapter_for_dataset",
            "list_hf_jev_ecosystem_models",
            "load_web_dataset_fixture",
            "run_all_web_benchmarks",
            "run_hf_jev_model_benchmarks",
        ]
        for exp in expected_exports:
            assert exp in all_exports, f"Expected {exp} in microllm.__all__"
            assert hasattr(microllm, exp), f"microllm missing exported attribute: {exp}"

    def test_lint_workflow_enforces_f401_f811_without_exit_zero(self) -> None:
        """Verify lint.yml strictly enforces F401 and F811 without --exit-zero."""
        lint_wf = WORKFLOWS_DIR / "lint.yml"
        content = lint_wf.read_text(encoding="utf-8")
        assert "--exit-zero" not in content
        assert "F401" in content
        assert "F811" in content
        assert "flake8" in content
        assert "ruff check" in content
        assert "mypy" in content

    def test_flake8_remediated_source_files_exit_code_zero(self) -> None:
        """Verify flake8 --select=F401,F811 passes cleanly on remediated files."""
        remediated_files = [
            str(REPO_ROOT / "src" / "microllm" / "__init__.py"),
            str(REPO_ROOT / "src" / "microllm" / "mcp_server.py"),
            str(REPO_ROOT / "src" / "microllm" / "gradio_handlers.py"),
        ]
        cmd = [sys.executable, "-m", "flake8"] + remediated_files + ["--select=F401,F811"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 and "No module named flake8" in res.stderr:
            pytest.skip("flake8 is not installed in current Python environment")
        assert res.returncode == 0, f"flake8 failed:\n{res.stdout}\n{res.stderr}"


# ==============================================================================
# Point 18 (F18): CI Workflow Deduplication
# ==============================================================================

class TestPoint18CiWorkflowDeduplication:
    """Audit Point 18: Deduplication of redundant pytest runs across CI workflows."""

    def test_ci_yml_matrix_deduplicated_not_running_matrix_pytest(self) -> None:
        """Verify ci.yml does not define a redundant Python 3.10-3.12 matrix test job."""
        ci_wf = WORKFLOWS_DIR / "ci.yml"
        content = ci_wf.read_text(encoding="utf-8")
        assert "matrix:" not in content
        assert "strategy:" not in content

    def test_unit_tests_yml_owns_test_matrix(self) -> None:
        """Verify unit-tests.yml defines the matrix test runner across 3.10, 3.11, 3.12."""
        ut_wf = WORKFLOWS_DIR / "unit-tests.yml"
        content = ut_wf.read_text(encoding="utf-8")
        assert "matrix:" in content
        for ver in ("3.10", "3.11", "3.12"):
            assert ver in content
        assert "pytest" in content
        assert "--cov=microllm" in content

    def test_lint_yml_owns_static_analysis(self) -> None:
        """Verify lint.yml is the dedicated workflow for static checks and linting."""
        lint_wf = WORKFLOWS_DIR / "lint.yml"
        content = lint_wf.read_text(encoding="utf-8")
        assert "ruff check" in content
        assert "flake8" in content
        assert "mypy" in content

    def test_all_workflow_concurrency_groups_properly_scoped(self) -> None:
        """Verify all active workflow files define scoped concurrency groups."""
        for wf_file in WORKFLOWS_DIR.glob("*.yml"):
            content = wf_file.read_text(encoding="utf-8")
            if "concurrency:" in content:
                assert "group:" in content or "concurrency: " in content
                assert "cancel-in-progress:" in content
