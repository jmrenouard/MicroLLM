"""Empirical Security & CI Matrix Challenge Test Suite for Milestone 4.

Author: Challenger M4-2 (Critic & Specialist)
Focus:
- Point 14 (F14): Jekyll Docker Workflow Security Posture & Least-Privilege Permissions
- Point 18 (F18): CI Workflow Deduplication, Matrix Distribution & Strict Linting
- Point 17 (F17): Dead Code Removal & Flake8 Strict Import/Redefinition Oracles
- Point 06 (F06): Hugging Face Space Sync Race Condition & Concurrency Serialization
- Point 13 (F13): Standalone Wheel Packaging, Isolated Execution & JSON Integrity
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


# ==============================================================================
# Challenge 1: Jekyll Docker Workflow Security Posture (Point 14 / F14)
# ==============================================================================

class TestEmpiricalJekyllDockerSecurity:
    """Adversarial stress-testing of Jekyll Docker workflow security."""

    def test_jekyll_yaml_valid_and_parses_cleanly(self) -> None:
        """Assert .github/workflows/jekyll-docker.yml is valid YAML."""
        wf_file = WORKFLOWS_DIR / "jekyll-docker.yml"
        assert wf_file.is_file(), "jekyll-docker.yml must exist"

        data = yaml.safe_load(wf_file.read_text(encoding="utf-8"))
        assert isinstance(data, dict), "Parsed workflow must be a mapping"
        assert "jobs" in data
        assert "build" in data["jobs"]

    def test_jekyll_least_privilege_permissions_strict(self) -> None:
        """Assert workflow declares strict least-privilege permissions: contents: read only."""
        wf_file = WORKFLOWS_DIR / "jekyll-docker.yml"
        data = yaml.safe_load(wf_file.read_text(encoding="utf-8"))

        # Top-level permissions must be defined
        assert "permissions" in data, "Workflow must declare top-level permissions"
        permissions = data["permissions"]
        assert permissions == {"contents": "read"}, (
            f"Permissions must be strictly {{'contents': 'read'}}, got {permissions}"
        )

        # Build job must not elevate permissions
        job_perms = data["jobs"]["build"].get("permissions")
        assert job_perms is None or job_perms == {"contents": "read"}

    def test_jekyll_zero_insecure_shell_patterns(self) -> None:
        """Assert zero chmod 777, zero sudo, zero eval, zero unrestricted file access."""
        wf_file = WORKFLOWS_DIR / "jekyll-docker.yml"
        content = wf_file.read_text(encoding="utf-8")
        data = yaml.safe_load(content)

        forbidden_patterns = [
            "chmod 777",
            "chmod -R 777",
            "chmod a+rwx",
            "chmod a+w",
            "sudo",
            "eval ",
            "curl -s | bash",
            "wget -O- | sh",
            "su ",
        ]

        steps = data["jobs"]["build"].get("steps", [])
        for step in steps:
            run_cmd = step.get("run", "")
            for pattern in forbidden_patterns:
                assert pattern not in run_cmd, (
                    f"Forbidden insecure shell pattern '{pattern}' detected in step '{step.get('name')}':\n{run_cmd}"
                )

        # Full file string assertion as defense-in-depth
        for pattern in ["chmod 777", "chmod -R 777", "sudo ", "eval "]:
            assert pattern not in content

    def test_jekyll_pinned_actions_and_docker_image(self) -> None:
        """Assert checkout is pinned to @v4 and Docker image is pinned to jekyll/builder:4.2.2."""
        wf_file = WORKFLOWS_DIR / "jekyll-docker.yml"
        data = yaml.safe_load(wf_file.read_text(encoding="utf-8"))

        steps = data["jobs"]["build"]["steps"]
        checkout_steps = [s for s in steps if "actions/checkout" in s.get("uses", "")]
        assert len(checkout_steps) == 1, "Expected exactly one checkout step"
        assert checkout_steps[0]["uses"] == "actions/checkout@v4", (
            f"Checkout must be pinned to actions/checkout@v4, got {checkout_steps[0]['uses']}"
        )

        run_steps = [s for s in steps if "docker run" in s.get("run", "")]
        assert len(run_steps) == 1, "Expected exactly one docker run step"
        run_script = run_steps[0]["run"]

        # Assert pinned image
        assert "jekyll/builder:4.2.2" in run_script, "Docker image must be pinned to jekyll/builder:4.2.2"
        assert ":latest" not in run_script, "Docker image must not use mutable :latest tag"

    def test_jekyll_docker_volume_mounts_security(self) -> None:
        """Assert workspace is mounted read-only (:ro) and destination is explicitly _site (:rw)."""
        wf_file = WORKFLOWS_DIR / "jekyll-docker.yml"
        data = yaml.safe_load(wf_file.read_text(encoding="utf-8"))

        steps = data["jobs"]["build"]["steps"]
        run_steps = [s for s in steps if "docker run" in s.get("run", "")]
        run_script = run_steps[0]["run"]

        # Parse command tokens
        assert "--rm" in run_script, "Docker container must run with --rm for ephemerality"
        assert "-v ${{ github.workspace }}:/srv/jekyll:ro" in run_script, (
            "Source repository must be mounted read-only (:ro) to prevent tamper"
        )
        assert "-v ${{ github.workspace }}/_site:/srv/jekyll/_site:rw" in run_script, (
            "Output directory must be explicitly mounted :rw for build artifact generation"
        )
        assert "--destination /srv/jekyll/_site" in run_script, (
            "Explicit destination flag must be passed to jekyll build"
        )

    def test_jekyll_step_timeout_and_concurrency(self) -> None:
        """Assert workflow has bounded timeout and concurrency cancellation."""
        wf_file = WORKFLOWS_DIR / "jekyll-docker.yml"
        data = yaml.safe_load(wf_file.read_text(encoding="utf-8"))

        job = data["jobs"]["build"]
        assert job.get("timeout-minutes") == 15, "Job build must have timeout-minutes: 15"

        concurrency = data.get("concurrency")
        assert isinstance(concurrency, dict)
        assert "group" in concurrency
        assert concurrency.get("cancel-in-progress") is True


# ==============================================================================
# Challenge 2: CI Workflow Deduplication & Matrix Specialization (Point 18 / F18)
# ==============================================================================

class TestEmpiricalCiWorkflowDeduplication:
    """Adversarial testing of workflow responsibilities and deduplication."""

    def test_ci_yml_does_not_contain_matrix_or_redundant_pytest(self) -> None:
        """Verify ci.yml does not run matrix testing or duplicate unit test suite."""
        ci_file = WORKFLOWS_DIR / "ci.yml"
        data = yaml.safe_load(ci_file.read_text(encoding="utf-8"))

        jobs = data["jobs"]
        # ci.yml must only define package-and-standalone-wheel-test
        assert "package-and-standalone-wheel-test" in jobs
        job = jobs["package-and-standalone-wheel-test"]

        # Must not define a matrix strategy
        assert "strategy" not in job, "ci.yml must not contain matrix strategy"

        # Must not run `pytest --cov=microllm` against repo
        steps = job.get("steps", [])
        for step in steps:
            run_cmd = step.get("run", "")
            assert "--cov=microllm" not in run_cmd, "ci.yml must not run full repo test coverage"
            assert "pytest tests/" not in run_cmd, "ci.yml must not duplicate repo unit tests"

        # Must run build, twine check, and isolated wheel integration
        step_runs = "\n".join(s.get("run", "") for s in steps)
        assert "python -m build" in step_runs
        assert "twine check --strict" in step_runs
        assert "mktemp -d" in step_runs

    def test_unit_tests_yml_strictly_owns_matrix_execution(self) -> None:
        """Verify unit-tests.yml defines the matrix test runner across 3.10, 3.11, 3.12."""
        ut_file = WORKFLOWS_DIR / "unit-tests.yml"
        data = yaml.safe_load(ut_file.read_text(encoding="utf-8"))

        jobs = data["jobs"]
        assert "unit-tests" in jobs
        job = jobs["unit-tests"]

        strategy = job.get("strategy", {})
        matrix = strategy.get("matrix", {})
        py_versions = matrix.get("python-version", [])

        assert "3.10" in py_versions
        assert "3.11" in py_versions
        assert "3.12" in py_versions

        steps = job.get("steps", [])
        step_runs = "\n".join(s.get("run", "") for s in steps)
        assert "pytest --cov=microllm" in step_runs
        assert "--cov-report=xml" in step_runs

        # Assert upload artifact step includes coverage.xml
        upload_steps = [s for s in steps if "upload-artifact" in s.get("uses", "")]
        assert len(upload_steps) == 1
        upload_path = upload_steps[0].get("with", {}).get("path", "")
        assert "coverage.xml" in upload_path
        assert "pytest-results.xml" in upload_path

    def test_lint_yml_strictly_owns_static_analysis_and_no_exit_zero(self) -> None:
        """Verify lint.yml owns linting and does NOT use --exit-zero."""
        lint_file = WORKFLOWS_DIR / "lint.yml"
        content = lint_file.read_text(encoding="utf-8")
        data = yaml.safe_load(content)

        assert "--exit-zero" not in content, "lint.yml must never use --exit-zero"

        jobs = data["jobs"]
        assert "lint" in jobs
        assert "typecheck" in jobs

        lint_steps = "\n".join(s.get("run", "") for s in jobs["lint"].get("steps", []))
        assert "ruff check" in lint_steps
        assert "flake8" in lint_steps
        assert "F401,F811" in lint_steps

        typecheck_steps = "\n".join(s.get("run", "") for s in jobs["typecheck"].get("steps", []))
        assert "mypy" in typecheck_steps

    def test_flake8_oracle_fails_on_synthetic_f401_and_f811(self) -> None:
        """Oracle test: prove that flake8 without --exit-zero fails when F401 or F811 is present."""
        with tempfile.TemporaryDirectory() as td:
            # Synthetic F401 (unused import)
            bad_f401 = Path(td) / "bad_f401.py"
            bad_f401.write_text("import sys\nx = 1\n", encoding="utf-8")

            res_f401 = subprocess.run(
                [sys.executable, "-m", "flake8", str(bad_f401), "--select=F401"],
                capture_output=True,
                text=True,
            )
            assert res_f401.returncode != 0, "flake8 must fail on synthetic F401"
            assert "F401" in res_f401.stdout

            # Synthetic F811 (redefinition)
            bad_f811 = Path(td) / "bad_f811.py"
            bad_f811.write_text("def foo():\n    pass\n\ndef foo():\n    pass\n", encoding="utf-8")

            res_f811 = subprocess.run(
                [sys.executable, "-m", "flake8", str(bad_f811), "--select=F811"],
                capture_output=True,
                text=True,
            )
            assert res_f811.returncode != 0, "flake8 must fail on synthetic F811"
            assert "F811" in res_f811.stdout


# ==============================================================================
# Challenge 3: Dead Code Absence & Strict Flake8 Conformance (Point 17 / F17)
# ==============================================================================

class TestEmpiricalDeadCodeAbsenceAndStrictFlake8:
    """Adversarial AST inspection and strict linter execution."""

    def test_mcp_server_ast_absence_of_dead_client_helpers(self) -> None:
        """Assert AST of src/microllm/mcp_server.py contains zero references to _get_client or _test_client."""
        mcp_path = REPO_ROOT / "src" / "microllm" / "mcp_server.py"
        content = mcp_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(mcp_path))

        # Check all function and method definitions
        func_names = [
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert "_get_client" not in func_names, "_get_client function/method must be deleted"
        assert "_test_client" not in func_names, "_test_client function/method must be deleted"

        # Check all attribute assignments and accesses
        attr_names = [
            node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        ]
        assert "_test_client" not in attr_names, "self._test_client attribute must not be accessed"
        assert "_get_client" not in attr_names, "self._get_client attribute must not be accessed"

        # Regex/substring check across full text
        assert "_get_client" not in content
        assert "_test_client" not in content

    def test_gradio_handlers_ast_no_duplicate_imports(self) -> None:
        """Assert AST of src/microllm/gradio_handlers.py contains zero duplicate imports of functools."""
        handlers_path = REPO_ROOT / "src" / "microllm" / "gradio_handlers.py"
        tree = ast.parse(handlers_path.read_text(encoding="utf-8"), filename=str(handlers_path))

        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.append(node.module)

        functools_count = imported_modules.count("functools")
        assert functools_count <= 1, f"Expected at most 1 functools import, found {functools_count}"

    def test_microllm_init_clean_all_export_parity(self) -> None:
        """Assert every top-level imported symbol in src/microllm/__init__.py is either used or exported in __all__."""
        init_path = REPO_ROOT / "src" / "microllm" / "__init__.py"
        tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))

        imported_names: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.append(alias.asname or alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.append(alias.asname or alias.name)

        # Locate __all__
        all_exports: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    all_exports.append(elt.value)

        assert len(all_exports) > 0, "__all__ must be defined in __init__.py"

        # Verify that all 15 previously missing symbols are in __all__
        missing_15 = [
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
        for sym in missing_15:
            assert sym in all_exports, f"Symbol '{sym}' must be exported in __all__"

    def test_flake8_passes_cleanly_on_all_remediated_modules(self) -> None:
        """Execute real flake8 process and assert exit code 0 on all remediated files."""
        files = [
            str(REPO_ROOT / "src" / "microllm" / "__init__.py"),
            str(REPO_ROOT / "src" / "microllm" / "mcp_server.py"),
            str(REPO_ROOT / "src" / "microllm" / "gradio_handlers.py"),
        ]
        cmd = [sys.executable, "-m", "flake8"] + files + ["--select=F401,F811"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"flake8 failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        assert res.stdout.strip() == "", f"flake8 reported unexpected warnings:\n{res.stdout}"


# ==============================================================================
# Challenge 4: Hugging Face Space Race Condition & Concurrency (Point 6 / F06)
# ==============================================================================

class TestEmpiricalHfSpaceRaceCondition:
    """Adversarial testing of HF Hub workflows serialization."""

    def test_sync_hf_readme_and_space_share_concurrency_group_and_cancel_in_progress(self) -> None:
        """Verify concurrency group and cancellation configuration between both HF workflows."""
        readme_file = WORKFLOWS_DIR / "sync-hf-readme.yml"
        space_file = WORKFLOWS_DIR / "sync-hf-space.yml"

        r_data = yaml.safe_load(readme_file.read_text(encoding="utf-8"))
        s_data = yaml.safe_load(space_file.read_text(encoding="utf-8"))

        r_conc = r_data.get("concurrency", {})
        s_conc = s_data.get("concurrency", {})

        assert r_conc.get("group") == "sync-hf-hub-${{ github.ref }}", (
            f"sync-hf-readme concurrency group mismatch: {r_conc.get('group')}"
        )
        assert s_conc.get("group") == "sync-hf-hub-${{ github.ref }}", (
            f"sync-hf-space concurrency group mismatch: {s_conc.get('group')}"
        )

        assert r_conc.get("cancel-in-progress") is True
        assert s_conc.get("cancel-in-progress") is True

    def test_sync_hf_readme_paths_filter_decoupled_from_code(self) -> None:
        """Verify sync-hf-readme paths trigger excludes python source files."""
        readme_file = WORKFLOWS_DIR / "sync-hf-readme.yml"
        r_data = yaml.safe_load(readme_file.read_text(encoding="utf-8"))

        push_cfg = r_data.get("on", {}).get("push", {})
        paths = push_cfg.get("paths", [])

        assert "src/microllm/registry.py" not in paths, (
            "registry.py must not be in sync-hf-readme paths"
        )
        for p in paths:
            assert not p.startswith("src/microllm"), f"Source path '{p}' should not trigger readme sync"


# ==============================================================================
# Challenge 5: Standalone Wheel Isolation & Attack Scenarios (Point 13 / F13)
# ==============================================================================

class TestEmpiricalStandaloneWheelIsolation:
    """Stress-test standalone wheel installation and execution in isolated sandbox."""

    @pytest.fixture(scope="class")
    def verified_wheel(self) -> Path:
        """Ensure a valid built wheel exists in dist/."""
        dist_dir = REPO_ROOT / "dist"
        wheels = list(dist_dir.glob("micro_jev-0.3.0-*.whl"))
        if wheels:
            return wheels[0]

        # Build if missing
        cmd = [sys.executable, "-m", "pip", "wheel", str(REPO_ROOT), "--no-deps", "-w", str(dist_dir)]
        subprocess.run(cmd, check=True, capture_output=True)
        wheels = list(dist_dir.glob("micro_jev-0.3.0-*.whl"))
        assert len(wheels) > 0
        return wheels[0]

    def test_wheel_packaged_json_files_all_valid_json(self, verified_wheel: Path) -> None:
        """Extract and validate that all 10 packaged JSON data files parse cleanly."""
        with zipfile.ZipFile(verified_wheel, "r") as zf:
            json_members = [
                n for n in zf.namelist()
                if n.startswith("microllm/data/") and n.endswith(".json")
            ]
            assert len(json_members) == 10, f"Expected 10 JSON files, found {len(json_members)}"

            for member in json_members:
                raw_bytes = zf.read(member)
                assert len(raw_bytes) > 0, f"Packaged JSON file '{member}' is empty"
                parsed = json.loads(raw_bytes.decode("utf-8"))
                assert isinstance(parsed, (dict, list)), f"JSON root in '{member}' must be dict or list"

    def test_wheel_isolated_execution_sandbox_no_git(self, verified_wheel: Path) -> None:
        """Run isolated execution of standalone wheel in a sandbox with zero git repository context."""
        with tempfile.TemporaryDirectory() as td:
            sandbox = Path(td).resolve()

            # Install wheel into sandbox
            install_cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                str(verified_wheel),
                "--no-deps",
                "-t",
                str(sandbox),
            ]
            res_inst = subprocess.run(install_cmd, capture_output=True, text=True)
            assert res_inst.returncode == 0, f"pip install into sandbox failed:\n{res_inst.stderr}"

            # Ensure zero .git in sandbox or parents
            assert not (sandbox / ".git").exists()

            # Execute validation script
            test_code = """
import os
import sys
from fastapi.testclient import TestClient

# 1. Verify import source
import microllm
print("Imported microllm from:", microllm.__file__)
assert sandbox_path in microllm.__file__, f"Import did not come from sandbox: {microllm.__file__}"

# 2. Verify JSON data load
from microllm.papers import list_academic_papers
papers = list_academic_papers()
assert len(papers) >= 10

from microllm.hf_jev_datasets import list_hf_jev_ecosystem_models
models = list_hf_jev_ecosystem_models()
assert len(models) >= 4

# 3. Verify server endpoints
from microllm.server import create_app
app = create_app()
client = TestClient(app)

# Healthz
r = client.get("/healthz")
assert r.status_code == 200

# Benchmark GET (tests path resolution fallback)
r_b = client.get("/v1/benchmark")
assert r_b.status_code == 200

# Documentation
r_d = client.get("/documentation")
assert r_d.status_code == 200

# SystemOne decision
r_s = client.post("/v1/systemone", json={
    "state": "User requested account PIN reset.",
    "questions": {
        "intent": {
            "type": "choice",
            "criteria": {
                "pin_reset": "User requesting password or PIN reset",
                "transfer": "User transferring funds"
            }
        }
    }
})
assert r_s.status_code == 200
data = r_s.json()
assert "answers" in data and "intent" in data["answers"]

print("CHALLENGER_ISOLATED_TEST_SUCCESS")
"""
            runner_script = f"sandbox_path = {repr(str(sandbox))}\n" + test_code
            env = os.environ.copy()
            env["PYTHONPATH"] = str(sandbox)
            env["MICROLLM_CACHE_DIR"] = str(sandbox / ".cache")

            res_run = subprocess.run(
                [sys.executable, "-c", runner_script],
                cwd=str(sandbox),
                env=env,
                capture_output=True,
                text=True,
            )

            assert res_run.returncode == 0, (
                f"Isolated sandbox execution failed with returncode {res_run.returncode}:\n"
                f"STDOUT:\n{res_run.stdout}\n"
                f"STDERR:\n{res_run.stderr}"
            )
            assert "CHALLENGER_ISOLATED_TEST_SUCCESS" in res_run.stdout
