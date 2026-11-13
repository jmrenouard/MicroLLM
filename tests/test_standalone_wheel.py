"""Integration tests for building, packaging, and executing the standalone wheel outside the git repository.

Verifies Audit Point 13 (Feature F13):
- Wheel builds cleanly with pyproject.toml package metadata.
- All 10 static data JSON files and static HTML assets are packaged.
- The wheel installs and runs inside a clean isolated directory outside the git repository checkout.
- /healthz, /v1/systemone, and /v1/benchmark endpoints function without git-relative path failures.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the standalone wheel cleanly in an isolated directory."""
    # Check if a pre-built wheel already exists in dist/
    dist_dir = REPO_ROOT / "dist"
    existing_wheels = list(dist_dir.glob("micro_jev-0.3.0-*.whl"))
    if existing_wheels:
        return existing_wheels[0]

    # Otherwise build wheel in an isolated directory
    build_staging = tmp_path_factory.mktemp("wheel_staging")
    src_copy = build_staging / "repo"
    shutil.copytree(str(REPO_ROOT / "src"), str(src_copy / "src"))
    shutil.copy2(str(REPO_ROOT / "pyproject.toml"), str(src_copy / "pyproject.toml"))
    shutil.copy2(str(REPO_ROOT / "README.md"), str(src_copy / "README.md"))
    shutil.copy2(str(REPO_ROOT / "LICENSE"), str(src_copy / "LICENSE"))

    dist_out = tmp_path_factory.mktemp("dist")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        str(src_copy),
        "--no-deps",
        "-w",
        str(dist_out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Failed to build wheel:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    wheels = list(dist_out.glob("*.whl"))
    assert len(wheels) == 1, f"Expected 1 wheel in {dist_out}, found {len(wheels)}"
    return wheels[0]


def test_standalone_wheel_archive_contains_all_package_data(built_wheel: Path) -> None:
    """Verify built .whl archive bundles all 10 JSON data files and static assets."""
    with zipfile.ZipFile(built_wheel, "r") as zf:
        namelist = zf.namelist()

        # 1. Verify core modules packaged
        assert "microllm/__init__.py" in namelist
        assert "microllm/server.py" in namelist
        assert "microllm/mcp_server.py" in namelist

        # 2. Verify static UI assets packaged
        assert "microllm/static/index.html" in namelist

        # 3. Verify all 10 packaged data JSON files exist
        expected_json_files = [
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
        for data_file in expected_json_files:
            assert data_file in namelist, f"Missing packaged data file in wheel: {data_file}"

        # 4. Verify distribution metadata and canonical name micro-jev
        meta_files = [n for n in namelist if n.endswith("METADATA")]
        assert len(meta_files) == 1
        meta_content = zf.read(meta_files[0]).decode("utf-8")
        assert "Name: micro-jev" in meta_content
        assert "Version: 0.3.0" in meta_content

        # 5. Verify console script entry points
        ep_files = [n for n in namelist if n.endswith("entry_points.txt")]
        assert len(ep_files) == 1
        ep_content = zf.read(ep_files[0]).decode("utf-8")
        assert "microllm = microllm.server:main" in ep_content
        assert "micro-jev = microllm.server:main" in ep_content


def test_standalone_wheel_executes_outside_git_repository(built_wheel: Path) -> None:
    """Install the standalone wheel in an isolated directory and test endpoints outside git repo."""
    with tempfile.TemporaryDirectory() as td:
        isolated_dir = Path(td).resolve()

        # 1. Install wheel into target directory outside git repo
        install_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            str(built_wheel),
            "--no-deps",
            "-t",
            str(isolated_dir),
        ]
        res_inst = subprocess.run(install_cmd, capture_output=True, text=True)
        assert res_inst.returncode == 0, f"pip install failed:\n{res_inst.stderr}"

        # 2. Run test script outside git checkout with cwd=isolated_dir
        test_script = """
import os
import sys
from fastapi.testclient import TestClient

# Verify microllm imports from isolated target
import microllm
assert microllm.__version__ == "0.3.0"

# Verify packaged data reads successfully without git repo root
from microllm.papers import list_academic_papers, get_academic_paper
papers = list_academic_papers()
assert len(papers) >= 10, f"Expected >= 10 papers, got {len(papers)}"
paper_p01 = get_academic_paper("2609.38850")
assert paper_p01 is not None and "OpenJev" in paper_p01.get("title", "")

# Verify FastAPI application instantiation and endpoints
from microllm.server import create_app
app = create_app()
client = TestClient(app)

# 1. Verify /healthz
r_health = client.get("/healthz")
assert r_health.status_code == 200, f"/healthz failed: {r_health.text}"
assert r_health.json().get("status") in ("ok", "healthy")

# 2. Verify /v1/systemone
r_sys1 = client.post("/v1/systemone", json={
    "state": "Customer asks about overdraft protection policy.",
    "questions": {
        "intent": {
            "type": "choice",
            "criteria": {
                "overdraft_inquiry": "Customer asking about overdraft protection",
                "deposit": "Customer depositing funds",
                "account_closure": "Customer closing account"
            }
        }
    }
})
assert r_sys1.status_code == 200, f"/v1/systemone failed: {r_sys1.text}"
sys1_data = r_sys1.json()
assert "answers" in sys1_data
assert "intent" in sys1_data["answers"]

# 3. Verify /v1/benchmark GET (verifies fallback path resolution when outside git repo)
r_bench_get = client.get("/v1/benchmark")
assert r_bench_get.status_code == 200, f"GET /v1/benchmark failed: {r_bench_get.text}"

# 4. Verify /documentation endpoint
r_docs = client.get("/documentation")
assert r_docs.status_code == 200, f"GET /documentation failed: {r_docs.text}"

print("STANDALONE_WHEEL_TEST_PASSED")
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(isolated_dir)
        env["MICROLLM_CACHE_DIR"] = str(isolated_dir / ".cache")

        res_exec = subprocess.run(
            [sys.executable, "-c", test_script],
            cwd=str(isolated_dir),
            env=env,
            capture_output=True,
            text=True,
        )

        assert res_exec.returncode == 0, (
            f"Execution inside isolated environment failed with code {res_exec.returncode}:\n"
            f"STDOUT:\n{res_exec.stdout}\n"
            f"STDERR:\n{res_exec.stderr}"
        )
        assert "STANDALONE_WHEEL_TEST_PASSED" in res_exec.stdout


def test_standalone_wheel_cli_help_outside_git_repo(built_wheel: Path) -> None:
    """Verify console CLI help works when executed in directory outside git repo."""
    with tempfile.TemporaryDirectory() as td:
        isolated_dir = Path(td).resolve()

        install_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            str(built_wheel),
            "--no-deps",
            "-t",
            str(isolated_dir),
        ]
        res_inst = subprocess.run(install_cmd, capture_output=True, text=True)
        assert res_inst.returncode == 0

        env = os.environ.copy()
        env["PYTHONPATH"] = str(isolated_dir)

        cli_cmd = [
            sys.executable,
            "-m",
            "microllm.cli",
            "--help",
        ]
        res_cli = subprocess.run(
            cli_cmd,
            cwd=str(isolated_dir),
            env=env,
            capture_output=True,
            text=True,
        )
        assert res_cli.returncode == 0, f"CLI help failed: {res_cli.stderr}"
        assert "MicroLLM" in res_cli.stdout or "usage:" in res_cli.stdout.lower()
