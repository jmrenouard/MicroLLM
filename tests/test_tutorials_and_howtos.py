"""Executable validation suite for Diataxis tutorials (01-03), how-to guides, CLIs, and diagram hygiene."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from microllm.registry import pull_profile, verify_profile
from microllm.server import create_app

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
SRC_DIR = REPO_ROOT / "src"

TUTORIAL_FILES: tuple[Path, ...] = (
    DOCS_DIR / "01-tutorials" / "01-local-server-quickstart.md",
    DOCS_DIR / "01-tutorials" / "02-decision-studio-and-calibration-lab.md",
    DOCS_DIR / "01-tutorials" / "03-guard-middleware-sandbox.md",
    DOCS_DIR / "01-tutorials" / "04-sakila-film-classification-and-tagging.md",
    DOCS_DIR / "01-tutorials" / "05-sirene-french-business-enrichment-lab.md",
    DOCS_DIR / "01-tutorials" / "06-guard-sql-agents-on-employees-db.md",
)

MERMAID_COL0_RE = re.compile(r"^```mermaid\s*$", re.MULTILINE)
INDENTED_MERMAID_RE = re.compile(r"^[ \t]+```mermaid\b", re.MULTILINE)
FENCED_BLOCK_RE = re.compile(r"^[ \t]*```([A-Za-z0-9_-]*)\r?\n(.*?)^[ \t]*```", re.MULTILINE | re.DOTALL)
ASCII_BOX_RE = re.compile(r"\+[=-]{3,}\+|[┌┐└┘╔╗╚╝][─━═]{2,}|[─━═]{2,}[┌┐└┘╔╗╚╝]")


def _cli_env() -> dict[str, str]:
    """Return environment with src/ prepended to PYTHONPATH for non-editable test runs."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{existing}" if existing else str(SRC_DIR)
    return env


def test_tutorial_1_local_server_quickstart_api_flow() -> None:
    """Execute Tutorial 1 HTTP probes (/healthz, /v1/models, /v1/systemone) via FastAPI TestClient."""
    client = TestClient(create_app())

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json().get("status") == "ok"

    models = client.get("/v1/models")
    assert models.status_code == 200
    model_names = {item["name"] for item in models.json().get("models", [])}
    assert "jev-latest" in model_names
    assert "microllm-gear1-fast" in model_names
    assert "microllm-gear2-anyjev-l0" in model_names

    resp = client.post(
        "/v1/systemone",
        json={
            "state": "Customer cannot log in after password reset; two-factor token rejected.",
            "questions": {
                "category": {
                    "type": "choice",
                    "instructions": "Route the support ticket.",
                    "criteria": {
                        "billing": "Invoices, refunds, charges, and subscriptions",
                        "technical_support": "Login failures, password resets, two-factor auth, and outages",
                        "sales": "Pricing, quotes, and enterprise upgrades",
                    },
                },
                "is_urgent": {
                    "type": "noul",
                    "instructions": "Does the issue block access or require urgent assistance?",
                    "criteria": {
                        "true": "User locked out or login blocked",
                        "false": "General inquiry or cosmetic question",
                    },
                },
            },
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    answers = payload["answers"]
    assert answers["category"]["choice"] == "technical_support"
    probs = answers["category"]["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-4
    assert 0.0 <= answers["is_urgent"]["noul"] <= 1.0
    assert "usage" in payload


def test_tutorial_2_studio_calibration_and_rca_all() -> None:
    """Execute Tutorial 2 endpoints (/v1/studio/analyze, /v1/calibrate, /v1/studio/rlcd-demo) and rca --all."""
    client = TestClient(create_app())

    analyze = client.post(
        "/v1/studio/analyze",
        json={
            "state": "My credit card was charged twice for invoice INV-9902.",
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "Triage ticket",
                    "criteria": {
                        "billing": "Payment issue",
                        "technical": "Bug",
                        "account": "Login",
                    },
                }
            },
        },
    )
    assert analyze.status_code == 200
    audit = analyze.json()["order_flip_audit"]["intent"]
    assert audit["order_flipped"] is False
    assert audit["max_probability_drift"] == 0.0

    cal = client.post("/v1/calibrate", json={"alpha": 0.10})
    assert cal.status_code == 200
    cal_body = cal.json()
    assert "temperature" in cal_body
    assert cal_body["calibrated_ece"] <= cal_body["raw_ece"]

    rlcd = client.post(
        "/v1/studio/rlcd-demo",
        json={"stage1_steps": 35, "stage2_steps": 25, "use_rlcd2_weights": True, "seed": 42},
    )
    assert rlcd.status_code == 200
    metrics = rlcd.json().get("metrics", rlcd.json())
    assert metrics["final_accuracy"] > metrics["initial_accuracy"]

    rca_proc = subprocess.run(
        [sys.executable, "-m", "microllm.cli", "rca", "--all", "--format", "json"],
        cwd=REPO_ROOT,
        env=_cli_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert rca_proc.returncode == 0, rca_proc.stderr
    rca_items = json.loads(rca_proc.stdout)
    assert len(rca_items) == 15
    assert all(item.get("non_regression_passed") is True for item in rca_items)


def test_tutorial_3_guard_cli_contract_flags_and_policy_stdin(tmp_path: Path) -> None:
    """Execute Tutorial 3 microllm-guard CLI contract, exit codes, and stdin --policy-file flows."""
    env = _cli_env()

    contract_proc = subprocess.run(
        [sys.executable, "-m", "microllm.guard_cli", "--contract"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert contract_proc.returncode == 0, contract_proc.stderr
    schema = json.loads(contract_proc.stdout)
    exit_codes = schema["$defs"]["CliExitCodes"]["properties"]
    assert exit_codes["allow"]["const"] == 0
    assert exit_codes["block"]["const"] == 2
    assert exit_codes["escalate"]["const"] == 3

    safe_proc = subprocess.run(
        [sys.executable, "-m", "microllm.guard_cli", "--tool", "bash", "--command", "ls -la", "--json"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert safe_proc.returncode == 0
    assert json.loads(safe_proc.stdout)["decision"] == "allow"

    danger_proc = subprocess.run(
        [sys.executable, "-m", "microllm.guard_cli", "--tool", "bash", "--command", "rm -rf /", "--json"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert danger_proc.returncode == 2
    assert json.loads(danger_proc.stdout)["decision"] == "block"

    policy_file = tmp_path / "sandbox-policy.json"
    policy_file.write_text(
        json.dumps({"allowed": ["SELECT count(*) FROM employees"], "forbidden": ["DROP TABLE", "salaries"]}),
        encoding="utf-8",
    )
    stdin_payload = json.dumps({"tool_name": "sql_exec", "command": "SELECT salary FROM salaries LIMIT 5;"})
    policy_proc = subprocess.run(
        [sys.executable, "-m", "microllm.guard_cli", "--policy-file", str(policy_file), "--stdin", "--json"],
        cwd=REPO_ROOT,
        env=env,
        input=stdin_payload,
        capture_output=True,
        text=True,
        check=False,
    )
    assert policy_proc.returncode == 2
    assert json.loads(policy_proc.stdout)["primary_threat"] == "policy_boundary_violation"


def test_howto_cli_profiles_scripts_and_guard_benchmark(tmp_path: Path) -> None:
    """Execute documented How-To commands safely in tmp_path without modifying tracked benchmarks."""
    env = _cli_env()

    list_proc = subprocess.run(
        [sys.executable, "-m", "microllm.cli", "list"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert list_proc.returncode == 0, list_proc.stderr
    profiles = json.loads(list_proc.stdout)
    assert len(profiles) >= 8

    cache_dir = tmp_path / "microllm-cache"
    pull_result = pull_profile("fast-cpu-int8", cache_dir=cache_dir)
    assert pull_result.get("profile") == "fast-cpu-int8"
    ver = verify_profile("fast-cpu-int8", cache_dir=cache_dir)
    assert ver.get("verified") is True
    assert ver.get("sha256_match") is True

    bench_out = tmp_path / "guard_judge.json"
    bench_proc = subprocess.run(
        [sys.executable, "-m", "microllm.cli", "benchmark", "--guard-judge", "--output", str(bench_out), "--json"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bench_proc.returncode == 0, bench_proc.stderr
    bench_report = json.loads(bench_out.read_text(encoding="utf-8"))
    assert bench_report["gates"]["all_gates_passed"] is True

    for py_script in (
        REPO_ROOT / "scripts" / "run_all_benchmarks.py",
        REPO_ROOT / "scripts" / "run_guard_vs_judge_benchmark.py",
        REPO_ROOT / "scripts" / "sync_hf_readmes.py",
        REPO_ROOT / "scripts" / "mkdocs_hooks.py",
        REPO_ROOT / "scripts" / "build_versioned_docs.py",
    ):
        assert py_script.is_file()
        ast.parse(py_script.read_text(encoding="utf-8"), filename=str(py_script))

    bash_bin = shutil.which("bash")
    if bash_bin is not None:
        sh_script = REPO_ROOT / "scripts" / "run_all_benchmarks.sh"
        res = subprocess.run([bash_bin, "-n"], input=sh_script.read_bytes().replace(b"\r\n", b"\n"), capture_output=True)
        assert res.returncode == 0, res.stderr.decode("utf-8", "replace")


def test_tutorials_have_col0_mermaid_and_zero_ascii_diagrams() -> None:
    """Require column-0 Mermaid diagrams in all 6 tutorials and forbid ASCII box diagrams in repo docs."""
    for tutorial in TUTORIAL_FILES:
        assert tutorial.is_file(), f"Missing tutorial: {tutorial}"
        assert MERMAID_COL0_RE.search(tutorial.read_text(encoding="utf-8")), (
            f"Tutorial missing column-0 Mermaid block: {tutorial.name}"
        )

    md_files = list(sorted(DOCS_DIR.rglob("*.md"))) + [
        REPO_ROOT / "README.md",
        REPO_ROOT / "BENCHMARKS.md",
        REPO_ROOT / "LLM_ENGINES_BENCHMARK_AND_COMPARISON.md",
    ]
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        rel = md_file.relative_to(REPO_ROOT).as_posix()
        assert not INDENTED_MERMAID_RE.search(text), f"Indented Mermaid fence found in {rel}"
        for match in FENCED_BLOCK_RE.finditer(text):
            lang = match.group(1).lower()
            body = match.group(2)
            if lang in {"mermaid", "diff", "patch"}:
                continue
            assert not ASCII_BOX_RE.search(body), (
                f"ASCII box diagram detected in {rel} (lang={lang or 'untagged'})"
            )
