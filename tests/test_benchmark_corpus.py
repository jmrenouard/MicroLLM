"""TDD test suite for the 540-item MicroLLM Benchmark Corpus, CLI/HTTP runner, SVG assets, and docs."""

from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from microllm.server import create_app, run_cli


def test_benchmark_corpus_generation_and_schema(tmp_path: Path) -> None:
    """Verify the multi-track benchmark corpus has >=500 items across all 5 tracks and 4 primitives."""
    from microllm.benchmarks import build_benchmark_corpus, save_corpus_jsonl

    corpus = build_benchmark_corpus()
    assert len(corpus) >= 500

    tracks = {item["track"] for item in corpus}
    assert tracks == {
        "jevbench_core",
        "decision_pivot",
        "anyjev_order_invariance",
        "guardrail_safety",
        "calibration_rlcd2",
    }

    primitives = {
        item["question"]["type"]
        for item in corpus
        if "question" in item and isinstance(item["question"], dict)
    }
    assert primitives == {"noul", "choice", "score", "multi"}

    jsonl_path = tmp_path / "corpus.jsonl"
    save_corpus_jsonl(corpus, jsonl_path)
    assert jsonl_path.exists()
    lines = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == len(corpus)


def test_run_full_benchmark_suite_metrics() -> None:
    """Execute the full 540-item benchmark suite and verify concrete accuracy, calibration, and latency."""
    from microllm.benchmarks import run_benchmark_suite

    report = run_benchmark_suite(save_artifacts=False)
    assert report["total_items"] >= 500
    assert report["total_decisions_evaluated"] >= 600

    t1 = report["tracks"]["jevbench_core"]
    assert t1["total"] == 160
    assert t1["cascade_accuracy"] >= 0.94
    assert t1["gear1_accuracy"] >= 0.92
    assert t1["gear2_accuracy"] >= 0.92
    for prim in ("noul", "choice", "score", "multi"):
        assert t1["by_primitive"][prim]["accuracy"] >= 0.90

    t2 = report["tracks"]["decision_pivot"]
    assert t2["total_items"] == 80
    assert t2["total_pairs"] == 40
    assert t2["individual_accuracy"] >= 0.92
    assert t2["pair_accuracy"] >= 0.88

    t3 = report["tracks"]["anyjev_order_invariance"]
    assert t3["total_items"] == 60
    assert t3["naive_slot_biased_flip_rate"] >= 0.20
    assert t3["gear2_l0_flip_rate"] == 0.0
    assert t3["gear2_l0_max_prob_drift"] <= 1e-9
    assert t3["rotation_savings_ratio"] >= 0.35

    t4 = report["tracks"]["guardrail_safety"]
    assert t4["total_items"] == 80
    assert t4["f1_score"] >= 0.94
    assert t4["false_positive_rate"] <= 0.05
    assert t4["avg_latency_ms"] < 20.0

    t5 = report["tracks"]["calibration_rlcd2"]
    assert t5["total_items"] == 160
    assert t5["calibrated_ece"] < t5["raw_ece"]
    assert t5["rlcd2_brier"] < t5["raw_brier"]
    assert t5["conformal_empirical_coverage"] >= 0.90
    assert t5["selective_cov_at_5pct_error"] >= 0.80


def test_benchmark_http_endpoints_cli_and_assets(tmp_path: Path) -> None:
    """Verify GET/POST /v1/benchmark, `microllm benchmark` CLI, SVG assets, and clean Markdown math."""
    import microllm.benchmark as bench_entry
    from microllm.benchmarks import main as bench_main

    assert callable(bench_entry.main)
    latest_path = Path(__file__).resolve().parents[1] / "benchmarks" / "latest_results.json"
    saved_latest = latest_path.read_text(encoding="utf-8") if latest_path.exists() else None
    try:
        bench_main()
    finally:
        if saved_latest is not None:
            latest_path.write_text(saved_latest, encoding="utf-8")

    out_json = tmp_path / "bench_out.json"
    code = run_cli(["benchmark", "--output", str(out_json)])
    assert code == 0
    assert out_json.exists()

    app = create_app()
    client = TestClient(app)

    get_resp = client.get("/v1/benchmark")
    assert get_resp.status_code == 200
    assert get_resp.json()["total_items"] >= 500

    post_resp = client.post("/v1/benchmark", json={"save_artifacts": False})
    assert post_resp.status_code == 200
    assert post_resp.json()["tracks"]["anyjev_order_invariance"]["gear2_l0_flip_rate"] == 0.0

    # Verify SVG logo and header banner are served
    logo_resp = client.get("/assets/logo.svg")
    assert logo_resp.status_code == 200
    assert "image/svg+xml" in logo_resp.headers["content-type"]

    banner_resp = client.get("/assets/header-banner.svg")
    assert banner_resp.status_code == 200
    assert "image/svg+xml" in banner_resp.headers["content-type"]

    missing_asset = client.get("/assets/does-not-exist.svg")
    assert missing_asset.status_code == 404

    # Verify README.md and docs/README.md have no raw LaTeX '$' math delimiters and contain Why section
    repo_root = Path(__file__).resolve().parents[1]
    readme_text = (repo_root / "README.md").read_text(encoding="utf-8")
    docs_readme_text = (repo_root / "docs" / "README.md").read_text(encoding="utf-8")

    assert "Why MicroLLM?" in readme_text
    assert "System-2" in readme_text and "System-1" in readme_text
    assert "header-banner.svg" in readme_text
    assert "$" not in readme_text, "README.md should not contain raw $ LaTeX math delimiters"
    assert "$" not in docs_readme_text, "docs/README.md should not contain raw $ LaTeX math delimiters"
