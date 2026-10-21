"""TDD Validation Suite for the Complete Test Plan (`docs/TEST_PLAN.md`) and 500-to-1,000 Structured Test Case Entries (`message`, `decision_type`, `rules`, `expected_result`)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_complete_test_plan_document_exists_and_covers_all_tracks() -> None:
    """Verify docs/TEST_PLAN.md exists and documents performance SLAs, quality gates, and the 1,000 test case entries."""
    plan_path = ROOT / "docs" / "TEST_PLAN.md"
    assert plan_path.is_file(), "docs/TEST_PLAN.md must exist"

    content = plan_path.read_text(encoding="utf-8")
    required_sections = [
        "Test Plan",
        "Performance",
        "Result Quality",
        "message",
        "decision_type",
        "rules",
        "expected_result",
        "Success / Fail Detection",
        "Latency",
        "Quality Gates",
        "RCA_REGISTER.md",
    ]
    for marker in required_sections:
        assert marker in content, f"Missing required section/marker '{marker}' in docs/TEST_PLAN.md"


def test_structured_test_case_entries_500_to_1000_schema(tmp_path: Path) -> None:
    """Verify build_structured_test_entries builds 500 to 1,000 entries with {message, decision_type, rules, expected_result}."""
    from microllm.quality_suite import build_structured_test_entries, save_test_entries_jsonl

    entries_500 = build_structured_test_entries(n_entries=500)
    assert len(entries_500) == 500

    entries_1000 = build_structured_test_entries(n_entries=1000)
    assert len(entries_1000) == 1000

    # Every single entry must have: id, domain, message, decision_type, rules, expected_result
    decision_types = set()
    domains = set()
    ids = set()
    for entry in entries_1000:
        assert isinstance(entry["id"], str) and entry["id"]
        assert isinstance(entry["domain"], str) and entry["domain"]
        assert isinstance(entry["message"], str) and len(entry["message"]) > 5
        assert entry["decision_type"] in {"noul", "choice", "score", "multi", "guardrail"}
        assert isinstance(entry["rules"], (dict, list)) and len(entry["rules"]) >= 1
        assert "expected_result" in entry
        ids.add(entry["id"])
        decision_types.add(entry["decision_type"])
        domains.add(entry["domain"])

    assert len(ids) == 1000
    assert decision_types == {"noul", "choice", "score", "multi", "guardrail"}
    assert len(domains) >= 6

    jsonl_out = tmp_path / "test_cases_1000.jsonl"
    save_test_entries_jsonl(entries_1000, jsonl_out)
    assert jsonl_out.is_file()
    lines = [ln for ln in jsonl_out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1000


def test_evaluate_1000_test_case_entries_and_publish_metrics(tmp_path: Path) -> None:
    """Evaluate 1,000 structured test cases and verify published accuracy, latency, success/fail detection, and calibration metrics."""
    from microllm.quality_suite import (
        evaluate_structured_test_cases,
        format_published_metrics_markdown,
    )

    out_json = tmp_path / "test_cases_1000_results.json"
    report = evaluate_structured_test_cases(n_entries=1000, output_path=out_json)

    assert out_json.is_file()
    assert report["total_entries"] == 1000
    assert report["success_count"] + report["fail_count"] == 1000
    assert report["accuracy"] >= 0.95
    assert report["all_gates_passed"] is True

    # Verify Success / Fail Detection metrics
    det = report["success_fail_detection"]
    assert det["success_count"] == report["success_count"]
    assert det["fail_count"] == report["fail_count"]
    assert 0.0 <= det["mean_confidence_on_success"] <= 1.0
    assert 0.0 <= det["selective_accuracy_above_tau"] <= 1.0
    assert det["selective_accuracy_above_tau"] >= report["accuracy"]

    # Verify Latency metrics
    lat = report["latency_metrics_ms"]
    assert lat["p50_ms"] <= 10.0
    assert lat["p95_ms"] <= 25.0
    assert lat["p99_ms"] <= 50.0
    assert lat["throughput_entries_per_sec"] >= 100.0

    # Verify Calibration & Probabilistic metrics
    cal = report["calibration_metrics"]
    assert 0.0 <= cal["brier_score"] <= 0.25
    assert 0.0 <= cal["expected_calibration_error"] <= 0.15
    assert cal["binary_f1_score"] >= 0.94

    # Verify Breakdown by decision_type and by domain
    by_type = report["by_decision_type"]
    for dtype in ("noul", "choice", "score", "multi", "guardrail"):
        assert dtype in by_type
        assert by_type[dtype]["total"] >= 100
        assert by_type[dtype]["accuracy"] >= 0.92
        assert by_type[dtype]["p50_latency_ms"] >= 0.0

    by_dom = report["by_domain"]
    assert len(by_dom) >= 6

    md = format_published_metrics_markdown(report)
    assert "1000" in md or "1,000" in md
    assert "Accuracy" in md
    assert "Latency" in md


def test_10000_product_independent_and_11000_total_test_cases(tmp_path: Path) -> None:
    """Verify 10,000 product-independent test cases (+ 1,000 baseline = 11,000 total) and published metrics."""
    from microllm.quality_suite import (
        build_product_independent_test_entries,
        build_structured_test_entries,
        evaluate_structured_test_cases,
    )

    indep_10000 = build_product_independent_test_entries(n_entries=10000)
    assert len(indep_10000) == 10000
    indep_domains = {e["domain"] for e in indep_10000}
    assert len(indep_domains) >= 10
    assert all(e.get("product_independent") is True for e in indep_10000)

    all_11000 = build_structured_test_entries(n_entries=11000)
    assert len(all_11000) == 11000
    assert len({e["id"] for e in all_11000}) == 11000

    out_11k = tmp_path / "test_cases_11000_results.json"
    report_11k = evaluate_structured_test_cases(n_entries=11000, output_path=out_11k)
    assert out_11k.is_file()
    assert report_11k["total_entries"] == 11000
    assert report_11k["product_independent_entries"] == 10000
    assert report_11k["accuracy"] >= 0.98
    assert report_11k["all_gates_passed"] is True
    assert report_11k["latency_metrics_ms"]["p50_ms"] <= 5.0
    assert report_11k["latency_metrics_ms"]["throughput_entries_per_sec"] >= 500.0
    assert len(report_11k["by_domain"]) >= 16


def test_execute_750_case_quality_and_performance_validation_suite(tmp_path: Path) -> None:
    """Execute all 750 test cases across the 8 tracks and verify all Result Quality and Performance SLA gates pass."""
    from microllm.quality_suite import format_quality_report_markdown, run_quality_validation_suite

    out_file = tmp_path / "quality_validation_report.json"
    report = run_quality_validation_suite(target_cases=750, output_path=out_file)

    assert out_file.is_file()
    assert report["total_test_cases"] == 750
    assert report["all_gates_passed"] is True, f"Failed gates: {report.get('failed_gates')}"
    md = format_quality_report_markdown(report)
    assert "750" in md
    assert "PASS" in md


def test_rca_register_and_action_logbook_updated_with_test_plan() -> None:
    """Verify docs/RCA_REGISTER.md and docs/ACTION_LOGBOOK.md record the test plan and 1,000-entry validation suite."""
    rca_content = (ROOT / "docs" / "RCA_REGISTER.md").read_text(encoding="utf-8")
    logbook_content = (ROOT / "docs" / "ACTION_LOGBOOK.md").read_text(encoding="utf-8")

    assert "TEST_PLAN.md" in rca_content
    assert "quality_suite.py" in rca_content
    assert "RCA-012" in rca_content
    assert "APP-007" in rca_content
    assert "IDEA-021" in rca_content
    assert "ACT-018" in logbook_content
    assert "ACT-019" in logbook_content
    assert "TEST_PLAN.md" in logbook_content


def test_benchmarks_md_exists_and_is_linked_from_readme() -> None:
    """Verify BENCHMARKS.md publishes the evaluation results and is linked from README.md."""
    benchmarks_path = ROOT / "BENCHMARKS.md"
    readme_path = ROOT / "README.md"

    assert benchmarks_path.is_file(), "BENCHMARKS.md must exist at repository root"
    bench_content = benchmarks_path.read_text(encoding="utf-8")
    readme_content = readme_path.read_text(encoding="utf-8")

    for required in (
        "11,000",
        "10,000",
        "1,000",
        "99.75%",
        "99.10%",
        "0.076 ms",
        "8,601.0",
        "Success / Fail Detection",
        "healthcare_clinical_triage",
        "legal_contract_clauses",
        "financial_risk_aml",
        "jevbench_core",
        "anyjev_order_invariance",
    ):
        assert required in bench_content, f"Missing '{required}' in BENCHMARKS.md"

    assert "BENCHMARKS.md" in readme_content, "README.md must link to BENCHMARKS.md"
