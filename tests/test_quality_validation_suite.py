"""TDD Validation Suite for the Complete Test Plan (`docs/TEST_PLAN.md`) and 750-to-1,000 Case Quality & Performance Runner."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_complete_test_plan_document_exists_and_covers_all_tracks() -> None:
    """Verify docs/TEST_PLAN.md exists and documents performance SLAs, quality gates, and the 8 tracks (750-1000 cases)."""
    plan_path = ROOT / "docs" / "TEST_PLAN.md"
    assert plan_path.is_file(), "docs/TEST_PLAN.md must exist"

    content = plan_path.read_text(encoding="utf-8")
    required_sections = [
        "Test Plan",
        "Performance",
        "Result Quality",
        "jevbench_core",
        "decision_pivot",
        "anyjev_order_invariance",
        "guardrail_safety",
        "calibration_rlcd2",
        "jevcompat_contract_and_boundaries",
        "multilingual_and_xai_fidelity",
        "performance_sla_and_batch_stress",
        "Quality Gates",
        "RCA_REGISTER.md",
    ]
    for marker in required_sections:
        assert marker in content, f"Missing required section/marker '{marker}' in docs/TEST_PLAN.md"


def test_quality_suite_builds_between_500_and_1000_test_cases() -> None:
    """Verify build_quality_test_cases generates 750 test cases by default and supports scaling up to 1000 cases."""
    from microllm.quality_suite import build_quality_test_cases

    cases_750 = build_quality_test_cases(target_cases=750)
    assert 500 <= len(cases_750) <= 1000
    assert len(cases_750) == 750

    tracks = {c["track"] for c in cases_750}
    assert tracks == {
        "jevbench_core",
        "decision_pivot",
        "anyjev_order_invariance",
        "guardrail_safety",
        "calibration_rlcd2",
        "jevcompat_contract_and_boundaries",
        "multilingual_and_xai_fidelity",
        "performance_sla_and_batch_stress",
    }

    # Verify unique IDs across all 750 test cases
    ids = [c["id"] for c in cases_750]
    assert len(set(ids)) == len(cases_750)

    # Verify scaling to 1,000 test cases also works cleanly
    cases_1000 = build_quality_test_cases(target_cases=1000)
    assert len(cases_1000) == 1000
    assert len({c["id"] for c in cases_1000}) == 1000


def test_execute_750_case_quality_and_performance_validation_suite(tmp_path: Path) -> None:
    """Execute all 750 test cases across the 8 tracks and verify all Result Quality and Performance SLA gates pass."""
    from microllm.quality_suite import format_quality_report_markdown, run_quality_validation_suite

    out_file = tmp_path / "quality_validation_report.json"
    report = run_quality_validation_suite(target_cases=750, output_path=out_file)

    assert out_file.is_file()
    assert report["total_test_cases"] == 750
    assert report["total_decisions_evaluated"] >= 1000
    assert report["all_gates_passed"] is True, f"Failed gates: {report.get('failed_gates')}"

    # Check individual track metrics
    tracks = report["tracks"]
    assert tracks["jevbench_core"]["cascade_accuracy"] >= 0.95
    assert tracks["decision_pivot"]["pair_accuracy"] >= 0.90
    assert tracks["anyjev_order_invariance"]["gear2_l0_flip_rate"] == 0.0
    assert tracks["guardrail_safety"]["f1_score"] >= 0.94
    assert tracks["calibration_rlcd2"]["calibrated_ece"] < tracks["calibration_rlcd2"]["raw_ece"]
    assert tracks["jevcompat_contract_and_boundaries"]["pass_rate"] == 1.0
    assert tracks["jevcompat_contract_and_boundaries"]["http_5xx_count"] == 0
    assert tracks["multilingual_and_xai_fidelity"]["pass_rate"] >= 0.95
    assert tracks["performance_sla_and_batch_stress"]["pass_rate"] == 1.0

    # Check Performance SLAs
    perf = report["performance_sla"]
    assert perf["p50_latency_ms"] <= 10.0
    assert perf["p95_latency_ms"] <= 25.0
    assert perf["throughput_decisions_per_sec"] >= 100.0
    assert perf["int8_quantization_mse"] <= 1e-3

    md = format_quality_report_markdown(report)
    assert "750" in md
    assert "PASS" in md


def test_rca_register_and_action_logbook_updated_with_test_plan() -> None:
    """Verify docs/RCA_REGISTER.md and docs/ACTION_LOGBOOK.md record the test plan and 750-case validation suite."""
    rca_content = (ROOT / "docs" / "RCA_REGISTER.md").read_text(encoding="utf-8")
    logbook_content = (ROOT / "docs" / "ACTION_LOGBOOK.md").read_text(encoding="utf-8")

    assert "TEST_PLAN.md" in rca_content
    assert "quality_suite.py" in rca_content
    assert "RCA-012" in rca_content
    assert "APP-007" in rca_content
    assert "IDEA-021" in rca_content
    assert "ACT-018" in logbook_content
    assert "TEST_PLAN.md" in logbook_content
