"""TDD validation suite for Product Documentation and Technical Specifications (Issue #70)."""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PRODUCT_DOC = ROOT / "docs" / "PRODUCT_DOCUMENTATION.md"
TECH_SPEC_DOC = ROOT / "docs" / "TECHNICAL_SPECIFICATIONS.md"
ROOT_README = ROOT / "README.md"
DOCS_README = ROOT / "docs" / "README.md"


JEVCOMPAT_32_MUST_IDS = [
    "http.models",
    "http.endpoint",
    "http.auth",
    "request.state-required",
    "request.state-types",
    "request.questions-required",
    "request.questions-non-empty",
    "request.model-optional",
    "response.top-level",
    "response.answer-keys-match",
    "response.usage-shape",
    "noul.minimal",
    "noul.criteria-optional",
    "noul.criteria-keys",
    "noul.answer-shape",
    "choice.criteria-required",
    "choice.criteria-bounds",
    "choice.option-values",
    "choice.answer-shape",
    "choice.choice-in-criteria",
    "choice.probabilities-keys",
    "choice.probabilities-valid",
    "choice.confidence-range",
    "score.criteria-required",
    "score.criteria-bounds",
    "score.answer-shape",
    "score.score-range",
    "score.legend-matches",
    "score.probabilities-keys",
    "score.probabilities-valid",
    "score.confidence-range",
    "errors.validation-shape",
    "errors.no-5xx",
    "semantics.question-id",
    "semantics.batching",
    "semantics.question-order",
]


ALL_18_DOMAINS = [
    "customer_support",
    "incident_triage",
    "ai_safety_guardrail",
    "policy_compliance",
    "multilingual_routing",
    "banking_20",
    "contrastive_pivot",
    "procurement_finance",
    "healthcare_clinical_triage",
    "legal_contract_clauses",
    "logistics_supply_chain",
    "hr_recruitment_policy",
    "insurance_claims_adjudication",
    "cybersecurity_soc_alerts",
    "education_academic_grading",
    "retail_ecommerce_moderation",
    "public_sector_civic_routing",
    "financial_risk_aml",
]


def test_product_documentation_exists_and_covers_all_product_pillars() -> None:
    """Verify `docs/PRODUCT_DOCUMENTATION.md` exists and covers all product features, domains, CLI & SDK."""
    assert PRODUCT_DOC.is_file(), "docs/PRODUCT_DOCUMENTATION.md must exist"
    content = PRODUCT_DOC.read_text(encoding="utf-8")
    assert len(content) >= 15000, "Product documentation must be comprehensive (>= 15,000 chars)"

    required_keywords = [
        "System-1",
        "System-2",
        "noul",
        "choice",
        "score",
        "multi",
        "x_conformal_set",
        "x_abstain",
        "prompt_injection",
        "destructive_operation",
        "pii_or_secret_leak",
        "hallucination_risk",
        "custom_rubrics",
        "explain_state_tokens",
        "explain_counterfactual",
        "Decision Studio",
        "MicroJevClient",
        "AsyncMicroJevClient",
        "Choice.from_enum",
        "Multi.from_enum",
        "system_one_router",
        "microllm serve",
        "microllm pull",
        "microllm list",
        "microllm benchmark",
        "microllm papers",
        "microllm rca",
        "fast-cpu-int8",
        "multilingual",
        "guardrail",
    ]
    for kw in required_keywords:
        assert kw in content, f"Missing expected product documentation keyword: {kw}"

    for domain in ALL_18_DOMAINS:
        assert domain in content, f"Missing domain use-case in PRODUCT_DOCUMENTATION.md: {domain}"


def test_technical_specifications_exists_and_covers_all_engineering_contracts() -> None:
    """Verify `docs/TECHNICAL_SPECIFICATIONS.md` covers architecture, math, 32/32 MUST, APIs, and schemas."""
    assert TECH_SPEC_DOC.is_file(), "docs/TECHNICAL_SPECIFICATIONS.md must exist"
    content = TECH_SPEC_DOC.read_text(encoding="utf-8")
    assert len(content) >= 20000, "Technical specifications must be comprehensive (>= 20,000 chars)"

    for req_id in JEVCOMPAT_32_MUST_IDS:
        assert req_id in content, f"Missing jevcompat 0.1 requirement ID in TECHNICAL_SPECIFICATIONS.md: {req_id}"

    required_endpoints = [
        "/v1/models",
        "/v1/systemone",
        "/v1/calibrate",
        "/v1/rlcd",
        "/v1/studio/analyze",
        "/v1/studio/rlcd-demo",
        "/v1/guardrail",
        "/v1/batch",
        "/v1/benchmark",
        "/v1/profiles",
        "/v1/profiles/pull",
        "/v1/papers",
        "/v1/metrics",
        "/documentation",
    ]
    for ep in required_endpoints:
        assert ep in content, f"Missing HTTP endpoint in TECHNICAL_SPECIFICATIONS.md: {ep}"

    required_symbols = [
        "FastEncoderBackend",
        "AnyJevPrefillBackend",
        "DualGearCascadeBackend",
        "PerKTemperatureScaler",
        "VectorScalingCalibrator",
        "SplitConformalCalibrator",
        "RLCDDecisionHeadTrainer",
        "RLCDReplayBuffer",
        "OnlinePriorTracker",
        "van_der_corput_order",
        "l0_circular_rotation_readout",
        "compute_dynamic_tau",
        "compute_adaptive_alpha",
        "run_hierarchical_tournament",
        "clopper_pearson_threshold",
        "tune_cascade_pareto",
        "verify_tabular_rlcd_unbiased_gradient",
        "quantize_matrix_int8",
        "verify_profile",
        "SystemOneRequest",
        "SystemOneResponse",
    ]
    for sym in required_symbols:
        assert sym in content, f"Missing technical symbol in TECHNICAL_SPECIFICATIONS.md: {sym}"


def test_documentation_hub_links_and_internal_references_resolve() -> None:
    """Verify `README.md` and `docs/README.md` link to the new docs and all relative file links resolve."""
    root_readme_text = ROOT_README.read_text(encoding="utf-8")
    docs_readme_text = DOCS_README.read_text(encoding="utf-8")

    assert "docs/PRODUCT_DOCUMENTATION.md" in root_readme_text
    assert "docs/TECHNICAL_SPECIFICATIONS.md" in root_readme_text
    assert "PRODUCT_DOCUMENTATION.md" in docs_readme_text
    assert "TECHNICAL_SPECIFICATIONS.md" in docs_readme_text

    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    comm_kit_doc = ROOT / "docs" / "RESULTS_COMMUNICATION_KIT.md"
    docs_to_check = (
        (PRODUCT_DOC, TECH_SPEC_DOC, comm_kit_doc)
        if comm_kit_doc.is_file()
        else (PRODUCT_DOC, TECH_SPEC_DOC)
    )
    for doc_path in docs_to_check:
        text = doc_path.read_text(encoding="utf-8")
        for match in link_pattern.finditer(text):
            target = match.group(1).strip()
            if (
                target.startswith("http://")
                or target.startswith("https://")
                or target.startswith("#")
                or target.startswith("mailto:")
            ):
                continue
            clean_target = target.split("#", 1)[0]
            if not clean_target:
                continue
            resolved = (doc_path.parent / clean_target).resolve()
            assert resolved.exists(), (
                f"Broken relative link '{target}' in {doc_path.name} -> {resolved}"
            )


def test_live_fastapi_routes_and_public_exports_documented_in_technical_specs() -> None:
    """Verify every `/v1/*` route in `create_app()` is documented in `docs/TECHNICAL_SPECIFICATIONS.md`."""
    from microllm.server import create_app

    app = create_app()
    tech_spec_text = TECH_SPEC_DOC.read_text(encoding="utf-8")

    v1_paths = {
        getattr(route, "path", "")
        for route in app.routes
        if getattr(route, "path", "").startswith("/v1/")
    }
    assert len(v1_paths) >= 12
    for route_path in sorted(v1_paths):
        assert route_path in tech_spec_text, (
            f"Live FastAPI route '{route_path}' missing from TECHNICAL_SPECIFICATIONS.md"
        )


def test_benchmark_procedure_and_resource_documentation_completeness() -> None:
    """Verify `BENCHMARKS.md`, `docs/EXTERNAL_DATASETS_BENCHMARKS.md`, and `docs/TEST_PLAN.md` document the full benchmark execution procedure and hardware/software resource profile."""
    benchmarks_doc = ROOT / "BENCHMARKS.md"
    ext_benchmarks_doc = ROOT / "docs" / "EXTERNAL_DATASETS_BENCHMARKS.md"
    test_plan_doc = ROOT / "docs" / "TEST_PLAN.md"

    assert benchmarks_doc.is_file()
    assert ext_benchmarks_doc.is_file()
    assert test_plan_doc.is_file()

    bench_text = benchmarks_doc.read_text(encoding="utf-8")
    ext_text = ext_benchmarks_doc.read_text(encoding="utf-8")
    plan_text = test_plan_doc.read_text(encoding="utf-8")

    required_hardware_specs = [
        "AMD Ryzen AI 9 365 w/ Radeon 880M",
        "10 physical cores / 20 logical threads",
        "32 GB",
        "24 MB L3",
        "Windows 11",
        "Python 3.14",
        "NumPy 2.5",
        "SciPy 1.18",
        "0% GPU",
    ]
    for spec in required_hardware_specs:
        assert spec in bench_text, f"Missing hardware/resource specification in BENCHMARKS.md: {spec}"
        assert spec in ext_text, f"Missing hardware/resource specification in EXTERNAL_DATASETS_BENCHMARKS.md: {spec}"
        assert spec in plan_text, f"Missing hardware/resource specification in TEST_PLAN.md: {spec}"

    required_procedure_items = [
        "microllm benchmark",
        "microllm benchmark --external",
        "microllm benchmark --external --scale-per-dataset 40",
        "microllm benchmark --external --fetch-live",
        "python -m microllm.quality_suite",
        "evaluate_structured_test_cases",
        "evaluate_all_12_hypotheses",
        "/v1/benchmark",
        "/v1/benchmark/external",
        "benchmarks/test_cases_11000_results.json",
        "benchmarks/test_cases_1000_results.json",
        "benchmarks/quality_validation_750_results.json",
        "benchmarks/latest_results.json",
        "benchmarks/web_benchmarks_results.json",
        "benchmarks/external_datasets_corpus.jsonl",
    ]
    for item in required_procedure_items:
        assert item in bench_text, f"Missing benchmark procedure item in BENCHMARKS.md: {item}"


def test_results_communication_kit_exists_and_covers_all_audiences_and_channels() -> None:
    """Verify `docs/RESULTS_COMMUNICATION_KIT.md` exists, is linked from READMEs, and covers executive and technical communication templates (Issue #85)."""
    comm_kit_doc = ROOT / "docs" / "RESULTS_COMMUNICATION_KIT.md"
    assert comm_kit_doc.is_file(), "docs/RESULTS_COMMUNICATION_KIT.md must exist"
    content = comm_kit_doc.read_text(encoding="utf-8")
    assert len(content) >= 12000, "Results Communication Kit must be comprehensive (>= 12,000 chars)"

    required_sections_and_metrics = [
        "Executive Summary",
        "114,018",
        "11,000",
        "99.75%",
        "65.92%",
        "86.43%",
        "0.076 ms",
        "0.14 ms",
        "0.369 ms",
        "0.000000",
        "32/32 MUST",
        "LinkedIn",
        "Hacker News",
        "Show HN",
        "Pitch Deck",
        "Slide 1",
        "Slide 10",
        "CTO",
        "ML",
        "OpenJev-RLCD",
        "AnyJev",
        "StartLux-Decision",
        "Decision Studio",
        "microllm benchmark",
    ]
    for item in required_sections_and_metrics:
        assert item in content, f"Missing expected section or metric in RESULTS_COMMUNICATION_KIT.md: {item}"

    root_readme_text = ROOT_README.read_text(encoding="utf-8")
    docs_readme_text = DOCS_README.read_text(encoding="utf-8")
    assert "docs/RESULTS_COMMUNICATION_KIT.md" in root_readme_text
    assert "RESULTS_COMMUNICATION_KIT.md" in docs_readme_text



