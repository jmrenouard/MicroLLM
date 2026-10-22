"""TDD validation suite for Product Documentation and Technical Specifications (Issue #70)."""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PRODUCT_DOC = ROOT / "docs" / "PRODUCT_DOCUMENTATION.md"
TECH_SPEC_DOC = ROOT / "docs" / "TECHNICAL_SPECIFICATIONS.md"
ROOT_README = ROOT / "README.md"
DOCS_README = ROOT / "docs" / "README.md"
RESULTS_KIT = ROOT / "docs" / "RESULTS_COMMUNICATION_KIT.md"


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


def test_lightpath_branded_slides_deck_exists_and_serves_over_http() -> None:
    """Verify `docs/slides.html` and `docs/assets/lightpath-logo.svg` exist with LightPath branding, 12 slides, and `/slides` route (Issue #93)."""
    from fastapi.testclient import TestClient
    from microllm.server import create_app

    slides_doc = ROOT / "docs" / "slides.html"
    lp_logo_svg = ROOT / "docs" / "assets" / "lightpath-logo.svg"
    assert slides_doc.is_file(), "docs/slides.html must exist"
    assert lp_logo_svg.is_file(), "docs/assets/lightpath-logo.svg must exist"

    logo_text = lp_logo_svg.read_text(encoding="utf-8")
    assert "<svg" in logo_text and "LightPath" in logo_text

    slides_html = slides_doc.read_text(encoding="utf-8")
    assert len(slides_html) >= 18000, "LightPath slide deck must be comprehensive (>= 18,000 chars)"

    required_branding_and_metrics = [
        "LightPath",
        "www.lightpath.fr",
        "Jean-Marie Renouard",
        "LP-PRES-2026-MICROLLM",
        "#0088cc",
        "#00f2fe",
        "#070d18",
        "114,018",
        "11,000",
        "99.75%",
        "87.49%",
        "0.076 ms",
        "0.000000",
        "32/32 MUST",
        "OpenJev-RLCD",
        "AnyJev",
        "StartLux-Decision",
        "data-slide=\"1\"",
        "data-slide=\"12\"",
        "@media print",
    ]
    for item in required_branding_and_metrics:
        assert item in slides_html, f"Missing expected LightPath branding or metric in docs/slides.html: {item}"

    client = TestClient(create_app())
    resp = client.get("/slides")
    assert resp.status_code == 200
    assert "LightPath" in resp.text
    assert "LP-PRES-2026-MICROLLM" in resp.text

    svg_resp = client.get("/assets/lightpath-logo.svg")
    assert svg_resp.status_code == 200
    assert "LightPath" in svg_resp.text


def test_core_algorithms_reference_document_exists_and_covers_all_foundational_algorithms() -> None:
    """Verify `docs/ALGORITHMS.md` exists and provides exhaustive mathematical and procedural specifications (Issue #98)."""
    algo_doc = ROOT / "docs" / "ALGORITHMS.md"
    assert algo_doc.is_file(), "docs/ALGORITHMS.md must exist"
    content = algo_doc.read_text(encoding="utf-8")
    assert len(content) >= 15000, "Algorithms documentation must be comprehensive (>= 15,000 chars)"

    required_algorithms_and_math = [
        "MicroLLM",
        "Algorithmic Reference Manual",
        "FastEncoder",
        "BLAKE2b",
        "Multi-Resolution Feature Hashing",
        "L2-Simplex Normalization",
        "choice",
        "noul",
        "score",
        "multi",
        "Proposition 1",
        "Cyclic Permutation",
        "L0 Flip",
        "Order-Bias Cancellation",
        "DecisionPivot",
        "Shifted Brier",
        "2 u_y - ||u||^2",
        "Expected Calibration Error",
        "Murphy Brier",
        "RLCD",
        "Beta(8,2)",
        "Split Conformal Prediction",
        "Mondrian",
        "Hierarchical Tournament",
        "Dual-Gear Cascade",
        "Value-of-Computation",
        "Epistemic vs. Aleatoric",
        "Multi-Rubric AI Safety Guardrail",
        "INT8",
        "tau = 0.45",
        "```mermaid",
    ]
    for token in required_algorithms_and_math:
        assert token in content, f"Missing required algorithmic section or token in docs/ALGORITHMS.md: {token}"

    root_readme = ROOT_README.read_text(encoding="utf-8")
    docs_readme = DOCS_README.read_text(encoding="utf-8")
    assert "docs/ALGORITHMS.md" in root_readme or "ALGORITHMS.md" in root_readme
    assert "ALGORITHMS.md" in docs_readme


def test_engineering_and_scientific_syllabus_document_exists_and_covers_all_modules() -> None:
    """Verify `docs/SYLLABUS.md` exists with structured 6-module curriculum, learning outcomes, and lab exercises (Issue #98)."""
    syllabus_doc = ROOT / "docs" / "SYLLABUS.md"
    assert syllabus_doc.is_file(), "docs/SYLLABUS.md must exist"
    content = syllabus_doc.read_text(encoding="utf-8")
    assert len(content) >= 15000, "Syllabus documentation must be comprehensive (>= 15,000 chars)"

    required_syllabus_structure = [
        "MicroLLM Engineering & Scientific Syllabus",
        "Module 1",
        "Module 2",
        "Module 3",
        "Module 4",
        "Module 5",
        "Module 6",
        "Course Learning Outcomes",
        "Prerequisites",
        "Hands-On Lab",
        "Mathematical Derivations",
        "Capstone Project",
        "Grading Rubric",
        "OpenJev-RLCD",
        "AnyJev",
        "StartLux-Decision",
        "jevcompat 0.1",
        "```mermaid",
    ]
    for token in required_syllabus_structure:
        assert token in content, f"Missing required syllabus structure token in docs/SYLLABUS.md: {token}"

    root_readme = ROOT_README.read_text(encoding="utf-8")
    docs_readme = DOCS_README.read_text(encoding="utf-8")
    assert "docs/SYLLABUS.md" in root_readme or "SYLLABUS.md" in root_readme
    assert "SYLLABUS.md" in docs_readme


def test_decision_studio_gif_screencast_exists_and_referenced_in_documentation() -> None:
    """Verify `docs/assets/decision_studio.gif` exists, is valid, and referenced in README and product docs (Issue #103)."""
    gif_path = ROOT / "docs" / "assets" / "decision_studio.gif"
    assert gif_path.is_file(), "docs/assets/decision_studio.gif must exist"
    assert gif_path.stat().st_size > 10000, "decision_studio.gif must be a real non-empty GIF asset (> 10 KB)"

    root_readme = ROOT_README.read_text(encoding="utf-8")
    product_doc = PRODUCT_DOC.read_text(encoding="utf-8")

    assert "docs/assets/decision_studio.gif" in root_readme or "assets/decision_studio.gif" in root_readme
    assert "assets/decision_studio.gif" in product_doc


def test_decision_studio_multi_view_images_exist_and_referenced_in_documentation() -> None:
    """Verify all 4 core Decision Studio screenshots exist (> 50 KB) and are referenced in README and docs (Issue #105)."""
    core_images = [
        "decision_studio_support_triage.png",
        "decision_studio_agent_guardrail.png",
        "decision_studio_multilabel_routing.png",
        "decision_studio_rlcd_and_benchmarks.png",
    ]
    assets_dir = ROOT / "docs" / "assets"
    root_readme = ROOT_README.read_text(encoding="utf-8")
    product_doc = PRODUCT_DOC.read_text(encoding="utf-8")

    for img_name in core_images:
        img_file = assets_dir / img_name
        assert img_file.is_file(), f"Image asset docs/assets/{img_name} must exist"
        assert img_file.stat().st_size > 50000, f"Image asset docs/assets/{img_name} must be > 50 KB, got {img_file.stat().st_size}"
        assert img_name in root_readme, f"Image {img_name} must be referenced in README.md"
        assert img_name in product_doc, f"Image {img_name} must be referenced in docs/PRODUCT_DOCUMENTATION.md"


def test_huggingface_spaces_and_benchmarks_links_exist_in_readme_and_docs() -> None:
    """Verify Hugging Face Spaces interactive demo badge and Benchmark repository links exist across documentation (Issue #107)."""
    root_readme = ROOT_README.read_text(encoding="utf-8")
    docs_readme = DOCS_README.read_text(encoding="utf-8")
    product_doc = PRODUCT_DOC.read_text(encoding="utf-8")
    results_kit = RESULTS_KIT.read_text(encoding="utf-8")

    hf_space_url = "https://huggingface.co/spaces/jmrenouard/microllm-decision-studio"
    hf_dataset_url = "https://huggingface.co/datasets/jmrenouard/microllm-benchmarks"

    assert hf_space_url in root_readme, "README.md must contain the Hugging Face Space URL"
    assert hf_dataset_url in root_readme, "README.md must contain the Hugging Face Benchmarks dataset URL"
    assert "Hugging Face" in root_readme, "README.md must contain Hugging Face badge"

    assert hf_space_url in docs_readme, "docs/README.md must contain the Hugging Face Space URL"
    assert hf_dataset_url in docs_readme, "docs/README.md must contain the Hugging Face Benchmarks dataset URL"

    assert hf_space_url in product_doc, "docs/PRODUCT_DOCUMENTATION.md must contain the Hugging Face Space URL"
    assert hf_dataset_url in product_doc, "docs/PRODUCT_DOCUMENTATION.md must contain the Hugging Face Benchmarks dataset URL"

    assert hf_space_url in results_kit, "docs/RESULTS_COMMUNICATION_KIT.md must contain the Hugging Face Space URL"
    assert hf_dataset_url in results_kit, "docs/RESULTS_COMMUNICATION_KIT.md must contain the Hugging Face Benchmarks dataset URL"


def test_four_official_huggingface_models_registered_and_documented(tmp_path: Path) -> None:
    """Verify the 4 official MicroLLM Hugging Face models are registered in strict order and documented across README & docs (Issue #112)."""
    from microllm.registry import MODEL_PROFILES, OFFICIAL_MICROLLM_HF_MODELS, pull_profile

    expected_models_in_order = [
        ("jmrenouard/microllm-fast-cpu-int8", "zero-shot-classification"),
        ("jmrenouard/microllm-guardrail-int8", "text-classification"),
        ("jmrenouard/microllm-multilingual-int8", "zero-shot-classification"),
        ("jmrenouard/microllm-rlcd2-decision-head", "tabular-classification"),
    ]
    expected_required_tags = [
        "llm",
        "system-1",
        "calibration",
        "conformal-prediction",
        "low-latency",
        "guardrails",
        "routing",
    ]
    assert len(OFFICIAL_MICROLLM_HF_MODELS) == 4
    for idx, (repo_id, pipeline_tag) in enumerate(expected_models_in_order, start=1):
        entry = OFFICIAL_MICROLLM_HF_MODELS[idx - 1]
        assert entry["order"] == idx
        assert entry["repo_id"] == repo_id
        assert entry["pipeline_tag"] == pipeline_tag
        assert entry["license"] == "apache-2.0"
        assert entry["tags"] == expected_required_tags
        assert entry["url"] == f"https://huggingface.co/{repo_id}"

    assert {"fast-cpu-int8", "guardrail", "multilingual", "rlcd2-decision-head"} <= set(MODEL_PROFILES.keys())
    manifest = pull_profile("rlcd2-decision-head", cache_dir=tmp_path, download_weights=False, from_hub=False)
    assert manifest["installed"] is True
    assert manifest["microllm_hf_repo"] == "jmrenouard/microllm-rlcd2-decision-head"

    # Also test pulling via full HF repo_id alias
    alias_manifest = pull_profile("jmrenouard/microllm-guardrail-int8", cache_dir=tmp_path, download_weights=False, from_hub=False)
    assert alias_manifest["profile"] == "guardrail"

    root_readme = ROOT_README.read_text(encoding="utf-8")
    docs_readme = DOCS_README.read_text(encoding="utf-8")
    product_doc = PRODUCT_DOC.read_text(encoding="utf-8")
    results_kit = RESULTS_KIT.read_text(encoding="utf-8")

    for repo_id, _ in expected_models_in_order:
        url = f"https://huggingface.co/{repo_id}"
        assert url in root_readme, f"README.md must link to {url}"
        assert url in docs_readme, f"docs/README.md must link to {url}"
        assert url in product_doc, f"docs/PRODUCT_DOCUMENTATION.md must link to {url}"
        assert url in results_kit, f"docs/RESULTS_COMMUNICATION_KIT.md must link to {url}"








