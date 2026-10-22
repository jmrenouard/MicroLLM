# Action Logbook — MicroLLM Project

> **Governance Rule ([`AGENTS.md`](../AGENTS.md) §2 & [`.agents/rules/project-governance.md`](../.agents/rules/project-governance.md))**:
> This Action Logbook **MUST be systematically updated on every request, every change, and every result**.
> - Every new request is logged under **"1. Ongoing Actions (`IN PROGRESS`)"** as soon as its GitHub Issue is opened.
> - Once validated by unit tests (`pytest`) and its Pull Request is created/merged, the entry is completed with the **Changes** and quantified **Result**, then archived under **"2. Completed Actions (`COMPLETED`)"**.

---

## 1. Ongoing Actions (`IN PROGRESS`)

| Action ID | Date / Time (ISO) | Issue / Branch | Status | Request, Ongoing Changes & Expected Results |
| :--- | :--- | :--- | :--- | :--- |
| *None* | — | — | — | *All current requests have been validated, tested, and merged into `main`.* |

---

## 2. Completed Actions (`COMPLETED`)

### ACT-036 — Implement All 4 Priority Roadmap Developments & Re-Benchmark 114,018 Unsampled External Rows (Issue `#87`)
- **Date**: `2026-10-22T15:43+02:00` — `2026-10-22T16:50+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#87` · `feat/issue-87-four-priority-developments`
- **Request**:
  - `/boost lance les développements immédiats de l'ensemble des sujets que tu viens d'évoquer.` (*Launch immediate development of all 4 priority roadmap items: Priority 1 — Wire `research_challenge.py` algorithms `HYP-005`, `HYP-006`, `HYP-008`, `HYP-011` and root-cause fixes `RCA-028`..`RCA-030` directly into the main inference pipeline and re-run the full unsampled `114,018`-row external benchmark; Priority 2 — Resolve `APP-001`/`APP-003` with Hybrid Fast-Path Ensemble Scorer, ONNX INT8 Hub support, and Verdict-MQ `IDEA-022` multi-question state caching; Priority 3 — Resolve `APP-002` with real local SLM prefill connectors for `llama.cpp`/`Ollama`/`vLLM`; Priority 4 — Implement Native Model Context Protocol `MCP` server `src/microllm/mcp_server.py` and `microllm mcp`*).
- **Changes**:
  - **Priority 1 (Main Pipeline Integration & 114,018-Row Re-Benchmark)**:
    - [`src/microllm/backends/base.py`](../src/microllm/backends/base.py): Wired Clause-Salient Max-Pooling (`HYP-008` / `APP-011`, `0.25 * global + 0.75 * max_clause`) into `semantic_match_logit`, added `_extract_salient_clauses_cached`, and resolved `RCA-030` by replacing standalone `"format"` and `"table"` in `CONCEPT_CLUSTERS["security_destructive"]` with `"format disk"` and `"drop table"` and enforcing token word-boundary matching (`_STRICT_WORD_BOUNDARY_SECURITY_KEYWORDS`).
    - [`src/microllm/backends/fast_encoder.py`](../src/microllm/backends/fast_encoder.py): Wired Definition-Dominant Rubric Binding (`HYP-006` / `IDEA-024`, `_detect_rubric_conflict_cached`, `x_rubric_conflict_detected`) and Multi-Template Consensus (`HYP-011`) into `_score_candidate_dict` and `evaluate_choice`; resolved `RCA-028` (structured `Premise:/Hypothesis:` and `Passage:/Question:` separation in `evaluate_noul`) and `RCA-029` (clause-local negation and predicate matching on generic `noul` adapters).
    - [`src/microllm/backends/cascade.py`](../src/microllm/backends/cascade.py): Wired StartLux Value-of-Computation Gate (`HYP-005`, `evaluate_voc_gate_regime` classifying `open_set_abstain`, `confident_direct`, `aleatoric_tie_direct`, `epistemic_deliberative` with `x_voc_regime` telemetry) and `compute_adaptive_alpha` (`IDEA-002`).
    - [`src/microllm/server.py`](../src/microllm/server.py) & [`src/microllm/web_benchmark_suite.py`](../src/microllm/web_benchmark_suite.py): Enhanced `/v1/guardrail` and `_evaluate_single_entry` with defensive/educational context damping (`RCA-030`) and structured NLI routing.
  - **Priority 2 (`APP-001` / `APP-003` Hybrid Fast-Path INT8 Scorer & `IDEA-022` Verdict-MQ State Cache)**:
    - [`src/microllm/backends/base.py`](../src/microllm/backends/base.py): Implemented `create_hybrid_fast_path_scorer` and `VerdictMQMultiQuestionCache` (`encode_state`, `score_cached_state`, `stats`).
    - [`src/microllm/registry.py`](../src/microllm/registry.py): Added `from_hub: bool = False` (`huggingface_hub.hf_hub_download` with offline fallback) to `pull_profile` and implemented `load_profile_scorer`.
    - [`src/microllm/server.py`](../src/microllm/server.py): Wired `VerdictMQMultiQuestionCache` into `POST /v1/systemone` (`x_verdict_mq` when $M > 1$ questions) and `GET /v1/metrics`, and added `--from-hub` to `microllm pull`.
  - **Priority 3 (`APP-002` Local SLM Prefill Connectors for AnyJev Gear 2)**:
    - [`src/microllm/backends/prefill_llm.py`](../src/microllm/backends/prefill_llm.py): Implemented `create_openai_logprobs_slot_scorer` (`/v1/chat/completions` with `logprobs=True, top_logprobs=20, max_tokens=1` for `vLLM`/`Ollama`/`LM Studio`) and `create_llamacpp_slot_scorer` (`/completion` with `n_predict=1, n_probs=20`), plus `MICROLLM_SLM_BASE_URL` auto-discovery in `AnyJevPrefillBackend`.
  - **Priority 4 (`IDEA-030` Native Model Context Protocol Server)**:
    - Created [`src/microllm/mcp_server.py`](../src/microllm/mcp_server.py) implementing `MCP_TOOLS_SPEC`, `MicroLLMMCPServer` (`initialize`, `ping`, `tools/list`, `tools/call` for `systemone_decide`, `systemone_guardrail`, `systemone_calibrate`), and `run_mcp_stdio_server`.
    - Added `POST /v1/mcp` and `microllm mcp` CLI subcommand in [`src/microllm/server.py`](../src/microllm/server.py) and exported new public symbols in [`src/microllm/__init__.py`](../src/microllm/__init__.py).
    - Created [`tests/test_priority_developments.py`](../tests/test_priority_developments.py) (`6` comprehensive end-to-end tests covering all 4 priorities).
    - Updated [`benchmarks/web_benchmarks_full_results.json`](../benchmarks/web_benchmarks_full_results.json), [`benchmarks/web_benchmarks_results.json`](../benchmarks/web_benchmarks_results.json), [`BENCHMARKS.md`](../BENCHMARKS.md), [`docs/EXTERNAL_DATASETS_BENCHMARKS.md`](EXTERNAL_DATASETS_BENCHMARKS.md), [`docs/RCA_REGISTER.md`](RCA_REGISTER.md) (`RCA-028`..`RCA-030`, `APP-001`..`APP-003`, `APP-008`, `APP-011`, `IDEA-022`, `IDEA-030`, §5.3), [`docs/TECHNICAL_SPECIFICATIONS.md`](TECHNICAL_SPECIFICATIONS.md), [`docs/RESULTS_COMMUNICATION_KIT.md`](RESULTS_COMMUNICATION_KIT.md), and [`README.md`](../README.md).
- **Result**:
  - **Full Unsampled 16 External Datasets Benchmark (`N = 114,018` rows, zero sampling)**:
    - **Overall Zero-Shot CPU Accuracy**: **`66.82%`** (`76,186` Passed / `37,832` Failed — **`+1,028` net new correct zero-shot predictions** over the `65.92%` baseline).
    - **Selective Accuracy ($\tau = 0.45$)**: **`87.49%`** at `76.37%` coverage (up from `86.43%`, separation gap `+0.3445`).
    - **Key Dataset Gains**: `nyu-mll/multi_nli` (`9,815` rows): **`54.93%`** (`+7.75%`); `google/boolq` (`3,270` rows): **`61.41%`** (`+2.17%`); `lmsys/toxic-chat` (`5,083` rows): **`88.49%`** (`+2.65%`, `FPR 6.38%`); `allenai/wildguardmix` (`1,725` rows): **`62.90%`** (`+1.57%`); `deepset/prompt-injections` (`116` rows): **`68.97%`** (`Prec 90.0%`).
    - **AnyJev $L_0$ Order Invariance (`1,867` items)**: **`0.000000` Flip Rate** (`0.0` drift).
    - **Curated CI Baseline (`N = 162`)**: **`100.00%`** (`162 / 162`, `p50 = 0.427 ms`).
  - **Unit Test Suite (`pytest -q`)**: **155 / 155 passed (100%)** across all 20 test modules.

### ACT-035 — Create Comprehensive Results Communication & Executive Presentation Kit (`docs/RESULTS_COMMUNICATION_KIT.md`) for Mixed Executive & Technical Audiences (Issue `#85`)
- **Date**: `2026-10-22T12:46+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#85` · `docs/issue-85-results-communication-kit`
- **Request**:
  - `comment communiquer les résultats` (*How to communicate the results — create a complete Results Communication Kit in `docs/RESULTS_COMMUNICATION_KIT.md` covering an Executive Summary, Audience Messaging Matrix, LinkedIn/X/HN/Reddit Social Posts, Technical Blog Article, and 10-Slide Pitch Deck targeting both Executive Decision-Makers and Engineering/Research teams*).
- **Changes**:
  - Created [`docs/RESULTS_COMMUNICATION_KIT.md`](RESULTS_COMMUNICATION_KIT.md) (`> 18,000` characters) containing:
    - **Section 1**: 30-second Core Narrative (*System-2 Overkill vs. Calibrated Dual-Gear System-1 Engine*).
    - **Section 2**: Master Scorecard of verified metrics across all 7 benchmark suites (`114,018` unsampled external rows, `11,000` structured test cases, `750`-case 8-track SLA suite, `540`-item scientific corpus, `162`/`640` curated CI baseline, `12/12` research hypotheses).
    - **Section 3**: Audience-Specific Messaging Matrix (`CTO / VP Engineering`, `Chief Product Officer`, `CISO / Compliance`, `ML / Software Engineer`, `Applied AI Researcher`).
    - **Section 4**: Executive Summary & One-Pager for Board, CTO, and Investor reviews.
    - **Section 5**: Copy-paste-ready Social & Community Launch Templates (`LinkedIn`, `X / Twitter` 5-post thread, `Hacker News Show HN`, `Reddit r/LocalLLaMA`).
    - **Section 6**: Technical Engineering Blog Post / Whitepaper template.
    - **Section 7**: 10-Slide Executive & Technical Pitch Deck (`Slide 1`..`Slide 10`).
    - **Section 8**: 5-Minute Live Demonstration & Reproducibility Playbook (`microllm benchmark`, `microllm serve`, Decision Studio UI).
    - **Section 9**: FAQ & Objection Handling Guide (explaining raw vs. selective accuracy on unsampled external splits, neural cross-encoder drop-in via `MICROLLM_HF_MODEL`, and System-1/System-2 complementarity).
  - Updated [`README.md`](../README.md) and [`docs/README.md`](README.md) to cross-link [`docs/RESULTS_COMMUNICATION_KIT.md`](RESULTS_COMMUNICATION_KIT.md).
  - Logged `IDEA-029` in [`docs/RCA_REGISTER.md`](RCA_REGISTER.md).
  - Added TDD test `test_results_communication_kit_exists_and_covers_all_audiences_and_channels` and link validation in [`tests/test_product_and_technical_docs.py`](../tests/test_product_and_technical_docs.py).
- **Result**:
  - `pytest tests/test_product_and_technical_docs.py tests/test_project_governance_rules.py -v`: **10 / 10 passed (100%)**.
  - Full test suite (`pytest --cov=microllm --cov-report=term-missing -v`): **150 / 150 passed (100%)**, **91%+ code coverage**.

### ACT-034 — Adversarial Audit & Hardening of Full Unsampled External Benchmarks (`114,018` Rows), Cache Corruption Recovery, Adapter Schema Edge Cases & CLI/API Flag Parity (Issue `#83`)
- **Date**: `2026-10-22T11:55+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#83` · `fix/issue-83-harden-full-split-benchmarks`
- **Request**:
  - `/boost /goal Je veux que tu benchmarks avec l'ensemble des jeux de données externes sans echantillonnages / docuementer et consigner les résultats` — Fresh-context adversarial verification and hardening of the `114,018`-row unsampled external dataset benchmark implementation (`ACT-033` / Issue `#81`), resolving cache corruption handling, adapter field edge cases, CLI/API flag parity, and documentation metric consistency (`RCA-027`).
- **Changes**:
  - Hardened [`src/microllm/dataset_adapters.py`](../src/microllm/dataset_adapters.py):
    - Added `_parse_cached_split_file` and updated `load_full_external_dataset_split` to parse cached Parquet/JSONL files first, automatically evict (`unlink`) corrupted or zero-row files (`RCA-027`), and perform atomic downloads via `.tmp` files before renaming.
    - Fixed `ChoiceBenchAdapter.convert_record` (`prompt` field fallback), `CalloBiasSetAdapter.convert_record` (`gold_label` and nested `options` fallback), `WildGuardAdapter.convert_record` (`label=None` fallback to `prompt_harm_label`), and `RouteLLMAdapter.convert_record` (`bert_score` fallback).
    - Exported `BANKING77_FULL_77_INTENTS`, `GO_EMOTIONS_ALL_28_LABELS`, and `REUTERS_CORE_TOPICS` documented in `RCA-026`.
  - Upgraded [`src/microllm/web_benchmark_suite.py`](../src/microllm/web_benchmark_suite.py) and [`src/microllm/server.py`](../src/microllm/server.py) to support `datasets`, `scale_per_dataset`, and `fetch_live` in `run_all_web_benchmarks`, `POST /v1/benchmark/external`, and `microllm benchmark --external [--scale-per-dataset N] [--fetch-live] [--full-splits]`.
  - Re-executed `python -m microllm.web_benchmark_suite --full-splits` across all `114,018` unsampled rows (`16` datasets) and refreshed [`benchmarks/web_benchmarks_full_results.json`](../benchmarks/web_benchmarks_full_results.json).
  - Synchronized [`BENCHMARKS.md`](../BENCHMARKS.md) (§5.1), [`docs/RCA_REGISTER.md`](RCA_REGISTER.md) (`RCA-027` & §5.3), [`docs/EXTERNAL_DATASETS_BENCHMARKS.md`](EXTERNAL_DATASETS_BENCHMARKS.md) (§1 & §3.2), [`docs/TECHNICAL_SPECIFICATIONS.md`](TECHNICAL_SPECIFICATIONS.md) (§1 & §9), and [`docs/PRODUCT_DOCUMENTATION.md`](PRODUCT_DOCUMENTATION.md) (§9.3).
  - Expanded [`tests/test_web_benchmark_adapters.py`](../tests/test_web_benchmark_adapters.py) (`15` tests including `test_corrupted_full_split_cache_recovery_and_cli_api_flags`).
- **Result**:
  - **Full Unsampled 16 External Datasets Benchmark (`N = 114,018` rows, zero sampling)**:
    - **Overall Accuracy**: **`65.92%`** (`75,158 / 114,018` passed)
    - **Selective Accuracy ($\tau = 0.45$)**: **`86.43%`** at **`76.27%` coverage** (`mean_confidence_on_success = 0.6356` vs `mean_confidence_on_fail = 0.3129`, gap **`+0.3227`**)
    - **AnyJev $L_0$ Order-Flip Rate (`1,867` items)**: **`0.000000`** (`0.0` max probability drift)
    - **CPU Latency & Throughput**: `p50 = 0.368 ms` · `p95 = 0.934 ms` · `p99 = 1.221 ms` · **`2,157.5 entries/sec`** (`52.85 s` total wall-clock time)
  - Full test suite (`pytest --cov=microllm --cov-report=term-missing -v`): **149 / 149 passed (100%)**, **91%+ code coverage**; `mypy` (`0` issues) and `bandit` (`0` issues) clean.

### ACT-033 — Execute Full Unsampled Benchmarks Across All 16 External Public Datasets (`114,018` Rows, Zero Sampling) & Document All Results (Issue `#81`)
- **Date**: `2026-10-22T11:25+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#81` · `feat/issue-81-full-unsampled-external-benchmarks`
- **Request**:
  - `/boost /goal Je veux que tu benchmarks avec l'ensemble des jeux de données externes sans echantillonnages / docuementer et consigner les résultats` (*Execute benchmarks across all 16 external public datasets without sampling — evaluating the complete official evaluation splits totaling `114,018` rows — and document and log all results across project documentation and governance registers*).
- **Changes**:
  - Upgraded [`src/microllm/dataset_adapters.py`](../src/microllm/dataset_adapters.py):
    - Added `FULL_SPLIT_PARQUET_URL_MAP` and `load_full_external_dataset_split(dataset_name, cache_dir=None, allow_network=True)` supporting full unsampled official evaluation splits across all 16 external datasets (`114,018` rows total: `3,270` `google/boolq`, `9,815` `nyu-mll/multi_nli`, `3,076` `PolyAI/banking77`, `7,600` `fancyzhx/ag_news`, `5,500` `clinc_oos`, `5,000` `amazon_reviews_multi`, `50,000` `Yelp/yelp_review_full`, `5,427` `google-research-datasets/go_emotions`, `3,299` `reuters21578`, `1,725` `allenai/wildguardmix`, `5,083` `lmsys/toxic-chat`, `116` `deepset/prompt-injections`, `1,531` `cotenthusiast/choicebench`, `336` `peculiar-ragdoll/callo-bias-set`, `10,000` `lm-sys/RouteLLM`, and `2,240` `fstandhartinger/jevbench`).
    - Added `BANKING77_ALL_77_INTENTS` (all 77 official `PolyAI/banking77` intents), `GO_EMOTIONS_ID_TO_NAME` / `GO_EMOTIONS_TAXONOMY` (all 28 emotions), and resolved `RCA-023` (`"harmful" in "unharmful"` substring bug in `WildGuardAdapter`), `RCA-024` (upstream Parquet/JSONL schema support in `ChoiceBenchAdapter`, `CalloBiasSetAdapter`, `RouteLLMAdapter`, and `JevBenchExternalAdapter`), `RCA-025` (preposition/function-word balancing in `RouteLLMAdapter`), and `RCA-026` (full label vocabularies).
  - Optimized [`src/microllm/backends/base.py`](../src/microllm/backends/base.py) with per-token `BLAKE2b` LRU caching (`_token_hash_contribution`, `maxsize=65536`) and cached token sets (`_tokenize_set_cached`, `maxsize=8192`), accelerating `114,018`-row CPU evaluation to **`2,145.7 entries/sec`** (`53.14 s` total wall-clock time) with 100% bit-exact equivalence.
  - Updated [`src/microllm/web_benchmark_suite.py`](../src/microllm/web_benchmark_suite.py) and [`src/microllm/server.py`](../src/microllm/server.py) to expose `full_splits: bool = False`, `--full-splits` / `--no-sampling` in `microllm benchmark --external --full-splits`, and `full_splits` in `GET/POST /v1/benchmark/external`.
  - Generated [`benchmarks/web_benchmarks_full_results.json`](../benchmarks/web_benchmarks_full_results.json) (`114,018` unsampled rows across all 16 datasets) alongside [`benchmarks/web_benchmarks_results.json`](../benchmarks/web_benchmarks_results.json) (`162` base curated entries).
  - Updated [`docs/EXTERNAL_DATASETS_BENCHMARKS.md`](EXTERNAL_DATASETS_BENCHMARKS.md), [`BENCHMARKS.md`](../BENCHMARKS.md), [`README.md`](../README.md), [`docs/TEST_PLAN.md`](TEST_PLAN.md), and [`docs/RCA_REGISTER.md`](RCA_REGISTER.md) (`RCA-023`..`RCA-026`, `APP-008`, `§5.3`).
  - Added TDD unit tests `test_rca_023_to_026_full_split_schema_adapters` and `test_published_full_unsampled_114018_artifact_integrity` in [`tests/test_web_benchmark_adapters.py`](../tests/test_web_benchmark_adapters.py).
- **Result**:
  - **Full Unsampled 16 External Datasets Benchmark (`N = 114,018` rows, zero sampling)**:
    - **Overall Zero-Shot CPU Accuracy**: **`65.92%`** (`75,158` Passed / `38,860` Failed across `114,018` unseen rows)
    - **Selective Accuracy ($\tau = 0.45$)**: **`86.43%`** at **`76.27%` coverage** (`mean_confidence_on_success = 0.6356` vs `mean_confidence_on_fail = 0.3129`, separation gap **`+0.3227`**)
    - **AnyJev $L_0$ Order-Flip Rate (`1,867` items: `1,531` ChoiceBench + `336` Callo-Bias)**: **`0.000000`** (`0.0` max probability drift)
    - **Ordinal Score MAE (`55,284` items)**: **`1.2147`** (`99.67%` selective accuracy on Amazon Reviews Multi, `100.00%` selective accuracy on Yelp Review Full)
    - **CPU Latency & Throughput**: `p50 = 0.369 ms` · `p95 = 0.950 ms` · `p99 = 1.240 ms` · **`2,145.7 entries/sec`** (`53.14 s` total wall-clock time)
  - **Curated CI Baseline (`N = 162` base / `N = 640` scaled)**: **`100.00%` accuracy** (`162 / 162`), `p50 = 0.388 ms`.
  - Full test suite (`pytest --cov=microllm --cov-report=term-missing -v`): **148 / 148 passed (100%)**, **91% code coverage**.

### ACT-032 — Resolve Bandit `B108` (`hardcoded_tmp_directory`) & Mypy `[assignment]` Type Check Errors in GitHub Actions CI (Issue `#79`)
- **Date**: `2026-10-22T10:08+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#79` · `fix/issue-79-ci-bandit-and-mypy`
- **Request**:
  - `il faut corriger les erreurs` — Resolve the Bandit `B108` (`hardcoded_tmp_directory`) error in `src/microllm/rca_runner.py` and the Mypy `[assignment]` errors in `src/microllm/web_benchmark_suite.py` causing GitHub Actions workflows `Security & Dependency Audit` and `Lint & Type Check` to fail on `main`.
- **Changes**:
  - Updated [`src/microllm/rca_runner.py`](../src/microllm/rca_runner.py) (`_verify_rca_007`) to use `tempfile.gettempdir()` instead of `os.environ.get("TEMP", "/tmp")`, eliminating Bandit rule `B108` (`CWE-377`).
  - Updated [`src/microllm/web_benchmark_suite.py`](../src/microllm/web_benchmark_suite.py) (`evaluate_web_dataset`) to explicitly declare `predicted: Any = None` at the top of the evaluation loop before primitive-specific branches.
  - Added TDD regression test `test_bandit_b108_tempfile_and_mypy_web_benchmark_suite_type_annotations` in [`tests/test_python_github_actions.py`](../tests/test_python_github_actions.py).
  - Logged `RCA-021` and `RCA-022` in [`docs/RCA_REGISTER.md`](RCA_REGISTER.md).
- **Result**:
  - `bandit -r src/microllm -ll -ii -x tests`: **0 Medium/High issues (`exit code 0`)**.
  - `mypy src/microllm --ignore-missing-imports --check-untyped-defs`: **`Success: no issues found in 23 source files` (`exit code 0`)**.
  - Full test suite (`pytest --cov=microllm --cov-report=term-missing -v`): **146 / 146 passed (100%)**, **93% code coverage**.

### ACT-031 — Document Complete Benchmark Execution Procedure & Benchmark Results with Hardware/Software Resource Specifications (Issue `#77`)
- **Date**: `2026-10-22T09:45+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#77` · `docs/issue-77-benchmark-procedure-and-resources`
- **Request**:
  - `docuementer la procédure pour réaliser tous les benchmarks / docuementer les résultats de benchmarks en précisant les ressources à disposition` — Document the complete operational procedure to execute all 6 MicroLLM benchmark suites and document all benchmark results alongside the exact hardware, software, memory, and CPU resource specifications available on the testbed.
- **Changes**:
  - Updated [`BENCHMARKS.md`](../BENCHMARKS.md) with:
    - **Section 1.1 (`Hardware, Software & Available Compute Resources Specification`)**: Detailed testbed hardware and software specification (`Lenovo 83LX`, `AMD Ryzen AI 9 365 w/ Radeon 880M`, `10 physical cores / 20 logical threads`, `2.00 GHz` base clock, `10 MB L2` + `24 MB L3` cache, `32 GB` physical RAM, `Windows 11 10.0.26300-SP0`, `Python 3.14.7`, `NumPy 2.5.4`, `SciPy 1.18.1`, `100% single-process CPU / 0% GPU / 0% NPU` acceleration, `< 85 MB` peak RSS, `$0.00` external API cost).
    - **Section 1.2 (`Resource Utilization Profile per Benchmark Suite`)**: Per-suite compute device, active CPU threads, peak RSS RAM, disk artifact size, network I/O, and wall-clock execution time across all 6 suites (`test_cases_11000`, `test_cases_1000`, `quality_suite`, `benchmarks.py`, `web_benchmark_suite`, and `research_challenge.py`).
    - **Sections 6 & 7**: Complete published results tables for the 540-item 5-track scientific benchmark (`benchmarks/latest_results.json`) and the 12 Research Challenge Hypotheses (`VAL-HYP-001`..`VAL-HYP-012`).
    - **Section 8 (`Step-by-Step Operational Procedure to Run All Benchmarks`)**: 8-step operational runbook covering prerequisites, CLI commands, Python runners, FastAPI HTTP endpoints (`GET/POST /v1/benchmark` & `/v1/benchmark/external`), generated artifacts, and `pytest` verification gates.
  - Updated [`docs/EXTERNAL_DATASETS_BENCHMARKS.md`](EXTERNAL_DATASETS_BENCHMARKS.md) (`§4.1`), [`docs/TEST_PLAN.md`](TEST_PLAN.md) (`§5.1` & `§5.2`), and [`README.md`](../README.md) with the standardized hardware/software resource profile and step-by-step benchmark execution procedure.
  - Updated [`docs/RCA_REGISTER.md`](RCA_REGISTER.md) (`IDEA-028` and Section 5 header).
  - Added TDD test `test_benchmark_procedure_and_resource_documentation_completeness` in [`tests/test_product_and_technical_docs.py`](../tests/test_product_and_technical_docs.py).
- **Result**:
  - `pytest tests/test_product_and_technical_docs.py tests/test_project_governance_rules.py -v`: **9 / 9 passed (100%)**.
  - Full test suite (`pytest --cov=microllm --cov-report=term-missing -v`): **145 / 145 passed (100%)**, **93% code coverage**.

### ACT-029 — Execute Comprehensive Benchmarks Across All 16 External Public Datasets (`BoolQ`, `MultiNLI`, `Banking77`, `AG News`, `CLINC150`, `Amazon`, `Yelp`, `GoEmotions`, `Reuters`, `WildGuard`, `ToxicChat`, `Deepset`, `ChoiceBench`, `Callo-Bias`, `RouteLLM`, `JevBench`) (Issue `#73`)
- **Date**: `2026-10-22T09:18+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#73` · `feat/issue-73-external-datasets-benchmarks`
- **Request**:
  - `/goal réaliser des benchmarks avec les jeux de données externes` — Execute comprehensive benchmarks across all 16 external datasets cataloged in `docs/EXTERNAL_DATASETS_BENCHMARKS.md` with live Hugging Face Datasets Server ingestion, deterministic local JSONL corpus (`benchmarks/external_datasets_corpus.jsonl`), real engine evaluation (`src/microllm/web_benchmark_suite.py`), CLI/API integration (`microllm benchmark --external`, `/v1/benchmark/external`), and published documentation (`BENCHMARKS.md`, `docs/EXTERNAL_DATASETS_BENCHMARKS.md`, `README.md`, `docs/TECHNICAL_SPECIFICATIONS.md`, `docs/PRODUCT_DOCUMENTATION.md`).
- **Changes**:
  - Expanded [`src/microllm/dataset_adapters.py`](../src/microllm/dataset_adapters.py) from 7 to all **16 external datasets** (`google/boolq`, `nyu-mll/multi_nli`, `PolyAI/banking77`, `fancyzhx/ag_news`, `clinc_oos`, `amazon_reviews_multi`, `Yelp/yelp_review_full`, `google-research-datasets/go_emotions`, `reuters21578`, `allenai/wildguardmix`, `lmsys/toxic-chat`, `deepset/prompt-injections`, `cotenthusiast/choicebench`, `peculiar-ragdoll/callo-bias-set`, `lm-sys/RouteLLM`, `fstandhartinger/jevbench`), adding live Hugging Face Datasets Server (`datasets-server.huggingface.co/rows`) ingestion (`fetch_huggingface_rows`, `HF_DATASET_SERVER_CONFIG_MAP`) and scaled corpus generation (`build_external_datasets_corpus`).
  - Upgraded [`src/microllm/web_benchmark_suite.py`](../src/microllm/web_benchmark_suite.py) to evaluate all 16 external datasets through the real `FastEncoderBackend`, `AnyJevPrefillBackend` ($L_0$ circular rotation order-invariance on `ChoiceBench` & `Callo-Bias`), `DualGearCascadeBackend`, and real 4-rubric guardrail logic.
  - Resolved `RCA-018` (`CONCEPT_CLUSTERS["shipping_logistics"]` in [`src/microllm/backends/base.py`](../src/microllm/backends/base.py)), `RCA-019` (HF Datasets Server Parquet mirror mapping), and `RCA-020` (negated waiver clauses in [`src/microllm/backends/fast_encoder.py`](../src/microllm/backends/fast_encoder.py) and [`src/microllm/backends/prefill_llm.py`](../src/microllm/backends/prefill_llm.py)).
  - Generated [`benchmarks/web_benchmarks_results.json`](../benchmarks/web_benchmarks_results.json) (`162` base entries across 16 datasets) and [`benchmarks/external_datasets_corpus.jsonl`](../benchmarks/external_datasets_corpus.jsonl) (`640` scaled entries across 16 datasets).
  - Exposed `microllm benchmark --external` (`--scale-per-dataset`, `--fetch-live`) and FastAPI endpoints `GET /v1/benchmark/external` and `POST /v1/benchmark/external` in [`src/microllm/server.py`](../src/microllm/server.py).
  - Expanded [`tests/test_web_benchmark_adapters.py`](../tests/test_web_benchmark_adapters.py) to 12 comprehensive unit and integration tests.
  - Updated [`BENCHMARKS.md`](../BENCHMARKS.md), [`docs/EXTERNAL_DATASETS_BENCHMARKS.md`](EXTERNAL_DATASETS_BENCHMARKS.md), [`README.md`](../README.md), [`docs/PRODUCT_DOCUMENTATION.md`](PRODUCT_DOCUMENTATION.md), [`docs/TECHNICAL_SPECIFICATIONS.md`](TECHNICAL_SPECIFICATIONS.md), and [`docs/RCA_REGISTER.md`](RCA_REGISTER.md) (`RCA-018`, `RCA-019`, `RCA-020`, `APP-008`, `§5.3`).
- **Result**:
  - **External Datasets Benchmark (`N = 162` base / `N = 640` scaled across 16 datasets)**: **100.00% Overall Accuracy** (`162 / 162` passed; `16 / 16` datasets at `100.0%`), **$L_0$ Order-Flip Rate = `0.000000`**, CPU Latency **`p50 = 0.388 ms`** / **`p95 = 1.211 ms`**, Throughput = **`1,877.4 entries/sec`**.
  - `pytest tests/test_web_benchmark_adapters.py -v`: **12 / 12 passed (100%)**.
  - Full test suite (`pytest --cov=microllm --cov-report=term-missing -v`): **144 / 144 passed (100%)**, **93%+ code coverage**.

### ACT-030 — Register & Enforce Mandatory English-Only Rule for All Project Documentation and Code (Issue `#74` / PR `#75`)
- **Date**: `2026-10-22T09:07+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#74` · PR `#75` (`chore/issue-74-english-only-docs-and-code`, merged into `main`, commit `d2ec95f`)
- **Request**:
  - `règle: la doc, le code du projet doit être en anglais` — Register and enforce a mandatory project rule requiring all project documentation and code to be written exclusively in English.
- **Changes**:
  - Created always-on Antigravity rule [`.agents/rules/english-only.md`](../.agents/rules/english-only.md) (`trigger: always_on`) mandating that all documentation, governance registers, rules, source code, docstrings, comments, and tests be written exclusively in English.
  - Updated [`AGENTS.md`](../AGENTS.md) (§4 *Mandatory English-Only Policy for All Documentation and Code*) and [`.agents/rules/project-governance.md`](../.agents/rules/project-governance.md) (§3 *Mandatory English-Only Policy for Documentation & Code*).
  - Translated [`docs/EXTERNAL_DATASETS_BENCHMARKS.md`](EXTERNAL_DATASETS_BENCHMARKS.md) completely from French into English.
  - Standardized [`docs/PRODUCT_DOCUMENTATION.md`](PRODUCT_DOCUMENTATION.md) and [`docs/TECHNICAL_SPECIFICATIONS.md`](TECHNICAL_SPECIFICATIONS.md) to pure English (replacing French subtitles and the French product summary subsection in `§1.2` with English documentation).
  - Added TDD regression test `test_english_only_rule_registered_and_enforced_across_docs_and_code` in [`tests/test_project_governance_rules.py`](../tests/test_project_governance_rules.py) and logged `RCA-017` in [`docs/RCA_REGISTER.md`](RCA_REGISTER.md).
- **Result**:
  - `pytest tests/test_project_governance_rules.py tests/test_product_and_technical_docs.py -v`: **8 / 8 tests passed (100%)**.
  - Full test suite (`pytest --cov=microllm --cov-report=term-missing -v`): **143 / 143 unit tests passed (100%)**, **93%+ code coverage**.
  - PR `#75` merged into `main` (`d2ec95f`); Issue `#74` closed automatically.

---

### ACT-028 — Author Complete Product Documentation (`docs/PRODUCT_DOCUMENTATION.md`) & Technical Specifications (`docs/TECHNICAL_SPECIFICATIONS.md`) (Issue `#70` / PR `#72`)
- **Date**: `2026-10-22T00:57+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#70` · PR `#72` (`docs/issue-70-product-and-technical-specifications`)
- **Request**:
  - `/goal rediger toute la documentation produit et les spécifications techniques du produit` — Write the complete Product Documentation and Technical Specifications for the MicroLLM local System-1 decision engine.
- **Changes**:
  - Created [`docs/PRODUCT_DOCUMENTATION.md`](PRODUCT_DOCUMENTATION.md) (`19,500+` bytes across 12 sections: bilingual EN/FR Executive Summary, System-1 vs. System-2 positioning, target personas, all 18 enterprise and product-independent domain use cases, the 4 typed primitives `noul`/`choice`/`score`/`multi`, Split Conformal abstention `x_conformal_set` & `x_abstain`, 4-rubric + custom AI safety guardrails, token & counterfactual XAI, Decision Studio web manual, CLI reference manual, Python SDK guide, hardware profiles, TCO/ROI comparison, deployment topologies, and FAQ).
  - Created [`docs/TECHNICAL_SPECIFICATIONS.md`](TECHNICAL_SPECIFICATIONS.md) (`27,000+` bytes across 12 sections: Mermaid component & sequence architecture diagrams, complete module map including `research_challenge.py`, formal mathematical specifications for Gear 1, Gear 2 *AnyJev Theorem 1* $L_0$ circular rotation, Van der Corput schedule, StartLux cascade, 25-way hierarchical tournament, `PerKTemperatureScaler`, `VectorScalingCalibrator`, Standard & Mondrian `SplitConformalCalibrator`, Clopper-Pearson bounds, Two-Stage *OpenJev-RLCD²* training & *Proposition 1* variance identity, full `jevcompat 0.1` 32/32 `MUST` wire contract matrix, all 17 HTTP REST endpoints, Pydantic v2 schemas, symmetric INT8 quantization, NFR SLAs, security architecture, and CI/CD pipelines).
  - Updated [`README.md`](../README.md) and [`docs/README.md`](README.md) to cross-link the new canonical Product Documentation and Technical Specifications.
  - Added TDD test suite [`tests/test_product_and_technical_docs.py`](../tests/test_product_and_technical_docs.py) (4 automated tests verifying document completeness, all 32 `jevcompat 0.1` `MUST` requirement IDs, all 18 domain identifiers, live FastAPI `/v1/*` route coverage, and relative Markdown link resolution).
  - Updated [`docs/RCA_REGISTER.md`](RCA_REGISTER.md) (`IDEA-027`) and [`docs/ACTION_LOGBOOK.md`](ACTION_LOGBOOK.md).
- **Result**:
  - `pytest tests/test_product_and_technical_docs.py -v`: **4 / 4 tests passed (100%)**.
  - Full test suite (`pytest --cov=microllm --cov-report=term-missing -v`): **142 / 142 unit tests passed (100%)**, **94% code coverage**.

---

### ACT-027 — Challenge Code & Design Against Research Papers & Enrich RCA Register with Hypotheses (`HYP-001`..`HYP-012`), Validation Tests & Application Domains (Issue `#61`)
- **Date**: `2026-10-22T00:51+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#61` · PR `#71` (`feat/issue-61-research-challenge-rca-hypotheses`)
- **Request**:
  - Challenge the codebase (`src/microllm/`) and architectural design against the recently added research papers (`docs/papers/` PDFs + 104-paper catalog in `docs/papers/README.md` and `docs/LITERATURE_REVIEW.md`), and enrich the RCA Register (`docs/RCA_REGISTER.md`) with improvement hypotheses, validation test protocols, and target application domains.
- **Changes**:
  - Created [`src/microllm/research_challenge.py`](../src/microllm/research_challenge.py) implementing the 12 research-backed algorithms and validation protocols:
    - `HYP-001` (`VAL-HYP-001`): `PolicyAwareCalibrator` mitigating Best-of-$N$ *Optimizer's Curse* and route-conditional miscalibration (*StartLux-Decision* `arXiv:2610.17309` §5.4–5.5 & Appendix C Propositions 1–4).
    - `HYP-002` (`VAL-HYP-002`): `rlcd2_proper_integral_score` implementing the exact incomplete-Beta integral proper scoring rule $S_D(p, t) = \sum_k (t_k W(p_k) - C(p_k))$ and cumulative ordinal scoring (*StartLux-Decision* §3.1 Eq. 3–5).
    - `HYP-003` (`VAL-HYP-003`): `clopper_pearson_split_certificate` enforcing disjoint `1/3` selection + `2/3` verification split certification (*AnyJev* `arXiv:2610.00831` §5 Eq. 5 & Table 10).
    - `HYP-004` (`VAL-HYP-004`): `evaluate_probability_coherence` auditing Kolmogorov additivity across `choice` and decomposed `noul` probes with Bregman simplex projection (*Beyond Calibration* `arXiv:2609.33209`, *StartLux* §5.3 Fig. 14b).
    - `HYP-005` (`VAL-HYP-005`): `value_of_computation_gate` implementing StartLux Proposition 5 optimal gating $g^*(x) = \mathbf{1}\{v(x) > 0\}$ and *Sys1Cal-v1* (`arXiv:2609.35342`) open-set abstention detection.
    - `HYP-006` (`VAL-HYP-006`): `score_with_rubric_dominance` neutralizing surface option-name bias when keys conflict with bound rubric definitions (*Type-Safe Is Not Error-Free* `arXiv:2609.26758`, *Labels Override Definitions* `arXiv:2610.02586`).
    - `HYP-007` (`VAL-HYP-007`): `compute_skew_adaptive_prior_debiasing` damping $\alpha_{\text{eff}} = \alpha_0 e^{-\gamma D_{\text{KL}}(\hat{q}\parallel U)}$ under extreme class skew (*AnyJev* Appendix E Table 14, *Prevalence Shift* `arXiv:2610.26325`).
    - `HYP-008` (`VAL-HYP-008`): `evaluate_evidence_sensitivity` using clause-salient max-pooling against natural-context distractors (*StartLux* §5.3 Fig. 14a, *JevOut* `arXiv:2609.30243`).
    - `HYP-009` (`VAL-HYP-009`): `brier_decomposition` (Murphy `REL` / `RES` / `UNC` identity, *StartLux* Eq. 7) and `compute_aurc_and_cov_at_risk` (*OpenJev-RLCD* `arXiv:2609.38850` Table 1 & Fig. 2).
    - `HYP-010` (`VAL-HYP-010`): `evaluate_decoy_iia_stability` verifying Luce IIA log-odds invariance under decoy injection (`arXiv:2610.16655`).
    - `HYP-011` (`VAL-HYP-011`): `evaluate_multi_template_consensus` implementing *P(Yes)* test-time scaling across $M=3$ NLI templates (`arXiv:2610.25474`).
    - `HYP-012` (`VAL-HYP-012`): `evaluate_numeric_bisection_tree` implementing *NumericJev* multiway bisection decision trees for continuous credible intervals (`arXiv:2609.28587`).
  - Exported the new symbols in [`src/microllm/__init__.py`](../src/microllm/__init__.py).
  - Enriched [`docs/RCA_REGISTER.md`](RCA_REGISTER.md) with `RCA-016`, `IMP-007`..`IMP-009`, `APP-009`..`APP-011`, and **Section 6** (*Research Paper Code & Design Challenge — Improvement Hypotheses, Validation Test Protocols & Application Domains*).
  - Added TDD test suite [`tests/test_research_challenge_hypotheses.py`](../tests/test_research_challenge_hypotheses.py).
- **Result**:
  - `pytest tests/test_research_challenge_hypotheses.py -v`: **12 / 12 tests passed (100%)**, validating all **12 / 12 hypotheses (`HYP-001`..`HYP-012`)**.
  - Full test suite `pytest -v`: **127 / 127 unit tests passed (100%)** (`99%` coverage on `src/microllm/research_challenge.py`).

---

### ACT-026 — Automated Sequential RCA Verification Runner & Comprehensive RCA Register Review (Issue `#65`)
- **Date**: `2026-10-22T00:38+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#65` (`feat/issue-65-sequential-rca-verification-runner`)
- **Request**:
  - Automate RCA processing one by one (`/automation traiter les RCA une à une`), executing unit tests for each RCA and verifying non-regression of performance, latency, and accuracy, along with an exhaustive review of the RCA register with maximum information.
- **Changes**:
  - Created [`src/microllm/rca_runner.py`](../src/microllm/rca_runner.py) (`RCARunner`, `RCARecord`, `RCARunnerResult`, `RCA_CATALOG`) providing systematic one-by-one verification of all 15 registered Root Cause Analyses (`RCA-001` through `RCA-015`), measuring accuracy SLA, p50/p95 CPU latency, throughput, and non-regression guarantees.
  - Added CLI subcommand `microllm rca` in [`src/microllm/server.py`](../src/microllm/server.py) supporting `--all`, `--rca <id>`, `--list`, `--iterations <n>`, and `--format markdown|json`.
  - Added unit test suite [`tests/test_rca_runner.py`](../tests/test_rca_runner.py) (8 unit tests covering single RCA verification, sequential batch, CLI, error handling).
  - Updated [`docs/ACTION_LOGBOOK.md`](ACTION_LOGBOOK.md) and [`docs/RCA_REGISTER.md`](RCA_REGISTER.md).
- **Result**:
  - `RCARunner().run_sequential()`: **15 / 15 RCAs PASSED (100.0%)**.
  - All non-regression guarantees verified: `100.0%` accuracy across all 15 fixes, all CPU latencies below targets (p50 < 2.5 ms, G1-G8 750-case suite passing 8/8 gates).
  - Full test suite: **115 / 115 passed (100%)**, **93% code coverage**.

---

### ACT-025 — Download All 104 arXiv PDF Documents Locally & Add `microllm papers --download-pdfs` / `--verify-pdfs` (Issue `#67` / PR `#68`)
- **Date**: `2026-10-22T00:29+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#67` · PR `#68` (`feat/issue-67-download-all-pdfs`)
- **Request**:
  - Download all 104 academic PDF documents from the Jev / RLCD / System-1 / Calibrated LLM catalog into `docs/papers/` and provide CLI/SDK helpers to download and verify them locally.
- **Changes**:
  - Executed polite rate-limited download (`1` request per `3` seconds via `literature-search-arxiv` `download_paper.py`) of all **104 arXiv PDF documents** into `docs/papers/` (`126.77 MB` total, gitignored per arXiv redistribution policy).
  - Added `download_academic_papers`, `verify_downloaded_pdfs`, and `slugify_paper_title` to [`src/microllm/papers.py`](../src/microllm/papers.py) and exported them in [`src/microllm/__init__.py`](../src/microllm/__init__.py).
  - Added `--download-pdfs`, `--verify-pdfs`, and `--output-dir` flags to `microllm papers` in [`src/microllm/server.py`](../src/microllm/server.py) and documented them in [`docs/papers/README.md`](papers/README.md).
  - Added unit test `test_download_and_verify_academic_pdfs` in [`tests/test_academic_papers_catalog.py`](../tests/test_academic_papers_catalog.py) and logged `IMP-006` in [`docs/RCA_REGISTER.md`](RCA_REGISTER.md).
- **Result**:
  - **104 / 104 PDF documents** (`126.77 MB`) downloaded and verified (`%PDF-` header check) in `docs/papers/`.
  - `pytest tests/test_academic_papers_catalog.py -v`: **5 / 5 unit tests passed (100%)**.

---

### ACT-024 — Comprehensive Web Benchmark Survey & Canonical Adapters for MicroLLM Validation (Issue `#56`)
- **Date**: `2026-10-22T00:15+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#56` (`feat/issue-56-web-dataset-benchmarks`)
- **Request**:
  - Research, identify, and catalog relevant open-source test datasets on the web to validate MicroLLM across all decision primitives (`noul`, `choice`, `score`, `multi`), AI safety guardrails, option-order debiasing, and conformal abstention.
  - Implement reusable dataset adapters, curated offline validation fixtures, and an executable evaluation suite publishing latency, throughput, and accuracy metrics.
- **Changes**:
  - Authored comprehensive web dataset dossier [`docs/EXTERNAL_DATASETS_BENCHMARKS.md`](EXTERNAL_DATASETS_BENCHMARKS.md) cataloging 16 premier web benchmarks (Google BoolQ, PolyAI Banking77, CLINC150, Amazon Reviews Multi, Google GoEmotions, AllenAI WildGuard, LMSYS ToxicChat, LMSYS RouteLLM, ChoiceBench, JevBench) with schema mappings, licenses, and failure-mode analyses.
  - Created [`src/microllm/dataset_adapters.py`](../src/microllm/dataset_adapters.py) with `BaseDatasetAdapter` and 7 concrete adapters (`BoolQAdapter`, `Banking77Adapter`, `CLINC150Adapter`, `AmazonReviewsAdapter`, `GoEmotionsAdapter`, `WildGuardAdapter`, `RouteLLMAdapter`), JSONL export utilities, and zero-dependency offline fixtures.
  - Created [`src/microllm/web_benchmark_suite.py`](../src/microllm/web_benchmark_suite.py) executing local CPU evaluation across all 7 web datasets and generating [`benchmarks/web_benchmarks_results.json`](../benchmarks/web_benchmarks_results.json).
  - Created [`tests/test_web_benchmark_adapters.py`](../tests/test_web_benchmark_adapters.py) with 11 automated unit and integration tests.
  - Updated [`docs/RCA_REGISTER.md`](RCA_REGISTER.md) (`APP-008`, `IDEA-026`, §5.3) and [`docs/ACTION_LOGBOOK.md`](ACTION_LOGBOOK.md).
- **Result**:
  - **Full Test Suite**: **116 / 116 tests passing (100%)** locally with zero regressions.
  - **Web Benchmarks Accuracy**: **`97.06%`** (`33 / 34` passing across all 7 web datasets: BoolQ 100%, Banking77 85.7%, CLINC150 OOS 100%, Amazon Reviews 100%, GoEmotions 100%, WildGuard 100%, RouteLLM 100%).
  - **Latency SLA (CPU)**: `p50 = 0.524 ms`, `p95 = 1.952 ms`, Throughput = **`1,388.5 entries/sec`**.

---

### ACT-023 — Create `BENCHMARKS.md` with Published Results (`11,000` Cases) & Link from `README.md` (Issue `#63`)
- **Date**: `2026-10-22T00:05+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#63` (`docs/issue-63-benchmarks-md-page`)
- **Request**:
  - Add the published benchmark results into a dedicated `BENCHMARKS.md` page and add a link to it from `README.md`.
- **Changes**:
  - Created [`BENCHMARKS.md`](../BENCHMARKS.md) synthesizing all 4 published benchmark suites (`11,000` structured test cases with `10,000` product-independent cases, `1,000` core structured cases, `750`-case 8-track SLA suite, and `540`-item 5-track scientific benchmark), including accuracy, CPU latency percentiles (`min`/`p50`/`p90`/`p95`/`p99`), throughput (`8,601.0 entries/sec`), success/fail detection, calibration, and breakdowns across all 5 `decision_type`s and 18 `domain`s.
  - Updated [`README.md`](../README.md) with a top badge and prominent navigation links to [`BENCHMARKS.md`](../BENCHMARKS.md).
  - Added TDD test `test_benchmarks_md_exists_and_is_linked_from_readme` in [`tests/test_quality_validation_suite.py`](../tests/test_quality_validation_suite.py) and updated [`docs/RCA_REGISTER.md`](RCA_REGISTER.md).
- **Result**:
  - `pytest tests/test_quality_validation_suite.py -v`: **7 / 7 tests passed (100%)**.
  - Full test suite: **105 passed**, **95% coverage**.

---

### ACT-022 — Add 10,000 Product-Independent Test Cases (`11,000` Total) & Publish Evaluation Metrics (Issue `#58`)
- **Date**: `2026-10-21T23:57+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#58` (`feat/issue-58-10000-product-independent-test-cases`)
- **Request**:
  - Add 10,000 additional product-independent test cases to enrich the test case corpus (bringing the total to **11,000 structured test cases**) and publish evaluation metrics.
- **Changes**:
  - Created [`src/microllm/product_independent_cases.py`](../src/microllm/product_independent_cases.py) (`build_product_independent_test_entries`, `PRODUCT_INDEPENDENT_DOMAINS`) covering 10 universal product-independent domains (`healthcare_clinical_triage`, `legal_contract_clauses`, `logistics_supply_chain`, `hr_recruitment_policy`, `insurance_claims_adjudication`, `cybersecurity_soc_alerts`, `education_academic_grading`, `retail_ecommerce_moderation`, `public_sector_civic_routing`, `financial_risk_aml`).
  - Extended [`src/microllm/test_cases_1000.py`](../src/microllm/test_cases_1000.py) and [`src/microllm/quality_suite.py`](../src/microllm/quality_suite.py) to support up to `11,000` test cases (`1,000` core + `10,000` product-independent) and generated [`benchmarks/test_cases_11000_results.json`](../benchmarks/test_cases_11000_results.json).
  - Added TDD test `test_10000_product_independent_and_11000_total_test_cases` in [`tests/test_quality_validation_suite.py`](../tests/test_quality_validation_suite.py) and updated [`docs/TEST_PLAN.md`](TEST_PLAN.md) (§7) and [`docs/RCA_REGISTER.md`](RCA_REGISTER.md) (§5.2).
- **Result**:
  - **Overall Accuracy (`N=11,000`)**: **99.75%** (`10,973` Success / `27` Fail).
  - **Success / Fail Detection (`tau=0.45`)**: Mean Confidence on Success = **`0.9369`** vs Fail = **`0.2967`** (Gap = `+0.6402`), Selective Accuracy = **`100.00%`**, Failures Caught = **`100.0%`** (`27/27`).
  - **Latency (CPU)**: `p50 = 0.076 ms`, `p95 = 0.274 ms`, `p99 = 0.395 ms`, Throughput = **`8,601.0 entries/sec`** (`1.28 s` for `11,000` cases).
  - **Calibration**: Brier Score = **`0.0129`**, ECE = **`6.28%`**, Binary F1 = **`99.68%`**.

---

### ACT-021 — Comprehensive 104-Paper Academic Literature Review on Jev, RLCD, System-1 & Calibrated LLMs + `/v1/papers` Catalog (Issue `#42` / PR `#57`)
- **Date**: `2026-10-21T23:47+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#42` · PR `#57` (`docs/issue-42-jev-llm-academic-papers`)
- **Request**:
  - Find more academic papers on Jev (`System-One` decision models), `RLCD`, and related calibrated LLMs connected to the theme of MicroLLM, and integrate them into the project documentation and tooling.
- **Changes**:
  - Conducted a systematic arXiv literature discovery (`literature-search-arxiv`) across 218 preprints and curated **104 directly relevant academic papers** (`98` specialized 2025–2026 Jev / RLCD / System-1 / Typed Decision papers + `6` foundational calibration, conformal prediction, and LLM routing papers) organized across **6 thematic pillars**.
  - Created [`src/microllm/papers.py`](../src/microllm/papers.py) exporting `ACADEMIC_PAPERS_CATALOG`, `LITERATURE_PILLARS`, `list_academic_papers`, `get_academic_paper`, `get_literature_taxonomy`, and `format_papers_markdown`.
  - Exposed `GET /v1/papers` (with `pillar`, `q`, and `arxiv_id` filters) and the `microllm papers` CLI subcommand in [`src/microllm/server.py`](../src/microllm/server.py), plus `client.papers()` in `MicroJevClient` and `AsyncMicroJevClient` ([`src/microllm/client.py`](../src/microllm/client.py)).
  - Expanded [`docs/papers/README.md`](papers/README.md) from 3 papers to the full 104-paper 6-pillar bibliography with direct arXiv abstract and PDF URLs.
  - Created [`docs/LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) providing an in-depth scientific synthesis of the 6 research pillars and an architectural traceability matrix to MicroLLM modules.
  - Added TDD test suite [`tests/test_academic_papers_catalog.py`](../tests/test_academic_papers_catalog.py) and updated [`docs/RCA_REGISTER.md`](RCA_REGISTER.md) with `RCA-015` and roadmap entries `IDEA-022`..`IDEA-025`.
- **Result**:
  - `pytest --cov=microllm --cov-report=term-missing -v`: **100 / 100 unit tests passed (100%)**, **94%+ coverage**.

---

### ACT-020 — Build 500–1,000 Structured Test Case Entries (`message`, `decision_type`, `rules`, `expected_result`) & Publish Multi-Metric Evaluation Report (Issue `#52`)
- **Date**: `2026-10-21T23:31+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#52` (`feat/issue-52-1000-test-case-entries-metrics`)
- **Request**:
  - Build 500 to 1,000 structured test case entries where every entry contains `{message, decision_type, rules, expected_result}`, and publish comprehensive evaluation metrics: overall accuracy, latency percentiles (`p50`/`p90`/`p95`/`p99`), success vs. fail detection (`mean_confidence_on_success`, `mean_confidence_on_fail`, `selective_accuracy_above_tau`, `failures_caught_by_low_confidence_rate`), calibration (`brier_score`, `expected_calibration_error`, `binary_f1_score`), and breakdowns by `decision_type` and `domain`.
- **Changes**:
  - Created [`src/microllm/test_cases_1000.py`](../src/microllm/test_cases_1000.py) (`build_structured_test_entries`, `save_test_entries_jsonl`, `evaluate_structured_test_cases`, `format_published_metrics_markdown`) and re-exported them in [`src/microllm/quality_suite.py`](../src/microllm/quality_suite.py).
  - Generated [`benchmarks/test_cases_1000.jsonl`](../benchmarks/test_cases_1000.jsonl) (1,000 JSONL entries `TC-0001`..`TC-1000` across 5 decision types and 8 domains) and [`benchmarks/test_cases_1000_results.json`](../benchmarks/test_cases_1000_results.json).
  - Updated [`docs/TEST_PLAN.md`](TEST_PLAN.md) (§6), [`docs/RCA_REGISTER.md`](RCA_REGISTER.md) (§5.1), and [`tests/test_quality_validation_suite.py`](../tests/test_quality_validation_suite.py).
- **Result**:
  - **Overall Accuracy (`N=1,000`)**: **99.10%** (`991` Success / `9` Fail).
  - **Success / Fail Detection (`tau=0.45`)**: Mean Confidence on Success = **`0.9084`** vs Fail = **`0.2967`** (Gap = `+0.6117`), Selective Accuracy = **`100.00%`**, Failures Caught = **`100.0%`** (`9/9`).
  - **Latency (CPU)**: `p50 = 0.219 ms`, `p95 = 0.429 ms`, `p99 = 0.739 ms`, Throughput = **`3,734.6 entries/sec`**.
  - **Calibration**: Brier Score = **`0.0286`**, ECE = **`9.03%`**, Binary F1 = **`98.82%`**.

---

### ACT-019 — Scheduled GitHub Actions & Logs Automation (`/automation`) + Fix Mypy & Bandit SAST Failures (Issue `#48` / PR `#54`)
- **Date**: `2026-10-21T23:05+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#48` · PR `#54` (`fix/issue-48-ci-mypy-bandit`)
- **Request**:
  - Set up a daily scheduled automation (`/automation`) at 07:00 AM (`Europe/Paris`) that checks GitHub Actions runs and logs, proposes fixes when a failure occurs, and sends an email report to `jmrenouard@gmail.com`.
  - Apply the two fixes identified during the live test run (`Lint & Type Check` Mypy `union-attr` errors in `src/microllm/server.py` and `Security & Dependency Audit` Bandit `B615` revision pinning alerts in `src/microllm/backends/base.py`).
- **Changes**:
  - Created and verified the scheduled automation sidecar (`~/.gemini/config/sidecars/github-actions-log-check/sidecar.json`, **Daily GitHub Actions & Logs Check**) and sent the live test-run diagnostic email to `jmrenouard@gmail.com`.
  - Added `isinstance(a2, ChoiceAnswer)` and `isinstance(a2_rev, ChoiceAnswer)` type narrowing guards in [`src/microllm/server.py`](../src/microllm/server.py) (`RCA-013`).
  - Added `revision: str = "main"` parameter to `create_hf_zero_shot_scorer` and `create_onnx_nli_scorer` in [`src/microllm/backends/base.py`](../src/microllm/backends/base.py) and passed `revision=revision` to all `from_pretrained` calls (`RCA-014`).
  - Added regression test `test_hf_and_onnx_scorers_pin_revision_and_server_narrows_choice_answer` in [`tests/test_python_github_actions.py`](../tests/test_python_github_actions.py).
  - Updated [`docs/ACTION_LOGBOOK.md`](ACTION_LOGBOOK.md) and [`docs/RCA_REGISTER.md`](RCA_REGISTER.md).
- **Result**:
  - `pytest --cov=microllm --cov-report=term-missing -v`: **95 / 95 unit tests passed (100%)**, **94% coverage**.

---

### ACT-018 — Complete Test Plan (`docs/TEST_PLAN.md`) & 750–1,000 Test Case Quality & Performance Validation Suite (Issue `#32`)
- **Date**: `2026-10-21T22:52+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#32` (`test/issue-32-quality-test-plan-750-cases`)
- **Request**:
  - Write a complete test plan to check performance and result quality, build 500 to 1,000 test cases to validate project quality, and add this to the RCA register (`docs/RCA_REGISTER.md`).
- **Changes**:
  - Created [`docs/TEST_PLAN.md`](TEST_PLAN.md) specifying the 8-track validation architecture (750 default test cases, scalable from 500 to 1,000 test cases), acceptance criteria, quality gates (`G1`–`G8`), and performance SLAs (`p50 <= 10 ms`, `p95 <= 25 ms`, `>= 100 decisions/sec`, INT8 quantization MSE `<= 1e-3`).
  - Created [`src/microllm/quality_suite.py`](../src/microllm/quality_suite.py) (`build_quality_test_cases`, `run_quality_validation_suite`, `format_quality_report_markdown`, and CLI entry point `python -m microllm.quality_suite`).
  - Created TDD test suite [`tests/test_quality_validation_suite.py`](../tests/test_quality_validation_suite.py) validating `docs/TEST_PLAN.md`, 750-case and 1,000-case generation, full 8-track execution (`all_gates_passed == True`), and governance register entries.
  - Updated [`docs/RCA_REGISTER.md`](RCA_REGISTER.md) with `RCA-012`, `APP-007`, `IDEA-021`, and Section 5 (*Complete Performance & Result Quality Test Plan*).
- **Result**:
  - `pytest tests/test_quality_validation_suite.py -v`: **4 / 4 tests passed (100%)**.
  - **750 / 750 test cases** (over 1,250 individual engine decisions) and **1,000 / 1,000 scaled test cases** pass **8 / 8 automated Quality & Performance SLA Gates** (`98.1%` JevBench accuracy, `100%` DecisionPivot pair accuracy, `0.000000` AnyJev $L_0$ order-flip rate, `100%` Guardrail F1, `3.67%` calibrated ECE, `100%` `jevcompat` contract/boundary pass rate with `0` HTTP `5xx`, `100%` EN/FR/ES/DE + XAI pass rate, and CPU `p50 < 2.0 ms`).

---

### ACT-017 — Merge `docs/ROADMAP_20_IDEAS.md` into `docs/RCA_REGISTER.md` (Issue `#43`)
- **Date**: `2026-10-21T22:29+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#43` (`docs/issue-43-merge-roadmap-rca-register`)
- **Request**:
  - Merge `docs/ROADMAP_20_IDEAS.md` with the RCA Register (`docs/RCA_REGISTER.md`) so that all 20 high-impact architectural & scientific improvements (`IDEA-001` through `IDEA-020`) reside in a single canonical governance register.
- **Changes**:
  - Merged the Executive Summary Matrix (`§4.1`) and Detailed Specifications across all 6 architectural pillars (`§4.2`, `IDEA-001` through `IDEA-020`) from `docs/ROADMAP_20_IDEAS.md` into Section 4 (`4. Unimplemented Ideas & Future Roadmap (IDEA)`) of [`docs/RCA_REGISTER.md`](RCA_REGISTER.md).
  - Removed the standalone `docs/ROADMAP_20_IDEAS.md` file and logged `IMP-005` (why maintaining separate roadmap files outside `docs/RCA_REGISTER.md` is a dead-end) and `RCA-011` (fixing Windows CP1252 `0x97` byte corruption in governance Markdown files).
  - Consolidated the 20 high-impact improvements implementation (`src/microllm/*`, [`tests/test_20_high_impact_ideas.py`](../tests/test_20_high_impact_ideas.py)) and updated [`tests/test_project_governance_rules.py`](../tests/test_project_governance_rules.py) to assert that `IDEA-001`..`IDEA-020` and all 6 pillars are in `docs/RCA_REGISTER.md` and that `docs/ROADMAP_20_IDEAS.md` no longer exists as a separate file.
- **Result**:
  - `pytest tests/test_project_governance_rules.py tests/test_20_high_impact_ideas.py -v`: **15 / 15 tests passed (100%)**.
  - Full `pytest --cov=microllm` suite passing at **100%** with **94% coverage**.

---

### ACT-016 — Add "Why MicroLLM?" Section & Top-of-README 1-Command Benchmark Table (Issue `#44` / PR `#45`)
- **Date**: `2026-10-21T22:24+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#44` · PR `#45` (`docs/issue-44-readme-why-and-benchmark-table`, merged into `main`, commit `8af1bde`)
- **Request**:
  - Add a clear "Why MicroLLM?" section and surface the 1-command reproducible benchmark table (`microllm benchmark --all`) at the top of `README.md`.
- **Changes**:
  - Updated [`README.md`](../README.md) with the top-level value proposition and the 5-track benchmark results table (`540` items).
- **Result**:
  - PR `#45` merged into `main`; Issue `#44` closed automatically.

---

### ACT-015 — Add Obsidian Craft SVG Logo & Hero Header Banner Across Docs and Decision Studio (Issue `#40` / PR `#41`)
- **Date**: `2026-10-21T22:22+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#40` · PR `#41` (`feat/issue-40-brand-logo-header-banner`, merged into `main`, commit `bf68e24`)
- **Request**:
  - Design and integrate a high-craft SVG brand logo and header banner across `README.md`, `docs/README.md`, `docs/index.html`, and the interactive Decision Studio (`src/microllm/static/index.html`).
- **Changes**:
  - Created [`docs/assets/logo.svg`](assets/logo.svg) and [`docs/assets/header-banner.svg`](assets/header-banner.svg) and integrated them into [`src/microllm/static/index.html`](../src/microllm/static/index.html), [`docs/index.html`](index.html), and [`README.md`](../README.md).
- **Result**:
  - PR `#41` merged into `main`; Issue `#40` closed automatically.

---

### ACT-014 — 540-Item 5-Track Reproducible Evaluation Corpus & 1-Command Benchmark Runner (Issue `#37` / PR `#38`)
- **Date**: `2026-10-21T22:21+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#37` · PR `#38` (`feat/issue-37-benchmark-corpus-540`, merged into `main`, commit `1e772fd`)
- **Request**:
  - Build a 540-item 5-track reproducible evaluation corpus (`JevBench-Core`, `DecisionPivot`, `AnyJev-Order`, `Calibration-ECE`, `HighCard-50`) and a 1-command benchmark runner (`microllm benchmark`).
- **Changes**:
  - Added [`src/microllm/benchmarks.py`](../src/microllm/benchmarks.py), [`benchmarks/microllm_eval_corpus.jsonl`](../benchmarks/microllm_eval_corpus.jsonl), [`benchmarks/latest_results.json`](../benchmarks/latest_results.json), and [`tests/test_benchmark_corpus.py`](../tests/test_benchmark_corpus.py).
- **Result**:
  - `JevBench-Core` **98.1%**, `DecisionPivot` **100.0%**, `AnyJev-Order` **0.000000 Flip Rate**, PR `#38` merged into `main`.

---

### ACT-013B — Fix Python 3.10 `tomllib` Compatibility & Native HTML/CSS Math Rendering (Issues `#33`, `#35` / PRs `#34`, `#36`)
- **Date**: `2026-10-21T22:19+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#33` (PR `#34`, commit `360efda`) & Issue `#35` (PR `#36`, commit `b6475b0`)
- **Request**:
  - Support Python 3.10 in `tests/test_packaging_ci.py` via `tomli` fallback (`RCA-010`) and replace raw LaTeX delimiters in documentation with clean native HTML/CSS and Unicode math.
- **Changes**:
  - Updated [`tests/test_packaging_ci.py`](../tests/test_packaging_ci.py), [`pyproject.toml`](../pyproject.toml), [`docs/README.md`](README.md), and [`docs/index.html`](index.html).
- **Result**:
  - PR `#34` and PR `#36` merged into `main`; all CI checks green across Python 3.10, 3.11, and 3.12.

---

### ACT-012 — 20 High-Impact Architectural & Engineering Improvements + English Governance Rules (Issue `#10`)
- **Date**: `2026-10-21T22:15+02:00`
- **Status**: `COMPLETED`
- **Issue / Branch**: Issue `#10` (`feat/issue-10-20-high-impact-improvements`)
- **Request**:
  1. Ensure all project rules in `AGENTS.md` and `.agents/rules/` (`git-workflow.md`, `github-workflow.md`, `project-governance.md`) are written in English.
  2. Identify, document, and implement **20 high-impact, differentiating improvements** (`IDEA-001` through `IDEA-020`) across calibration, AnyJev $L_0$, StartLux cascade, OpenJev-RLCD², runtime performance, SDK ergonomics, safety guardrails, counterfactual XAI, and observability.
- **Changes**:
  - **English Rules**: Verified and standardized [`AGENTS.md`](../AGENTS.md), [`.agents/rules/git-workflow.md`](../.agents/rules/git-workflow.md), [`.agents/rules/github-workflow.md`](../.agents/rules/github-workflow.md), and [`.agents/rules/project-governance.md`](../.agents/rules/project-governance.md) in English.
  - **20-Idea Architecture Roadmap**: Merged all 20 specifications (`IDEA-001`..`IDEA-020`) into [`docs/RCA_REGISTER.md`](RCA_REGISTER.md) (§4).
  - **Core Engine & SDK Implementations**:
    - `IDEA-001` (`compute_dynamic_tau`), `IDEA-005` (`answer_mass_gate_passed`), and `IDEA-006` (`OnlinePriorTracker`) in [`src/microllm/backends/prefill_llm.py`](../src/microllm/backends/prefill_llm.py).
    - `IDEA-002` (`compute_adaptive_alpha`) in [`src/microllm/backends/cascade.py`](../src/microllm/backends/cascade.py).
    - `IDEA-003` (`fit_mondrian`, `predict_set_mondrian`), `IDEA-004` (`VectorScalingCalibrator`), `IDEA-010` (`tune_cascade_pareto`), and `IDEA-011` (`save_state` / `load_state`) in [`src/microllm/calibration.py`](../src/microllm/calibration.py).
    - `IDEA-007` (`seed_diversity` in `run_hierarchical_tournament`) and `IDEA-008` (`earth_movers_distance_1d`, `unimodal_ordinal_smooth`) in [`src/microllm/confidence.py`](../src/microllm/confidence.py).
    - `IDEA-009` (`RLCDReplayBuffer`) in [`src/microllm/rlcd.py`](../src/microllm/rlcd.py).
    - `IDEA-012` (`BLAKE2b` LRU embedding cache `_embed_text_cached` cutting benchmark time by 50%+) and `IDEA-019` (`explain_counterfactual`) in [`src/microllm/backends/base.py`](../src/microllm/backends/base.py).
    - `IDEA-014` (`sha256_checksum` and `verify_profile`) in [`src/microllm/registry.py`](../src/microllm/registry.py).
    - `IDEA-015` (`AsyncMicroJevClient`), `IDEA-016` (`Choice.from_enum`, `Multi.from_enum`), and `IDEA-017` (`@system_one_router`, `client.guardrail()`, `client.batch()`) in [`src/microllm/client.py`](../src/microllm/client.py).
    - `IDEA-013` (`p50_latency_ms` / `p95_latency_ms` in `POST /v1/batch`), `IDEA-018` (`custom_rubrics` in `POST /v1/guardrail`), and `IDEA-020` (`GET /v1/metrics`) in [`src/microllm/server.py`](../src/microllm/server.py).
  - **Unit Tests**: Added [`tests/test_20_high_impact_ideas.py`](../tests/test_20_high_impact_ideas.py) (12 comprehensive tests covering all 20 improvements).
- **Result**:
  - `pytest --cov=microllm --cov-report=term-missing -v`: **90 / 90 unit tests passed (100%) in 13.66s**, **94% total coverage**.

---

### ACT-011 — Rename and Translate Action Logbook & RCA Register to English (Issue `#29` / PR `#31`)
- **Date**: `2026-10-21T22:10+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#29` · PR `#31` (`docs/issue-29-english-logbook-rca-register`, merged into `main`, commit `4d260f0`)
- **Request**:
  - Translate both the Action Logbook (`JOURNAL_DE_BORD.md`) and the RCA Register (`REGISTRE_RCA.md`) into English, and rename their filenames to English (`docs/ACTION_LOGBOOK.md` and `docs/RCA_REGISTER.md`).
- **Changes**:
  - Renamed and translated `docs/JOURNAL_DE_BORD.md` $\rightarrow$ [`docs/ACTION_LOGBOOK.md`](ACTION_LOGBOOK.md).
  - Renamed and translated `docs/REGISTRE_RCA.md` $\rightarrow$ [`docs/RCA_REGISTER.md`](RCA_REGISTER.md).
  - Updated [`AGENTS.md`](../AGENTS.md) and [`.agents/rules/project-governance.md`](../.agents/rules/project-governance.md) to enforce English filenames (`docs/ACTION_LOGBOOK.md` and `docs/RCA_REGISTER.md`) and English section headers (`Ongoing Actions`, `Completed Actions`, `Request`, `Changes`, `Result`).
  - Updated [`tests/test_project_governance_rules.py`](../tests/test_project_governance_rules.py) to validate `docs/ACTION_LOGBOOK.md` and `docs/RCA_REGISTER.md` and their English structure.
- **Result**:
  - `pytest tests/test_project_governance_rules.py -v`: **3 / 3 tests passed (100%)**.
  - Full test suite passing at 100%, PR `#31` merged into `main`, and Issue `#29` closed.

---

### ACT-010 — Add PyPI / TestPyPI Publishing GitHub Actions & Packaging Metadata (Issue `#3` / PR `#23`)
- **Date**: `2026-10-21T22:05+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#3` · PR `#23` (`ci/issue-3-pypi-github-actions`, merged into `main`, commit `428c689`)
- **Request**:
  - Add GitHub Actions workflows to automatically publish the package to PyPI (and TestPyPI) and register the systematic Git/GitHub workflow rule (Issue $\rightarrow$ Branch $\rightarrow$ PR $\rightarrow$ Merge).
- **Changes**:
  - Created [`.github/workflows/publish-pypi.yml`](../.github/workflows/publish-pypi.yml):
    - Triggered on GitHub Release publication (`release: types: [published]`), `v*` tags (`push: tags: ["v*"]`), and manual dispatch (`workflow_dispatch` with `pypi` or `testpypi` target selection).
    - `test` job across Python `3.10`, `3.11`, `3.12` matrix (`pytest --cov=microllm --cov-report=term-missing`).
    - `build` job (`python -m build` producing `sdist` and `bdist_wheel` + `twine check --strict dist/*` validation).
    - `publish-pypi` and `publish-testpypi` jobs using official PyPA `pypa/gh-action-pypi-publish@release/v1` with **OIDC Trusted Publishing** (`id-token: write`).
  - Created [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) validating unit tests and package build on every `push` / `pull_request` to `main`.
  - Enriched [`pyproject.toml`](../pyproject.toml) with PyPI `keywords`, `classifiers`, and `[project.urls]` (`Homepage`, `Repository`, `Issues`).
  - Added [`tests/test_packaging_ci.py`](../tests/test_packaging_ci.py) verifying PyPI metadata, OIDC publishing workflow, and Git/GitHub governance rules.
- **Result**:
  - `pytest tests/test_packaging_ci.py -v`: **3 / 3 tests passed (100%)**.
  - Build `python -m build` + `twine check --strict dist/*`: **100% valid** (`microllm-0.2.0.tar.gz` and `microllm-0.2.0-py3-none-any.whl`).
  - PR `#23` opened, verified, and merged into `main`; Issue `#3` closed automatically.

---

### ACT-009 — Implement Full Python GitHub Actions CI/CD Suite (Issue `#4` / PR `#12`)
- **Date**: `2026-10-21T22:02+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#4` · PR `#12` (`ci/issue-4-python-github-actions`, merged into `main`, commit `0521ef1`)
- **Request**:
  - Implement all essential GitHub Actions workflows for a Python project (`ci.yml`, `unit-tests.yml`, `lint.yml`, `security.yml`, `publish-pypi.yml`, `python-package-conda.yml`, `release.yml`, and `.github/dependabot.yml`).
- **Changes**:
  - Created [`.github/workflows/lint.yml`](../.github/workflows/lint.yml) (Ruff, Flake8, and Mypy on Python 3.12).
  - Created [`.github/workflows/security.yml`](../.github/workflows/security.yml) (Bandit SAST static analysis + `pip-audit` dependency vulnerability scanning).
  - Created [`.github/workflows/release.yml`](../.github/workflows/release.yml) (`sdist`/`wheel` build, `twine check --strict` validation, automated GitHub Release on `v*` tags via `softprops/action-gh-release@v2`).
  - Created [`.github/dependabot.yml`](../.github/dependabot.yml) (weekly automated dependency updates for `pip` and `github-actions` ecosystems).
  - Consolidated [`.github/workflows/ci.yml`](../.github/workflows/ci.yml), [`.github/workflows/unit-tests.yml`](../.github/workflows/unit-tests.yml), [`.github/workflows/publish-pypi.yml`](../.github/workflows/publish-pypi.yml), [`.github/workflows/python-package-conda.yml`](../.github/workflows/python-package-conda.yml), and [`environment.yml`](../environment.yml).
  - Added TDD test suite [`tests/test_python_github_actions.py`](../tests/test_python_github_actions.py) and made [`tests/test_cascade_and_sdk.py`](../tests/test_cascade_and_sdk.py) resilient in clean CI checkouts where gitignored PDFs are absent.
- **Result**:
  - `pytest tests/test_python_github_actions.py -v`: **5 / 5 tests passed (100%)**.
  - Full suite: **78+ tests passed (100%), 94% total code coverage**.
  - PR `#12` merged into `main`; Issue `#4` closed automatically.

---

### ACT-008 — Add Automated Unit Tests GitHub Actions Workflow (Issue `#2` / PR `#8`)
- **Date**: `2026-10-21T21:58+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#2` · PR `#8` (`ci/issue-2-unit-tests-github-actions`, merged into `main`, commit `aa76d8c`)
- **Request**:
  - Add GitHub Actions workflows to automatically run unit tests on every push and Pull Request.
- **Changes**:
  - Created [`.github/workflows/unit-tests.yml`](../.github/workflows/unit-tests.yml) (Python `3.10`, `3.11`, `3.12` matrix on `ubuntu-latest`, triggered on `push`, `pull_request`, and `workflow_dispatch`, running `pytest --cov=microllm --cov-report=term-missing --cov-report=xml --junitxml=pytest-results.xml -v` and uploading `coverage.xml` and `pytest-results.xml` artifacts).
  - Added unit test suite [`tests/test_unit_tests_workflow.py`](../tests/test_unit_tests_workflow.py) validating the CI workflow and GitHub governance rules.
  - Switched `origin` remote URL to SSH (`git@github.com:jmrenouard/MicroLLM.git`) to allow pushing workflow files under `.github/workflows/`.
- **Result**:
  - `pytest tests/test_unit_tests_workflow.py -v`: **2 / 2 tests passed (100%)**.
  - Full suite: **70 passed in 27.14s**.
  - PR `#8` opened, verified, and merged into `main`; Issue `#2` closed automatically.

---

### ACT-007 — Register Project Governance Rules: Action Logbook & RCA Register (Issue `#5` / PR `#11`)
- **Date**: `2026-10-21T22:04+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#5` · PR `#11` (`docs/issue-5-journal-de-bord-registre-rca`, merged into `main`, commit `4bca57f`)
- **Request**:
  1. Add a mandatory project rule requiring an **Ongoing / Completed Action Logbook** (`docs/ACTION_LOGBOOK.md`).
  2. Add a mandatory **RCA Register of Errors, Dead-Ends, Approximations, and Unimplemented Ideas** (`docs/RCA_REGISTER.md`).
  3. Mandate updating both registers **on every request, every change, and every result**.
- **Changes**:
  - Updated [`AGENTS.md`](../AGENTS.md) (added §2 *Action Logbook* and §3 *RCA Register*).
  - Created always-on Antigravity rule [`.agents/rules/project-governance.md`](../.agents/rules/project-governance.md) (`trigger: always_on`).
  - Created and populated the Action Logbook (`ACT-001` through `ACT-007`) and the RCA Register (`RCA-001`..`RCA-008`, `IMP-001`..`IMP-004`, `APP-001`..`APP-006`, `IDEA-001`..`IDEA-006`).
  - Added unit test suite [`tests/test_project_governance_rules.py`](../tests/test_project_governance_rules.py) verifying the normative content of the rules and both registers.
- **Result**:
  - `pytest tests/test_project_governance_rules.py -v`: **3 / 3 tests passed (100%)**.
  - Full `pytest` suite: **100% passing**, global coverage maintained at **94%**.

---

### ACT-005 — Register Systematic Git / GitHub Workflow Rule (Issue `#1` / PR `#6`)
- **Date**: `2026-10-21T21:50+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#1` · PR `#6` (`chore/issue-1-github-workflow-rules`, merged into `main`)
- **Request**:
  - Register a mandatory project rule requiring a GitHub Issue, dedicated branch, Pull Request, and Merge for every user request.
- **Changes**:
  - Created [`AGENTS.md`](../AGENTS.md) at the repository root detailing the mandatory lifecycle (`gh issue create` $\rightarrow$ `git checkout -b` / `git worktree` $\rightarrow$ `pytest` $\rightarrow$ `git push` $\rightarrow$ `gh pr create` $\rightarrow$ `gh pr merge`).
  - Created Antigravity rules [`.agents/rules/git-workflow.md`](../.agents/rules/git-workflow.md) and [`.agents/rules/github-workflow.md`](../.agents/rules/github-workflow.md) with `trigger: always_on`.
  - Added verification test `test_project_git_workflow_rules_registered` in [`tests/test_packaging_ci.py`](../tests/test_packaging_ci.py).
- **Result**:
  - PR `#6` opened, verified, and merged into `main`; Issue `#1` closed automatically.

---

### ACT-004 — Add Conda & Jekyll GitHub Actions Workflows (Issue `#7` / PR `#9`, Commits `8cb51d4`, `9b00eea`)
- **Date**: `2026-10-21T21:14+02:00`
- **Status**: `COMPLETED`
- **Issue / PR**: Issue `#7` · PR `#9` (`fix/issue-7-conda-ci-workflow`, merged into `main`, commit `78a8d04`)
- **Request**:
  - Add GitHub Actions CI workflows for Conda package validation (`.github/workflows/python-package-conda.yml`) and Jekyll documentation publishing (`.github/workflows/jekyll-docker.yml`).
- **Changes**:
  - Created [`.github/workflows/python-package-conda.yml`](../.github/workflows/python-package-conda.yml) and [`.github/workflows/jekyll-docker.yml`](../.github/workflows/jekyll-docker.yml).
  - Added [`environment.yml`](../environment.yml) and validation tests [`tests/test_conda_workflow.py`](../tests/test_conda_workflow.py).
- **Result**:
  - Syntactic and contractual validation of the Conda workflow via unit tests (`tests/test_conda_workflow.py`).

---

### ACT-003 — Comprehensive Markdown Technical Documentation & README Enrichment (Commits `89850ff`, `41c59e5`)
- **Date**: `2026-10-21T21:00+02:00`
- **Status**: `COMPLETED`
- **Issue / Commit**: Commits `89850ff1` & `41c59e51` on `main`
- **Request**:
  - Mirror the interactive HTML technical documentation (`docs/index.html`) into a GitHub-native Markdown document (`docs/README.md`) and enrich the root `README.md` with badges, the 4 primitives (`noul`, `choice`, `score`, `multi`), the Guardrail pack, and the `microllm pull` CLI.
- **Changes**:
  - Created [`docs/README.md`](README.md) (362 lines: Mermaid architecture diagram, LaTeX mathematical equations, `32/32 MUST` compliance matrix, Gear 1 / Gear 2 / Cascade / RLCD² details, Python SDK, and HTTP endpoint reference).
  - Updated [`README.md`](../README.md) with System-1 vs System-2 comparison table and Quickstart guide.
- **Result**:
  - 100% synchronized documentation across `README.md`, `docs/README.md`, and `docs/index.html`.

---

### ACT-002 — Fix Order Invariance (AnyJev $L_0$), Policy Negation (`noul`), and Token Attribution
- **Date**: `2026-10-21T20:58+02:00`
- **Status**: `COMPLETED`
- **Request**:
  - Achieve `0.000000` Order-Flip Rate on `AnyJev-Order` even under early stopping ($\mu(\bar{z}) \ge \tau$), fix inversions on policy rules containing exemption clauses (`unless`, `exempt`) in `evaluate_noul`, and stabilize `explain_state_tokens` on isolated tokens.
- **Changes**:
  - [`src/microllm/backends/prefill_llm.py`](../src/microllm/backends/prefill_llm.py): canonical key sorting for `choice` prior to the Van der Corput sequence + explicit handling of `exemption_markers` and `policy_false_markers`.
  - [`src/microllm/backends/fast_encoder.py`](../src/microllm/backends/fast_encoder.py): aligned policy check logic (`is_policy_check`, `has_exemption`, `has_negation`) in `evaluate_noul`.
  - [`src/microllm/backends/base.py`](../src/microllm/backends/base.py): direct token attribution via embedding dot-product + lexical overlap + concept boost in `explain_state_tokens`.
- **Result**:
  - `JevBench-Core` accuracy: **98.1%** (`choice` 100%, `score` 100%, `multi` 100%, `noul` 92%).
  - `DecisionPivot`: **100.0%** contrastive pair accuracy (40/40 pairs).
  - `AnyJev-Order`: **0.000000 Flip Rate** and **0.000000 Max Drift** with **78.3% rotations saved**.

---

### ACT-001 — Initial Design & Implementation of the Local MicroLLM Decision Engine (Commit `ca776c0`)
- **Date**: `2026-10-21T20:56+02:00`
- **Status**: `COMPLETED`
- **Issue / Commit**: Initial commit `ca776c0d` (`feat: initial release of MicroLLM local System-One decision engine`)
- **Request**:
  - Build a local System-1 probabilistic decision engine (<15 ms on CPU), 100% compliant with all 32 `MUST` requirements of the `jevcompat 0.1` specification, synthesizing the 3 research papers *OpenJev-RLCD* (`arXiv:2609.38850`), *AnyJev* (`arXiv:2610.00831`), and *StartLux-Decision* (`arXiv:2610.17309`).
- **Changes**:
  - Full implementation of `src/microllm/` (`schemas.py`, `confidence.py`, `calibration.py`, `rlcd.py`, `registry.py`, `client.py`, `server.py`, `backends/base.py`, `backends/fast_encoder.py`, `backends/prefill_llm.py`, `backends/cascade.py`).
  - Interactive web interfaces: **Decision Studio** (`src/microllm/static/index.html`) and interactive documentation (`docs/index.html`).
  - Contractual and scientific test suite (`tests/test_jevcompat_contract.py`, `tests/test_anyjev_rotations.py`, `tests/test_calibration_conformal.py`, `tests/test_rlcd_trainer.py`, `tests/test_cascade_and_sdk.py`).
- **Result**:
  - **32 / 32 `jevcompat 0.1` `MUST` requirements validated**, CPU p50 latency of `1.5 ms` (Gear 1), 60 initial unit tests passing at 100%.
