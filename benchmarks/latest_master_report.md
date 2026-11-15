# MicroLLM Unified Master Benchmark Report

- **Timestamp (UTC)**: `2026-11-15T03:10:12.070727+00:00`
- **Environment**: `Windows-11-10.0.26300-SP0` · Python `3.14.10` · NumPy `2.5.4` · `20` logical cores (`CPU Only (0% GPU / 0% NPU)`)
- **Suites Executed**: **12 / 12 PASS**
- **Total Evaluated Items / Decisions**: **`128,713` items** (`131,487` decisions)
- **Total Wall-Clock Duration**: **`66.73 s`** (`66,729.7 ms`)

| Suite ID | Benchmark Suite | Items | Primary Accuracy | Secondary Metrics / Gates | CPU `p50` / `p95` (ms) | Wall-Clock (ms) | Status | Artifact |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| `eval_540` | 5-Track Scientific Paper Benchmark (540 Items) | 540 | **97.50%** | Pivot 100.0% · L0 Flip 0.000000 · Cal ECE 6.15% | `0.114` / `0.676` | `420.9` | **PASS** | `benchmarks/latest_results.json` |
| `quality_750` | 8-Track Quality & Performance SLA Suite (750 Cases) | 750 | **97.50%** | Gates: 8/8 PASS · Throughput 2153.3 dec/s | `0.043` / `0.257` | `665.5` | **PASS** | `benchmarks/quality_validation_750_results.json` |
| `quality_1000` | 8-Track Quality & Performance SLA Suite (1000 Cases) | 1,000 | **97.50%** | Gates: 8/8 PASS · Throughput 2475.7 dec/s | `0.027` / `0.245` | `866.0` | **PASS** | `benchmarks/quality_validation_1000_results.json` |
| `test_cases_1000` | Structured Enterprise & Product-Independent Test Cases (1,000 Entries) | 1,000 | **99.10%** | Selective Acc: 100.00% · Throughput: 9577.2 entries/s | `0.084` / `0.165` | `104.4` | **PASS** | `benchmarks/test_cases_1000_results.json` |
| `test_cases_11000` | Structured Enterprise & Product-Independent Test Cases (11,000 Entries) | 11,000 | **99.75%** | Selective Acc: 100.00% · Throughput: 38917.0 entries/s | `0.013` / `0.078` | `282.6` | **PASS** | `benchmarks/test_cases_11000_results.json` |
| `web_external_curated` | 16 External Public Datasets — Curated CI Baseline (162 Entries) | 162 | **100.00%** | Selective Acc: 100.00% (100.0% cov) · L0 Flip: 0.000000 | `0.197` / `0.592` | `47.3` | **PASS** | `benchmarks/web_benchmarks_results.json` |
| `hf_jev_models_ecosystem` | 12 Hugging Face JEV Model Datasets — Curated & Scaled Corpus (120 Curated / 480 Corpus Entries) | 120 | **100.00%** | Selective Acc: 100.00% (100.0% cov) · L0 Flip: 0.000000 · 25 HF Models | `0.330` / `0.745` | `47.4` | **PASS** | `benchmarks/hf_jev_models_benchmarks_results.json` |
| `web_external_full_splits` | 16 External Public Datasets — Full Unsampled Splits (114,018 Rows) | 114,018 | **66.51%** | Selective Acc: 87.54% (76.0% cov) · L0 Flip: 0.000000 | `0.293` / `1.656` | `62,975.7` | **PASS** | `benchmarks/web_benchmarks_full_results.json` |
| `research_hypotheses` | 12 Research Challenge Hypotheses (VAL-HYP-001..VAL-HYP-012) | 12 | **100.00%** | 12/12 Hypotheses Verified | `3.829` / `5.744` | `46.0` | **PASS** | `src/microllm/research_challenge.py` |
| `rca_non_regression` | 15 Sequential RCA Non-Regression Verifications (RCA-001..RCA-015) | 15 | **100.00%** | 15/15 RCAs Non-Regression PASS | `0.136` / `9.508` | `795.9` | **PASS** | `src/microllm/rca_runner.py` |
| `multilingual_eval` | Quadrilingual Evaluation Runner (EN, FR, ES, PT across Fast, Neural, Auto) | 48 | **100.00%** | Fast: 100.0% · Neural MiniLM: 100.0% · Cascade Rate: 10.4% | `0.169` / `0.325` | `84.8` | **PASS** | `benchmarks/multilingual_benchmarks_results.json` |
| `pareto_all_engines` | Comparative Multi-Engine Pareto & Scaling Benchmark (All Engines) | 48 | **100.00%** | Cascade p50 0.018ms · Speedup 1.5x · Esc 10.4% · Optimal: True | `0.018` / `0.055` | `59.1` | **PASS** | `benchmarks/llm_engines_comparative_benchmark.json` |
