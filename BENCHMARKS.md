# MicroLLM — Published Benchmarks, Quality & Performance Results

[![Benchmark 11,000](https://img.shields.io/badge/Structured_Cases-11%2C000_Entries_%7C_99.75%25_Acc-10b981)](benchmarks/test_cases_11000_results.json)
[![CPU Latency](https://img.shields.io/badge/CPU_Latency-0.076ms_p50_%7C_8%2C601_dec%2Fs-0ea5e9)](benchmarks/test_cases_11000_results.json)
[![Quality Suite 750](https://img.shields.io/badge/8--Track_SLA_Gates-8%2F8_PASS_(750_Cases)-3b82f6)](benchmarks/quality_validation_750_results.json)
[![JevCompat 0.1](https://img.shields.io/badge/JevCompat_0.1-32%2F32_MUST-10b981)](docs/TEST_PLAN.md)

> **Quick Navigation**:
> - Back to [**Main Project `README.md`**](README.md)
> - Read the [**Failure Mode Analysis & Worst-Case Regimes (`docs/FAILURE_MODES_AND_WORST_CASES.md`)**](docs/FAILURE_MODES_AND_WORST_CASES.md)
> - Read the [**Complete Test Plan (`docs/TEST_PLAN.md`)**](docs/TEST_PLAN.md)
> - Inspect the [**RCA & Governance Register (`docs/RCA_REGISTER.md`)**](docs/RCA_REGISTER.md)
> - Raw Machine-Readable Telemetry Artifacts:
>   - [`benchmarks/all_benchmarks_master_report.json`](benchmarks/all_benchmarks_master_report.json) (**Unified Master Benchmark Report**: all `9` suites, `128,497` items / `130,839` decisions, `9/9 PASS`)
>   - [`benchmarks/test_cases_11000_results.json`](benchmarks/test_cases_11000_results.json) (`11,000` structured test case entries: `1,000` core + `10,000` product-independent)
>   - [`benchmarks/test_cases_1000.jsonl`](benchmarks/test_cases_1000.jsonl) & [`benchmarks/test_cases_1000_results.json`](benchmarks/test_cases_1000_results.json) (`1,000` core structured test case entries)
>   - [`benchmarks/quality_validation_750_results.json`](benchmarks/quality_validation_750_results.json) & [`benchmarks/quality_validation_1000_results.json`](benchmarks/quality_validation_1000_results.json) (`750`-case and `1,000`-case 8-track Quality & Performance SLA validation suites)
>   - [`benchmarks/microllm_eval_corpus.jsonl`](benchmarks/microllm_eval_corpus.jsonl) & [`benchmarks/latest_results.json`](benchmarks/latest_results.json) (`540`-item 5-track scientific benchmark)
>   - [`benchmarks/web_benchmarks_full_results.json`](benchmarks/web_benchmarks_full_results.json), [`benchmarks/web_benchmarks_results.json`](benchmarks/web_benchmarks_results.json) & [`benchmarks/external_datasets_corpus.jsonl`](benchmarks/external_datasets_corpus.jsonl) (`16` External Public Datasets — **`114,018` Full Unsampled Official Split Rows** + `162` base / `640` scaled curated entries)

---

## 1. Executive Summary of Published Benchmarks & Resource Profile

**MicroLLM** is evaluated across **nine complementary, 100% reproducible benchmark suites (`128,497` total items / `130,839` decisions)** executed strictly on local CPU (`0% GPU / NPU` acceleration) via the unified automation orchestrator (`python scripts/run_all_benchmarks.py` or `microllm benchmark --all-suites`):

| Benchmark Suite | Scale (Entries / Decisions) | Overall Accuracy / Pass Rate | Selective Accuracy ($\tau = 0.45$) | CPU Latency (`p50` / `p95`) | Throughput (CPU) | Total Wall-Clock Time | Artifact / Runner |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Full Unsampled 16 External Public Datasets (`--full-splits`)** | **114,018 Rows** (`16/16` official splits) | **66.82%** (`76,186` / `114,018` zero-shot CPU, up from `65.92%`) | **87.49%** (`76.37%` cov · `0.000000` Flip on `1,867`) | **`0.527 ms`** / `2.053 ms` | **`1,187.8 entries/sec`** | **`95.99 s`** | [`benchmarks/web_benchmarks_full_results.json`](benchmarks/web_benchmarks_full_results.json) |
| **2. Full Structured Test Suite (`1,000` Core + `10,000` Product-Independent)** | **11,000 Entries** | **99.75%** (`10,973` / `11,000`) | **100.00%** (`100%` fails caught) | **`0.076 ms`–`0.190 ms`** / `0.274 ms`–`0.639 ms` | **`3,725–8,601 entries/sec`** | **`1.28 s`–`2.95 s`** | [`benchmarks/test_cases_11000_results.json`](benchmarks/test_cases_11000_results.json) |
| **3. Core Enterprise Structured Test Cases (`TC-0001`..`TC-1000`)** | **1,000 Entries** | **99.10%** (`991` / `1,000`) | **100.00%** (`100%` fails caught) | **`0.174 ms`–`0.379 ms`** / `0.429 ms`–`1.021 ms` | **`2,084–3,945 entries/sec`** | **`0.25 s`–`0.48 s`** | [`benchmarks/test_cases_1000_results.json`](benchmarks/test_cases_1000_results.json) |
| **4. 8-Track Quality & Performance SLA Suite (`G1`..`G8`, `750` & `1,000` Cases)** | **750 & 1,000 Cases** (`1,433`–`2,144` decisions) | **8 / 8 SLA Gates PASS** (**`98.75%`** Core Acc) | Cov@5% Error: **100.0%** | **`0.157 ms`–`0.433 ms`** / `0.610 ms`–`1.427 ms` | **`288–616 decisions/sec`** | **`2.91 s`–`5.79 s`** | [`benchmarks/quality_validation_750_results.json`](benchmarks/quality_validation_750_results.json) & [`benchmarks/quality_validation_1000_results.json`](benchmarks/quality_validation_1000_results.json) |
| **5. 5-Track Scientific Paper Benchmark (`JevBench` + `AnyJev` + `RLCD²`)** | **540 Items** (`1,040` decisions) | **98.75%** (`JevBench`, up from `98.1%`) · **100%** (`Pivot`) | **100.0%** Certified / **95.0%** Empirical Cov | **`0.266 ms`** / `0.917 ms` | **`> 1,000 decisions/sec`** | **`2.47 s`–`3.07 s`** | [`benchmarks/latest_results.json`](benchmarks/latest_results.json) |
| **6. 16 External Public Datasets Curated CI Baseline (`BoolQ`, `Banking77`, ...)** | **162 Base** / **640 Scaled** | **100.00%** (`162 / 162` · `16/16` datasets) | **100.00%** (`0.000000` Order-Flip) | **`0.388 ms`–`0.751 ms`** / `1.211 ms`–`2.633 ms` | **`1,671–1,877 entries/sec`** | **`0.097 s`–`0.172 s`** | [`benchmarks/web_benchmarks_results.json`](benchmarks/web_benchmarks_results.json) |
| **7. 12 Research Challenge Hypotheses & 15 Sequential RCAs** | **12 Protocols** + **15 RCAs** | **12/12 Hypotheses** & **15/15 RCAs PASS** (`100%`) | Route-Certified ($\alpha \le 5\%$) | **`0.306 ms`–`0.441 ms`** `p50` | **`> 1,500 trials/sec`** | **`0.05 s` + `3.53 s`** | [`src/microllm/research_challenge.py`](src/microllm/research_challenge.py) & [`src/microllm/rca_runner.py`](src/microllm/rca_runner.py) |

### 1.1 Hardware, Software & Available Compute Resources Specification

All published benchmark numbers in this repository were measured on the following standardized local testbed without any discrete GPU, cloud accelerator, or external API dependency:

| Resource Dimension | Specification / Measured Value | Operational Notes |
| :--- | :--- | :--- |
| **Host System** | `Lenovo 83LX` (`x86_64` / `AMD64` Workstation Laptop) | Standard consumer/developer hardware |
| **Processor (CPU)** | **AMD Ryzen AI 9 365 w/ Radeon 880M** (`Zen 5` / `Zen 5c`) | **10 physical cores / 20 logical threads**, base clock `2.00 GHz` |
| **CPU Cache Hierarchy** | `10 MB L2` (`10,240 KB`) · **`24 MB L3`** (`24,576 KB`) | All 256-D deterministic embeddings and INT8 weight tables fit entirely inside L3 cache |
| **System Memory (RAM)** | **32 GB** (`33,597,796,352 bytes` / `31.29 GiB` physical RAM) | Peak process RSS during benchmarks is **`< 85 MB`** (`< 195 MB` during `114,018`-row full Parquet run) |
| **Integrated GPU / NPU** | `AMD Radeon 880M Graphics` (`512 MB` UMA, Driver `32.0.22024.3004`) | **0% GPU / 0% NPU utilization** — all benchmarks run strictly on **CPU** |
| **Operating System** | **Windows 11** (`10.0.26300-SP0`, 64-bit `AMD64`) | Cross-platform compatible (`Linux`, `macOS`, `Windows`) |
| **Python Runtime** | **Python 3.14.7** (`MSC v.1944 64-bit AMD64`) | Single-process synchronous execution (`1` active Python worker thread) |
| **Scientific & Web Stack** | **NumPy 2.5.4** · **SciPy 1.18.1** · `FastAPI` · `Pydantic v2` | Vectorized `float64`/`int8` matrix ops + `BLAKE2b` LRU cache (`maxsize=65536` token / `4096` text) |
| **Network & API Budget** | **`0` external LLM API calls** · **`$0.00` token cost** | Cached Parquet splits (`~/.cache/microllm/external_datasets`, `~49 MB`) + offline fixtures |

### 1.2 Resource Utilization Profile per Benchmark Suite

| Benchmark Suite | Active Compute Device | CPU Threads Used | Peak Process RSS (RAM) | Disk Artifact Size | Network I/O (Default) | Total Wall-Clock Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. `web_benchmark_suite --full-splits` (`N = 114,018`)** | CPU Only (`0% GPU`) | `1` thread (single-process) | `~195 MB` | `~21 KB` JSON (`~49 MB` Parquet cache) | `0 B` (cached splits) | **`53.14 s`** |
| **1. `test_cases_11000` (`N = 11,000`)** | CPU Only (`0% GPU`) | `1` thread (single-process) | `~78 MB` | `~45 KB` JSON | `0 B` (100% offline) | **`1.28 s`** |
| **2. `test_cases_1000` (`N = 1,000`)** | CPU Only (`0% GPU`) | `1` thread (single-process) | `~64 MB` | `~360 KB` JSONL + JSON | `0 B` (100% offline) | **`0.27 s`** |
| **3. `quality_suite` (`N = 750`..`1,000`)** | CPU Only (`0% GPU`) | `1` thread (single-process) | `~82 MB` | `~18 KB` JSON | `0 B` (100% offline) | **`1.85 s`** |
| **4. `benchmarks.py` (`N = 540`)** | CPU Only (`0% GPU`) | `1` thread (single-process) | `~68 MB` | `~210 KB` JSONL + JSON | `0 B` (100% offline) | **`0.45 s`** |
| **5. `web_benchmark_suite` (`N = 162` / `640`)** | CPU Only (`0% GPU`) | `1` thread (single-process) | `~72 MB` | `~310 KB` JSONL + JSON | `0 B` (`~180 KB` with `--fetch-live`) | **`0.086 s`** / **`0.34 s`** |
| **6. `research_challenge` (`12` Hypotheses)** | CPU Only (`0% GPU`) | `1` thread (single-process) | `~74 MB` | In-memory report | `0 B` (100% offline) | **`1.10 s`** |

---

## 2. Benchmark 1 — 11,000 Structured Test Case Entries (`1,000` Core + `10,000` Product-Independent)

Every test case entry in [`src/microllm/test_cases_1000.py`](src/microllm/test_cases_1000.py) and [`src/microllm/product_independent_cases.py`](src/microllm/product_independent_cases.py) follows the uniform schema:
```json
{
  "id": "TC-01001",
  "domain": "healthcare_clinical_triage",
  "message": "[Indep #1001] Urgent triage: patient presents with acute chest pain and critical oxygen saturation drop!",
  "decision_type": "noul",
  "instructions": "Does this patient require immediate emergency resuscitation triage?",
  "rules": {
    "true": "Affirmative, yes, active emergency, compliant, permitted, approved, refund or escalation required",
    "false": "Negative, no, calm, normal, denied, prohibited, missing requirement, no issue"
  },
  "expected_result": true,
  "product_independent": true
}
```

### 2.1 Headline Metrics (`N = 11,000` Test Cases)

| Metric Category | Metric | Measured Result (`N = 11,000`) | SLA / Quality Gate |
| :--- | :--- | :---: | :---: |
| **Accuracy & Outcomes** | **Overall Accuracy** (`success_count` / `fail_count`) | **99.75%** (`10,973` Success / `27` Fail) | `>= 98.0%` (**PASS**) |
| **Success / Fail Detection** | **Mean Confidence on Success** | **`0.9369`–`0.9381`** | High certainty on valid decisions |
| **Success / Fail Detection** | **Mean Confidence on Fail** (Ambiguous Edge Cases) | **`0.2967`** | Low confidence on uncertain inputs |
| **Success / Fail Detection** | **Confidence Separation Gap** ($\overline{c}_{\text{succ}} - \overline{c}_{\text{fail}}$) | **`+0.6402`–`+0.6415`** | Strong separation between pass and fail |
| **Success / Fail Detection** | **Selective Accuracy** (Retained above $\tau = 0.45$) | **100.00%** (`99.45%`–`99.75%` coverage) | `100.0%` accuracy on non-abstained calls |
| **Success / Fail Detection** | **Failures Caught by Low-Confidence Abstention** | **100.0%** (`27 / 27` caught) | Automatic failure detection |
| **CPU Latency & Throughput** | **Latency Percentiles (`min` / `p50` / `p90` / `p95` / `p99`)** | `0.056` / **`0.076`–`0.190`** / `0.189` / **`0.274`–`0.639`** / **`0.395 ms`** | `p50 <= 5.0 ms`, `p95 <= 15.0 ms` (**PASS**) |
| **CPU Latency & Throughput** | **Wall-Clock Execution Time & Throughput** | **`1,278.9 ms` (`1.28 s`–`2.95 s`)** · **`3,725.2`–`8,601.0 entries/sec`** | `>= 500 entries/sec` (**PASS**) |
| **Calibration & Probabilistic** | **Brier Score / ECE / Binary F1 Score** | **Brier `0.0129`** · **ECE `6.16%`–`6.28%`** · **F1 `99.68%`** | `Brier <= 0.25`, `F1 >= 94%` (**PASS**) |

### 2.2 Breakdown by `decision_type` (`N = 11,000`)

| Decision Type | Total Entries | Success | Fail | Accuracy | Mean Confidence | Latency `p50` (ms) | Latency `p95` (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `noul` (Binary Proposition) | 2,420 | 2,408 | 12 | **99.50%** | `0.9207` | `0.061` | `0.261` |
| `choice` (Single-Choice $K=3..20$) | 3,520 | 3,505 | 15 | **99.57%** | `0.9860` | `0.070` | `0.225` |
| `score` (Ordinal Likert Rubric) | 1,760 | 1,760 | 0 | **100.00%** | `0.9886` | `0.067` | `0.182` |
| `multi` (Multi-Label Tagging) | 1,650 | 1,650 | 0 | **100.00%** | `0.7854` | `0.083` | `0.212` |
| `guardrail` (4-Rubric AI Safety) | 1,650 | 1,650 | 0 | **100.00%** | `0.9500` | `0.168` | `0.317` |
| **TOTAL** | **11,000** | **10,973** | **27** | **99.75%** | **`0.9381` (Succ) / `0.2967` (Fail)** | **`0.076`–`0.190`** | **`0.274`–`0.639`** |

### 2.3 Breakdown Across All 18 Domains (`8` Core + `10` Product-Independent Universal Domains)

| Domain | Scope | Entries | Success | Fail | Accuracy | Mean Confidence | Latency `p50` (ms) | Latency `p95` (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `financial_risk_aml` | Product-Independent | 1,340 | 1,330 | 10 | **99.25%** | `0.8915` | `0.082` | `0.183` |
| `cybersecurity_soc_alerts` | Product-Independent | 1,186 | 1,186 | 0 | **100.00%** | `0.9948` | `0.163` | `0.282` |
| `education_academic_grading` | Product-Independent | 1,107 | 1,099 | 8 | **99.28%** | `0.9394` | `0.068` | `0.186` |
| `legal_contract_clauses` | Product-Independent | 1,060 | 1,060 | 0 | **100.00%** | `0.9301` | `0.103` | `0.211` |
| `logistics_supply_chain` | Product-Independent | 1,040 | 1,040 | 0 | **100.00%** | `0.8609` | `0.082` | `0.166` |
| `healthcare_clinical_triage` | Product-Independent | 1,027 | 1,027 | 0 | **100.00%** | `0.9936` | `0.066` | `0.092` |
| `retail_ecommerce_moderation` | Product-Independent | 820 | 820 | 0 | **100.00%** | `0.8790` | `0.083` | `0.089` |
| `hr_recruitment_policy` | Product-Independent | 807 | 807 | 0 | **100.00%** | `0.9312` | `0.070` | `0.164` |
| `public_sector_civic_routing` | Product-Independent | 807 | 807 | 0 | **100.00%** | `0.9936` | `0.068` | `0.073` |
| `insurance_claims_adjudication` | Product-Independent | 806 | 806 | 0 | **100.00%** | `0.9870` | `0.067` | `0.074` |
| `customer_support` | Core Suite | 244 | 244 | 0 | **100.00%** | `0.8773` | `0.210` | `0.352` |
| `incident_triage` | Core Suite | 194 | 193 | 1 | **99.48%** | `0.9444` | `0.207` | `0.277` |
| `ai_safety_guardrail` | Core Suite | 150 | 150 | 0 | **100.00%** | `0.9571` | `0.301` | `0.348` |
| `policy_compliance` | Core Suite | 107 | 105 | 2 | **98.13%** | `0.8748` | `0.280` | `0.631` |
| `multilingual_routing` | Core Suite | 96 | 95 | 1 | **98.96%** | `0.7482` | `0.223` | `0.320` |
| `banking_20` | Core Suite | 80 | 75 | 5 | **93.75%** | `0.9025` | `0.380` | `0.501` |
| `contrastive_pivot` | Core Suite | 68 | 68 | 0 | **100.00%** | `0.9965` | `0.212` | `0.294` |
| `procurement_finance` | Core Suite | 61 | 61 | 0 | **100.00%** | `0.9623` | `0.206` | `0.288` |

---

## 3. Benchmark 2 — 1,000 Core Structured Test Case Entries (`benchmarks/test_cases_1000.jsonl`)

Evaluating the standalone 1,000 core test case entries (`TC-0001`..`TC-1000`) with zero synthetic prefix stripping (`benchmarks/test_cases_1000_results.json`):
- **Overall Accuracy**: **99.10%** (`991` Success / `9` Fail)
- **Success / Fail Detection (`tau = 0.45`)**: Mean Confidence on Success = **`0.9105`** (`0.9084`–`0.9105`) vs Fail = **`0.2967`** (Gap = **`+0.6138`**) · Selective Accuracy = **`100.00%`** · Failures Caught = **`100.0%`** (`9/9`)
- **Latency Distribution (CPU)**: `min = 0.145 ms` · **`p50 = 0.219 ms`–`0.379 ms`** · `p90 = 0.367 ms`–`0.830 ms` · **`p95 = 0.429 ms`–`1.021 ms`** · **`p99 = 0.739 ms`–`1.416 ms`** · Throughput = **`2,084.3`–`3,734.6 entries/sec`**
- **Calibration**: Brier Score = **`0.0284`** (`0.0284`–`0.0286`) · ECE = **`8.83%`** (`8.83%`–`9.03%`) · Binary F1 Score = **`98.82%`**

---

## 4. Benchmark 3 — 750–1,000 Case 8-Track Quality & Performance SLA Validation Matrix

Executed via `python -m microllm.quality_suite` ([`src/microllm/quality_suite.py`](src/microllm/quality_suite.py), [`benchmarks/quality_validation_750_results.json`](benchmarks/quality_validation_750_results.json), and [`benchmarks/quality_validation_1000_results.json`](benchmarks/quality_validation_1000_results.json)):

| Gate ID | Validation Track | Cases (`N=750` / `N=1000`) | Primary Metric & Measured Value | SLA Target | Gate Status |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **`G1`** | `jevbench_core` | 160 / 160 | Cascade Accuracy: **98.75%** (`choice` 100%, `score` 100%, `multi` 100%, `noul` 95.0%) | `>= 95.0%` | **PASS** |
| **`G2`** | `decision_pivot` | 80 (40 pairs) | Contrastive Pair Accuracy: **100.0%** (`40/40` pairs) | `>= 90.0%` | **PASS** |
| **`G3`** | `anyjev_order_invariance` | 60 ($K=2..20$) | Gear 2 $L_0$ Order-Flip Rate: **`0.000000`** (`0.000000` drift, **78.3% fewer rotations**) | `== 0.000000` | **PASS** |
| **`G4`** | `guardrail_safety` | 80 / 80 | 4-Rubric Safety F1: **100.0%** · Recall: **100.0%** · False Positive Rate: **0.0%** | `F1 >= 94.0%` | **PASS** |
| **`G5`** | `calibration_rlcd2` | 160 / 160 | Calibrated ECE: **`6.15%`** (vs `15.85%` raw) · RLCD² Brier: **`0.1525`** (vs `0.3275` raw) · Cov@5%: **100.0%** | `< Raw Baseline` | **PASS** |
| **`G6`** | `jevcompat_contract_and_boundaries` | 70 / 153 | Contract & Adversarial 422/401/Tournament Pass Rate: **100.0%** (`0` HTTP 5xx) | `100% / 0 5xx` | **PASS** |
| **`G7`** | `multilingual_and_xai_fidelity` | 70 / 153 | EN/FR/ES/DE Routing + Token Attribution & Counterfactual XAI: **100.0%** | `>= 95.0%` | **PASS** |
| **`G8`** | `performance_sla_and_batch_stress` | 70 / 154 | CPU `p50 = 0.157 ms–0.433 ms`, `p95 = 0.610 ms–1.427 ms`, `BLAKE2b` Cache Hits, INT8 `MSE = 4.1e-6` | `p50 <= 10 ms` | **PASS** |

---

## 5. Benchmark 4 — 16 External Public Web Datasets Suite (`114,018` Full Unsampled Rows & `162`/`640` Curated Baseline)

Executed via `microllm benchmark --external --full-splits` (full unsampled splits in [`benchmarks/web_benchmarks_full_results.json`](benchmarks/web_benchmarks_full_results.json)) and `microllm benchmark --external` (curated CI baseline in [`benchmarks/web_benchmarks_results.json`](benchmarks/web_benchmarks_results.json), full dossier in [`docs/EXTERNAL_DATASETS_BENCHMARKS.md`](docs/EXTERNAL_DATASETS_BENCHMARKS.md)) across all **16 external public datasets** from Hugging Face and GitHub:

### 5.1 Regime A — Full Unsampled Official Evaluation Splits (`N = 114,018` Rows, Zero Sampling)
- **Total Unsampled Rows Evaluated**: **`114,018`** (`100%` of the official evaluation splits across all 16 datasets)
- **Overall Zero-Shot CPU Accuracy**: **`66.82%`** (`76,186` Passed / `37,832` Failed across 114,018 unseen natural-language rows — **`+1,028` net new correct zero-shot predictions** over the `65.92%` baseline following `RCA-028`..`RCA-030` and `HYP-005`/`HYP-006`/`HYP-008`/`HYP-011`)
- **Selective Accuracy ($\tau = 0.45$)**: **`87.49%`** at **`76.37%` coverage** (`mean_confidence_on_success = 0.6358` vs `mean_confidence_on_fail = 0.2913`, separation gap **`+0.3445`**)
- **AnyJev $L_0$ Order-Flip Rate (`1,867` items: `1,531` ChoiceBench + `336` Callo-Bias)**: **`0.000000`** (`0.0000000000` max probability drift)
- **Ordinal Score MAE (`55,284` items: `5,000` Amazon + `50,000` Yelp + `284` JevBench)**: **`1.2086`** (`99.24%` selective accuracy on Amazon, `99.97%` selective accuracy on Yelp)
- **CPU Latency & Throughput (`N = 114,018`)**: `p50 = 0.527 ms` · `p95 = 2.053 ms` · `p99 = 2.720 ms` · **`1,187.8 entries/sec`** (`95.99 s` total wall-clock time)

| # | External Public Dataset | Official Split | Primitive | Unsampled Rows | Passed | Failed | Raw Acc | Selective Acc ($\tau=0.45$) | `p50` (ms) | `p95` (ms) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `google/boolq` | `validation` | `noul` | **3,270** | 2,008 | 1,262 | **61.41%** | `61.41%` (`100%` cov) | `0.532` | `0.918` |
| 2 | `nyu-mll/multi_nli` | `validation_matched` | `noul` | **9,815** | 5,391 | 4,424 | **54.93%** | `54.93%` (`100%` cov) | `0.107` | `0.215` |
| 3 | `PolyAI/banking77` | `test` | `choice` | **3,076** | 1,877 | 1,199 | **61.02%** | **98.74%** (`61.8%` cov) | `1.488` | `1.712` |
| 4 | `fancyzhx/ag_news` | `test` | `choice` | **7,600** | 3,897 | 3,703 | **51.28%** | **99.51%** (`51.5%` cov) | `0.623` | `0.805` |
| 5 | `clinc_oos` | `test` | `choice (OOS)` | **5,500** | 4,749 | 751 | **86.35%** | **91.47%** (`94.4%` cov) | `0.950` | `1.122` |
| 6 | `amazon_reviews_multi` | `test` | `score` | **5,000** | 3,373 | 1,627 | **67.46%** | **99.24%** (`68.0%` cov) | `0.319` | `0.551` |
| 7 | `Yelp/yelp_review_full` | `test` | `score` | **50,000** | 33,708 | 16,292 | **67.42%** | **99.97%** (`67.4%` cov) | `0.486` | `2.182` |
| 8 | `google-research-datasets/go_emotions` | `test` | `multi` | **5,427** | 3,460 | 1,967 | **63.76%** | **83.37%** (`76.5%` cov) | `0.763` | `0.884` |
| 9 | `reuters21578` | `test` | `multi` | **3,299** | 2,393 | 906 | **72.54%** | **95.34%** (`76.1%` cov) | `0.718` | `3.337` |
| 10 | `allenai/wildguardmix` | `wildguardtest` | `guardrail` | **1,725** | 1,085 | 640 | **62.90%** | `62.90%` (`Prec 68.27%`) | `0.523` | `2.255` |
| 11 | `lmsys/toxic-chat` | `test` | `guardrail` | **5,083** | 4,498 | 585 | **88.49%** | **88.49%** (`FPR 6.38%`) | `0.432` | `1.729` |
| 12 | `deepset/prompt-injections` | `test` | `guardrail` | **116** | 80 | 36 | **68.97%** | `68.97%` (`Prec 90.00%`) | `0.442` | `0.719` |
| 13 | `cotenthusiast/choicebench` | `validation` | `choice (L0)` | **1,531** | 355 | 1,176 | **23.19%** | **`0.000000` Flip Rate** | `0.693` | `2.111` |
| 14 | `peculiar-ragdoll/callo-bias-set` | `default` | `choice (L0)` | **336** | 176 | 160 | **52.38%** | **`0.000000` Flip Rate** | `0.526` | `0.882` |
| 15 | `lm-sys/RouteLLM` | `train` | `choice (cascade)` | **10,000** | 7,775 | 2,225 | **77.75%** | **92.60%** (`84.0%` cov) | `0.394` | `1.793` |
| 16 | `fstandhartinger/jevbench` | `test` | `jevcompat (4x)` | **2,240** | 1,361 | 879 | **60.76%** | **67.31%** (`90.3%` cov) | `0.600` | `2.318` |
| **TOTAL** | **16 Unsampled Official Splits** | — | **All 5 Modes** | **114,018** | **76,186** | **37,832** | **66.82%** | **87.49% (`76.37%` cov)** | **`0.527`** | **`2.053`** |

### 5.2 Regime B — Curated Gold-Standard CI Baseline (`N = 162` Base / `640` Scaled Entries, `benchmarks/web_benchmarks_results.json`)
- **Overall Accuracy**: **`100.00%`** (`162 / 162` across all 16 external datasets; `640 / 640` on scaled corpus)
- **Selective Accuracy ($\tau = 0.45$)**: **`100.00%`** (`100.0%` coverage) · **AnyJev $L_0$ Order-Flip Rate**: **`0.000000`**
- **CPU Latency & Throughput**: `p50 = 0.388 ms–0.751 ms` · `p95 = 1.211 ms–2.633 ms` · **`1,671–1,877 entries/sec`**

---

## 6. Benchmark 5 — 540-Item 5-Track Scientific Paper Benchmark (`benchmarks/latest_results.json`)

Executed via `microllm benchmark` ([`src/microllm/benchmarks.py`](src/microllm/benchmarks.py)), evaluating **540 items (`1,040` decisions)** exported to [`benchmarks/microllm_eval_corpus.jsonl`](benchmarks/microllm_eval_corpus.jsonl) and [`benchmarks/latest_results.json`](benchmarks/latest_results.json):

| Scientific Track | Items | Gear 1 (`FastEncoder`) | Gear 2 (`AnyJevPrefill`) | Dual-Gear Cascade (`StartLux`) | Latency `p50` / `p95` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Track 1 — `JevBench-Core`** (`noul`, `choice`, `score`, `multi`) | 160 | `96.25%` (`noul` 85%, `choice`/`score`/`multi` 100%) | **`98.75%`** (`noul` 95%, `choice`/`score`/`multi` 100%) | **`98.75%`** (`noul` 95%, `choice`/`score`/`multi` 100%) | **`0.266 ms`** / `0.917 ms` |
| **Track 2 — `Decision-Pivot`** (`40` contrastive pairs) | 80 | `100.0%` (`40/40` pairs) | `100.0%` (`40/40` pairs) | **`100.0%`** (`40/40` pairs) | `0.15 ms` / `0.42 ms` |
| **Track 3 — `AnyJev Order-Invariance`** ($K \in \{2,4,8,12,20\}$) | 60 | `0.000000` Flip Rate | **`0.000000` Flip Rate** (`0.0` drift, **78.3% fewer rotations**) | Naive baseline: `50.0%` Flip Rate | `0.42 ms` / `1.35 ms` |
| **Track 4 — `Guardrail Safety`** (`4` rubrics + grounded controls) | 80 | — | — | **F1 `100.0%`** · **Recall `100.0%`** · **FPR `0.0%`** | `0.56 ms` / `0.79 ms` |
| **Track 5 — `Calibration, Conformal & RLCD²`** | 160 | Raw ECE: `15.85%` (`Brier 0.3275`) | Temp-Scaled ECE: **`6.15%`** (`Brier 0.2577`) | **RLCD² Brier: `0.1525`** · **Certified Cov: `100.0%`** (`95.0%` emp) | `< 35 ms` total fit |

---

## 7. Benchmark 6 — 12 Research Challenge Hypotheses (`VAL-HYP-001`..`VAL-HYP-012`)

Executed via `evaluate_all_12_hypotheses()` in [`src/microllm/research_challenge.py`](src/microllm/research_challenge.py) and verified in [`tests/test_research_challenge_hypotheses.py`](tests/test_research_challenge_hypotheses.py) (full mathematical specifications in [`docs/RCA_REGISTER.md` §6](docs/RCA_REGISTER.md)):

| Hypothesis ID | Validation Protocol | Challenged Mechanism | Measured Result | Status |
| :--- | :--- | :--- | :--- | :---: |
| **`HYP-001`** | `VAL-HYP-001` | Policy-Aware & Route-Conditioned Calibration ($N=16$ Best-of-$N$) | Overconfidence gap reduced by **`> 65%`**, route-wise risk certified | **PASS** |
| **`HYP-002`** | `VAL-HYP-002` | Exact Beta-Integral Proper Score $S_D(p,t)$ (`StartLux` Eq. 3–5) | Unique maximizer at $q^* = \pi$ (`\|q* - pi\| = 0.0000`) | **PASS** |
| **`HYP-003`** | `VAL-HYP-003` | Disjoint `1/3` Selection + `2/3` Verification Clopper-Pearson Split | Multiplicity-free finite-sample risk certificate verified | **PASS** |
| **`HYP-004`** | `VAL-HYP-004` | Kolmogorov Probability Additivity & Bregman Simplex Projection | Coherence deviation reduced to **`< 1e-12`** | **PASS** |
| **`HYP-005`** | `VAL-HYP-005` | Value-of-Computation (`VoC`) Gate vs. Aleatoric/Open-Set Escalation | `100%` open-set detection, **`>= 50%` aleatoric escalations saved** | **PASS** |
| **`HYP-006`** | `VAL-HYP-006` | Bound Rubric Definition Dominance vs. Adversarial Option Keys | Conflict detected (`100%`), follows bound rubric (`> 60%` prob) | **PASS** |
| **`HYP-007`** | `VAL-HYP-007` | KL Skew-Adaptive Prior Debiasing (`85/10/5` class skew) | Preserves minority & majority accuracy (`Acc_adapt >= Acc_naive`) | **PASS** |
| **`HYP-008`** | `VAL-HYP-008` | Clause-Salient Max-Pooling under Long Distractor Contexts (`JevOut`) | Evidence removal drop `>= 0.20`, distractor drift `<= 0.05`, **SNR `>= 4.0x`** | **PASS** |
| **`HYP-009`** | `VAL-HYP-009` | Murphy Brier Decomposition (`REL - RES + UNC`) & `AURC`/`Cov@10%` | Exact identity residual `< 0.05`, `RES > REL`, `Cov@10% = 100%` | **PASS** |
| **`HYP-010`** | `VAL-HYP-010` | Luce IIA Stability under Dominated Decoy Option Injection | Top-1 preserved (`100%`), pairwise log-odds drift `< 1e-6` | **PASS** |
| **`HYP-011`** | `VAL-HYP-011` | Log-Space Multi-Template NLI Framing Consensus ($M=3$) | Cross-template variance `< 0.05`, consensus matches ground truth | **PASS** |
| **`HYP-012`** | `VAL-HYP-012` | `NumericJev` Multiway Bisection Decision Tree ($D=3$, 8 leaves) | Continuous target recovered (`MAE <= 12.5`) inside `90%` credible interval | **PASS** |

---

## 8. Step-by-Step Operational Procedure to Run All Benchmarks

This section documents the complete, end-to-end procedure to execute every benchmark suite locally or in CI, regenerate all JSON/JSONL telemetry artifacts in `benchmarks/`, and verify all quality gates.

### Step 0 — Environment Prerequisites, Installation & 1-Command Unified Automation Orchestrator

1. **System Requirements**:
   - Any 64-bit CPU (`x86_64` / `AMD64` or `ARM64` / Apple Silicon), `>= 512 MB` free RAM, **no GPU required** (`0% GPU`).
   - Python `>= 3.10` (tested up to `Python 3.14.7` with `NumPy 2.5` and `SciPy 1.18`).
2. **Install MicroLLM in Editable Mode with Dev Dependencies**:
   ```bash
   git clone https://github.com/jmrenouard/MicroLLM.git
   cd MicroLLM
   pip install -e ".[dev]"
   ```
3. **One-Command Unified Automation Runner (`microllm.benchmark_runner` & `scripts/run_all_benchmarks.*`)**:
   To avoid running individual benchmark modules manually, use the unified automation orchestrator ([`src/microllm/benchmark_runner.py`](src/microllm/benchmark_runner.py)) or the cross-platform wrapper scripts in [`scripts/`](scripts/) to execute all **9 benchmark suites** (`eval_540`, `quality_750`, `quality_1000`, `test_cases_1000`, `test_cases_11000`, `web_external_curated`, `web_external_full_splits`, `research_hypotheses`, `rca_non_regression`) and generate [`benchmarks/all_benchmarks_master_report.json`](benchmarks/all_benchmarks_master_report.json):
   ```bash
   # Run all 9 benchmark suites (including 114,018-row full external splits) and save all JSON/JSONL artifacts
   python scripts/run_all_benchmarks.py
   # Or via CLI:
   microllm benchmark --all-suites
   # Or via OS wrapper scripts:
   # Windows PowerShell: .\scripts\run_all_benchmarks.ps1
   # Linux / macOS:      ./scripts/run_all_benchmarks.sh

   # Fast CI mode (runs 8 suites = 14,479 items in ~15s, skipping the 114,018-row full Parquet splits)
   python scripts/run_all_benchmarks.py --fast
   microllm benchmark --all-suites --fast
   ```

---

### Step 1 — Run Benchmark Suite 5: 540-Item 5-Track Scientific Benchmark (`microllm benchmark`)

Evaluates `JevBench-Core` (160), `Decision-Pivot` (80), `AnyJev Order-Invariance` (60), `Guardrail Safety` (80), and `Calibration & RLCD²` (160):

```bash
# Via CLI (prints Markdown report to stdout and writes benchmarks/latest_results.json + benchmarks/microllm_eval_corpus.jsonl)
microllm benchmark --output benchmarks/latest_results.json

# Or directly via Python module
python -m microllm.benchmarks
```
- **Generated Artifacts**:
  - [`benchmarks/microllm_eval_corpus.jsonl`](benchmarks/microllm_eval_corpus.jsonl) (`540` evaluation records)
  - [`benchmarks/latest_results.json`](benchmarks/latest_results.json) (per-track metrics, Brier, ECE, latency percentiles)

---

### Step 2 — Run Benchmark Suites 1 & 6: 16 External Public Datasets (`--full-splits` 114,018 Rows & Curated Baseline)

Evaluates all **16 external datasets** (`google/boolq`, `nyu-mll/multi_nli`, `PolyAI/banking77`, `fancyzhx/ag_news`, `clinc_oos`, `amazon_reviews_multi`, `Yelp/yelp_review_full`, `google-research-datasets/go_emotions`, `reuters21578`, `allenai/wildguardmix`, `lmsys/toxic-chat`, `deepset/prompt-injections`, `cotenthusiast/choicebench`, `peculiar-ragdoll/callo-bias-set`, `lm-sys/RouteLLM`, `fstandhartinger/jevbench`):

```bash
# 2A. Full unsampled official evaluation splits across all 16 datasets (114,018 rows, zero sampling -> ~53s on CPU)
microllm benchmark --external --full-splits --output benchmarks/web_benchmarks_full_results.json

# 2B. Standard deterministic offline run across all 16 datasets (162 base curated entries)
microllm benchmark --external --output benchmarks/web_benchmarks_results.json

# 2C. Scaled 640-entry run (40 entries per dataset across all 16 datasets)
microllm benchmark --external --scale-per-dataset 40 --output benchmarks/web_benchmarks_results.json

# 2D. Live Hugging Face Datasets Server ingestion (fetches live rows from https://datasets-server.huggingface.co/rows with automatic offline fallback)
microllm benchmark --external --fetch-live --output benchmarks/web_benchmarks_results.json

# 2E. Standalone Python module execution (exports benchmarks/external_datasets_corpus.jsonl + benchmarks/web_benchmarks_results.json)
python -m microllm.web_benchmark_suite
python -m microllm.web_benchmark_suite --full-splits
```
- **Generated Artifacts**:
  - [`benchmarks/web_benchmarks_full_results.json`](benchmarks/web_benchmarks_full_results.json) (**`114,018` unsampled rows**, `66.82%` raw / `87.49%` selective accuracy at `76.37%` coverage, `0.000000` $L_0$ order-flip rate on `1,867` items, `p50 = 0.527 ms`–`0.938 ms`)
  - [`benchmarks/web_benchmarks_results.json`](benchmarks/web_benchmarks_results.json) (`162` base curated entries, `100.00%` accuracy, `p50 = 0.388 ms`–`0.751 ms`)
  - [`benchmarks/external_datasets_corpus.jsonl`](benchmarks/external_datasets_corpus.jsonl) (`640` normalized `jevcompat 0.1` entries across 16 datasets)

---

### Step 3 — Run Benchmark Suite 3: 750–1,000 Case 8-Track Quality & Performance SLA Suite

Evaluates all 8 automated SLA gates (`G1`..`G8`: accuracy, contrastive pivot, $L_0$ order invariance, safety guardrails, RLCD² calibration, `32/32 MUST` `jevcompat 0.1` contract + HTTP `422`/`401` + $K>25$ tournament, `EN`/`FR`/`ES`/`DE` + XAI, and CPU latency + `BLAKE2b` cache + INT8 quantization):

```bash
# Default 750-case run (writes benchmarks/quality_validation_750_results.json)
python -m microllm.quality_suite

# Or run at maximum 1,000-case scale from Python
python -c "from pathlib import Path; from microllm.quality_suite import run_quality_validation_suite, format_quality_suite_markdown; r = run_quality_validation_suite(target_cases=1000, output_path=Path('benchmarks/quality_validation_1000_results.json')); print(format_quality_suite_markdown(r))"
```
- **Generated Artifacts**:
  - [`benchmarks/quality_validation_750_results.json`](benchmarks/quality_validation_750_results.json) (`750` cases, `8/8` Gates PASS, `98.75%` core acc)
  - [`benchmarks/quality_validation_1000_results.json`](benchmarks/quality_validation_1000_results.json) (`1,000` cases, `8/8` Gates PASS, `98.75%` core acc)

---

### Step 4 — Run Benchmark Suites 1 & 2: 1,000 Core & 11,000 Full Structured Test Cases

Evaluates the `1,000` core enterprise structured test cases (`TC-0001`..`TC-1000`) and the `10,000` product-independent cross-industry test cases (`TC-1001`..`TC-11000` across 18 domains):

```bash
# 4A. Evaluate the 1,000 core structured test cases (writes benchmarks/test_cases_1000.jsonl and benchmarks/test_cases_1000_results.json)
python -m microllm.test_cases_1000

# 4B. Evaluate all 11,000 structured test cases (1,000 core + 10,000 product-independent -> writes benchmarks/test_cases_11000_results.json)
python -c "from pathlib import Path; from microllm.test_cases_1000 import evaluate_structured_test_cases, format_published_metrics_markdown; r = evaluate_structured_test_cases(11000, Path('benchmarks/test_cases_11000_results.json')); print(format_published_metrics_markdown(r))"
```
- **Generated Artifacts**:
  - [`benchmarks/test_cases_1000.jsonl`](benchmarks/test_cases_1000.jsonl) & [`benchmarks/test_cases_1000_results.json`](benchmarks/test_cases_1000_results.json)
  - [`benchmarks/test_cases_11000_results.json`](benchmarks/test_cases_11000_results.json)

---

### Step 5 — Run Benchmark Suite 6: 12 Research Challenge Hypotheses (`VAL-HYP-001`..`VAL-HYP-012`)

Executes the 12 falsifiable mathematical and empirical validation protocols (`PolicyAwareCalibrator`, `rlcd2_proper_integral_score`, `clopper_pearson_split_certificate`, Kolmogorov coherence, Value-of-Computation gate, rubric dominance, skew-adaptive prior debiasing, clause-salient max-pooling, Murphy Brier decomposition, decoy IIA stability, multi-template NLI consensus, and `NumericJev` bisection trees):

```bash
python -c "import json; from microllm.research_challenge import evaluate_all_12_hypotheses; r = evaluate_all_12_hypotheses(); print(f'Passed: {r[\"passed_count\"]}/{r[\"total_hypotheses\"]} (all_passed={r[\"all_passed\"]})')"
```

---

### Step 6 — Run Benchmarks via the FastAPI HTTP REST Server & Decision Studio Web UI

Start the local server (`microllm serve --port 8080`) and trigger either the internal 5-track benchmark or the 16-dataset external benchmark over HTTP:

```bash
# 1. Start the MicroLLM FastAPI server
microllm serve --host 127.0.0.1 --port 8080

# 2. Trigger the 540-item scientific benchmark via REST API
curl -s http://127.0.0.1:8080/v1/benchmark | python -m json.tool

# 3. Trigger the 16-dataset external benchmark via GET or POST
curl -s "http://127.0.0.1:8080/v1/benchmark/external?scale_per_dataset=0&fetch_live=false" | python -m json.tool

curl -s -X POST http://127.0.0.1:8080/v1/benchmark/external \
  -H "Content-Type: application/json" \
  -d '{"datasets": ["google/boolq", "PolyAI/banking77", "clinc_oos"], "scale_per_dataset": 10, "fetch_live": false}' | python -m json.tool
```

---

### Step 7 — Run Automated `pytest` Verification & Non-Regression Gates

```bash
# Run dedicated benchmark & quality gate test suites
pytest tests/test_benchmark_runner.py tests/test_benchmark_corpus.py tests/test_quality_validation_suite.py tests/test_web_benchmark_adapters.py tests/test_research_challenge_hypotheses.py -v

# Run the complete 168-test repository suite with code coverage
pytest --cov=microllm --cov-report=term-missing -v
```

---

## 9. Worst-Case Regimes, Failure Mode Analysis & Architectural Limitations

For an exhaustive, transparent scientific breakdown of MicroLLM's worst-case performance, refer to the dedicated monograph:
👉 [**`docs/FAILURE_MODES_AND_WORST_CASES.md`**](docs/FAILURE_MODES_AND_WORST_CASES.md)

### 9.1 Summary of Lowest-Accuracy Regimes (`114,018` Unsampled Public Rows)
1. **Adversarial Deductive Reasoning**: `cotenthusiast/choicebench` (**`23.19%`** raw zero-shot accuracy). Without autoregressive token generation or deep multi-head cross-attention, FastEncoder cannot perform multi-step counterfactual deduction (though AnyJev $L_0$ order invariance remains certified at `0.000000`).
2. **Severe Lexical Overlap Across Topics**: `fancyzhx/ag_news` (**`51.28%`** raw). Conformal thresholding ($\tau = 0.45$) lifts accuracy to **`99.51%`** at `51.53%` coverage.
3. **Cognitive Bias Syllogisms**: `peculiar-ragdoll/callo-bias-set` (**`52.38%`** raw).
4. **Three-Way NLI Entailment**: `nyu-mll/multi_nli` (**`54.93%`** raw).
5. **High-Cardinality Banking Intents**: `PolyAI/banking77` (**`61.02%`** raw across 77 intents, lifting to **`98.74%`** selective accuracy at $\tau = 0.45$).
6. **Wikipedia Fact Verification**: `google/boolq` (**`61.41%`** raw).

### 9.2 Real-World Safety Vulnerability (Wild Web Jailbreaks)
Across `6,924` real-world adversarial attacks (`allenai/wildguardmix`, `lmsys/toxic-chat`, `deepset/prompt-injections`):
- **Safety Recall**: **`27.04%`** (missed `858` narrative and role-playing jailbreaks).
- **Safety Precision**: **`44.11%`** (generated `403` false alarms on colloquial banter).
- **F1-Score**: **`33.53%`**.
- **Architectural Remedy**: System-1 guardrails provide fast Tier-1 triage (< 0.5 ms), but critical endpoints require the StartLux cascade to a neural System-2 guardrail.

### 9.3 Ordinal Star-Level Granularity vs. Polar Sentiment
On `55,284` Amazon and Yelp customer reviews, exact star-match accuracy is **`27.64%`** (Amazon) and **`30.12%`** (Yelp), with a Mean Absolute Error of **`1.20` stars**. The model reliably separates positive from negative reviews, but confuses adjacent ratings (4★ vs. 5★). Conformal filtering yields **`99.97%`** selective accuracy on unambiguous reviews.

---

## 10. Hugging Face JEV Models Ecosystem Benchmark Suite (`12` Datasets · `25` HF JEV Models, Issue #108)

MicroLLM catalogs **25 active System-One / JEV models** published on the Hugging Face Hub ([`https://huggingface.co/models?search=jev`](https://huggingface.co/models?search=jev)) in [`src/microllm/hf_jev_datasets.py`](src/microllm/hf_jev_datasets.py) and evaluates **12 linked JEV ecosystem datasets** (`ALL_12_HF_JEV_MODEL_DATASETS`, expanding total supported external datasets to **28** via `ALL_28_EXTERNAL_DATASETS`):

- **Execution Command**: `microllm benchmark --hf-jev-models` (or `python -m microllm.web_benchmark_suite --hf-jev-models`)
- **HTTP Endpoints**: `GET /v1/jev-models`, `GET /v1/benchmark/hf-jev-models`, `POST /v1/benchmark/hf-jev-models`
- **Artifacts Generated**: [`benchmarks/hf_jev_models_benchmarks_results.json`](benchmarks/hf_jev_models_benchmarks_results.json) (`120` curated entries) and [`benchmarks/hf_jev_models_corpus.jsonl`](benchmarks/hf_jev_models_corpus.jsonl) (`480` scaled entries)
- **Measured Metrics**: **`100.00%` accuracy (`120/120`)**, **`0.000000` AnyJev $L_0$ order-flip rate**, **`100.0%` Guardrail F1 (`0.0%` FPR)**, **`0.0000` Ordinal MAE**, and **`p50 = 0.557 ms`** CPU latency (`p95 = 1.230 ms`).

| # | Hugging Face JEV Dataset | Linked HF JEV Models | MicroLLM Primitive | Entries | Accuracy | Latency `p50` |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | `SargeDev/jev-distill-corpus-v3` | `autotrust/JEV-27B`, `JEV-9B`, `Bonsai-JEV` | `noul` / `choice` / `score` | 10 | **100.00%** | `0.935 ms` |
| 2 | `ZefanCai/Open-Jev` | `autotrust/JEV-27B`, `TokenRhythm/NeoHorse-Jev-4B` | `choice` / `noul` / `score` | 10 | **100.00%** | `0.602 ms` |
| 3 | `tasksource/tasksource-jev-typed-decisions` | `scairesearch/kieve-jev` | `noul` / `choice` / `score` | 10 | **100.00%** | `0.639 ms` |
| 4 | `JevAgentHarness/AgentDecisionBench` | `JevAgentHarness/AgentDecision-4B` | `choice` (Agent Tools) | 10 | **100.00%** | `0.686 ms` |
| 5 | `LocalLLaMA/typed-decisions` | `J33-AI/j33-jev` | `choice` / `noul` / `score` | 10 | **100.00%** | `0.709 ms` |
| 6 | `3nesdeniz/agentic-prompt-injection-5k` | `patronus-studio/wolf-defender-jev` | `guardrail` | 10 | **100.00%** | `0.426 ms` |
| 7 | `MaziyarPanahi/Jev-RAG-Claim-Verification-130K` | `hubin/jev-guard-v3.2-2b` | `noul` (RAG Groundedness) | 10 | **100.00%** | `0.246 ms` |
| 8 | `MaziyarPanahi/Jev-Safety-Refusal-Decisions-150K` | `hubin/jev-guard-v3.2-2b` | `guardrail` | 10 | **100.00%** | `0.473 ms` |
| 9 | `SetFit/sst5` | `com-kotobalabs/open-jev-deberta-v3-large` | `score` (5-level SST-5) | 10 | **100.00%** | `0.286 ms` |
| 10 | `google-research-datasets/paws` | `shibing624/train-jev-4b-lora` | `noul` (Paraphrase NLI) | 10 | **100.00%** | `0.373 ms` |
| 11 | `akhilaaa3/decision-bench` | `akhilaaa3/Jev-Omni` | `choice` | 10 | **100.00%** | `0.740 ms` |
| 12 | `Hangtao/JevAdvBench` | `arXiv:2609.31142` Adversarial Suite | `choice` ($L_0$ Invariance) | 10 | **100.00%** | `0.657 ms` |



