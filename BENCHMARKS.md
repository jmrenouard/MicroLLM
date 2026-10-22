# MicroLLM — Published Benchmarks, Quality & Performance Results

[![Benchmark 11,000](https://img.shields.io/badge/Structured_Cases-11%2C000_Entries_%7C_99.75%25_Acc-10b981)](benchmarks/test_cases_11000_results.json)
[![CPU Latency](https://img.shields.io/badge/CPU_Latency-0.076ms_p50_%7C_8%2C601_dec%2Fs-0ea5e9)](benchmarks/test_cases_11000_results.json)
[![Quality Suite 750](https://img.shields.io/badge/8--Track_SLA_Gates-8%2F8_PASS_(750_Cases)-3b82f6)](benchmarks/quality_validation_750_results.json)
[![JevCompat 0.1](https://img.shields.io/badge/JevCompat_0.1-32%2F32_MUST-10b981)](docs/TEST_PLAN.md)

> **Quick Navigation**:
> - Back to [**Main Project `README.md`**](README.md)
> - Read the [**Complete Test Plan (`docs/TEST_PLAN.md`)**](docs/TEST_PLAN.md)
> - Inspect the [**RCA & Governance Register (`docs/RCA_REGISTER.md`)**](docs/RCA_REGISTER.md)
> - Raw Machine-Readable Telemetry Artifacts:
>   - [`benchmarks/test_cases_11000_results.json`](benchmarks/test_cases_11000_results.json) (`11,000` structured test case entries: `1,000` core + `10,000` product-independent)
>   - [`benchmarks/test_cases_1000.jsonl`](benchmarks/test_cases_1000.jsonl) & [`benchmarks/test_cases_1000_results.json`](benchmarks/test_cases_1000_results.json) (`1,000` core structured test case entries)
>   - [`benchmarks/quality_validation_750_results.json`](benchmarks/quality_validation_750_results.json) (`750`-case 8-track Quality & Performance SLA validation suite)
>   - [`benchmarks/microllm_eval_corpus.jsonl`](benchmarks/microllm_eval_corpus.jsonl) & [`benchmarks/latest_results.json`](benchmarks/latest_results.json) (`540`-item 5-track scientific benchmark)
>   - [`benchmarks/web_benchmarks_results.json`](benchmarks/web_benchmarks_results.json) & [`benchmarks/external_datasets_corpus.jsonl`](benchmarks/external_datasets_corpus.jsonl) (`16` External Public Datasets — `162` base / `640` scaled entries)

---

## 1. Executive Summary of Published Benchmarks

**MicroLLM** is evaluated across **five complementary, 100% reproducible benchmark suites** on CPU:

| Benchmark Suite | Scale (Entries / Cases) | Overall Accuracy / Pass Rate | Selective Accuracy ($\tau = 0.45$) | CPU Latency (`p50` / `p95`) | Throughput (CPU) | Artifact / Runner |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Full Structured Test Suite (`1,000` Core + `10,000` Product-Independent)** | **11,000 Entries** | **99.75%** (`10,973` / `11,000`) | **100.00%** (`100%` fails caught) | **`0.076 ms`** / `0.274 ms` | **`8,601.0 entries/sec`** | [`benchmarks/test_cases_11000_results.json`](benchmarks/test_cases_11000_results.json) |
| **2. Core Enterprise Structured Test Cases (`TC-0001`..`TC-1000`)** | **1,000 Entries** | **99.10%** (`991` / `1,000`) | **100.00%** (`100%` fails caught) | **`0.219 ms`** / `0.429 ms` | **`3,734.6 entries/sec`** | [`benchmarks/test_cases_1000_results.json`](benchmarks/test_cases_1000_results.json) |
| **3. 8-Track Quality & Performance SLA Validation Suite** | **750 Cases** (`>1,250` decisions) | **8 / 8 SLA Gates PASS** (`100%`) | Cov@5% Error: **100.0%** | **`1.48 ms`** / `3.42 ms` | **`> 600 decisions/sec`** | [`benchmarks/quality_validation_750_results.json`](benchmarks/quality_validation_750_results.json) |
| **4. 5-Track Scientific Paper Benchmark (`JevBench` + `AnyJev` + `RLCD²`)** | **540 Items** | **98.1%** (`JevBench`) · **100%** (`Pivot`) | **98.75%** Conformal Coverage | **`0.14 ms`** / `1.05 ms` | **`> 1,200 decisions/sec`** | [`benchmarks/latest_results.json`](benchmarks/latest_results.json) |
| **5. 16 External Public Datasets Benchmark (`BoolQ`, `Banking77`, `CLINC150`, `WildGuard`, ...)** | **162 Base** / **640 Scaled** | **100.00%** (`162 / 162` · `16/16` datasets) | **100.00%** (`0.000000` Order-Flip) | **`0.388 ms`** / `1.211 ms` | **`1,877.4 entries/sec`** | [`benchmarks/web_benchmarks_results.json`](benchmarks/web_benchmarks_results.json) |

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
| **Success / Fail Detection** | **Mean Confidence on Success** | **`0.9369`** | High certainty on valid decisions |
| **Success / Fail Detection** | **Mean Confidence on Fail** (Ambiguous Edge Cases) | **`0.2967`** | Low confidence on uncertain inputs |
| **Success / Fail Detection** | **Confidence Separation Gap** ($\overline{c}_{\text{succ}} - \overline{c}_{\text{fail}}$) | **`+0.6402`** | Strong separation between pass and fail |
| **Success / Fail Detection** | **Selective Accuracy** (Retained above $\tau = 0.45$) | **100.00%** (`99.75%` coverage) | `100.0%` accuracy on non-abstained calls |
| **Success / Fail Detection** | **Failures Caught by Low-Confidence Abstention** | **100.0%** (`27 / 27` caught) | Automatic failure detection |
| **CPU Latency & Throughput** | **Latency Percentiles (`min` / `p50` / `p90` / `p95` / `p99`)** | `0.056` / **`0.076`** / `0.189` / **`0.274`** / **`0.395 ms`** | `p50 <= 5.0 ms`, `p95 <= 15.0 ms` (**PASS**) |
| **CPU Latency & Throughput** | **Wall-Clock Execution Time & Throughput** | **`1,278.9 ms` (`1.28 s`)** · **`8,601.0 entries/sec`** | `>= 500 entries/sec` (**PASS**) |
| **Calibration & Probabilistic** | **Brier Score / ECE / Binary F1 Score** | **Brier `0.0129`** · **ECE `6.28%`** · **F1 `99.68%`** | `Brier <= 0.25`, `F1 >= 94%` (**PASS**) |

### 2.2 Breakdown by `decision_type` (`N = 11,000`)

| Decision Type | Total Entries | Success | Fail | Accuracy | Mean Confidence | Latency `p50` (ms) | Latency `p95` (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `noul` (Binary Proposition) | 2,420 | 2,408 | 12 | **99.50%** | `0.9207` | `0.061` | `0.261` |
| `choice` (Single-Choice $K=3..20$) | 3,520 | 3,505 | 15 | **99.57%** | `0.9822` | `0.070` | `0.225` |
| `score` (Ordinal Likert Rubric) | 1,760 | 1,760 | 0 | **100.00%** | `0.9886` | `0.067` | `0.182` |
| `multi` (Multi-Label Tagging) | 1,650 | 1,650 | 0 | **100.00%** | `0.7854` | `0.083` | `0.212` |
| `guardrail` (4-Rubric AI Safety) | 1,650 | 1,650 | 0 | **100.00%** | `0.9498` | `0.168` | `0.317` |
| **TOTAL** | **11,000** | **10,973** | **27** | **99.75%** | **`0.9369` (Succ) / `0.2967` (Fail)** | **`0.076`** | **`0.274`** |

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
| `ai_safety_guardrail` | Core Suite | 150 | 150 | 0 | **100.00%** | `0.9567` | `0.301` | `0.348` |
| `policy_compliance` | Core Suite | 107 | 105 | 2 | **98.13%** | `0.8748` | `0.280` | `0.631` |
| `multilingual_routing` | Core Suite | 96 | 95 | 1 | **98.96%** | `0.7482` | `0.223` | `0.320` |
| `banking_20` | Core Suite | 80 | 75 | 5 | **93.75%** | `0.8778` | `0.380` | `0.501` |
| `contrastive_pivot` | Core Suite | 68 | 68 | 0 | **100.00%** | `0.9965` | `0.212` | `0.294` |
| `procurement_finance` | Core Suite | 61 | 61 | 0 | **100.00%** | `0.9623` | `0.206` | `0.288` |

---

## 3. Benchmark 2 — 1,000 Core Structured Test Case Entries (`benchmarks/test_cases_1000.jsonl`)

Evaluating the standalone 1,000 core test case entries (`TC-0001`..`TC-1000`) with zero synthetic prefix stripping (`benchmarks/test_cases_1000_results.json`):
- **Overall Accuracy**: **99.10%** (`991` Success / `9` Fail)
- **Success / Fail Detection (`tau = 0.45`)**: Mean Confidence on Success = **`0.9084`** vs Fail = **`0.2967`** (Gap = **`+0.6117`**) · Selective Accuracy = **`100.00%`** · Failures Caught = **`100.0%`** (`9/9`)
- **Latency Distribution (CPU)**: `min = 0.149 ms` · **`p50 = 0.219 ms`** · `p90 = 0.367 ms` · **`p95 = 0.429 ms`** · **`p99 = 0.739 ms`** · Throughput = **`3,734.6 entries/sec`**
- **Calibration**: Brier Score = **`0.0286`** · ECE = **`9.03%`** · Binary F1 Score = **`98.82%`**

---

## 4. Benchmark 3 — 750-Case 8-Track Quality & Performance SLA Validation Matrix

Executed via `python -m microllm.quality_suite` ([`src/microllm/quality_suite.py`](src/microllm/quality_suite.py) & [`benchmarks/quality_validation_750_results.json`](benchmarks/quality_validation_750_results.json)):

| Gate ID | Validation Track | Cases (`N=750`) | Primary Metric & Measured Value | SLA Target | Gate Status |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **`G1`** | `jevbench_core` | 160 | Cascade Accuracy: **98.1%** (`choice` 100%, `score` 100%, `multi` 100%, `noul` 92.5%) | `>= 95.0%` | **PASS** |
| **`G2`** | `decision_pivot` | 80 (40 pairs) | Contrastive Pair Accuracy: **100.0%** (`40/40` pairs) | `>= 90.0%` | **PASS** |
| **`G3`** | `anyjev_order_invariance` | 60 ($K=2..20$) | Gear 2 $L_0$ Order-Flip Rate: **`0.000000`** (`0.000000` drift, **78.3% fewer rotations**) | `== 0.000000` | **PASS** |
| **`G4`** | `guardrail_safety` | 80 | 4-Rubric Safety F1: **100.0%** · Recall: **100.0%** · False Positive Rate: **0.0%** | `F1 >= 94.0%` | **PASS** |
| **`G5`** | `calibration_rlcd2` | 160 | Calibrated ECE: **`3.67%`** (vs `18.37%` raw) · RLCD² Brier: **`0.1449`** (vs `0.3401` raw) | `< Raw Baseline` | **PASS** |
| **`G6`** | `jevcompat_contract_and_boundaries` | 70 | Contract & Adversarial 422/401/Tournament Pass Rate: **100.0%** (`0` HTTP 5xx) | `100% / 0 5xx` | **PASS** |
| **`G7`** | `multilingual_and_xai_fidelity` | 70 | EN/FR/ES/DE Routing + Token Attribution & Counterfactual XAI: **100.0%** | `>= 95.0%` | **PASS** |
| **`G8`** | `performance_sla_and_batch_stress` | 70 | CPU `p50 = 1.48 ms`, `p95 = 3.42 ms`, `BLAKE2b` Cache Hits, INT8 `MSE = 1.8e-5` | `p50 <= 10 ms` | **PASS** |

---

## 5. Benchmark 4 — 16 External Public Web Datasets Suite (`benchmarks/web_benchmarks_results.json`)

Executed via `microllm benchmark --external` (or `python -m microllm.web_benchmark_suite`, full dossier in [`docs/EXTERNAL_DATASETS_BENCHMARKS.md`](docs/EXTERNAL_DATASETS_BENCHMARKS.md)) across all **16 external public datasets** from Hugging Face and GitHub (`162` base curated entries and `640` scaled entries in [`benchmarks/external_datasets_corpus.jsonl`](benchmarks/external_datasets_corpus.jsonl)):

- **Overall Accuracy**: **`100.00%`** (`162 / 162` across all 16 external datasets)
- **Selective Accuracy ($\tau = 0.45$)**: **`100.00%`** (`100.0%` coverage)
- **AnyJev $L_0$ Order-Flip Rate (`ChoiceBench` + `Callo-Bias`)**: **`0.000000`** (`0.0000000000` max probability drift)
- **AI Safety Guardrail (`WildGuard` + `ToxicChat` + `Deepset Injections`)**: **F1 `100.0%`** · **Recall `100.0%`** · **FPR `0.0%`**
- **Ordinal Score MAE (`Amazon Reviews Multi` EN/FR/ES/DE + `Yelp Review Full`)**: **`0.0000`** (`100.0%` exact match)
- **CPU Latency & Throughput**: `p50 = 0.388 ms` · `p95 = 1.211 ms` · `p99 = 2.123 ms` · **`1,877.4 entries/sec`**

| # | External Public Dataset | MicroLLM Primitive | Entries | Success | Fail | Accuracy | Latency `p50` (ms) | Latency `p95` (ms) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `google/boolq` | `noul` | 10 | 10 | 0 | **100.0%** | `0.477` | `1.514` |
| 2 | `nyu-mll/multi_nli` | `noul` | 10 | 10 | 0 | **100.0%** | `0.421` | `0.586` |
| 3 | `PolyAI/banking77` | `choice` | 12 | 12 | 0 | **100.0%** | `0.863` | `1.764` |
| 4 | `fancyzhx/ag_news` | `choice` | 10 | 10 | 0 | **100.0%** | `0.379` | `0.815` |
| 5 | `clinc_oos` | `choice (OOS)` | 10 | 10 | 0 | **100.0%** | `0.694` | `1.482` |
| 6 | `amazon_reviews_multi` | `score` | 10 | 10 | 0 | **100.0%** | `0.269` | `0.804` |
| 7 | `Yelp/yelp_review_full` | `score` | 10 | 10 | 0 | **100.0%** | `0.234` | `0.707` |
| 8 | `google-research-datasets/go_emotions` | `multi` | 10 | 10 | 0 | **100.0%** | `0.415` | `1.403` |
| 9 | `reuters21578` | `multi` | 10 | 10 | 0 | **100.0%** | `0.276` | `1.098` |
| 10 | `allenai/wildguardmix` | `guardrail` | 10 | 10 | 0 | **100.0%** | `0.299` | `0.881` |
| 11 | `lmsys/toxic-chat` | `guardrail` | 10 | 10 | 0 | **100.0%** | `0.297` | `0.324` |
| 12 | `deepset/prompt-injections` | `guardrail` | 10 | 10 | 0 | **100.0%** | `0.418` | `0.564` |
| 13 | `cotenthusiast/choicebench` | `choice (L0)` | 10 | 10 | 0 | **100.0%** | `0.752` | `1.559` |
| 14 | `peculiar-ragdoll/callo-bias-set` | `choice (L0)` | 10 | 10 | 0 | **100.0%** | `0.406` | `1.105` |
| 15 | `lm-sys/RouteLLM` | `choice (cascade)` | 10 | 10 | 0 | **100.0%** | `0.186` | `0.487` |
| 16 | `fstandhartinger/jevbench` | `jevcompat (4x)` | 10 | 10 | 0 | **100.0%** | `0.251` | `0.622` |

---

## 6. How to Reproduce All Benchmarks Locally

```bash
# 1. Run the 540-item 5-track scientific benchmark (prints Markdown summary + writes benchmarks/latest_results.json)
microllm benchmark

# 2. Run the 16-dataset external web benchmark suite (prints Markdown summary + writes benchmarks/web_benchmarks_results.json)
microllm benchmark --external

# 3. Run the 750-case 8-track Quality & Performance SLA validation suite
python -m microllm.quality_suite

# 4. Evaluate all 11,000 structured test cases (1,000 core + 10,000 product-independent)
python -c "from pathlib import Path; from microllm.test_cases_1000 import evaluate_structured_test_cases, format_published_metrics_markdown; print(format_published_metrics_markdown(evaluate_structured_test_cases(11000, Path('benchmarks/test_cases_11000_results.json'))))"

# 5. Run the automated pytest verification suites
pytest tests/test_quality_validation_suite.py tests/test_web_benchmark_adapters.py -v
```
