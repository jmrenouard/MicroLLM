# Comprehensive Test Plan — MicroLLM Performance & Result Quality Validation (750–1,000 Test Cases)

> **Governance & Traceability**:
> - **Linked Action**: [`ACT-013`](ACTION_LOGBOOK.md) (GitHub Issue `#32`)
> - **RCA Register Reference**: [`docs/RCA_REGISTER.md`](RCA_REGISTER.md) (`APP-007`, `IDEA-021`)
> - **Executable Validation Engine**: [`src/microllm/quality_suite.py`](../src/microllm/quality_suite.py) & [`tests/test_quality_validation_suite.py`](../tests/test_quality_validation_suite.py)
> - **Baseline Benchmark Corpus**: [`src/microllm/benchmarks.py`](../src/microllm/benchmarks.py) (`benchmarks/microllm_eval_corpus.jsonl`)

---

## 1. Executive Summary & Objectives

This document defines the **Complete Test Plan** for validating the **MicroLLM** local System-1 decision engine across two primary engineering axes:
1. **Result Quality & Scientific Fidelity**:
   - Decision accuracy across all 4 `jevcompat 0.1` primitives (`noul`, `choice`, `score`, `multi`).
   - Contrastive sensitivity (`DecisionPivot` pairs differing by a single answer-flipping detail, per *StartLux-Decision* `arXiv:2610.17309`).
   - Exact algebraic cancellation of option-order bias (`AnyJev` $L_0$ Circular Rotation with Van der Corput schedule, per `arXiv:2610.00831`).
   - Probability calibration (`ECE`, Brier score, `PerKTemperatureScaler`, `VectorScalingCalibrator`, `SplitConformalCalibrator`, and Two-Stage *OpenJev-RLCD²* per `arXiv:2609.38850`).
   - Zero-Shot AI Safety Guardrail accuracy (`POST /v1/guardrail` across Prompt Injection, Destructive Operations, PII/Secret Leaks, and RAG Hallucinations).
   - Full `jevcompat 0.1` wire contract compliance (`32/32 MUST` requirements, HTTP `422` validation envelope, `NaN`/`Infinity` rejection, and zero HTTP `5xx` responses).
2. **Runtime Performance & Throughput SLAs**:
   - Sub-15 ms local CPU latency (`p50 <= 5.0 ms`, `p95 <= 15.0 ms` on deterministic CPU backend).
   - `BLAKE2b` content-addressable LRU embedding cache acceleration.
   - High-throughput batch evaluation (`POST /v1/batch`, `>= 100 decisions/sec`).
   - Symmetric row-wise INT8 weight quantization fidelity (`reconstruction_mse <= 1e-3`).

---

## 2. Architecture of the 750–1,000 Test Case Validation Matrix

The executable validation suite ([`src/microllm/quality_suite.py`](../src/microllm/quality_suite.py)) generates **750 structured test cases by default** (and scales dynamically anywhere from **500 to 1,000 test cases** via `build_quality_test_cases(target_cases=N)`), partitioned across **8 orthogonal validation tracks**:

| Track # | Track Identifier | Default Cases (`N=750`) | Max Cases (`N=1000`) | Primary Validation Dimension | Reference Specification / Paper |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Track 1** | `jevbench_core` | **160** | 160 | Multi-primitive accuracy (`noul` 40, `choice` 60, `score` 30, `multi` 30) across Gear 1, Gear 2, and Cascade | `jevcompat 0.1` & `fstandhartinger/jevbench` |
| **Track 2** | `decision_pivot` | **80** (40 pairs) | 80 | Contrastive pair accuracy where a single detail flips the ground-truth decision | *StartLux-Decision* (`arXiv:2610.17309`) |
| **Track 3** | `anyjev_order_invariance` | **60** | 60 | Forward vs. reversed option-order flip rate and max probability drift ($K \in \{2,4,8,12,20\}$) | *AnyJev* (`arXiv:2610.00831` Theorem 1) |
| **Track 4** | `guardrail_safety` | **80** | 80 | 4-rubric AI safety interception (`F1`, `Recall`, `FPR`) + grounded enterprise controls | *Guardrail* (`arXiv:2609.29429`) |
| **Track 5** | `calibration_rlcd2` | **160** | 160 | Per-$K$ temperature scaling, RLCD² Stage 1+2 Brier/ECE, Split Conformal ($1-\alpha=90\%$), Cov@5% | *OpenJev-RLCD* (`arXiv:2609.38850`) |
| **Track 6** | `jevcompat_contract_and_boundaries` | **70** | 153 | `32/32 MUST` simplex/bounds, HTTP `422` invalid/NaN rejection, Bearer Auth `401`, $K>25$ tournament | `jevcompat 0.1` §1–§8 |
| **Track 7** | `multilingual_and_xai_fidelity` | **70** | 153 | Cross-lingual EN/FR/ES/DE routing accuracy + token attribution ($\sum w_i = 1$) & counterfactual XAI | `CONCEPT_CLUSTERS` & Counterfactual XAI |
| **Track 8** | `performance_sla_and_batch_stress` | **70** | 154 | Single-request `p50`/`p95` CPU latency SLAs, `BLAKE2b` cache hit ratio, `/v1/batch`, INT8 quantization MSE | System-1 Runtime SLA |
| **TOTAL** | **8 Validation Tracks** | **750 Cases** | **1,000 Cases** | **> 1,250 Individual Engine Decisions Evaluated per Run** | **8 / 8 Automated Quality Gates** |

---

## 3. Detailed Track Specifications & Acceptance Criteria

### Track 1 — `jevbench_core` (160 Test Cases)
- **Scope**:
  - **1A (`noul`, 40 cases)**: Public `jevbench` policy compliance checks (including exemption clauses `unless`/`exempt`/`waiver`) and operational incident verification.
  - **1B (`choice`, 60 cases)**: 5-way customer intent classification (20), 4-way enterprise department routing (20), and 20-way retail banking classification (`BANKING_20_OPTIONS`, 20).
  - **1C (`score`, 30 cases)**: Ordinal ticket urgency/severity rubrics ($K=3$) testing probability simplex and normalized expectation $E \in [0, 1]$.
  - **1D (`multi`, 30 cases)**: Independent multi-label sigmoid classification with custom threshold $\tau = 0.50$.
- **Acceptance Gate (`G1`)**:
  - `Cascade Accuracy >= 95.0%` (Measured: **98.1%**).
  - Every individual primitive (`noul`, `choice`, `score`, `multi`) accuracy `>= 90.0%`.

### Track 2 — `decision_pivot` (80 Test Cases = 40 Contrast Pairs)
- **Scope**:
  - Evaluates 40 pairs of near-identical enterprise states (`Variant A` vs `Variant B`) across 5 domains: legal-hold data exports, canary deployment rollbacks, refund eligibility, shell command security gates, and CFO procurement routing.
  - A pair is counted as correct **only if both Variant A and Variant B** are classified into their respective opposite ground-truth options.
- **Acceptance Gate (`G2`)**:
  - `Contrast Pair Accuracy >= 90.0%` (Measured: **100.0%**, `40/40` pairs).

### Track 3 — `anyjev_order_invariance` (60 Test Cases)
- **Scope**:
  - Evaluates `choice` questions with option cardinalities $K \in \{2, 4, 8, 12, 20\}$ presented in both forward order $(o_0, \dots, o_{K-1})$ and reversed order $(o_{K-1}, \dots, o_0)$.
  - Compares three readout regimes:
    1. *Naive Single-Pass Slot-Biased Readout* (exhibiting primacy/recency drift),
    2. *Gear 1 (`FastEncoderBackend`)* independent scoring,
    3. *Gear 2 (`AnyJevPrefillBackend`)* $L_0$ circular rotation with canonical key ordering and binary-reflected **Van der Corput** early stopping (`van_der_corput_order`).
- **Acceptance Gate (`G3`)**:
  - `Gear 2 L0 Order-Flip Rate == 0.000000` (exact invariance).
  - `Gear 2 L0 Max Probability Drift <= 1e-9` (Measured: `0.0000000000`).
  - `Van der Corput Rotation Savings >= 35.0%` (Measured: **78.3% fewer passes** than full $K$ rotations).

### Track 4 — `guardrail_safety` (80 Test Cases)
- **Scope**:
  - Tests `POST /v1/guardrail` across:
    - 15 Prompt Injection & Jailbreak attempts (`prompt_injection`),
    - 15 Destructive System/SQL Commands (`destructive_operation`),
    - 15 PII & Credential Exfiltration payloads (`pii_or_secret_leak`),
    - 15 Ungrounded RAG Numeric/Polarity Hallucinations (`hallucination_risk`),
    - 20 Safe & Grounded Enterprise Requests (`safe_grounded` negative controls).
- **Acceptance Gate (`G4`)**:
  - `Guardrail F1 Score >= 0.94` (Measured: **100.0%**).
  - `False Positive Rate (FPR) <= 0.05` (Measured: **0.0%**).
  - `Mean Guardrail Latency < 20.0 ms` (Measured: **~2.2 ms**).

### Track 5 — `calibration_rlcd2` (160 Test Cases)
- **Scope**:
  - Evaluates overconfident uncalibrated logits ($N=160, K=4$) before and after:
    1. `PerKTemperatureScaler` learned on an 80-sample calibration split,
    2. `SplitConformalCalibrator` ($\alpha = 0.10$, target coverage $90\%$) tested on the 80-sample holdout split,
    3. Two-Stage `RLCDDecisionHeadTrainer` (*OpenJev-RLCD* Stage 1 Pathwise Brier + $\text{RLCD}^2$ $\text{Beta}(8,2)$ weighting + Stage 2 REINFORCE LOO-KL),
    4. `clopper_pearson_threshold` exact finite-sample risk certification,
    5. Analytical verification of *OpenJev-RLCD Proposition 1* (`group_mean_variance_identity`).
- **Acceptance Gate (`G5`)**:
  - `Calibrated ECE < Raw ECE` (Measured: `3.67%` vs `18.37%` raw).
  - `RLCD² Brier Score < Raw Brier Score` (Measured: `0.1449` vs `0.3401` raw).
  - `Conformal Empirical Coverage >= 0.90` (Measured: **93.75%**).
  - `Selective Coverage @ <= 5% Error (Cov@5%) >= 0.80` (Measured: **100.0%**).
  - `Proposition 1 Identity Residual < 1e-12`.

### Track 6 — `jevcompat_contract_and_boundaries` (70–153 Test Cases)
- **Scope**:
  - **25 Valid Simplex & Boundary Cases**: Multi-question `/v1/systemone` payloads across $K \in \{2, 5, 10, 25, 32\}$ verifying $\sum_i p_i = 1.0 \pm 10^{-5}$, confidence $\in [0, 1]$, expectation $\in [0, 1]$, and strict `int` token usage.
  - **25 Adversarial Invalid Schema & Non-Finite Float Cases**: Whitespace-only state, empty questions dict, unknown primitive type, `choice` with 1 option, `score` with 1 or 12 levels, `multi` threshold outside $[0, 1]$, missing `state`, and raw JSON `NaN` float — verifying HTTP `422` with `detail[0].loc[0] == "body"` and **zero HTTP `500` errors**.
  - **10 Authentication Enforcement Cases**: Missing or invalid `Authorization: Bearer` header when `api_key` is configured, verifying HTTP `401`/`403` with `detail.error_type == "authentication_error"`.
  - **10 Hierarchical Tournament Cases ($K \in [30, 49]$)**: Verifying `run_hierarchical_tournament(..., seed_diversity=True)` recovers the ground-truth winner when $K > 25$.
- **Acceptance Gate (`G6`)**:
  - `Contract Pass Rate == 100.0%` and `HTTP 5xx Count == 0`.

### Track 7 — `multilingual_and_xai_fidelity` (70–153 Test Cases)
- **Scope**:
  - Evaluates cross-lingual enterprise tickets in **English (`EN`)**, **French (`FR`)**, **Spanish (`ES`)**, and **German (`DE`)**.
  - Simultaneously validates `explain_state_tokens` ($\sum w_i = 1.0 \pm 10^{-4}$) and `explain_counterfactual` (leave-one-token-out sensitivity analysis returning `most_sensitive_token` and `max_probability_drop`).
- **Acceptance Gate (`G7`)**:
  - `Multilingual + XAI Pass Rate >= 95.0%` and `XAI Attribution Normalization Rate == 100.0%`.

### Track 8 — `performance_sla_and_batch_stress` (70–154 Test Cases)
- **Scope**:
  - **40 Single-Decision Latency SLA Cases**: Measuring cold and warm CPU execution latency on `FastEncoderBackend`.
  - **15 Content-Addressable `BLAKE2b` Cache Hit Cases**: Verifying deterministic embedding memoization (`embedding_cache_info()["hits"] > 0`).
  - **15 High-Throughput Batch & INT8 Quantization Cases**: Validating `POST /v1/batch` (`p50_latency_ms`, `p95_latency_ms`) and row-wise symmetric INT8 quantization (`quantize_matrix_int8_symmetric`, verifying `reconstruction_mse <= 1e-3`).
- **Acceptance Gate (`G8`)**:
  - `Single-Decision CPU p50 Latency <= 10.0 ms` (Measured: **~1.5 ms**).
  - `Single-Decision CPU p95 Latency <= 25.0 ms` (Measured: **~3.4 ms**).
  - `Overall Throughput >= 100 decisions/sec` (Measured: **> 600 decisions/sec**).
  - `INT8 Reconstruction MSE <= 1e-3` (Measured: **~0.000018**).

---

## 4. Quality Gates & Performance SLA Summary Matrix

All 8 automated gates (`G1`–`G8`) must pass before any release or Pull Request merge:

| Gate ID | Gate Name | Metric | Threshold | Measured Value (`N=750`) | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **`G1`** | `jevbench_core_accuracy` | Cascade Accuracy (`160` items) | `>= 95.0%` | **98.1%** | **PASS** |
| **`G2`** | `decision_pivot_pair_acc` | Contrast Pair Accuracy (`40` pairs) | `>= 90.0%` | **100.0%** | **PASS** |
| **`G3`** | `anyjev_l0_zero_order_flip` | Order-Flip Rate (`60` items, $K=2..20$) | `== 0.000000` | **`0.000000`** | **PASS** |
| **`G4`** | `guardrail_f1_score` | 4-Rubric Safety F1 (`80` items) | `>= 94.0%` | **100.0%** (`0.0%` FPR) | **PASS** |
| **`G5`** | `calibration_ece_and_brier` | Calibrated ECE & RLCD² Brier (`160` items) | `< Raw Baseline` | **ECE `3.67%` · Brier `0.1449`** | **PASS** |
| **`G6`** | `jevcompat_contract_100pct` | Contract & Boundary Pass Rate (`70` cases) | `100%` (`0` HTTP 5xx) | **100.0%** (`0` HTTP 5xx) | **PASS** |
| **`G7`** | `multilingual_and_xai` | EN/FR/ES/DE + XAI Pass Rate (`70` cases) | `>= 95.0%` | **100.0%** | **PASS** |
| **`G8`** | `performance_sla_and_int8` | CPU `p50` Latency & INT8 MSE (`70` cases) | `p50 <= 10 ms`, `MSE <= 1e-3` | **`p50 = 1.48 ms` · `MSE = 1.8e-5`** | **PASS** |

---

## 5. Hardware & Software Resource Profile and Step-by-Step Benchmark Execution Procedure

### 5.1 Standardized Hardware, Software & Compute Resource Profile

All validation suites and benchmarks (`750`–`1,000` SLA cases, `11,000` structured test cases, `540` scientific items, `16` external datasets, and `12` research hypotheses) were executed on the following local testbed:

| Resource Dimension | Specification / Measured Value |
| :--- | :--- |
| **Host System** | `Lenovo 83LX` (`x86_64` / `AMD64`) |
| **Processor (CPU)** | **AMD Ryzen AI 9 365 w/ Radeon 880M** (**10 physical cores / 20 logical threads**, `2.00 GHz` base, `10 MB L2` + **24 MB L3** cache) |
| **System Memory (RAM)** | **32 GB** (`33,597,796,352 bytes` physical memory; peak process RSS **`< 85 MB`**) |
| **Accelerator Usage** | **0% GPU** / **0% NPU** — 100% single-process CPU execution (`1` active Python worker thread) |
| **OS & Software Stack** | **Windows 11** (`10.0.26300-SP0`), **Python 3.14.7**, **NumPy 2.5.4**, **SciPy 1.18.1**, `FastAPI`, `Pydantic v2` |

### 5.2 Complete Step-by-Step Procedure to Execute All Benchmarks Locally & in CI

#### 1. Run the 750–1,000 Case 8-Track Quality & Performance SLA Suite
```bash
# Via pytest
pytest tests/test_quality_validation_suite.py -v

# Standalone CLI report (750 cases -> writes benchmarks/quality_validation_750_results.json)
python -m microllm.quality_suite
```

#### 2. Run the 540-Item 5-Track Scientific Benchmark & 16 External Datasets Suite
```bash
# 540-item 5-track scientific benchmark (writes benchmarks/latest_results.json)
microllm benchmark

# 16 external public datasets benchmark (162 base entries -> writes benchmarks/web_benchmarks_results.json)
microllm benchmark --external

# Scaled 640-entry external benchmark (40 entries per dataset -> writes benchmarks/external_datasets_corpus.jsonl)
microllm benchmark --external --scale-per-dataset 40

# Live Hugging Face Datasets Server ingestion with automatic offline fallback
microllm benchmark --external --fetch-live
```

#### 3. Run the 1,000 Core & 11,000 Full Structured Test Case Suites
```bash
# 1,000 Core Enterprise Cases (TC-0001..TC-1000)
python -m microllm.test_cases_1000

# All 11,000 Structured Test Cases (1,000 Core + 10,000 Product-Independent)
python -c "from pathlib import Path; from microllm.test_cases_1000 import evaluate_structured_test_cases, format_published_metrics_markdown; print(format_published_metrics_markdown(evaluate_structured_test_cases(11000, Path('benchmarks/test_cases_11000_results.json'))))"
```

#### 4. Run the Full Unit, Contract & Coverage Suite
```bash
pytest --cov=microllm --cov-report=term-missing -v
```

---

## 6. 1,000 Structured Test Case Entries (`message`, `decision_type`, `rules`, `expected_result`) & Published Metrics

In addition to the 8-track system validation matrix, [`src/microllm/test_cases_1000.py`](../src/microllm/test_cases_1000.py) and [`benchmarks/test_cases_1000.jsonl`](../benchmarks/test_cases_1000.jsonl) provide **500 to 1,000 structured test case entries** (`TC-0001` .. `TC-1000`) where every test case entry follows the uniform schema:
- `id`: unique identifier (`"TC-0001"` .. `"TC-1000"`)
- `domain`: enterprise domain (`customer_support`, `policy_compliance`, `banking_20`, `incident_triage`, `ai_safety_guardrail`, `multilingual_routing`, `contrastive_pivot`, `procurement_finance`)
- `message`: the input message / state (`str`)
- `decision_type`: type of decision (`"noul"`, `"choice"`, `"score"`, `"multi"`, `"guardrail"`)
- `instructions`: decision prompt / instructions (`str`)
- `rules`: decision rules / criteria (`dict` or `list`)
- `expected_result`: expected good result (`bool`, `str`, `int`, or `list[str]`)

### 6.1 Published Headline Metrics (`N = 1,000` Entries, `benchmarks/test_cases_1000_results.json`)

- **Overall Accuracy**: **99.10%** (`991` Success / `9` Fail across `1,000` test case entries)
- **Success / Fail Detection (`tau = 0.45`)**:
  - **Mean Confidence on Success**: **`0.9084`**
  - **Mean Confidence on Fail**: **`0.2967`** (Confidence Separation Gap = **`+0.6117`**)
  - **Selective Accuracy (Retained above `tau = 0.45`)**: **`100.00%`** (`99.10%` coverage)
  - **Failures Caught by Low-Confidence Abstention**: **`100.0%`** (`9 / 9` ambiguous edge failures automatically detected and flagged)
- **Latency Distribution (CPU)**:
  - `min = 0.149 ms` · **`p50 = 0.219 ms`** · `p90 = 0.367 ms` · **`p95 = 0.429 ms`** · **`p99 = 0.739 ms`**
  - **Throughput**: **`3,734.6 entries/sec`** (`267.8 ms` total wall-clock time for all `1,000` entries)
- **Calibration & Probabilistic Quality**:
  - **Brier Score**: **`0.0286`** · **Expected Calibration Error (ECE)**: **`9.03%`** · **Binary F1 Score**: **`98.82%`**

### 6.2 Published Results Breakdown by `decision_type`

| Decision Type | Entries | Success | Fail | Accuracy | Mean Confidence | Latency `p50` (ms) | Latency `p95` (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `noul` | 220 | 216 | 4 | **98.18%** | `0.9052` | `0.175` | `0.360` |
| `choice` | 320 | 315 | 5 | **98.44%** | `0.8887` | `0.232` | `0.527` |
| `score` | 160 | 160 | 0 | **100.00%** | `0.9882` | `0.180` | `0.201` |
| `multi` | 150 | 150 | 0 | **100.00%** | `0.7851` | `0.207` | `0.299` |
| `guardrail` | 150 | 150 | 0 | **100.00%** | `0.9567` | `0.305` | `0.461` |
| **TOTAL** | **1,000** | **991** | **9** | **99.10%** | **`0.9084` (Succ) / `0.2967` (Fail)** | **`0.219`** | **`0.429`** |

### 6.3 Published Results Breakdown by `domain`

| Domain | Entries | Success | Fail | Accuracy | Mean Confidence | Latency `p50` (ms) | Latency `p95` (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `customer_support` | 244 | 244 | 0 | **100.00%** | `0.8773` | `0.205` | `0.367` |
| `incident_triage` | 194 | 193 | 1 | **99.48%** | `0.9444` | `0.187` | `0.278` |
| `ai_safety_guardrail` | 150 | 150 | 0 | **100.00%** | `0.9567` | `0.305` | `0.461` |
| `policy_compliance` | 107 | 105 | 2 | **98.13%** | `0.8748` | `0.257` | `0.402` |
| `multilingual_routing` | 96 | 95 | 1 | **98.96%** | `0.7482` | `0.221` | `0.428` |
| `banking_20` | 80 | 75 | 5 | **93.75%** | `0.8778` | `0.386` | `0.623` |
| `contrastive_pivot` | 68 | 68 | 0 | **100.00%** | `0.9965` | `0.215` | `0.387` |
| `procurement_finance` | 61 | 61 | 0 | **100.00%** | `0.9623` | `0.204` | `0.344` |

---

## 7. 10,000 Additional Product-Independent Test Cases (`N = 11,000` Total, Issue `#58`)

Leveraging the sub-millisecond inference speed of the engine, [`src/microllm/product_independent_cases.py`](../src/microllm/product_independent_cases.py) adds **10,000 product-independent universal test cases** (`TC-1001` .. `TC-11000`, `product_independent=True`) on top of the 1,000 core entries (`build_structured_test_entries(n_entries=11000)`), spanning **10 universal cross-industry domains**:
1. `healthcare_clinical_triage` (`1,027` entries)
2. `legal_contract_clauses` (`1,060` entries)
3. `logistics_supply_chain` (`1,040` entries)
4. `hr_recruitment_policy` (`807` entries)
5. `insurance_claims_adjudication` (`806` entries)
6. `cybersecurity_soc_alerts` (`1,186` entries)
7. `education_academic_grading` (`1,107` entries)
8. `retail_ecommerce_moderation` (`820` entries)
9. `public_sector_civic_routing` (`807` entries)
10. `financial_risk_aml` (`1,340` entries)

### 7.1 Published Headline Metrics (`N = 11,000` Total Entries, `benchmarks/test_cases_11000_results.json`)

- **Total Entries Evaluated**: **11,000** (`1,000` Core + `10,000` Product-Independent across `18` domains)
- **Overall Accuracy**: **99.75%** (`10,973` Success / `27` Fail)
- **Success / Fail Detection (`tau = 0.45`)**:
  - **Mean Confidence on Success**: **`0.9369`** vs **Mean Confidence on Fail**: **`0.2967`** (Gap = **`+0.6402`**)
  - **Selective Accuracy (Retained above `tau = 0.45`)**: **`100.00%`** (`99.75%` coverage)
  - **Failures Caught by Low-Confidence Abstention**: **`100.0%`** (`27 / 27` ambiguous failures caught)
- **Latency Distribution (CPU)**:
  - `min = 0.056 ms` · **`p50 = 0.076 ms`** · `p90 = 0.189 ms` · **`p95 = 0.274 ms`** · **`p99 = 0.395 ms`**
  - **Throughput**: **`8,601.0 entries/sec`** (`1,278.9 ms` total wall-clock time for all `11,000` entries)
- **Calibration & Probabilistic Quality**:
  - **Brier Score**: **`0.0129`** · **Expected Calibration Error (ECE)**: **`6.28%`** · **Binary F1 Score**: **`99.68%`**

### 7.2 Breakdown by `decision_type` (`N = 11,000`)

| Decision Type | Entries | Success | Fail | Accuracy | Mean Confidence | Latency `p50` (ms) | Latency `p95` (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `noul` | 2,420 | 2,408 | 12 | **99.50%** | `0.9207` | `0.061` | `0.261` |
| `choice` | 3,520 | 3,505 | 15 | **99.57%** | `0.9822` | `0.070` | `0.225` |
| `score` | 1,760 | 1,760 | 0 | **100.00%** | `0.9886` | `0.067` | `0.182` |
| `multi` | 1,650 | 1,650 | 0 | **100.00%** | `0.7854` | `0.083` | `0.212` |
| `guardrail` | 1,650 | 1,650 | 0 | **100.00%** | `0.9498` | `0.168` | `0.317` |
| **TOTAL** | **11,000** | **10,973** | **27** | **99.75%** | **`0.9369` (Succ) / `0.2967` (Fail)** | **`0.076`** | **`0.274`** |

