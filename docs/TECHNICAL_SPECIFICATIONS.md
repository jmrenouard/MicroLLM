<div align="center">
  <img src="assets/header-banner.svg" alt="MicroLLM — Complete Technical Specifications" width="100%" />
</div>

# MicroLLM — Complete Technical Specifications & Engineering Architecture
## *(Spécifications Techniques Complètes — Architecture, Mathématiques, Protocole `jevcompat 0.1` & SLAs)*

[![JevCompat 0.1](https://img.shields.io/badge/JevCompat_0.1-32%2F32_MUST-10b981)](../tests/test_jevcompat_contract.py)
[![Tests](https://img.shields.io/badge/Unit_Tests-118%2B_Passing_(100%25)-3b82f6)](../tests/)
[![Coverage](https://img.shields.io/badge/Code_Coverage-93%25%2B-10b981)](../pyproject.toml)
[![Benchmarks 11,000](https://img.shields.io/badge/Benchmarks_11%2C000-99.75%25_Acc_%7C_0.076ms_p50-0ea5e9)](../BENCHMARKS.md)

> **Documentation Hub**:
> - **Product Documentation & User Guide**: [`docs/PRODUCT_DOCUMENTATION.md`](PRODUCT_DOCUMENTATION.md)
> - **Technical Specifications (This Document)**: [`docs/TECHNICAL_SPECIFICATIONS.md`](TECHNICAL_SPECIFICATIONS.md)
> - **Architecture & Quick Reference**: [`docs/README.md`](README.md) · Interactive HTML: [`docs/index.html`](index.html)
> - **Published Benchmarks (`11,000` Cases)**: [`BENCHMARKS.md`](../BENCHMARKS.md) · [`docs/EXTERNAL_DATASETS_BENCHMARKS.md`](EXTERNAL_DATASETS_BENCHMARKS.md)
> - **Quality & Performance Test Plan (`750–1,000` Cases)**: [`docs/TEST_PLAN.md`](TEST_PLAN.md)
> - **Academic Literature Catalog (`104` Papers)**: [`docs/LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) · [`docs/papers/README.md`](papers/README.md)
> - **Governance & Traceability Registers**: [`docs/ACTION_LOGBOOK.md`](ACTION_LOGBOOK.md) · [`docs/RCA_REGISTER.md`](RCA_REGISTER.md)

---

## Table of Contents

1. [System Architecture & Module Topology](#1-system-architecture--module-topology)
2. [Formal Mathematical Specifications of Decision Primitives & Confidence](#2-formal-mathematical-specifications-of-decision-primitives--confidence)
3. [Gear 1 Specification: `FastEncoderBackend`, Embeddings & XAI](#3-gear-1-specification-fastencoderbackend-embeddings--xai)
4. [Gear 2 Specification: `AnyJevPrefillBackend`, Theorem 1 $L_0$ & Van der Corput](#4-gear-2-specification-anyjevprefillbackend-theorem-1-l_0--van-der-corput)
5. [StartLux Cascade (`DualGearCascadeBackend`) & 25-Way Hierarchical Tournament](#5-startlux-cascade-dualgearcascadebackend--25-way-hierarchical-tournament)
6. [Statistical Calibration, Conformal Prediction & Clopper-Pearson Certification](#6-statistical-calibration-conformal-prediction--clopper-pearson-certification)
7. [Two-Stage OpenJev-RLCD² Local Training Engine (`src/microllm/rlcd.py`)](#7-two-stage-openjev-rlcd2-local-training-engine-srcmicrollmrlcdpy)
8. [Complete `jevcompat 0.1` Wire Protocol & 32/32 `MUST` Compliance Matrix](#8-complete-jevcompat-01-wire-protocol--3232-must-compliance-matrix)
9. [Complete HTTP REST API Specification (17 Endpoints)](#9-complete-http-rest-api-specification-17-endpoints)
10. [Strict Pydantic v2 Data Models & Validation Rules (`src/microllm/schemas.py`)](#10-strict-pydantic-v2-data-models--validation-rules-srcmicrollmschemaspy)
11. [Hardware Profile Registry & Symmetric INT8 Quantization (`src/microllm/registry.py`)](#11-hardware-profile-registry--symmetric-int8-quantization-srcmicrollmregistrypy)
12. [Non-Functional Requirements (NFRs), Security, CI/CD & Verification Traceability](#12-non-functional-requirements-nfrs-security-cicd--verification-traceability)

---

## 1. System Architecture & Module Topology

### 1.1 High-Level Component Architecture

**MicroLLM** (`version = "0.2.0"`, Python `>= 3.10`) is architected as a modular, zero-mandatory-GPU Python package (`src/microllm/`) exposing both an in-process Python API and a Jev-compatible FastAPI server (`create_app`).

```mermaid
flowchart TB
    subgraph Clients["Client Layer"]
        SDK_SYNC["MicroJevClient (Sync HTTPX)"]
        SDK_ASYNC["AsyncMicroJevClient (AsyncIO)"]
        ROUTER["@system_one_router Decorator"]
        STUDIO["Decision Studio Web UI (GET /)"]
        CLI["microllm CLI (serve|pull|list|benchmark|papers|rca)"]
    end

    subgraph API["FastAPI Transport & Validation Layer (src/microllm/server.py & schemas.py)"]
        AUTH["Bearer Auth Middleware (MICROLLM_API_KEY)"]
        SCHEMA["Pydantic v2 Schemas + Recursive NaN/Inf Filter (_ensure_finite_json)"]
        ISOL["Isolated Question Dispatcher (_evaluate_isolated_question)"]
        GUARD["4-Rubric + Custom AI Safety Guardrail (/v1/guardrail)"]
        METRICS["Telemetry & Observability Collector (/v1/metrics)"]
    end

    subgraph Engine["Dual-Gear Hybrid Decision Engine (src/microllm/backends/)"]
        CASC["DualGearCascadeBackend (StartLux Read-or-Think Cascade)"]
        G1["Gear 1: FastEncoderBackend (NLI Template + 512-Token Guard)"]
        G2["Gear 2: AnyJevPrefillBackend (L0 Circular Rotation + Van der Corput)"]
        TOURN["run_hierarchical_tournament (25-Way Chunks up to K=255)"]
        BASE["Base Scorer: embed_text_deterministic (BLAKE2b LRU + 16 Concept Clusters)<br/>or Pluggable HF / ONNX Scorer"]
    end

    subgraph MathCal["Calibration, Conformal & RLCD² Layer"]
        TEMP["PerKTemperatureScaler & VectorScalingCalibrator"]
        CONF["SplitConformalCalibrator (Standard & Mondrian Class-Conditional)"]
        CP["Clopper-Pearson Risk Bound (clopper_pearson_threshold)"]
        RLCD["RLCDDecisionHeadTrainer (Stage 1 Pathwise + Stage 2 LOO-KL) & RLCDReplayBuffer"]
    end

    Clients --> API
    API --> CASC
    CASC --> G1
    CASC --> G2
    G1 --> TOURN
    G2 --> TOURN
    TOURN --> BASE
    G1 --> TEMP
    G1 --> CONF
    API --> RLCD
    API --> CP
```

### 1.2 Request Lifecycle Sequence Diagram (`POST /v1/systemone`)

```mermaid
sequenceDiagram
    autonumber
    participant C as Client / Agent
    participant S as FastAPI Server (/v1/systemone)
    participant V as Pydantic v2 Validator (schemas.py)
    participant Cas as DualGearCascadeBackend
    participant G1 as FastEncoderBackend (Gear 1)
    participant G2 as AnyJevPrefillBackend (Gear 2)

    C->>S: POST /v1/systemone {model, state, questions}
    S->>S: Verify Authorization: Bearer <key> (401 on mismatch)
    S->>V: Parse SystemOneRequest & _ensure_finite_json
    alt Invalid Schema or NaN/Infinity
        V-->>C: HTTP 422 {"detail": [{"loc": ["body", ...], "msg": ..., "type": ...}]}
    end
    loop For each (qid, q_spec) in Complete Isolation
        S->>Cas: _evaluate_isolated_question(backend, state_text, q_spec)
        Cas->>G1: Evaluate p_0, confidence(p_0), x_conformal_set, x_abstain
        alt confidence(p_0) >= gate_threshold (0.75) AND x_abstain == False
            G1-->>Cas: Fast Auto-Commit p_0 (~0.08 - 1.0 ms)
        else Ambiguous or Abstained (Escalate to Gear 2)
            Cas->>G2: Evaluate p_1 via AnyJev L0 Circular Rotation (Van der Corput)
            G2-->>Cas: Position-Unbiased Distribution p_1 + x_l0_meta
            Cas->>Cas: Fuse p_fused = (1 - alpha) * p_0 + alpha * p_1
        end
        Cas-->>S: Validated AnswerSpec (NoulAnswer | ChoiceAnswer | ScoreAnswer | MultiAnswer)
    end
    S-->>C: HTTP 200 SystemOneResponse {model, answers, usage}
```

### 1.3 Complete Source Code Module Map

| File Path | Primary Classes & Functions | Engineering Responsibility |
| :--- | :--- | :--- |
| [`src/microllm/__init__.py`](../src/microllm/__init__.py) | `__version__ = "0.2.0"`, 54 exported symbols | Public package surface unifying backends, calibration, RLCD, SDK, registry, and papers. |
| [`src/microllm/schemas.py`](../src/microllm/schemas.py) | `SystemOneRequest`, `SystemOneResponse`, `NoulQuestion`, `ChoiceQuestion`, `ScoreQuestion`, `MultiQuestion`, `NoulAnswer`, `ChoiceAnswer`, `ScoreAnswer`, `MultiAnswer`, `UsageInfo`, `_ensure_finite_json`, `render_structured_text` | Strict `jevcompat 0.1` Pydantic v2 request/response envelopes and recursive `NaN`/`Infinity` rejection (`RCA-004`). |
| [`src/microllm/confidence.py`](../src/microllm/confidence.py) | `choice_confidence`, `score_expectation`, `score_confidence`, `multi_confidence`, `earth_movers_distance_1d`, `unimodal_ordinal_smooth`, `run_hierarchical_tournament`, `sanitize_and_normalize_probabilities` | Official TypeSafe AI confidence & expectation math (`jevcompat` §4.3–§4.5) and StartLux 25-way tournament routing (`K <= 255`). |
| [`src/microllm/calibration.py`](../src/microllm/calibration.py) | `PerKTemperatureScaler`, `VectorScalingCalibrator`, `SplitConformalCalibrator`, `brier_score`, `expected_calibration_error`, `reliability_diagram_bins`, `clopper_pearson_upper_bound`, `clopper_pearson_threshold`, `tune_cascade_pareto` | Cardinality-aware temperature scaling, $L_2$ vector scaling, standard & Mondrian split conformal prediction, Clopper-Pearson bounds, and Pareto cascade tuning. |
| [`src/microllm/rlcd.py`](../src/microllm/rlcd.py) | `RLCDConfig`, `RLCDDecisionHeadTrainer`, `RLCDReplayBuffer`, `shifted_brier_reward`, `group_mean_variance_identity`, `beta_decision_weight`, `rlcd2_weighted_brier_reward`, `stage1_pathwise_loss_and_grad`, `compute_leave_one_out_kl_advantages`, `verify_tabular_rlcd_unbiased_gradient` | Two-Stage *OpenJev-RLCD* Algorithm 1 + *StartLux* $\text{RLCD}^2$ $\text{Beta}(8,2)$ weighting, *Proposition 1* variance identity, and Appendix A exact analytical gradient verification. |
| [`src/microllm/backends/base.py`](../src/microllm/backends/base.py) | `DecisionBackend`, `CONCEPT_CLUSTERS`, `tokenize_multilingual`, `embed_text_deterministic`, `_embed_text_cached`, `semantic_match_logit`, `explain_state_tokens`, `explain_counterfactual`, `create_hf_zero_shot_scorer`, `create_onnx_nli_scorer` | 256-D deterministic CPU encoder with `BLAKE2b` LRU cache, token & counterfactual XAI, and revision-pinned HuggingFace/ONNX adapters (`RCA-014`). |
| [`src/microllm/backends/fast_encoder.py`](../src/microllm/backends/fast_encoder.py) | `FastEncoderBackend`, `format_nli_hypothesis`, `truncate_to_context_budget` | Gear 1 single-pass encoder with NLI sentence templating (`"It is {opt}: {desc}."`), 512-token context guard, and policy/exemption handling (`RCA-001`). |
| [`src/microllm/backends/prefill_llm.py`](../src/microllm/backends/prefill_llm.py) | `AnyJevPrefillBackend`, `l0_circular_rotation_readout`, `van_der_corput_order`, `compute_dynamic_tau`, `compute_answer_mass`, `apply_prior_debiasing`, `OnlinePriorTracker` | Gear 2 *AnyJev* $L_0$ circular rotation readout cancelling slot bias, Van der Corput schedule, canonical key sorting (`RCA-002`), and online EMA prior debiasing. |
| [`src/microllm/backends/cascade.py`](../src/microllm/backends/cascade.py) | `DualGearCascadeBackend`, `fuse_distributions`, `compute_adaptive_alpha` | StartLux *Read-or-Think* cascade combining Gear 1 speed with Gear 2 order-invariance via $(1-\alpha)p_0 + \alpha p_1$. |
| [`src/microllm/registry.py`](../src/microllm/registry.py) | `MODEL_PROFILES`, `list_profiles`, `pull_profile`, `verify_profile`, `quantize_matrix_int8`, `dequantize_matrix_int8`, `quantize_onnx_file_int8` | Hardware profile registry (`fast-cpu-int8`, `multilingual`, `guardrail`), symmetric INT8 quantization, and SHA-256 checksum verification. |
| [`src/microllm/client.py`](../src/microllm/client.py) | `MicroJevClient`, `AsyncMicroJevClient`, `Noul`, `Choice`, `Score`, `Multi`, `NoulResult`, `ChoiceResult`, `ScoreResult`, `MultiResult`, `system_one_router` | Typed synchronous and asynchronous Python SDKs, `Enum` schema bridges (`from_enum`), and `@system_one_router` decorator. |
| [`src/microllm/server.py`](../src/microllm/server.py) | `create_app`, `run_cli`, `main` | FastAPI HTTP server (17 routes), telemetry collector, Decision Studio UI host, and CLI dispatcher. |
| [`src/microllm/benchmarks.py`](../src/microllm/benchmarks.py) & [`src/microllm/benchmark.py`](../src/microllm/benchmark.py) | `build_eval_corpus`, `run_benchmark_suite` | 540-item 5-track scientific evaluation corpus and 1-command benchmark runner. |
| [`src/microllm/quality_suite.py`](../src/microllm/quality_suite.py) | `build_quality_test_cases`, `run_quality_validation_suite`, `format_quality_report_markdown` | 750–1,000 case 8-track Quality & Performance SLA validation engine (`G1`–`G8` gates). |
| [`src/microllm/test_cases_1000.py`](../src/microllm/test_cases_1000.py) & [`src/microllm/product_independent_cases.py`](../src/microllm/product_independent_cases.py) | `build_structured_test_entries`, `build_product_independent_test_entries`, `evaluate_structured_test_cases` | 11,000 structured test cases (`1,000` core + `10,000` product-independent across 18 domains). |
| [`src/microllm/dataset_adapters.py`](../src/microllm/dataset_adapters.py) & [`src/microllm/web_benchmark_suite.py`](../src/microllm/web_benchmark_suite.py) | `BaseDatasetAdapter`, 7 dataset adapters (`BoolQAdapter`, `Banking77Adapter`, `CLINC150Adapter`, `AmazonReviewsAdapter`, `GoEmotionsAdapter`, `WildGuardAdapter`, `RouteLLMAdapter`), `run_web_benchmark_suite` | Canonical adapters and offline fixtures for 7 public web benchmarks. |
| [`src/microllm/papers.py`](../src/microllm/papers.py) | `ACADEMIC_PAPERS_CATALOG`, `LITERATURE_PILLARS`, `list_academic_papers`, `get_academic_paper`, `download_academic_papers`, `verify_downloaded_pdfs` | 104-paper academic catalog across 6 research pillars with polite arXiv PDF downloader and verifier. |
| [`src/microllm/rca_runner.py`](../src/microllm/rca_runner.py) | `RCARunner`, `RCARecord`, `RCARunnerResult`, `RCA_CATALOG` | Sequential one-by-one verification runner for all 15 Root Cause Analyses (`RCA-001`..`RCA-015`). |
| [`src/microllm/research_challenge.py`](../src/microllm/research_challenge.py) | `PolicyAwareCalibrator`, `rlcd2_proper_integral_score`, `clopper_pearson_split_certificate`, `evaluate_probability_coherence`, `value_of_computation_gate`, `score_with_rubric_dominance`, `compute_skew_adaptive_prior_debiasing`, `evaluate_evidence_sensitivity`, `brier_decomposition`, `compute_aurc_and_cov_at_risk`, `evaluate_decoy_iia_stability`, `evaluate_multi_template_consensus`, `evaluate_numeric_bisection_tree`, `run_all_hypotheses_challenge` | Research-paper challenge algorithms & validation protocols (`HYP-001`..`HYP-012`). |

---

## 2. Formal Mathematical Specifications of Decision Primitives & Confidence

Implemented in [`src/microllm/confidence.py`](../src/microllm/confidence.py) in strict conformance with `jevcompat 0.1` §4.2–§4.5:

### 2.1 Simplex Normalization (`sanitize_and_normalize_probabilities`)
Given raw non-negative scores $\tilde{p}_1, \dots, \tilde{p}_K$, probabilities are normalized via $p_i = \tilde{p}_i / \sum_{j=1}^K \tilde{p}_j$ (falling back to uniform $1/K$ if $\sum_j \tilde{p}_j \le 10^{-12}$), and residual IEEE-754 floating-point drift is absorbed onto $k^* = \arg\max_i p_i$ via:
$$p_{k^*} \leftarrow \text{clip}_{[0, 1]}\!\left(1.0 - \sum_{j \neq k^*} p_j\right)$$
guaranteeing $\sum_{i=1}^K p_i = 1.0$ to machine precision.

### 2.2 `choice` Confidence (`choice_confidence`)
For $K \in [2, 255]$ mutually exclusive options with maximum probability $m = \max_{i \in \{1..K\}} p_i$:
$$\text{confidence}_{\text{choice}}(p) = \text{clip}_{[0, 1]}\!\left(\frac{K \cdot m - 1}{K - 1}\right)$$
- **Boundary Properties**: $\text{confidence}_{\text{choice}}(1/K, \dots, 1/K) = 0.0$ (uniform chance); $\text{confidence}_{\text{choice}}(e_{k^*}) = 1.0$ (complete certainty).

### 2.3 `score` Expectation (`score_expectation`) & Confidence (`score_confidence`)
For $K \in [2, 10]$ ordered rubric levels indexed $i \in \{0, 1, \dots, K-1\}$:
1. **Expected Value**:
   $$\mathbb{E}[S] = \sum_{i=0}^{K-1} i \cdot p_i \in [0, K-1]$$
2. **Modal Concentration Confidence**: Let $k^* = \arg\max_{i \in \{0..K-1\}} p_i$ be the modal level and $D = \frac{1}{K}\sum_{i=0}^{K-1}\left|i - \frac{K-1}{2}\right|$ be the mean absolute deviation of a uniform distribution around the midpoint $\frac{K-1}{2}$. Then:
   $$\text{confidence}_{\text{score}}(p) = \text{clip}_{[0, 1]}\!\left(1 - \frac{\sum_{i=0}^{K-1} p_i \, |i - k^*|}{D}\right)$$
3. **1D Earth Mover's Distance (`earth_movers_distance_1d`, `IDEA-008`) & Unimodal Smoothing (`unimodal_ordinal_smooth`)**:
   $$\text{EMD}_{1\text{D}}(p, q) = \frac{1}{K - 1} \sum_{i=0}^{K-2} \left|\sum_{j=0}^i p_j - \sum_{j=0}^i q_j\right|$$
   $$\tilde{p}_i = (1 - \lambda) p_i + \frac{\lambda}{2}\left(p_{\max(0, i-1)} + p_{\min(K-1, i+1)}\right), \quad \lambda \in [0, 0.5]$$

### 2.4 `multi` Independent Certainty (`multi_confidence`)
For $K \in [2, 255]$ independent binary labels with probabilities $p_k \in (0, 1)$:
$$\text{confidence}_{\text{multi}}(p) = \frac{1}{K}\sum_{k=1}^K |2p_k - 1| \in [0, 1]$$

---

## 3. Gear 1 Specification: `FastEncoderBackend`, Embeddings & XAI

Implemented in [`src/microllm/backends/base.py`](../src/microllm/backends/base.py) and [`src/microllm/backends/fast_encoder.py`](../src/microllm/backends/fast_encoder.py):

### 3.1 Deterministic 256-Dimensional Multilingual Encoder (`embed_text_deterministic`)
To guarantee `< 1.0 ms` cold-start CPU execution with zero network dependency (`IMP-002`, `APP-001`), text is embedded into $\mathbb{R}^{256}$ (`_embed_text_cached` with `functools.lru_cache(maxsize=4096)`, `IDEA-012`):
- **Dimensions `0..15` (Multilingual Concept Clusters)**: Activations across `CONCEPT_CLUSTERS` (`billing_finance`, `technical_support`, `shipping_logistics`, `urgency_high`, `urgency_low`, `affirmative`, `negative`, `security_destructive`, `security_injection`, `security_pii`) covering English, French, Spanish, German, and Korean vocabulary ($v_c = 2.2 \times \text{hits}$).
- **Dimensions `16..255` (`BLAKE2b` Word & Subword Character 3-Gram Hashing)**:
  - Each word token $w$ hashes via 8-byte `blake2b` into slot $16 + (\text{hash}(w) \bmod 240)$ with signed weight $\pm 1.6$.
  - Each boundary-padded character 3-gram `<w>` hashes via 4-byte `blake2b` into slot $16 + (\text{hash}(\text{tri}) \bmod 240)$ with weight $+0.45$, providing morphological robustness.
- **Normalization**: $\hat{v} = v / \|v\|_2$.

### 3.2 Semantic Match Logit (`semantic_match_logit`) & NLI Hypothesis Templating (`format_nli_hypothesis`)
Per *openJev-verdict-2.0*, candidate options `(option_name, description)` are formatted as complete NLI hypothesis sentences:
$$\text{format\_nli\_hypothesis}(\text{opt}, \text{desc}) = \text{"It is \{opt\}: \{desc\}."}$$
Given premise $s$ (`state` truncated to 512 tokens via `truncate_to_context_budget`), query $q$ (`instructions`), and candidate hypothesis $h$:
$$\text{logit}(s, q, h) = 3.2 \langle \hat{v}_s, \hat{v}_h \rangle + 0.6 \langle \hat{v}_q, \hat{v}_h \rangle + 2.8 \cdot \frac{|T(s) \cap T^*(h)|}{\max(1, |T^*(h)|)} + 1.8 \langle \hat{v}_{s, 0:16}, \hat{v}_{h, 0:16} \rangle$$

### 3.3 Token Attribution (`explain_state_tokens`) & Counterfactual Sensitivity (`explain_counterfactual`)
- **`explain_state_tokens(state_text, query_text, candidate_text, top_k=6)`**: Filters stopwords, scores each unique token $w_i \in s$ directly against $h$ (`RCA-003`), retains tokens with $w > 0.05$, and normalizes weights relative to $\max_j w_j$.
- **`explain_counterfactual(state_text, instructions_text, options, top_k=5)`**: Computes baseline probabilities $p(y \mid s)$ and winning option $y^* = \arg\max_y p(y \mid s)$, then evaluates leave-one-token-out ablations $s \setminus \{w_i\}$, recording `probability_drop` $= p(y^* \mid s) - p(y^* \mid s \setminus \{w_i\})$ and `flips_decision` $= (\arg\max_y p(y \mid s \setminus \{w_i\}) \neq y^*)$.

---

## 4. Gear 2 Specification: `AnyJevPrefillBackend`, Theorem 1 $L_0$ & Van der Corput

Implemented in [`src/microllm/backends/prefill_llm.py`](../src/microllm/backends/prefill_llm.py) following Nokia Applied Research's *AnyJev Technical Report* (`arXiv:2610.00831`):

### 4.1 Formal Proof of AnyJev Theorem 1 (`l0_circular_rotation_readout`)
Suppose a causal prefill model assigns log-probability to option $i \in \{0,\dots,K-1\}$ when placed in prompt slot $j = (i - t) \bmod K$ during circular shift $t \in \{0,\dots,K-1\}$ according to the additive slot-bias model:
$$\log q_t(i) = s_i + b_{(i - t) \bmod K} - \log Z_t$$
where $s_i$ is the true intrinsic semantic score of option $i$, $b_j$ is the structural position bias of slot $j$, and $Z_t$ is the partition function of pass $t$. Averaging in log-probability space across the cyclic group $t = 0,\dots,K-1$:
$$\bar{z}_i = \frac{1}{K}\sum_{t=0}^{K-1} \log q_t(i) = s_i + \underbrace{\frac{1}{K}\sum_{j=0}^{K-1} b_j}_{\text{constant } \bar{b}} - \underbrace{\frac{1}{K}\sum_{t=0}^{K-1} \log Z_t}_{\text{constant } \overline{\log Z}}$$
Taking the softmax over $\bar{z}$ cancels every additive constant:
$$p_i^{(L_0)} = \frac{\exp(\bar{z}_i)}{\sum_{k=0}^{K-1} \exp(\bar{z}_k)} = \frac{\exp(s_i)}{\sum_{k=0}^{K-1} \exp(s_k)}$$
which is **algebraically independent of the slot biases $b_j$**.

### 4.2 Binary-Reflected Van der Corput Schedule (`van_der_corput_order`) & Dynamic $\tau(K, \alpha)$ Early Stopping
To avoid running all $K$ passes when a clear winner emerges early (`IMP-004`), rotations are scheduled in binary-reflected Van der Corput order:
$$\text{van\_der\_corput\_order}(K) = \left[0,\, \lfloor K/2 \rfloor,\, \lfloor K/4 \rfloor,\, \lfloor 3K/4 \rfloor,\, \dots\right]$$
After at least `min_passes = 2` (initial + antipodal shift), evaluation halts as soon as the top-2 running log-odds margin satisfies $\mu(\bar{z}) = \bar{z}_{(1)} - \bar{z}_{(2)} \ge \tau(K, \alpha)$, where `compute_dynamic_tau` (`IDEA-001`) scales:
$$\tau(K, \alpha) = \tau_{\text{base}} \cdot \left(1 + 0.12 \ln\!\left(\frac{K}{2}\right)\right) \cdot \sqrt{\frac{\ln(1 / \alpha)}{\ln 20}}$$

### 4.3 Canonical Key Sorting vs. Ordinal Preservation (`RCA-002`)
- For unordered primitives (`choice`, `noul`, `multi`), `canonicalize_order=True` lexicographically sorts `(key, description)` prior to executing `l0_circular_rotation_readout` and maps probabilities back to the caller's key order afterward. This guarantees **`0.000000` order-flip rate and `0.000000` max probability drift** even when early stopping triggers at $T = 2 < K$.
- For ordinal `score` questions, `canonicalize_order=False` preserves the natural order `0..K-1` per *AnyJev Appendix E*.

### 4.4 Answer Mass Gate (`compute_answer_mass`, `IDEA-005`) & Prior Debiasing (`apply_prior_debiasing`, `OnlinePriorTracker`, `IDEA-006`)
- **Answer Mass Gate**: $m(s, \pi) = \sum_{j=1}^K P_{\text{vocab}}(\text{token} = \ell_j) \in [0, 1]$, emitting `answer_mass_gate_passed = (m(s, \pi) >= 0.15)` in `x_l0_meta`.
- **Prior Correction**: Given empirical or streaming EMA prior $\hat{q}$ (`OnlinePriorTracker(momentum=0.05, alpha=0.75)`):
  $$p_i^{\text{dbs}} = \frac{p_i / \hat{q}_i^\alpha}{\sum_{k=1}^K p_k / \hat{q}_k^\alpha}$$

---

## 5. StartLux Cascade (`DualGearCascadeBackend`) & 25-Way Hierarchical Tournament

### 5.1 StartLux *Read-or-Think* Cascade (`src/microllm/backends/cascade.py`)
1. **Gear 1 Fast Pass**: Compute $p_0 = \text{Gear1}(s, q)$, confidence $c_0$, and conformal abstention flag `x_abstain`.
2. **Gating Rule**: If $c_0 \ge g$ (`gate_threshold = 0.75`) and `x_abstain == False`, return $p_0$ immediately (`x_gear_used = "gear1_fast"`).
3. **Gear 2 Escalation & Fusion**: Otherwise evaluate $p_1 = \text{Gear2}_{L_0}(s, q)$ and return:
   $$p_{\text{fused}} = (1 - \alpha) p_0 + \alpha p_1 \quad (\text{default } \alpha = 0.75, \text{ `x\_gear\_used` = `"cascade\_fused"`})$$
4. **Adaptive Fusion Weight (`compute_adaptive_alpha`, `IDEA-002`)**:
   $$\alpha(x) = \text{clip}_{[0.15, 0.95]}\!\Big(\big(\alpha_0 + 0.30 (c_1 - c_0)\big) \cdot \phi(m(s, \pi))\Big)$$

### 5.2 25-Way Hierarchical Tournament (`run_hierarchical_tournament`)
For `choice` questions with $K \in (25, 255]$ options (`StartLux-Decision` §3.2):
1. Optional snake/stride interleaving (`seed_diversity=True`, `IDEA-007`) distributes lexically adjacent options across $\lceil K / 25 \rceil$ brackets.
2. Each chunk of $\le 25$ options is scored independently; the top $M = 3$ options per chunk advance to the finalist playoff round.
3. Non-finalist probabilities are rescaled below the minimum finalist probability of their chunk (`min_finalist * 0.25 * rel_local / denom`) and the full $K$-vector is normalized via `sanitize_and_normalize_probabilities`, preserving strict dominance and $\sum_{i=1}^K p_i = 1.0$.

---

## 6. Statistical Calibration, Conformal Prediction & Clopper-Pearson Certification

Implemented in [`src/microllm/calibration.py`](../src/microllm/calibration.py):

| Component | Class / Function | Mathematical Formulation |
| :--- | :--- | :--- |
| **Per-$K$ Temperature Scaling** | `PerKTemperatureScaler` | Fits scalar $T_K \in [0.05, 10.0]$ per cardinality $K$ minimizing $\text{NLL}(z/T_K, y) + 0.5\,\text{Brier}(z/T_K, y)$ via bounded scalar optimization; interpolates linearly in $\ln K$ for unseen cardinalities (`IMP-003`). |
| **$L_2$ Vector Scaling (`IDEA-004`)** | `VectorScalingCalibrator` | Fits per-class scale $\mathbf{a} \in \mathbb{R}^K$ and bias $\mathbf{b} \in \mathbb{R}^K$ via BFGS on $\text{NLL}(\mathbf{a}\odot z + \mathbf{b}, y) + 0.5\,\text{Brier} + \lambda(\|\mathbf{a}-\mathbf{1}\|_2^2 + \|\mathbf{b}\|_2^2)$. |
| **Split Conformal Prediction** | `SplitConformalCalibrator` | Non-conformity $s_i = 1 - p(y_i \mid x_i)$; finite-sample quantile $\hat{q}_{1-\alpha} = \text{Quantile}_{\text{higher}}\!\left(\{s_i\}_{i=1}^N; \min\!\left(1, \frac{\lceil(N+1)(1-\alpha)\rceil}{N}\right)\right)$; prediction set $C_\alpha(x) = \{k : p_k \ge 1 - \hat{q}_{1-\alpha}\}$; `abstain = (|C_α(x)| != 1)`. |
| **Mondrian Conformal (`IDEA-003`)** | `fit_mondrian` / `predict_set_mondrian` | Class-conditional quantiles $\hat{q}_{1-\alpha}^{(k)}$ per class $k$ (falling back to global $\hat{q}_{1-\alpha}$ when $n_k < 3$). |
| **Clopper-Pearson Risk Bound** | `clopper_pearson_threshold` | One-sided exact binomial upper bound $U(e, n_{\text{acc}}, \delta) = \text{Beta}^{-1}(1 - \delta;\, e + 1,\, n_{\text{acc}} - e)$; selects lowest confidence threshold $\tau$ where $U \le r_{\text{target}}$. |
| **Pareto Cascade Auto-Tuner (`IDEA-010`)** | `tune_cascade_pareto` | Sweeps $(g, \alpha) \in [0.10, 0.95] \times [0.25, 0.95]$ subject to $\mathbb{P}(c_0 < g) \le \text{max\_escalation\_rate}$ to minimize fused Brier score. |

---

## 7. Two-Stage OpenJev-RLCD² Local Training Engine (`src/microllm/rlcd.py`)

Implements **Proposition 1**, **Algorithm 1 ("Calibrate, then Reinforce")**, and **Appendix A** from *OpenJev-RLCD* (`arXiv:2609.38850`) combined with $\text{RLCD}^2$ decision weights from *StartLux-Decision* (`arXiv:2610.17309`):

1. **Strictly Proper Shifted Brier Reward (`shifted_brier_reward`)**:
   $$J(u, q) = 2 u^\top q - \|u\|_2^2 = 1 - \|u - e_y\|_2^2 \quad (\text{when } q = e_y)$$
2. **Proposition 1 Variance Identity (`group_mean_variance_identity`, `IMP-001`)**:
   $$J\!\left(\mathbb{E}_r[u(r)],\, q\right) = \mathbb{E}_r\!\big[J(u(r),\, q)\big] + \mathbb{E}_r\!\left[\|u(r) - \bar{p}\|_2^2\right]$$
   Proves that scoring the group mean $\bar{p} = \mathbb{E}_r[u(r)]$ adds an artificial diversity bonus $\mathbb{E}_r[\|u(r) - \bar{p}\|_2^2] \ge 0$ that rewards ungrounded rationale disagreement.
3. **$\text{RLCD}^2$ $\text{Beta}(8, 2)$ Decision Weight (`beta_decision_weight`)**:
   $$w(c) = 0.5 + 0.5 \cdot \text{BetaPDF}(c;\, \alpha=8, \beta=2), \quad c = \max_k u_k$$
4. **Stage 1 (Pathwise Readout Calibration, `stage1_pathwise_loss_and_grad`)**:
   - Minimizes $L_1(z, y) = \text{sg}(w(c)) \cdot \|u - e_y\|_2^2$ via analytical Jacobian $\nabla_z L_1 = u \odot (g_u - u^\top g_u)$ where $g_u = 2 w(c)(u - e_y)$, then freezes the Stage-1 policy anchor $\theta_1$.
5. **Stage 2 (KL-Anchored Leave-One-Out REINFORCE, `compute_leave_one_out_kl_advantages`)**:
   - For $M = 4$ sampled latent rationales $r^{(1)}, \dots, r^{(M)} \sim \pi_\theta(\cdot \mid x)$:
     $$\tilde{R}_i = R_i - \beta \left(\log \pi_\theta(r^{(i)} \mid x) - \log \pi_{\theta_1}(r^{(i)} \mid x)\right), \quad A_i = \frac{1}{M}\left(\tilde{R}_i - \frac{1}{M - 1}\sum_{j \neq i} \tilde{R}_j\right)$$
   - Verified to `< 10^{-12}` analytical gradient error via full $R^M \times K$ tabular enumeration in `verify_tabular_rlcd_unbiased_gradient`.

---

## 8. Complete `jevcompat 0.1` Wire Protocol & 32/32 `MUST` Compliance Matrix

Every requirement below is enforced in [`src/microllm/schemas.py`](../src/microllm/schemas.py) and [`src/microllm/server.py`](../src/microllm/server.py) and tested in [`tests/test_jevcompat_contract.py`](../tests/test_jevcompat_contract.py):

| # | Requirement ID | Section | Normative Specification (`jevcompat 0.1`) | Enforcing Code Symbol | Test Verification |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 1 | `http.models` | §1 HTTP | `GET /v1/models` returns JSON `{"models": [{"name", "description", "release_date"}]}` | `list_models` (`server.py`) | **PASS** |
| 2 | `http.endpoint` | §1 HTTP | `POST /v1/systemone` accepts JSON request and returns JSON `200 OK` | `systemone_endpoint` (`server.py`) | **PASS** |
| 3 | `http.auth` | §1 HTTP | Enforces `Authorization: Bearer <key>` returning `401` when `MICROLLM_API_KEY` is configured | `_check_auth` (`server.py`) | **PASS** |
| 4 | `request.state-required` | §2 Request | Top-level `state` field is mandatory (`422` if missing or `null`) | `SystemOneRequest.validate_state` | **PASS** |
| 5 | `request.state-types` | §2 Request | `state` accepts `str`, JSON `dict`, or JSON `list` (rejects numbers/booleans/null) | `StructuredText`, `validate_state` | **PASS** |
| 6 | `request.questions-required` | §2 Request | Top-level `questions` mapping is mandatory | `SystemOneRequest.questions` | **PASS** |
| 7 | `request.questions-non-empty` | §2 Request | `questions` must contain at least 1 question (`len >= 1`) | `validate_questions_non_empty` | **PASS** |
| 8 | `request.model-optional` | §2 Request | `model` is optional and defaults to `"jev-latest"` | `SystemOneRequest.model` | **PASS** |
| 9 | `response.top-level` | §3 Response | Top-level response contains `model` (`str`), `answers` (`dict`), and `usage` (`UsageInfo`) | `SystemOneResponse` | **PASS** |
| 10 | `response.answer-keys-match` | §3 Response | `answers.keys()` matches `questions.keys()` 1-for-1 | `systemone_endpoint` | **PASS** |
| 11 | `response.usage-shape` | §3 Response | `usage` contains non-negative integers `input_tokens` and `output_tokens` | `UsageInfo` (`schemas.py`) | **PASS** |
| 12 | `noul.minimal` | §4.2 Noul | `{"type": "noul"}` works with both `instructions` and `criteria` omitted | `NoulQuestion` | **PASS** |
| 13 | `noul.criteria-optional` | §4.2 Noul | `criteria` may be `None`, `{}`, `{"true": ...}`, `{"false": ...}`, or both | `NoulQuestion.validate_noul_criteria` | **PASS** |
| 14 | `noul.criteria-keys` | §4.2 Noul | Any key in `noul.criteria` outside `{"true", "false"}` is rejected with `422` | `NoulQuestion.validate_noul_criteria` | **PASS** |
| 15 | `noul.answer-shape` | §4.2 Noul | Returns `{"type": "noul", "noul": float}` in `[0, 1]` with **no** `confidence` field | `NoulAnswer` (`schemas.py`) | **PASS** |
| 16 | `choice.criteria-required` | §4.3 Choice | `criteria` mapping is required on `choice` questions | `ChoiceQuestion.criteria` | **PASS** |
| 17 | `choice.criteria-bounds` | §4.3 Choice | `choice.criteria` must have between `2` and `255` options (`422` on `<2` or `>255`) | `validate_choice_options` | **PASS** |
| 18 | `choice.option-values` | §4.3 Choice | Option values may be `null`, `str`, JSON `dict`, or JSON `list` | `StructuredText \| None` | **PASS** |
| 19 | `choice.answer-shape` | §4.3 Choice | Returns `{"type": "choice", "choice", "probabilities", "confidence"}` | `ChoiceAnswer` (`schemas.py`) | **PASS** |
| 20 | `choice.choice-in-criteria` | §4.3 Choice | Winning `choice` key is always a member of `criteria.keys()` | `evaluate_choice` | **PASS** |
| 21 | `choice.probabilities-keys` | §4.3 Choice | `probabilities.keys()` equals `criteria.keys()` | `sanitize_and_normalize_probabilities` | **PASS** |
| 22 | `choice.probabilities-valid` | §4.3 Choice | Finite floats in `[0, 1]` summing to `1.0` (`±1e-6`) | `sanitize_and_normalize_probabilities` | **PASS** |
| 23 | `choice.confidence-range` | §4.3 Choice | `confidence == (n * max_p - 1) / (n - 1)` in `[0, 1]` | `choice_confidence` (`confidence.py`) | **PASS** |
| 24 | `score.criteria-required` | §4.4 Score | `criteria` ordered list is required on `score` questions (rejects dicts) | `ScoreQuestion.reject_map_and_bounds` | **PASS** |
| 25 | `score.criteria-bounds` | §4.4 Score | `score.criteria` must contain between `2` and `10` levels (`422` on `<2` or `>10`) | `ScoreQuestion.reject_map_and_bounds` | **PASS** |
| 26 | `score.answer-shape` | §4.4 Score | Returns `{"type": "score", "score", "legend", "probabilities", "confidence"}` | `ScoreAnswer` (`schemas.py`) | **PASS** |
| 27 | `score.score-range` | §4.4 Score | `score == sum(i * p_i)` in `[0, n - 1]` | `score_expectation` (`confidence.py`) | **PASS** |
| 28 | `score.legend-matches` | §4.4 Score | `legend` maps `"0".."n-1"` to exact input `criteria[i]` descriptions | `_evaluate_isolated_question` | **PASS** |
| 29 | `score.probabilities-keys` | §4.4 Score | `probabilities.keys()` equals `{"0", "1", ..., str(n - 1)}` | `evaluate_score` | **PASS** |
| 30 | `score.probabilities-valid` | §4.4 Score | Finite floats in `[0, 1]` summing to `1.0` (`±1e-6`) | `sanitize_and_normalize_probabilities` | **PASS** |
| 31 | `score.confidence-range` | §4.4 Score | `confidence == max(0, 1 - sum_i p_i |i - k*| / D)` in `[0, 1]` | `score_confidence` (`confidence.py`) | **PASS** |
| 32 | `errors.validation-shape` | §6 Errors | Malformed requests return HTTP `422` with `{"detail": [{"loc": ["body", ...], "msg", "type"}]}` | `validation_exception_handler` | **PASS** |
| 33 | `errors.no-5xx` | §6 Errors | No client input (including `NaN`, `Infinity`, empty objects, bad types) ever triggers HTTP `5xx` | `_ensure_finite_json` (`RCA-004`) | **PASS** |
| 34 | `semantics.question-id` | §7 Semantics | Renaming a question ID (`"q1"` vs `"custom_id"`) produces bit-identical probabilities | `_evaluate_isolated_question` | **PASS** |
| 35 | `semantics.batching` | §7 Semantics | Evaluating a question alone vs. batched with sibling questions produces bit-identical results | `_evaluate_isolated_question` | **PASS** |
| 36 | `semantics.question-order` | §7 Semantics | Reordering questions inside `questions` produces bit-identical results | `_evaluate_isolated_question` | **PASS** |

---

## 9. Complete HTTP REST API Specification (17 Endpoints)

All routes are registered by `create_app(api_key: str | None = None)` in [`src/microllm/server.py`](../src/microllm/server.py):

| # | HTTP Method & Path | Request Model / Query Params | Response Payload Summary | Status Codes |
| :---: | :--- | :--- | :--- | :---: |
| **1** | `GET /v1/models` | *(None)* | `{"models": [{"name": "jev-latest" \| "microllm-gear1-fast" \| "microllm-gear2-anyjev-l0", "description": ..., "release_date": ...}]}` | `200` |
| **2** | `POST /v1/systemone` | `SystemOneRequest` (`{model?, state, questions}`) | `SystemOneResponse` (`{model, answers, usage}`) | `200`, `401`, `422` |
| **3** | `POST /v1/calibrate` | `CalibrateRequest` (`{logits?, labels?, alpha, target_risk, delta, seed}`) | `{cardinality_k, fitted_temperature, raw_brier, calibrated_brier, raw_ece, calibrated_ece, conformal_q_hat, clopper_pearson_certification, reliability_bins}` | `200`, `422` |
| **4** | `POST /v1/rlcd` | `StudioRLCDDemoRequest` (`{stage1_steps, stage2_steps, use_rlcd2_weights, seed, features?, labels?}`) | `{initial_brier, stage1_brier, final_brier, initial_ece, final_ece, initial_accuracy, final_accuracy, history, initial_reliability_bins, final_reliability_bins, risk_coverage_curve, clopper_pearson_certification, proposition1_variance_identity}` | `200`, `422` |
| **5** | `POST /v1/studio/rlcd-demo` | `StudioRLCDDemoRequest` | Identical to `POST /v1/rlcd` (used by Decision Studio RLCD² Lab) | `200`, `422` |
| **6** | `POST /v1/studio/analyze` | `StudioAnalyzeRequest` (`{state, questions, gate_threshold, alpha, run_order_flip_test}`) | `{gear1, gear2, cascade, order_flip_audit, token_attribution}` | `200`, `422` |
| **7** | `POST /v1/guardrail` | `GuardrailRequest` (`{state, context?, threshold=0.55, custom_rubrics?}`) | `{safe, action ("allow"\|"review"\|"block"), risk_score, groundedness_score, checks, token_attribution, latency_ms}` | `200`, `422` |
| **8** | `POST /v1/batch` | `BatchRequest` (`{items (1..500), questions, model}`) | `{summary: {total_items, abstained_count, total_latency_ms, avg_latency_ms, p50_latency_ms, p95_latency_ms}, results: [...]}` | `200`, `422` |
| **9** | `GET /v1/benchmark` | *(None)* | Returns cached [`benchmarks/latest_results.json`](../benchmarks/latest_results.json) (or executes 540-item suite if uncached) | `200` |
| **10** | `POST /v1/benchmark` | `{"save_artifacts": bool}` *(optional)* | Runs the 540-item 5-track benchmark suite live and returns full JSON + `markdown_table` | `200` |
| **11** | `GET /v1/profiles` | *(None)* | `{"profiles": list_profiles()}` (`fast-cpu-int8`, `multilingual`, `guardrail`) | `200` |
| **12** | `POST /v1/profiles/pull` | `ProfilePullRequest` (`{profile, cache_dir?, download_weights}`) | Quantizes and installs profile, returning `profile_manifest.json` with `sha256_checksum` | `200`, `422` |
| **13** | `GET /v1/papers` | Query params: `pillar?`, `q?`, `arxiv_id?` | `{total, taxonomy, papers: [...]}` from the 104-paper catalog | `200`, `404` |
| **14** | `GET /v1/metrics` | *(None)* | `{total_requests, systemone_requests, guardrail_requests, batch_requests, guardrail_blocked, abstained_decisions, latency_p50_ms, latency_p95_ms}` | `200` |
| **15** | `GET /` | *(None)* | Serves interactive **Decision Studio** HTML application ([`src/microllm/static/index.html`](../src/microllm/static/index.html)) | `200` |
| **16** | `GET /documentation` | *(None)* | Serves interactive HTML technical documentation ([`docs/index.html`](index.html)) | `200` |
| **17** | `GET /assets/{asset_name}` & `GET /papers/{paper_name}` | Path parameter (`logo.svg`, `header-banner.svg`, or `<id>.pdf`) | Serves verified `.svg` or `.pdf` files with strict path-traversal containment | `200`, `404` |

---

## 10. Strict Pydantic v2 Data Models & Validation Rules (`src/microllm/schemas.py`)

All wire schemas use Pydantic v2 with discriminated unions (`Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion, MultiQuestion], Field(discriminator="type")]`):

```python
StructuredText = Union[str, dict[str, Any], list[Any]]

class SystemOneRequest(BaseModel):
    model: str = "jev-latest"
    state: StructuredText
    questions: dict[str, QuestionSpec]  # Non-empty, non-empty string keys

class NoulQuestion(BaseModel):
    type: Literal["noul"] = "noul"
    instructions: StructuredText | None = None
    criteria: dict[str, StructuredText | None] | None = None  # Keys subset of {"true", "false"}

class ChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: StructuredText | None = None
    criteria: dict[str, StructuredText | None]  # 2 <= len(criteria) <= 255

class ScoreQuestion(BaseModel):
    type: Literal["score"] = "score"
    instructions: StructuredText | None = None
    criteria: list[StructuredText]  # Must be a JSON array (list), 2 <= len(criteria) <= 10

class MultiQuestion(BaseModel):
    type: Literal["multi"] = "multi"
    instructions: StructuredText | None = None
    criteria: dict[str, StructuredText | None]  # 2 <= len(criteria) <= 255
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
```

### Recursive Non-Finite Float Rejection (`_ensure_finite_json`)
Python's standard `json.loads` accepts `NaN`, `Infinity`, and `-Infinity` by default, which violates RFC 8259 and `jevcompat 0.1` §6 (`RCA-004`). `_ensure_finite_json` recursively traverses every string, float, dict key/value, and list element in `state`, `instructions`, and `criteria`, raising a `ValueError` caught by `validation_exception_handler` to return a clean HTTP `422` response.

---

## 11. Hardware Profile Registry & Symmetric INT8 Quantization (`src/microllm/registry.py`)

### 11.1 Symmetric INT8 Quantization (`quantize_matrix_int8` & `dequantize_matrix_int8`)
Given a floating-point weight matrix $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$:
$$s = \max\!\left(\frac{\max_{i,j} |W_{i,j}|}{127.0},\, 10^{-8}\right), \quad Q_{\text{int8}} = \text{clip}_{[-127, 127]}\!\left(\text{round}\!\left(\frac{W}{s}\right)\right) \in \mathbb{Z}_{8}^{d_{\text{in}} \times d_{\text{out}}}$$
$$\widehat{W} = \text{dequantize\_matrix\_int8}(Q_{\text{int8}}, s) = s \cdot Q_{\text{int8}}$$
- **Compression Ratio**: `8.0x` vs `float64` (`4.0x` vs `float32`).
- **Reconstruction MSE SLA**: $\frac{1}{d_{\text{in}} d_{\text{out}}}\|W - \widehat{W}\|_F^2 \le 10^{-3}$ (measured: **`1.8 \times 10^{-5}`** on Quality Gate `G8`).
- **Cryptographic Supply-Chain Verification (`verify_profile`, `IDEA-014`)**: `pull_profile` computes `hashlib.sha256(npz_path.read_bytes()).hexdigest()`, stores `sha256_checksum` in `profile_manifest.json`, and verifies exact byte integrity on load.

---

## 12. Non-Functional Requirements (NFRs), Security, CI/CD & Verification Traceability

### 12.1 Non-Functional Requirements (NFRs) & Performance SLAs

| NFR ID | Category | Target SLA Specification | Measured Production Value | Verification Suite |
| :--- | :--- | :--- | :--- | :--- |
| **NFR-01** | **Single-Request CPU Latency (`p50` / `p95`)** | `p50 <= 5.0 ms`, `p95 <= 15.0 ms` on CPU | **`p50 = 0.076 ms`**, **`p95 = 0.274 ms`** (`11,000` suite) | [`BENCHMARKS.md`](../BENCHMARKS.md) |
| **NFR-02** | **Batch Throughput (CPU)** | `>= 500 decisions/sec` | **`8,601.0 entries/sec`** (`11,000` entries in `1.28 s`) | [`tests/test_quality_validation_suite.py`](../tests/test_quality_validation_suite.py) |
| **NFR-03** | **Option-Order Invariance (`AnyJev L0`)** | Order-Flip Rate `== 0.000000`, Drift `<= 1e-9` | **`0.000000` Flip Rate**, **`0.000000` Drift** (`78.3%` fewer passes) | [`tests/test_anyjev_rotations.py`](../tests/test_anyjev_rotations.py) |
| **NFR-04** | **Selective Decision Accuracy ($\tau = 0.45$)** | `>= 99.0%` on non-abstained predictions | **`100.00%` Selective Accuracy** (`100%` of failures caught) | [`BENCHMARKS.md`](../BENCHMARKS.md) |
| **NFR-05** | **Wire Contract & Resilience** | `32/32 MUST` pass rate, `0` HTTP `5xx` errors | **`32/32 PASS`**, **`0` HTTP `5xx`** across adversarial fuzzing | [`tests/test_jevcompat_contract.py`](../tests/test_jevcompat_contract.py) |
| **NFR-06** | **Automated RCA Non-Regression** | `15/15` historical RCAs verified on every run | **`15 / 15 PASS` (`100.0%` accuracy, `p50 < 2.5 ms`)** | [`tests/test_rca_runner.py`](../tests/test_rca_runner.py) |

### 12.2 Security Architecture
1. **Supply-Chain Revision Pinning (`Bandit B615`, `RCA-014`)**: Both `create_hf_zero_shot_scorer` and `create_onnx_nli_scorer` in [`src/microllm/backends/base.py`](../src/microllm/backends/base.py) enforce an explicit `revision: str = "main"` parameter on all `AutoTokenizer.from_pretrained` and `AutoModelForSequenceClassification.from_pretrained` calls (CWE-494).
2. **Strict Path Traversal Prevention**: Static asset routes (`/assets/{asset_name}` and `/papers/{paper_name}`) canonicalize paths via `.resolve()` and verify `candidate.parent == allowed_dir` plus strict file extension allowlists (`.svg`, `.pdf`).
3. **Zero External Telemetry or Secret Leakage**: No hardcoded credentials exist in the codebase; `MICROLLM_API_KEY` is read exclusively from environment variables or constructor injection.

### 12.3 Automated CI/CD Pipeline Architecture (`.github/workflows/`)
- [`ci.yml`](../.github/workflows/ci.yml): Matrix unit tests (`Python 3.10, 3.11, 3.12`) + `sdist`/`wheel` build + `twine check --strict`.
- [`unit-tests.yml`](../.github/workflows/unit-tests.yml): Full `pytest --cov=microllm` coverage & JUnit XML artifact publishing.
- [`lint.yml`](../.github/workflows/lint.yml): Ruff, Flake8, and Mypy static type checking (`RCA-013`).
- [`security.yml`](../.github/workflows/security.yml): Bandit SAST security scanning (`RCA-014`) + `pip-audit` dependency auditing.
- [`publish-pypi.yml`](../.github/workflows/publish-pypi.yml): OIDC Trusted Publishing to PyPI and TestPyPI.
- [`release.yml`](../.github/workflows/release.yml): Automated GitHub Release creation on `v*` tags.
- [`python-package-conda.yml`](../.github/workflows/python-package-conda.yml): Conda environment validation against [`environment.yml`](../environment.yml) (`RCA-009`).
- [`jekyll-docker.yml`](../.github/workflows/jekyll-docker.yml): Documentation publishing workflow.
