# MicroLLM Engineering & Scientific Syllabus
## Architecture, Calibration, and Systems of Sub-Millisecond Probabilistic Decision Engines

> **Governance & Traceability ([`AGENTS.md`](../AGENTS.md) §2 & [`.agents/rules/project-governance.md`](../.agents/rules/project-governance.md))**:
> - **Document ID**: `LP-SYLLABUS-2026-MICROLLM`
> - **Linked Actions**: [`ACT-041`](ACTION_LOGBOOK.md) (GitHub Issue `#98`), [`ACT-039`](ACTION_LOGBOOK.md) (GitHub Issue `#92`), [`ACT-038`](ACTION_LOGBOOK.md) (GitHub Issue `#93`)
> - **RCA Register References**: [`docs/RCA_REGISTER.md`](RCA_REGISTER.md) (`RCA-001`–`RCA-038`, `APP-001`–`APP-012`, `IDEA-001`–`IDEA-034`)
> - **Core Reference Literature**: **OpenJev-RLCD** ([*arXiv:2609.38850*](https://arxiv.org/abs/2609.38850)), **AnyJev** ([*arXiv:2610.00831*](https://arxiv.org/abs/2610.00831)), **StartLux-Decision** ([*arXiv:2610.17309*](https://arxiv.org/abs/2610.17309)), **`jevcompat 0.1`** ([`SPEC.md`](../SPEC.md))

---

## Course Overview & Metadata

- **Course Code**: `ML-ENG-801`
- **Course Title**: *Engineering Sub-Millisecond Probabilistic Decision Systems: From Kahneman System-1 to Certified Micro-Architectures*
- **Level**: Advanced Graduate / Senior Enterprise ML Engineering Masterclass
- **Duration**: 6 Modules (Flexible 6-week intensive or 12-week semester format)
- **Target Audience**: Machine Learning Engineers, LLM Systems Architects, Quant Developers, Edge AI & Robotics Engineers, Technical Leads.
- **Hardware & Compute Prerequisites**:
  - Standard development laptop or server CPU (x86_64 or ARM64).
  - **0 GPUs required** ($0.00 cloud compute cost; 100% bit-exact reproducible on standard CPUs).
  - Python 3.10+ with standard scientific libraries (`numpy`, `scipy`, `fastapi`, `pytest`).

```mermaid
flowchart LR
    M1["Mod 1: System-1 Foundations & Protocols"] --> M2["Mod 2: Feature Hashing & Fast Encoders"]
    M2 --> M3["Mod 3: Typed Decision Operators"]
    M3 --> M4["Mod 4: Calibration & Conformal Safety"]
    M4 --> M5["Mod 5: Invariance Axioms & Robustness"]
    M5 --> M6["Mod 6: Dual-Gear Cascades & Production"]
    M6 --> CAP["Capstone Project: Certified Pipeline"]
```

---

## Course Learning Outcomes (CLOs)

Upon successful completion of this course, students will be able to:

1. **CLO-1 (Architectural Design)**: Deconstruct the economic and latency failure modes of oversized autoregressive LLMs, and architect sub-millisecond non-autoregressive System-1 alternatives compliant with the `jevcompat 0.1` specification.
2. **CLO-2 (Feature Engineering & Fast Representation)**: Implement deterministic, multilingual subword feature hashing (`BLAKE2b`), character $n$-gram shingling, and concept subspace projections with strict $\mathcal{O}(L \cdot D)$ time complexity.
3. **CLO-3 (Probabilistic Calibration)**: Derive and implement strictly proper scoring rules (Shifted Brier score), temperature scaling, and StartLux RLCD² decision-weighted training algorithms.
4. **CLO-4 (Conformal Safety Certification)**: Formulate and evaluate Split Conformal Prediction algorithms that guarantee finite-sample coverage $P(Y \in \mathcal{C}(X)) \ge 1 - \alpha$ and establish optimal selective abstention thresholds ($\tau = 0.45$).
5. **CLO-5 (Invariance Theory)**: Prove group-theoretic order-bias cancellation theorems (AnyJev Propositions 1 & 2), implement minimal rotation subgroups to eliminate positional flip rates, and calibrate contrastive decision pivot vectors.
6. **CLO-6 (System Serving & Cascading)**: Deploy dual-gear "Read-or-Think" cascades with Value-of-Computation (VOC) gating, integrate Model Context Protocol (MCP) servers, and benchmark 9-suite evaluation pipelines on 128,000+ test items.

---

## Detailed Curriculum by Module

### Module 1: The System-1 Paradigm & Probabilistic Decision Foundations

#### 1.1 Lecture Topics
- **The LLM Efficiency Crisis**: The hidden compute and latency overhead of autoregressive generation for deterministic, typed decisions (`choice`, `noul`, `score`, `multi`).
- **Kahneman's Dual-Process Theory Applied to Software**:
  - System-1: Fast, automatic, low-cost, intuitive, non-autoregressive pattern matching.
  - System-2: Slow, deliberative, high-cost, autoregressive multi-step reasoning.
- **Contract-Driven Engineering**: The `jevcompat 0.1` open specification.
  - Strict input validation: state requirements, criteria bounds, question order.
  - Strict output typing: probability distributions summing to $1.0$, confidence bounds, machine precision drift correction.
  - The 32/32 `MUST` compliance assertions.

#### 1.2 Mathematical Derivations
- **Computational Complexity Divergence**:
  $$\text{Complexity}_{\text{System-2}} = \mathcal{O}\left(N_{\text{layers}} \cdot L_{\text{gen}} \cdot d_{\text{model}}^2 + L_{\text{ctx}} \cdot L_{\text{gen}} \cdot d_{\text{model}}\right)$$
  $$\text{Complexity}_{\text{System-1}} = \mathcal{O}\left(L_{\text{prompt}} \cdot D + K \cdot D\right)$$
  Prove that for fixed latent dimension $D=256$ and candidate cardinality $K \le 25$, System-1 latency is strictly $\mathcal{O}(L)$, rendering it invariant to model size.

#### 1.3 Hands-On Lab 1
- **Objective**: Install `microllm`, explore the Python SDK and CLI, and verify protocol compliance.
- **Tasks**:
  1. Initialize `MicroJevClient` and query the live server via Unix socket or HTTP.
  2. Implement an automated compliance validator that asserts all 32 `MUST` rules of `jevcompat 0.1` against `/v1/decision`.
  3. Measure latency percentiles ($p50, p95, p99$) across 1,000 synthetic queries.

---

### Module 2: Deterministic Feature Spaces, Subword Hashing & Fast Encoders

#### 2.1 Lecture Topics
- **Token-Free and GPU-Free Semantic Representations**: Overcoming the cold-start and memory footprint of 100k-token BPE vocabularies.
- **Universal Cryptographic Feature Hashing**:
  - Properties of `BLAKE2b` universal hashing: avalanche effect, uniform distribution, 64-bit speed on modern CPUs.
  - Character 3-gram boundary shingling ($\langle t \rangle$) for typo and morphology tolerance.
- **Multilingual Tokenization without Heavy Dependency Trees**: Regex Unicode script extraction for cross-lingual support (`EN`, `FR`, `ES`, `DE`, CJK).
- **Subspace Decomposition & Semantic Concept Boosting**:
  - Partitioning $\mathbb{R}^{256}$ into Concept slots $[0..15]$ and Hash slots $[16..255]$.
  - Dictionary keyword projection with boundary-sensitive security token matching.
- **L2-Simplex Normalization & Directional Cosine Geometry**.

#### 2.2 Mathematical Derivations
- **Johnson-Lindenstrauss Lemma**: For any set of $M$ points in high-dimensional space, random projection into $D = \mathcal{O}(\epsilon^{-2} \log M)$ dimensions preserves pairwise Euclidean distances within factor $1 \pm \epsilon$.
- **Hashing Collision Probability**: Derive the upper bound on feature slot collision probability across $T$ distinct tokens in $D-16$ hash bins.

#### 2.3 Hands-On Lab 2
- **Objective**: Implement and benchmark the core components of `FastEncoderBackend`.
- **Tasks**:
  1. Build a custom `_token_hash_contribution` function utilizing `hashlib.blake2b` and LRU cache.
  2. Implement Clause-Salient Max-Pooling (`HYP-008`) to isolate decisive evidence in distractor-padded customer tickets.
  3. Evaluate vector similarity accuracy between semantically identical phrases across English, French, and Spanish.

---

### Module 3: Typed Decision Routing & Contract Satisfaction

#### 3.1 Lecture Topics
- **Natural Language Inference (NLI) Hypothesis Templating**:
  - Formatting candidate labels as declarative sentences (`"It is {option}: {description}."`).
  - Aligning inputs with latent semantic geometry to lift zero-shot discrimination.
- **The Choice Operator (`choice`)**:
  - Per-cardinality temperature scaling ($T_K$).
  - Derivation of TypeSafe Choice Confidence: $(K \cdot \max_k p_k - 1) / (K - 1)$.
- **The Score Operator (`score`)**:
  - Expected value vs mode extraction for ordinal review scales (0..4).
  - Unimodal ordinal regularizers (`IDEA-008`) and Wasserstein-1 (Earth Mover's Distance) evaluation.
- **The Noul (`noul`) and Multi-Label (`multi`) Operators**:
  - Margin-based calibrated sigmoid boundaries.
  - Aggregated multi-label binary certainty.

#### 3.2 Mathematical Derivations
- **Score Confidence Dispersion Bound**:
  Prove that for an ordinal probability distribution $p \in \Delta^{N-1}$ and mode $k^* = \operatorname{argmax}_i p_i$, the score confidence:
  $$\text{conf}(p) = \max\left(0, 1 - \frac{\sum_{i=0}^{N-1} p_i |i - k^*|}{D_{\text{uniform}}}\right)$$
  satisfies $\text{conf}(p) = 0$ if and only if $p$ is uniform, and $\text{conf}(p) = 1$ if and only if $p = e_{k^*}$.

#### 3.3 Hands-On Lab 3
- **Objective**: Implement custom routing logic for multi-class classification and ordinal scoring.
- **Tasks**:
  1. Implement a 5-star customer review evaluator utilizing `evaluate_score`.
  2. Implement a multi-rubric document classifier utilizing `evaluate_multi`.
  3. Evaluate accuracy and calibration across the 10 universal domains in `product_independent_cases.py`.

---

### Module 4: Proper Scoring Rules, Calibration & Conformal Safety

#### 4.1 Lecture Topics
- **The Calibration Dilemma in Modern Machine Learning**:
  - Overconfidence in uncalibrated deep models.
  - Measuring calibration: Reliability Diagrams, Expected Calibration Error (ECE), and Maximum Calibration Error (MCE).
- **Strictly Proper Scoring Rules**:
  - Shifted Brier score: $J(u, y) = 2 u_y - \|u\|_2^2 = 1 - \|u - e_y\|_2^2$.
  - Proposition 1: Group-Mean Variance Identity and the artificial diversity bonus of ensemble averaging.
- **StartLux RLCD² Decision-Weighted Brier Optimization**:
  - Weighting errors based on automated impact: $w(c) = 0.5 + 0.5 \cdot \text{Beta}(c; 8, 2)$.
  - Stage 1 Pathwise Calibration vs Stage 2 LOO REINFORCE with KL anchoring.
- **Finite-Sample Split Conformal Prediction**:
  - Non-conformity scores $s_i = 1 - \hat{P}(y_i \mid x_i)$.
  - Guaranteed finite-sample coverage $P(Y \in \mathcal{C}(X)) \ge 1 - \alpha$.
  - Class-conditional Mondrian conformal prediction (`IDEA-003`).
- **Selective Prediction & The $\tau = 0.45$ Abstention Threshold**:
  - Catching 100% of residual failures via selective abstention.

#### 4.2 Mathematical Derivations
- **Properness of Shifted Brier Score**:
  Prove that for any true label distribution $q \in \Delta^{K-1}$:
  $$\mathbb{E}_{y \sim q}[J(u, y)] = 2 u^\top q - \|u\|_2^2 = -\|u - q\|_2^2 + \|q\|_2^2$$
  which is strictly and uniquely maximized when $u = q$.
- **Conformal Coverage Guarantee**:
  Prove via rank exchangeability that the finite-sample prediction set $\mathcal{C}(X_{n+1})$ achieves exact validity:
  $$P(Y_{n+1} \in \mathcal{C}(X_{n+1})) \ge 1 - \alpha$$

#### 4.3 Hands-On Lab 4
- **Objective**: Implement calibration metrics, train a temperature scaler, and verify conformal coverage.
- **Tasks**:
  1. Write a function to compute ECE and generate a reliability diagram across 10 confidence bins.
  2. Fit `SplitConformalCalibrator` on calibration splits and verify coverage at $\alpha = 0.10$ ($90\%$ coverage).
  3. Verify that setting abstention threshold $\tau = 0.45$ catches all 9 residual errors in `test_cases_1000.py`.

---

### Module 5: Invariance Axioms, Permutation Symmetry & Adversarial Robustness

#### 5.1 Lecture Topics
- **The Positional Bias Flaw in Autoregressive LLMs**:
  - Positional recency and primacy biases causing up to $50\%$ order flip rate.
  - The definition of Order-Flip Rate ($L_0$ Flip).
- **AnyJev Group Invariance Theorems**:
  - AnyJev Proposition 1: Full permutation group averaging $\mathfrak{S}_K$.
  - AnyJev Proposition 2: Minimal Cyclic Rotation Subgroup $C_K$.
  - Proving that $\mathcal{O}(K)$ cyclic rotations mathematically eliminate first-order positional bias.
- **Contrastive Decision Pivoting (`DecisionPivot`)**:
  - Counterfactual robustness and differential projection vectors $\vec{w}_{\Delta}$.
- **Multi-Rubric AI Safety Guardrail Architecture**:
  - Fast-path lexical automata for Prompt Injection, Destructive Commands, PII Leaks, and Hallucination.

#### 5.2 Mathematical Derivations
- **Symmetry Cancellation under Cyclic Permutations**:
  Prove that for any positional scoring function with additive rank bias $f(c_i, \text{pos}) = s(c_i) + \beta(\text{pos})$, the cyclic average over all $K$ rotations satisfies:
  $$\frac{1}{K}\sum_{r=0}^{K-1} f(c_i, (i + r) \pmod K) = s(c_i) + \frac{1}{K}\sum_{p=0}^{K-1} \beta(p)$$
  which is completely independent of candidate presentation index $i$.

#### 5.3 Hands-On Lab 5
- **Objective**: Benchmark order invariance and evaluate contrastive robustness.
- **Tasks**:
  1. Run `AnyJev-Order` across option counts $K = 2 \dots 20$ and verify $0.000000$ flip rate.
  2. Implement a contrastive pivot detector for legal contract clauses (`ContractNLI`).
  3. Benchmark the 4-rubric safety guardrail against adversarial prompt injection attacks (`WildGuardMix`).

---

### Module 6: Dual-Gear Cascades, Value-of-Computation Gates & Production Deployment

#### 6.1 Lecture Topics
- **The StartLux Dual-Gear Cascade ("Read-or-Think")**:
  - Architectural blueprint: Gear 1 (MicroLLM CPU) + Gear 2 (Local SLM / vLLM System-2).
  - Latency and cost trade-offs in high-throughput enterprise pipelines.
- **Proposition 5: Value-of-Computation (VOC) Gate**:
  - Epistemic vs. Aleatoric uncertainty decomposition.
  - Automated escalation vs open-set safe abstention.
- **Probability Fusion**: Blending distributions via $p_{\text{fused}} = (1 - \alpha) p_0 + \alpha p_1$ with $\alpha = 0.75$.
- **Hierarchical Tournament Routing for Large Action Spaces**:
  - Scaling to 255 options with competitive brackets of size $B = 8$.
- **Production Serving & Ecosystem Integration**:
  - FastAPI HTTP server, streaming Unix domain sockets, and native Model Context Protocol (MCP) server.
  - Automated 9-suite benchmark runner (`scripts/run_all_benchmarks.py`).

#### 6.2 Mathematical Derivations
- **Cascade Optimal Stopping Formulation**:
  Derive the threshold condition for the Value of Computation gate where escalating to Gear 2 maximizes expected operational utility minus compute cost:
  $$\text{VOC}(x) = \mathbb{E}_{y \sim P_1}[\mathcal{U}(y, y^*)] - \max_k P_0(k \mid x) - \lambda_{\text{compute}} > 0$$

#### 6.3 Hands-On Lab 6
- **Objective**: Deploy MicroLLM as an MCP server, configure dual-gear escalation, and run the master benchmark suite.
- **Tasks**:
  1. Configure `microllm mcp` in Claude Desktop or Antigravity IDE and execute typed decisions directly from an AI agent.
  2. Run the unified 9-suite benchmark orchestrator (`.\scripts\run_all_benchmarks.ps1 -Quick`) and inspect the generated JSON artifacts.
  3. Deploy a production containerized instance with Prometheus monitoring and sub-millisecond SLA alerting.

---

## Capstone Project Specification

### Title: *Certified Enterprise Real-Time Decision Pipeline*

Each student or engineering team must build, evaluate, and package a production-grade sub-millisecond decision pipeline on an approved domain (e.g., Financial Fraud Triage, Clinical Urgency Triage, or Cybersecurity SOC Alert Routing).

### Capstone Deliverables
1. **Model Implementation**:
   - Custom intent taxonomy with $\ge 10$ fine-grained classes.
   - FastEncoder feature representations with custom domain concept clusters.
2. **Calibration & Conformal Certification**:
   - Fit `SplitConformalCalibrator` with certified $90\%$ coverage guarantee.
   - Establish optimal selective abstention threshold $\tau$.
3. **Automated Test & Benchmark Suite**:
   - Comprehensive test suite with $\ge 500$ structured test cases.
   - Verification of $0.000000$ order flip rate and $100\%$ abstention on edge cases.
   - Benchmark report verifying $p50 < 0.5 \text{ ms}$ on single-thread CPU.
4. **Production Deployment**:
   - Containerized FastAPI deployment + Native MCP integration.
   - Executive presentation slide deck (using `docs/slides.html` template).

---

## Grading Rubric & Assessment Weights

| Assessment Component | Weight | Criteria |
| :--- | :--- | :--- |
| **Laboratory Exercises (Labs 1–6)** | **30%** | Code correctness, test coverage, benchmark execution, documentation adherence. |
| **Theoretical & Mathematical Problem Sets** | **20%** | Rigorous proofs of proper scoring rules, conformal coverage, and cyclic group invariance. |
| **Midterm Evaluation (Benchmarks Audit)** | **15%** | Execution and analysis of the 9 benchmark suites with failure mode diagnostics. |
| **Capstone Project & Codebase** | **25%** | Architecture quality, performance SLA ($< 1$ ms), conformal certification, test plan. |
| **Capstone Presentation & Defense** | **10%** | Technical clarity, live demonstration via MCP/Decision Studio, defense of architectural trade-offs. |

---

## Required & Canonical Literature

1. **OpenJev-RLCD**: *Reinforcement Learning with Calibrated Decisions via Strictly Proper Scoring Rules*, arXiv:2609.38850.
2. **AnyJev**: *Order-Invariant Non-Autoregressive Decision Routing and Minimal Permutation Subgroups*, arXiv:2610.00831.
3. **StartLux-Decision**: *Read-or-Think Dual-Gear Cascades and Decision-Weighted Proper Scoring*, arXiv:2610.17309.
4. **jevcompat 0.1**: *Unified System-One TypeSafe Protocol & Evaluation Benchmark Specification*, MicroLLM Open Standard.
5. **ModernBERT / GLiClass**: *OpenJev-Verdict-2.0 Non-Autoregressive Classifier Enhancements*, arXiv:2412.13663.
6. **VerdictML & poorjev**: *Finite-Sample Split Conformal Prediction in Enterprise Routing*, arXiv:2609.37470.
