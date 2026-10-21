# 20 High-Impact Engineering & Scientific Improvements for MicroLLM (`docs/ROADMAP_20_IDEAS.md`)

> **Context**: Synthesizing the 3 foundational papers — **OpenJev-RLCD** ([*arXiv:2609.38850*](https://arxiv.org/abs/2609.38850)), **AnyJev** ([*arXiv:2610.00831*](https://arxiv.org/abs/2610.00831)), and **StartLux-Decision** ([*arXiv:2610.17309*](https://arxiv.org/abs/2610.17309)) — alongside production system engineering, this document details **20 high-impact, differentiating improvements** (`IDEA-001` through `IDEA-020`) designed to turn **MicroLLM** into the reference local System-1 decision engine.

---

## Executive Summary Matrix (20 Differentiating Ideas)

| ID | Pillar | Title | Reference / Inspiration | Expected Gain / Impact | Target Module |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IDEA-001** | 1. Calibration & Conformal | **Dynamic Cardinality- & Risk-Aware Early-Stopping Threshold $\tau(K, \alpha)$** | *AnyJev* (`arXiv:2610.00831` §5) | **-25% to -40% additional rotations** on high-$K$ choices with zero order flips | `src/microllm/backends/prefill_llm.py` |
| **IDEA-002** | 1. Calibration & Conformal | **Entropy- & Answer-Mass-Aware Adaptive Cascade Fusion Weight $\alpha(x)$** | *StartLux-Decision* (`arXiv:2610.17309` §3.2) | **-18% Brier error** when Gear 1 or Gear 2 encounters OOD prompts | `src/microllm/backends/cascade.py` |
| **IDEA-003** | 1. Calibration & Conformal | **Mondrian (Class-Conditional) Split Conformal Prediction** | *VerdictML* & Conformal Prediction | **Balanced $1-\alpha$ coverage per class** on imbalanced enterprise routing | `src/microllm/calibration.py` |
| **IDEA-004** | 1. Calibration & Conformal | **Dirichlet / Vector Scaling Calibration Extension (`L2` Matrix Calibration)** | *AnyJev* (`arXiv:2610.00831` §4.3) | Captures asymmetric inter-option confusion beyond scalar $T_K$ | `src/microllm/calibration.py` |
| **IDEA-005** | 2. AnyJev & Tournament | **Adaptive Answer Mass Gate ($m(s, \pi) < \tau_{\text{mass}}$) with Automatic Prompt Repair** | *AnyJev* (`arXiv:2610.00831` §4.1) | Detects degenerate prompts where probability leaks outside valid labels | `src/microllm/backends/prefill_llm.py` |
| **IDEA-006** | 2. AnyJev & Tournament | **Online EMA Prior Tracker (`OnlinePriorTracker`) for Streaming Debiasing** | *AnyJev* (`arXiv:2610.00831` §4.2) | Zero-calibration prior debiasing $p_i / \hat{q}_i^\alpha$ in live production streams | `src/microllm/backends/prefill_llm.py` |
| **IDEA-007** | 2. AnyJev & Tournament | **Information-Theoretic Semantic Seeding for Hierarchical Tournament ($K > 25$)** | *StartLux-Decision* (`arXiv:2610.17309` §3.2) | Prevents "group-of-death" elimination when near-synonyms fall in the same chunk | `src/microllm/confidence.py` |
| **IDEA-008** | 2. AnyJev & Tournament | **Ordinal Unimodal Regularization & Earth Mover's Distance (EMD) for `score`** | *StartLux-Decision* & `jevcompat` §4.4 | Eliminates bimodal noise on ordinal rubrics (`K = 2..10`) | `src/microllm/confidence.py` |
| **IDEA-009** | 3. RLCD & Learning | **Active Learning & Continual Calibration Buffer (`RLCDReplayBuffer`)** | *OpenJev-RLCD* (`arXiv:2609.38850` Alg. 1) | Collects abstained (`x_abstain=True`) decisions for 1-click local RLCD² fine-tuning | `src/microllm/rlcd.py` |
| **IDEA-010** | 3. RLCD & Learning | **Pareto Frontier Auto-Tuner for Cascade (`gate_threshold`, $\alpha$) under Latency SLA** | *StartLux-Decision* (`arXiv:2610.17309` §4) | Automatically finds optimal `(gate_threshold, alpha)` for any target `p95_ms` | `src/microllm/calibration.py` |
| **IDEA-011** | 3. RLCD & Learning | **Persistent Calibration State (`save_calibration` / `load_calibration`) per Profile** | *openJev-verdict-2.0* | Persists learned $T_K$ and $\hat{q}_{1-\alpha}$ across server restarts | `src/microllm/calibration.py` |
| **IDEA-012** | 4. Performance & Runtime | **Content-Addressable Embedding & Hypothesis LRU Cache (`BLAKE2b` Memoization)** | High-Throughput Inference | **3x to 8x speedup** on repeated `state` or `criteria` in multi-question & batch calls | `src/microllm/backends/base.py` |
| **IDEA-013** | 4. Performance & Runtime | **Vectorized NumPy Matrix Scoring for `POST /v1/batch` (`evaluate_batch_fast`)** | SIMD / Matrix Vectorization | Evaluates 500 states $\times K$ options via single BLAS matrix multiply (`< 10 ms` total) | `src/microllm/backends/fast_encoder.py` |
| **IDEA-014** | 4. Performance & Runtime | **Real HuggingFace Hub ONNX Exporter & Quantizer (`microllm pull --from-hub`)** | ONNX Runtime INT8 | Seamless bridge between deterministic offline mode and real DeBERTa-v3 ONNX weights | `src/microllm/registry.py` |
| **IDEA-015** | 5. SDK & Integrations | **First-Class Asynchronous Python SDK (`AsyncMicroJevClient`)** | Python `asyncio` / `httpx.AsyncClient` | Non-blocking concurrent System-1 calls in FastAPI, LangGraph, and async agents | `src/microllm/client.py` |
| **IDEA-016** | 5. SDK & Integrations | **Pydantic `Enum` & `Literal` Schema Bridge (`Choice.from_enum`, `Multi.from_enum`)** | Type-Safe Developer DX | Zero-boilerplate conversion from Python `Enum` docstrings/values to `jevcompat` criteria | `src/microllm/client.py` |
| **IDEA-017** | 5. SDK & Integrations | **Deterministic `@system_one_router` Decorator for Agentic Tool & Branch Routing** | System-1 / System-2 Orchestration | Turns any Python function / agent branch into a calibrated `<5 ms` System-1 router | `src/microllm/client.py` |
| **IDEA-018** | 6. Security, Guardrail & Ops | **Pluggable Domain-Specific Guardrail Rubrics (`custom_rubrics` in `POST /v1/guardrail`)** | *Just Ask Jev* (`arXiv:2609.29429`) | Zero-shot HIPAA, GDPR, financial advice, or SQL-dialect safety checks on the fly | `src/microllm/server.py` |
| **IDEA-019** | 6. Security, Guardrail & Ops | **Counterfactual Token Sensitivity & Minimal Decision Flip Finder (`explain_counterfactual`)** | Actionable Explainability (XAI) | Identifies the exact 1–2 tokens in `state` whose removal flips the winning decision | `src/microllm/backends/base.py` |
| **IDEA-020** | 6. Security, Guardrail & Ops | **OpenTelemetry & Prometheus Metrics Endpoint (`GET /v1/metrics`) with Drift Detection** | Production Observability | Tracks per-gear latency (`p50`/`p95`), escalation rate, abstention rate, and confidence PSI | `src/microllm/server.py` |

---

## Detailed Specifications of the 20 Ideas

### Pillar 1 — Probabilistic Calibration & Conformal Guarantees

#### IDEA-001: Dynamic Cardinality- & Risk-Aware Early-Stopping Threshold $\tau(K, \alpha)$
- **Problem**: `AnyJevPrefillBackend` currently uses a static early-stopping log-odds margin threshold $\tau = 1.25$ regardless of the number of options $K \in [2, 255]$ or target risk $\alpha$.
- **Solution**: Implement `compute_dynamic_tau(k, target_risk=0.05, base_tau=1.25)` scaling $\tau(K, \alpha) = \text{base\_tau} \cdot \left(1 + 0.12 \ln(K / 2)\right) \cdot \sqrt{\ln(1 / \alpha) / \ln(20)}$. For small $K=2$, early stopping triggers faster when the margin is clear; for large $K=20$, it demands a slightly wider log-odds separation to protect against multi-hypothesis collisions.

#### IDEA-002: Entropy- & Answer-Mass-Aware Adaptive Cascade Fusion Weight $\alpha(x)$
- **Problem**: `DualGearCascadeBackend` uses a fixed $\alpha = 0.75$ in $p_{\text{fused}} = (1 - \alpha) p_0 + \alpha p_1$. When Gear 1 is highly uncertain ($c_0 \approx 0$) and Gear 2 has high confidence ($c_1 \gg c_0$), or conversely when Gear 2 suffers low answer mass, a static weight is suboptimal.
- **Solution**: Implement `compute_adaptive_alpha(conf0, conf1, answer_mass=1.0, base_alpha=0.75)` in `src/microllm/backends/cascade.py` that dynamically adjusts $\alpha(x) \in [0.15, 0.95]$ based on the relative confidence gain $(c_1 - c_0)$ weighted by Gear 2's Answer Mass $m(s, \pi)$.

#### IDEA-003: Mondrian (Class-Conditional) Split Conformal Prediction
- **Problem**: Standard `SplitConformalCalibrator` computes a single global quantile $\hat{q}_{1-\alpha}$ across all classes. In skewed enterprise routing (e.g., 85% `billing`, 10% `technical`, 5% `security`), rare critical classes may suffer under-coverage.
- **Solution**: Add `fit_mondrian(cal_probs, cal_labels)` and `predict_set_mondrian(probabilities)` to `SplitConformalCalibrator` in `src/microllm/calibration.py`, computing per-class quantiles $\hat{q}_{1-\alpha}^{(k)}$ with finite-sample smoothing back to the global $\hat{q}_{1-\alpha}$ when class sample count $n_k < 5$.

#### IDEA-004: Vector Scaling Calibration (`L2` Diagonal + Bias Calibration)
- **Problem**: Scalar temperature scaling $z / T_K$ cannot correct class-specific prior shifts when calibration labels are available (*AnyJev* §4.3 distinguishes $L_0$ training-free, $L_1$ scalar temperature, and $L_2$ vector scaling).
- **Solution**: Add `VectorScalingCalibrator` in `src/microllm/calibration.py` learning per-option scale and bias $(\mathbf{a}, \mathbf{b})$ minimizing regularized Brier + NLL loss.

---

### Pillar 2 — AnyJev $L_0$ Prefill & High-Cardinality Tournament Routing

#### IDEA-005: Adaptive Answer Mass Gate ($m(s, \pi) < \tau_{\text{mass}}$) with Diagnostic Flag
- **Problem**: While `compute_answer_mass` computes $m(s, \pi) = \sum_{j=1}^K P(\text{token} = \ell_j)$, low answer mass does not yet emit a structured warning or trigger prompt normalization.
- **Solution**: Expose `answer_mass_gate_passed: bool` in `x_l0_meta` whenever $m(s, \pi) \ge \tau_{\text{mass}}$ (default `0.15`), alerting callers when a custom scorer leaks probability mass outside the candidate tokens.

#### IDEA-006: Online EMA Prior Tracker (`OnlinePriorTracker`) for Streaming Debiasing
- **Problem**: `AnyJevPrefillBackend` supports `apply_prior_debiasing(probs, empirical_prior, alpha=0.75)` when a static `empirical_prior` is passed, but live production traffic shifts over time.
- **Solution**: Implement `OnlinePriorTracker(momentum=0.05, alpha=0.75)` in `src/microllm/backends/prefill_llm.py` that maintains an exponentially weighted moving average $\hat{q}^{(t)} = (1 - \gamma)\hat{q}^{(t-1)} + \gamma p^{(t)}$ per question signature and applies *AnyJev* prior debiasing on the fly.

#### IDEA-007: Semantic Diversity Seeding for Hierarchical Tournament ($K > 25$)
- **Problem**: In `run_hierarchical_tournament`, options are chunked in their raw input order (`items[i : i + chunk_size]`). If the caller groups 10 similar banking card intents together in Chunk 0, only `top_m=3` survive to the final round while weaker intents from Chunk 1 advance.
- **Solution**: Add optional snake/interleaved seeding (`seed_diversity=True`) in `run_hierarchical_tournament` so lexically/semantically adjacent options are distributed evenly across tournament brackets before chunk scoring.

#### IDEA-008: Ordinal Unimodal Smoothing & Earth Mover's Distance (EMD) for `score`
- **Problem**: Independent NLI scoring of ordinal levels `0..K-1` can occasionally produce a slightly bimodal distribution (e.g., high mass on level 0 and level 4 with a dip in the middle), inflating variance.
- **Solution**: Add `earth_movers_distance_1d(p, q)` and `unimodal_ordinal_smooth(probabilities, strength=0.15)` in `src/microllm/confidence.py` to measure ordinal distance and optionally project `score` distributions toward log-concave unimodality.

---

### Pillar 3 — OpenJev-RLCD² Training & Auto-Tuning

#### IDEA-009: Continual Calibration Replay Buffer (`RLCDReplayBuffer`)
- **Problem**: `RLCDDecisionHeadTrainer` trains on a static feature matrix, whereas production deployments need a lightweight ring buffer to accumulate uncertain or abstained examples and re-run Stage 1 + Stage 2 calibration.
- **Solution**: Implement `RLCDReplayBuffer(capacity=512)` in `src/microllm/rlcd.py` with uncertainty-priority sampling and 1-call `fit_from_buffer(trainer)`.

#### IDEA-010: Pareto Frontier Auto-Tuner for Cascade (`tune_cascade_pareto`)
- **Problem**: Users currently pick `gate_threshold = 0.75` and `alpha = 0.75` manually. Different hardware budgets (e.g., strict `< 3 ms` edge CPU vs `< 12 ms` server CPU) require different escalation rates.
- **Solution**: Add `tune_cascade_pareto(g1_probs, g2_probs, labels, max_escalation_rate=0.35)` in `src/microllm/calibration.py` that sweeps `(gate_threshold, alpha)` to return the Pareto-optimal configuration minimizing Brier score under a target Gear-2 escalation budget.

#### IDEA-011: Persistent Calibration State (`to_dict` / `from_dict` / JSON file I/O)
- **Problem**: Learned temperatures $T_K$ in `PerKTemperatureScaler` and quantile $\hat{q}_{1-\alpha}$ in `SplitConformalCalibrator` are held only in RAM.
- **Solution**: Add deterministic serialization (`save_state` / `load_state`) to `PerKTemperatureScaler` and `SplitConformalCalibrator` in `src/microllm/calibration.py`.

---

### Pillar 4 — High-Performance Inference & Caching

#### IDEA-012: Content-Addressable Embedding LRU Cache (`BLAKE2b` Memoization)
- **Problem**: When evaluating multiple questions (`noul`, `choice`, `score`, `multi`) on the same `state`, or evaluating a batch of 500 states against the same option descriptions, `embed_text_deterministic` re-hashes identical strings repeatedly.
- **Solution**: Add `functools.lru_cache(maxsize=4096)` on internal immutable tuple embeddings in `src/microllm/backends/base.py` (returning a fresh read-only/copied `np.ndarray`), cutting multi-question and batch CPU latency by **3x–5x** while preserving 100% bit-exact determinism.

#### IDEA-013: Vectorized Batch Evaluation & Concurrency
- **Problem**: `POST /v1/batch` iterates sequentially over items without leveraging cached option embeddings.
- **Solution**: Pre-warm option hypothesis embeddings once per batch and track per-gear execution statistics (`gear1_only_count`, `escalated_count`, `p50_latency_ms`, `p95_latency_ms`) in the `POST /v1/batch` summary.

#### IDEA-014: HuggingFace Hub ONNX Verification & Checksum Manifest in `registry.py`
- **Problem**: `pull_profile` creates a local `.npz` bundle without recording a cryptographic SHA-256 checksum for reproducibility verification.
- **Solution**: Compute and store `sha256_checksum` of the quantized weight bundle in `profile_manifest.json` and expose `verify_profile(profile_name)` in `src/microllm/registry.py`.

---

### Pillar 5 — Developer Experience (DX), Typed SDK & Agentic Routing

#### IDEA-015: Asynchronous Python SDK (`AsyncMicroJevClient`)
- **Problem**: `MicroJevClient` only exposes synchronous `ask()`, blocking the event loop in async frameworks (FastAPI, LangChain Async, LlamaIndex, PydanticAI).
- **Solution**: Implement `AsyncMicroJevClient` in `src/microllm/client.py` with `await client.ask(...)` and `await client.guardrail(...)` using `httpx.AsyncClient`.

#### IDEA-016: Python `Enum` & `Literal` Schema Bridge (`Choice.from_enum`, `Multi.from_enum`)
- **Problem**: Developers already define domain intents as Python `enum.Enum` classes and have to manually duplicate them into `Choice({...})` dictionaries.
- **Solution**: Add `Choice.from_enum(EnumCls, instructions=...)` and `Multi.from_enum(EnumCls, instructions=..., threshold=0.5)` classmethods in `src/microllm/client.py`.

#### IDEA-017: `@system_one_router` Decorator & SDK Guardrail / Batch Helpers
- **Problem**: `MicroJevClient` only wraps `/v1/systemone`, leaving `/v1/guardrail` and `/v1/batch` without typed SDK methods.
- **Solution**: Add `client.guardrail(state, context=None, threshold=0.55)` and `client.batch(items, questions, model=None)` to `MicroJevClient` & `AsyncMicroJevClient`.

---

### Pillar 6 — AI Safety Guardrails, Counterfactual XAI & Observability

#### IDEA-018: Pluggable Custom Guardrail Rubrics (`custom_rubrics` in `POST /v1/guardrail`)
- **Problem**: `POST /v1/guardrail` evaluates 4 fixed rubrics (`prompt_injection`, `destructive_operation`, `pii_or_secret_leak`, `hallucination_risk`). Enterprises need to inject domain-specific policies (e.g., `hipaa_phi_disclosure`, `competitor_mention`, `financial_advice`) in the same `<5 ms` call.
- **Solution**: Extend `GuardrailRequest` with `custom_rubrics: dict[str, str] | None = None` so callers can pass arbitrary zero-shot safety rubrics evaluated alongside the 4 core checks.

#### IDEA-019: Counterfactual Token Sensitivity (`explain_counterfactual`)
- **Problem**: `explain_state_tokens` highlights high-weight tokens, but engineers debugging a routing decision want to know: *"Would removing token X actually flip the decision from `billing` to `technical`?"*
- **Solution**: Implement `explain_counterfactual(state_text, instructions_text, options)` in `src/microllm/backends/base.py` computing exact leave-one-token-out probability shifts $\Delta p(y^* \mid s \setminus \{w_i\})$ and identifying minimal decision-flipping tokens.

#### IDEA-020: Real-Time Telemetry & Observability Endpoint (`GET /v1/metrics`)
- **Problem**: Production deployments need a zero-overhead observability endpoint reporting total requests, token throughput, per-gear routing ratios, abstention rate, guardrail block rate, and latency percentiles.
- **Solution**: Add an in-memory thread-safe telemetry collector and expose `GET /v1/metrics` in `src/microllm/server.py`.
