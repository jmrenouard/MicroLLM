# Comprehensive Academic Literature Review: Jev, RLCD, System-One Typed Decision Models & Calibrated LLMs

> **Document Role**: Canonical scientific synthesis and architectural mapping of **116 peer-reviewed and preprint papers** (`110` specialized Jev / System-One / RLCD papers from 2025–2026 + `6` foundational calibration, conformal prediction, and LLM routing papers) to the **MicroLLM** local decision engine.
> **Related Resources**: [`docs/papers/README.md`](papers/README.md) · [`src/microllm/papers.py`](../src/microllm/papers.py) · `GET /v1/papers` · [`docs/RCA_REGISTER.md`](RCA_REGISTER.md)

---

## 1. Executive Synthesis: From Autoregressive Generation to Calibrated System-One Decisions

Traditional Large Language Models (LLMs) treat structured decision-making—such as binary verification (`noul`), multi-way intent routing (`choice`), ordinal rubric evaluation (`score`), and multi-label tagging (`multi`)—as **autoregressive text generation** (`System 2`). Across the 116 papers surveyed in this review (and further validated in industry by Google DeepMind's November 2026 `gemini-decision-1` endpoint via `POST /v1beta/decisions`), four structural flaws of generative decision-making recur consistently:

1. **Severe Latency and Token Overhead**: Generating JSON schemas or chain-of-thought (CoT) tokens requires dozens to hundreds of sequential decode steps (`200 ms` to `3,000 ms`), whereas a typed decision requires only a categorical probability vector over `K` predefined options ([*Harness Tokenomics*, `arXiv:2609.28919`](https://arxiv.org/abs/2609.28919); [*Verdict: Encode Once, Decide Many*, `arXiv:2610.20778`](https://arxiv.org/abs/2610.20778)).
2. **Overconfidence Induced by RLHF / RLVR**: Reinforcement Learning from Verifiable Rewards (`RLVR`, e.g., GRPO on binary correctness) optimizes the group-mixture reward without a diversity term, driving reasoning models toward extreme, uncalibrated probabilities near `0` or `1` ([*OpenJev-RLCD*, `arXiv:2609.38850`](https://arxiv.org/abs/2609.38850); [*Beyond Binary Rewards*, `arXiv:2507.16806`](https://arxiv.org/abs/2507.16806); [*Don't Think Twice!*, `arXiv:2508.15050`](https://arxiv.org/abs/2508.15050)).
3. **Permutation & Option-Order Fragility**: Reading next-token logits from a causal LLM exhibits up to a **33% Order-Flip Rate** when candidate options are permuted, due to coupled label-token priors and primacy/recency slot bias ([*AnyJev*, `arXiv:2610.00831`](https://arxiv.org/abs/2610.00831); [*LLMs Are Not Robust Multiple Choice Selectors*, `arXiv:2309.00667`](https://arxiv.org/abs/2309.00667); [*Jeva*, `arXiv:2610.18773`](https://arxiv.org/abs/2610.18773)).
4. **Conflation of Aleatoric Tie Uncertainty and Epistemic Ignorance**: A flat softmax distribution (`[0.5, 0.5]`) can mean either that two candidates are genuinely equally plausible or that the model has never seen the domain and "doesn't know" ([*Sys1Cal-v1*, `arXiv:2609.35342`](https://arxiv.org/abs/2609.35342); [*When the Right Answer Is Missing*, `arXiv:2609.39496`](https://arxiv.org/abs/2609.39496)).

**MicroLLM** synthesizes the state of the art across all six research pillars into a local, `<2 ms` CPU hybrid engine that satisfies `32/32 MUST` requirements of the `jevcompat 0.1` wire specification.

---

## 2. Deep-Dive Synthesis Across the Six Thematic Pillars

### 2.1 Pillar 1 — Core Jev, System-One & Typed Decision Architectures (21 Papers)

The first wave of System-One literature establishes how to compute calibrated categorical distributions in a single forward pass without autoregressive decoding:

- **Single-Pass Encoders & Encode-Once-Decide-Many**:
  - **StartLux-Decision Instant** ([`arXiv:2610.17309`](https://arxiv.org/abs/2610.17309)), **Koa-decision** ([`arXiv:2609.36115`](https://arxiv.org/abs/2609.36115)), **Laya** ([`arXiv:2609.33843`](https://arxiv.org/abs/2609.33843)), **this-that-model-1.0** ([`arXiv:2609.23886`](https://arxiv.org/abs/2609.23886)), and **SBERT2S1** ([`arXiv:2610.02486`](https://arxiv.org/abs/2610.02486)) reformulate typed decisions as bidirectional cross-attention or NLI entailment scoring (`format_nli_hypothesis` in [`src/microllm/backends/fast_encoder.py`](../src/microllm/backends/fast_encoder.py)).
  - **Verdict-MQ** ([`arXiv:2610.20778`](https://arxiv.org/abs/2610.20778)) and **Jeva** ([`arXiv:2610.18773`](https://arxiv.org/abs/2610.18773)) prove that encoding the `state` once and scoring multiple independent questions or cached option embeddings avoids redundant state encoding while guaranteeing zero cross-question leakage (`semantics.batching-invariance`) and exact permutation invariance. MicroLLM implements this in [`src/microllm/backends/base.py`](../src/microllm/backends/base.py) via content-addressable `BLAKE2b` embedding memoization (`IDEA-012`) and isolated per-question evaluation (`_evaluate_isolated_question` in [`src/microllm/server.py`](../src/microllm/server.py)).
- **Training-Free Prefill Readout & Algebraic Slot-Bias Cancellation**:
  - **AnyJev** ([`arXiv:2610.00831`](https://arxiv.org/abs/2610.00831)), **Releasing the Inner Jev** ([`arXiv:2610.17205`](https://arxiv.org/abs/2610.17205)), and **LLM-as-Jev** ([`arXiv:2610.02076`](https://arxiv.org/abs/2610.02076)) show that instruction-tuned LLMs already contain strong System-One representations at the first prefill token, provided two defects are removed: (i) static label bias (corrected via `OnlinePriorTracker` `p_i / q_i^alpha`) and (ii) positional slot bias `b_j` (algebraically cancelled in log-space by averaging across the cyclic group `C_K` in `l0_circular_rotation_readout`, accelerated by the binary-reflected Van der Corput schedule).
- **Looped, Narrow & Continual Decision Models**:
  - **LEV** ([`arXiv:2610.20951`](https://arxiv.org/abs/2610.20951)), **Narrow Decision Models** ([`arXiv:2610.25553`](https://arxiv.org/abs/2610.25553)), **FastJEV** ([`arXiv:2610.11379`](https://arxiv.org/abs/2610.11379)), and **FunDM** ([`arXiv:2610.17175`](https://arxiv.org/abs/2610.17175)) motivate MicroLLM's INT8 quantized CPU profiles (`fast-cpu-int8`, `multilingual`, `guardrail` in [`src/microllm/registry.py`](../src/microllm/registry.py)) and confidence-gated two-gear cascade.
  - **Jev-DLC** ([`arXiv:2610.31521`](https://arxiv.org/abs/2610.31521)) demonstrates backprop-free, history-free continual decision adaptation from typed revocation feedback pairs (`rejected -> preferred`), shaping roadmap item `IDEA-053`.

---

### 2.2 Pillar 2 — Probability Calibration, Coherence, Scoring Rules & Uncertainty (21 Papers)

A central contribution of the October 2026 literature is proving that **post-hoc ECE calibration alone is not sufficient** for reliable automated decision-making:

1. **Kolmogorov Probability Axioms & Coherence** ([*Beyond Calibration*, `arXiv:2609.33209`](https://arxiv.org/abs/2609.33209); [*Do System One Decisions Add Up?*, `arXiv:2609.33971`](https://arxiv.org/abs/2609.33971)):
   - Hosted black-box decision models frequently violate binary complementation (`P(yes) + P(no) != 1` when phrased positively vs. negatively) and partition invariance (splitting one option `A` into `A1, A2` inflates total probability mass).
   - MicroLLM enforces strict simplex normalization (`sum(p_i) = 1.0`), Per-K temperature scaling (`PerKTemperatureScaler` in [`src/microllm/calibration.py`](../src/microllm/calibration.py)), and hierarchical tournament re-normalization (`run_hierarchical_tournament` in [`src/microllm/confidence.py`](../src/microllm/confidence.py)).
2. **Epistemic Ignorance vs. Aleatoric Tie Uncertainty** ([*Sys1Cal-v1*, `arXiv:2609.35342`](https://arxiv.org/abs/2609.35342); [*When the Right Answer Is Missing*, `arXiv:2609.39496`](https://arxiv.org/abs/2609.39496); [*Beyond Answer Confidence*, `arXiv:2610.01006`](https://arxiv.org/abs/2610.01006)):
   - *Sys1Cal-v1* demonstrates that standard softmax outputs cannot express *"I don't know"* when the right option is missing or the state is uninformative. MicroLLM pairs `SplitConformalCalibrator` (`x_abstain=True` when the conformal prediction set `C_{1-alpha}(x)` is empty or has cardinality `!= 1`) with the AnyJev **Answer Mass Gate** (`answer_mass_gate_passed` in [`src/microllm/backends/prefill_llm.py`](../src/microllm/backends/prefill_llm.py), `IDEA-005`).
3. **Finite-Sample Probability Contracts** ([*Probability Contracts*, `arXiv:2609.37470`](https://arxiv.org/abs/2609.37470); [* Gentle Introduction to Conformal Prediction*, `arXiv:2107.07511`](https://arxiv.org/abs/2107.07511)):
   - Directly implemented in [`src/microllm/calibration.py`](../src/microllm/calibration.py) via `clopper_pearson_threshold(confidences, correct, target_risk, delta)`, certifying the selective risk bound `P(Y_hat != Y | C(X) >= tau) <= r` with probability `>= 1 - delta`.
4. **Ordinal-Scale Bias & Prevalence Shift** ([*More Choices, Fewer Decisions*, `arXiv:2609.38827`](https://arxiv.org/abs/2609.38827); [*Prevalence Shift*, `arXiv:2610.26325`](https://arxiv.org/abs/2610.26325); [*PRISM*, `arXiv:2610.18389`](https://arxiv.org/abs/2610.18389)):
   - Addressed in MicroLLM via `unimodal_ordinal_smooth` + `earth_movers_distance_1d` (`IDEA-008`), `VectorScalingCalibrator` (`IDEA-004`), and `OnlinePriorTracker` (`IDEA-006`).
5. **Proper Scoring Rules & Two-Stage RLCD²** ([*OpenJev-RLCD*, `arXiv:2609.38850`](https://arxiv.org/abs/2609.38850); [*StartLux-Decision*, `arXiv:2610.17309`](https://arxiv.org/abs/2610.17309)):
   - Implemented in [`src/microllm/rlcd.py`](../src/microllm/rlcd.py) (`RLCDDecisionHeadTrainer`, `group_mean_variance_identity`, `shifted_brier_reward`, and `RLCDReplayBuffer`).
6. **Multi-Valid Allocation, Shift Monitoring, Cross-Primitive Consistency & Logit-Space Attribution** ([`arXiv:2610.29217`](https://arxiv.org/abs/2610.29217); [*SHIFT-JEV*, `arXiv:2610.34329`](https://arxiv.org/abs/2610.34329); [*Cross-Primitive Consistency*, `arXiv:2610.38623`](https://arxiv.org/abs/2610.38623); [*S1-SHAP Audit*, `arXiv:2611.12521`](https://arxiv.org/abs/2611.12521)):
   - **Asymmetric Co-Correct Allocation** (`arXiv:2610.29217`) proves forced single-winner softmax assigns under `1/K` mass to valid co-answers in `28%–50%` of sets, validating MicroLLM's independent binary sigmoid `multi` primitive.
   - **SHIFT-JEV** (`arXiv:2610.34329`) catches `95%–99%` of class-prior shifts via a chi-squared predicted-label test (`IDEA-053`), while **Cross-Primitive Consistency** (`arXiv:2610.38623`) predicts decision failures at `0.91` ROC-AUC from `noul`/`choice`/`score` disagreement (`PrimaCons`).
   - **Logit-Space `S1-SHAP`** (`arXiv:2611.12521`) eliminates post-softmax saturation masking on disjunctive `OR` triggers (raising feature recovery from `57%–82%` to `89%–100%`), informing `logit_drop` and `original_log_odds` in `explain_counterfactual` (`IDEA-055`).

---

### 2.3 Pillar 3 — Benchmarks, Empirical Audits & Structural Failure Modes (18 Papers)

Recent benchmark and audit papers expose subtle failure modes in naive typed-decision implementations:

- **37-Dataset Audits & 14-Item Evaluation Checklist** ([`arXiv:2609.37647`](https://arxiv.org/abs/2609.37647); [`arXiv:2609.32160`](https://arxiv.org/abs/2609.32160); [*System1-Arena*, `arXiv:2610.21244`](https://arxiv.org/abs/2610.21244); [*TypedBench*, `arXiv:2610.11392`](https://arxiv.org/abs/2610.11392)):
  - Establish that decision models must be audited simultaneously on accuracy, calibration (`ECE`, `Brier`), permutation invariance (`Order-Flip Rate`), contrastive sensitivity (`DecisionPivot`), and tail latency (`p50`/`p95`). MicroLLM's 8-track, 750–1,000 case validation suite ([`src/microllm/quality_suite.py`](../src/microllm/quality_suite.py) and [`docs/TEST_PLAN.md`](TEST_PLAN.md)) implements all 14 checklist items.
- **Option Name vs. Definition Conflict** ([*Type-Safe Is Not Error-Free*, `arXiv:2609.26758`](https://arxiv.org/abs/2609.26758); [*Labels Override Definitions*, `arXiv:2610.02586`](https://arxiv.org/abs/2610.02586)):
  - Demonstrate that when an option key (e.g., `"urgent"`) contradicts or narrows its criteria definition, many models ignore the definition and decide purely on the key label (dropping accuracy by up to 50 percentage points). MicroLLM's `format_nli_hypothesis(key, description)` in [`src/microllm/backends/fast_encoder.py`](../src/microllm/backends/fast_encoder.py) explicitly binds both the option key and its full criteria description into the NLI hypothesis.
- **Contrastive Sensitivity (`DecisionPivot`) & Irrelevant Alternatives (`IIA`)** ([`arXiv:2610.28522`](https://arxiv.org/abs/2610.28522); [`arXiv:2610.16655`](https://arxiv.org/abs/2610.16655)):
  - MicroLLM achieves **100.0% contrastive pair accuracy** (40/40 pairs) on `DecisionPivot` by combining exemption-aware policy parsing (`RCA-001`) with Gear 2 `AnyJev` escalation.
- **Multilingual Policy Transfer & Reliability Arenas** ([*XDecideBench*, `arXiv:2611.01370`](https://arxiv.org/abs/2611.01370); [*PolyRoute*, `arXiv:2611.13843`](https://arxiv.org/abs/2611.13843); [*SO-Arena*, `arXiv:2611.15777`](https://arxiv.org/abs/2611.15777)):
  - **XDecideBench** (122 languages) and **PolyRoute** (478 native + Hinglish/Spanglish/Arabizi code-mixed routes) prove English-fitted confidence cutoffs do not transfer directly to French, German, or code-mixed inputs (`IDEA-054`), while **SO-Arena** benchmarks 21 System-One models on option-set stability and missing-answer abstention.

---

### 2.4 Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration (22 Papers)

This pillar studies how System-One models interact with slower System-Two reasoners and multi-agent workflows:

- **Accept When Confident, Escalate When Unsure**:
  - **JEV-as-a-Judge** ([`arXiv:2609.26550`](https://arxiv.org/abs/2609.26550)), **REFLEX** ([`arXiv:2609.26532`](https://arxiv.org/abs/2609.26532)), **FrugalGPT** ([`arXiv:2305.05176`](https://arxiv.org/abs/2305.05176)), and **RouteLLM** ([`arXiv:2406.18665`](https://arxiv.org/abs/2406.18665)) formalize two-tier routing: accept Gear 1 when confidence exceeds `gate_threshold`, and escalate uncertain cases to Gear 2. MicroLLM implements this in `DualGearCascadeBackend` ([`src/microllm/backends/cascade.py`](../src/microllm/backends/cascade.py)) with adaptive entropy/answer-mass fusion `compute_adaptive_alpha` (`IDEA-002`) and Pareto frontier auto-tuning `tune_cascade_pareto` (`IDEA-010`).
- **Agent Harnesses, Memory Compaction & Continuous System-2 Monitoring**:
  - **JevAgentHarness** ([`arXiv:2610.21412`](https://arxiv.org/abs/2610.21412)), **Jev-LiteMAS** ([`arXiv:2610.21411`](https://arxiv.org/abs/2610.21411)), **Jarvis** ([`arXiv:2610.23804`](https://arxiv.org/abs/2610.23804)), **SOMA** ([`arXiv:2610.26793`](https://arxiv.org/abs/2610.26793)), **Jev-Mem** ([`arXiv:2609.23986`](https://arxiv.org/abs/2609.23986)), **Source-Judged Compaction** ([`arXiv:2610.20927`](https://arxiv.org/abs/2610.20927)), and **The Value of Thinking (`CDPO`)** ([`arXiv:2611.16961`](https://arxiv.org/abs/2611.16961)) show that replacing LLM-generated control flow with `<5 ms` System-One decorators and zero-or-T Value-of-Computation policies reduces agent orchestration token costs by over 90% ([*Harness Tokenomics*, `arXiv:2609.28919`](https://arxiv.org/abs/2609.28919)).

---

### 2.5 Pillar 5 — AI Safety, Guardrails, Alignment Detection & Adversarial Robustness (19 Papers)

- **Zero-Shot Safety & Alignment Detection**:
  - **Just Ask Jev / RLCDAlignBench** ([`arXiv:2609.29429`](https://arxiv.org/abs/2609.29429)), **Jev-Guard** ([`arXiv:2610.18326`](https://arxiv.org/abs/2610.18326)), **Calibrated Prompt-Injection Deferral** ([`arXiv:2610.20907`](https://arxiv.org/abs/2610.20907)), and **SecJev** ([`arXiv:2610.03073`](https://arxiv.org/abs/2610.03073)) demonstrate that calibrated `noul` probes detect prompt injections, destructive shell/SQL operations, PII/credential leaks, and ungrounded hallucinations in a single pass.
  - MicroLLM implements this directly in `POST /v1/guardrail` ([`src/microllm/server.py`](../src/microllm/server.py)) with three-way calibrated action routing (`allow` / `review` / `block`), groundedness checking (`arXiv:2610.20634`), and pluggable `custom_rubrics` (`IDEA-018`).
- **Adversarial Attacks on Decision Models & Policy-Boundary Gates**:
  - **JevAdvBench** ([`arXiv:2609.31142`](https://arxiv.org/abs/2609.31142)), **Decision Hijacking** ([`arXiv:2609.28613`](https://arxiv.org/abs/2609.28613)), **Nudge the Judge** ([`arXiv:2610.18612`](https://arxiv.org/abs/2610.18612)), **DICE Interface Attacks** ([`arXiv:2610.14404`](https://arxiv.org/abs/2610.14404)), and **JevOut** ([`arXiv:2609.30243`](https://arxiv.org/abs/2609.30243)) reveal how option-order manipulation and distractor injection can tilt uncalibrated judges. MicroLLM neutralizes option-order interface attacks (`DICE`) via AnyJev `L0` canonicalized circular rotation (`0.000000` flip rate) and surfaces token-level counterfactual sensitivity via `explain_counterfactual` (`IDEA-019`).
  - **Ranked but Not Stopped** ([`arXiv:2611.02522`](https://arxiv.org/abs/2611.02522)) and **S1-Evo** ([`arXiv:2611.16937`](https://arxiv.org/abs/2611.16937)) expose certainty-claiming and evidence-competition hijacks (`57.47%` high-confidence ASR) that fool naive low-confidence gates, motivating `detect_evidence_competition_hijack` (`IDEA-056`). Meanwhile, **Decision Models as Policy Gates for Coding Agents** ([`arXiv:2611.02155`](https://arxiv.org/abs/2611.02155)) demonstrates that explicit allowed/forbidden boundary pairs cut coding-agent false alarms from `30.6%` to `4.8%` while lifting recall to `98.2%` (`IDEA-057`).

---

### 2.6 Pillar 6 — Multimodal, Numeric, Retrieval & Domain-Specific System-One Models (15 Papers)

- **Numeric & Tree-Structured Decisions**: **NumericJev** ([`arXiv:2609.28587`](https://arxiv.org/abs/2609.28587)) generalizes binary `noul` and `K`-way `choice` into balanced multiway decision trees for continuous interval estimation.
- **Embedding-Free Retrieval & Reranking**: **SAG2** ([`arXiv:2610.22588`](https://arxiv.org/abs/2610.22588)), **Jev-as-a-Judge for IR Reranking** ([`arXiv:2610.23346`](https://arxiv.org/abs/2610.23346)), and **Multimodal Reranking** ([`arXiv:2610.28435`](https://arxiv.org/abs/2610.28435)) pair fast lexical retrieval with batched System-One verification (`POST /v1/batch`).
- **High-Stakes Clinical & Real-Time 6G Edge Control**: **System-One in Medicine** ([`arXiv:2610.18529`](https://arxiv.org/abs/2610.18529)), **Clinical Triage Probabilities** ([`arXiv:2610.24692`](https://arxiv.org/abs/2610.24692)), **CovNav Conformal Adaptive Inference** ([`arXiv:2610.27677`](https://arxiv.org/abs/2610.27677)), and **6G Open RAN RIC Intent Interpretation** ([`arXiv:2609.23136`](https://arxiv.org/abs/2609.23136)) validate MicroLLM's `<2 ms` deterministic CPU latency and Clopper-Pearson / Conformal safety guarantees.

---

## 3. Architectural Traceability Matrix: Research Papers to MicroLLM Modules

| Research Concept / Paper | arXiv Reference | MicroLLM Module & Symbol | Verification / Benchmark Metric |
| :--- | :--- | :--- | :--- |
| **RLCD Stage 1 + Stage 2 LOO-KL & Variance Identity** | [`arXiv:2609.38850`](https://arxiv.org/abs/2609.38850) | `src/microllm/rlcd.py` (`RLCDDecisionHeadTrainer`, `group_mean_variance_identity`) | Exact `< 1e-6` gradient match; Brier `-24%` |
| **AnyJev `L0` Circular Rotation & Van der Corput Stopping** | [`arXiv:2610.00831`](https://arxiv.org/abs/2610.00831) | `src/microllm/backends/prefill_llm.py` (`l0_circular_rotation_readout`) | `0.000000` Order-Flip Rate; `78.3%` rotations saved |
| **Dual-Gear Instant/AutoThink Cascade & `RLCD^2` Beta(8,2)** | [`arXiv:2610.17309`](https://arxiv.org/abs/2610.17309) | `src/microllm/backends/cascade.py` (`DualGearCascadeBackend`) | `98.1%` JevBench-Core; `100%` DecisionPivot |
| **Encode-Once-Decide-Many & Option Caching** | [`arXiv:2610.20778`](https://arxiv.org/abs/2610.20778), [`arXiv:2610.18773`](https://arxiv.org/abs/2610.18773) | `src/microllm/backends/base.py` (`_embed_text_cached` `BLAKE2b` LRU) | `p50 = 1.48 ms` CPU latency; `3x–5x` batch speedup |
| **Finite-Sample Probability Contracts (Clopper-Pearson)** | [`arXiv:2609.37470`](https://arxiv.org/abs/2609.37470) | `src/microllm/calibration.py` (`clopper_pearson_threshold`) | Certified selective risk bound at `delta = 0.05` |
| **Epistemic Abstention & Conformal Prediction Sets** | [`arXiv:2609.35342`](https://arxiv.org/abs/2609.35342), [`arXiv:2610.27677`](https://arxiv.org/abs/2610.27677) | `src/microllm/calibration.py` (`SplitConformalCalibrator`) | `>= 90%` empirical coverage (`alpha = 0.10`) |
| **Zero-Shot AI Safety Guardrails & Policy Trees** | [`arXiv:2609.29429`](https://arxiv.org/abs/2609.29429), [`arXiv:2610.18326`](https://arxiv.org/abs/2610.18326) | `src/microllm/server.py` (`POST /v1/guardrail`, `custom_rubrics`) | `100.0%` Guardrail F1 (`0.0%` FPR on 80 cases) |
| **Counterfactual Probing & Minimal Flip Attribution** | [`arXiv:2610.18389`](https://arxiv.org/abs/2610.18389) | `src/microllm/backends/base.py` (`explain_counterfactual`) | Exact leave-one-token-out probability delta |
| **Ordinal Scale Smoothing & Wasserstein EMD** | [`arXiv:2609.38827`](https://arxiv.org/abs/2609.38827) | `src/microllm/confidence.py` (`unimodal_ordinal_smooth`, `earth_movers_distance_1d`) | `100.0%` `score` accuracy on JevBench-Core |
| **Logit-Space `S1-SHAP` Counterfactual Attribution** | [`arXiv:2611.12521`](https://arxiv.org/abs/2611.12521) | `src/microllm/backends/base.py` (`explain_counterfactual`, `logit_drop`) | Resolves saturated disjunctive `OR` triggers (`IDEA-055`) |
| **Zero-or-T Value-of-Thinking (`CDPO`)** | [`arXiv:2611.16961`](https://arxiv.org/abs/2611.16961) | `src/microllm/backends/cascade.py` (`DualGearCascadeBackend`, `HYP-005`) | Cost-aware local System-1 / System-2 gating |
| **Evidence-Competition & Certainty-Claim Hijack Defense** | [`arXiv:2611.02522`](https://arxiv.org/abs/2611.02522), [`arXiv:2611.16937`](https://arxiv.org/abs/2611.16937) | `src/microllm/backends/base.py` (`detect_evidence_competition_hijack`, `IDEA-056`) | Inter-clause competition + override escalation |
| **Explicit Policy-Boundary Coding-Agent Gates** | [`arXiv:2611.02155`](https://arxiv.org/abs/2611.02155) | `src/microllm/server.py`, `src/microllm/mcp_server.py` (`policy_boundaries`, `IDEA-057`) | False alarm suppression (`30.6% -> 4.8%`) with hard hazard floor |
