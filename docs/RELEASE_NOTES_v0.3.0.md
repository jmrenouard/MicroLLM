# MicroLLM v0.3.0 — Calibrated Multilingual System-1 Decision Engine

We are thrilled to announce the production release of **MicroLLM v0.3.0** (`micro-jev`), delivering a complete, high-performance, calibrated **Multilingual System-1 Decision Engine** across English (EN), French (FR), Spanish (ES), and Portuguese (PT).

MicroLLM v0.3.0 eliminates the heavy latency, memory, and operational costs of generative autoregressive LLMs for deterministic classification, workflow routing, and AI safety guardrails, executing certified probabilistic decisions locally on standard CPU hardware with sub-millisecond speed and mathematically bounded confidence.

---

## 🚀 Key Highlights & Verified Production Metrics

- **Dual-Gear Multilingual Architecture**:
  - **Strategy 1 (Fast Lightweight Engine / Gear 1)**: Pure Python stdlib execution (`unicodedata`, `re`, `functools`) with zero third-party runtime dependencies. Delivers sub-millisecond latency (`p50 < 0.2 ms`, SLA `< 1.0 ms`) with localized concept clusters, fast Unicode unaccenting (`NFKD`), rule-based light suffix stemming, fast language detection (`detect_language_fast`), and localized stopword/negation/connector tables across EN, FR, ES, and PT.
  - **Strategy 2 (Neural Multilingual Models / Gear 2)**: Dual ONNX INT8 cross-encoder models—`multilingual-MiniLMv2-L6-mnli-xnli` (~48 MB INT8, `< 8 ms`) and `mDeBERTa-v3-base-mnli-xnli` (~134 MB INT8, `< 20 ms`). Dynamic `label2id` entailment index resolution (`RCA-075`) and localized source-language hypothesis templates eliminate cross-lingual transfer drift.
- **Unified Strategy Cascade & Serving**:
  - Flexible strategy selection via `multilingual_strategy` (`"fast"`, `"neural"`, `"auto"`) and `neural_model` (`"minilm"`, `"mdeberta"`), fully accessible across FastAPI (`POST /v1/systemone`), Python SDK (`MicroJevClient`), CLI (`microllm serve`), and Model Context Protocol (`microllm mcp`).
- **Interactive Multilingual Decision Studio**:
  - Full multilingual strategy selection and ticket evaluation presets (billing, shipping, technical support) in the Gradio web application (`src/microllm/gradio_app.py`), synchronized with the Hugging Face Space [`jmrenouard/microllm-decision-studio`](https://huggingface.co/spaces/jmrenouard/microllm-decision-studio).
- **Comprehensive E2E Opaque-Box & Adversarial Test Validation**:
  - **913 executable unit tests passed (100% pass rate)**, including 194 comprehensive opaque-box and adversarial tests (`tests/e2e_multilingual/`) verifying end-user contracts across all 4 languages.
  - Zero static typing errors under `mypy src/microllm --ignore-missing-imports --check-untyped-defs`.
  - 100% LaTeX math formatting compliance verified by `pytest tests/test_docs_math_syntax.py`.
- **Reproducible Multilingual Benchmarks**:
  - Synthetic probe tickets across billing, shipping, and tech support in EN, FR, ES, and PT with isolated benchmark telemetry recorded in `benchmarks/multilingual_benchmarks_results.json` and documented in `BENCHMARKS.md §11`.
  - Hugging Face Hub dataset publishing utilities under `jmrenouard/microllm-multilingual-benchmarks` with `--dry-run` and live token resolution.
- **In-Depth Failure Modes & Architectural Critiques**:
  - Comparative critique of lightweight vs neural strategies and detailed worst-case failure mode analysis documented in `docs/FAILURE_MODES_AND_WORST_CASES.md` and `docs/MULTILINGUAL_STRATEGIES_SUMMARY.md`.

---

## 🛠 Features & Capabilities

### 1. Strategy 1: Lightweight Multilingual Engine
- **Localized Lexicons**: Package `microllm.lexicons` containing curated lexicons for `en.py`, `fr.py`, `es.py`, and `pt.py`.
- **Multilingual Normalizer**: Fast unaccenting (`strip_accents`), rule-based stemming (`light_stem`), normalized tokenization (`tokenize_multilingual_normalized`), and O(1) heuristic language detection (`detect_language_fast`).
- **Concept Clusters**: Expanded `CONCEPT_CLUSTERS` spanning 10 domain categories activating the first 16 slots of the 256-D embedding space.
- **Negation & Connectors**: Multi-word negation phrases, double negations, and localized sentence connectors preventing false-positive sentiment and intent flips.

### 2. Strategy 2: Neural Multilingual Models
- **Dynamic Entailment Resolution**: Resolves entailment label indices dynamically via ONNX `config.json` (`label2id`), resolving hardcoded index assumptions.
- **Native Hypothesis Templates**: Localized templates in `format_nli_hypothesis` across EN (`"It is {opt}: {desc}."`), FR (`"Il s'agit de {opt}: {desc}."`), ES (`"Se trata de {opt}: {desc}."`), and PT (`"Trata-se de {opt}: {desc}."`).
- **Dual Profiles**: Support for `multilingual-minilm-int8` (ultra-fast INT8) and `multilingual-mdeberta-int8` (high-capacity multilingual NLI).

### 3. Dual-Gear Cascade
- Automatically commits Gear 1 when confidence exceeds the calibration threshold $\tau$.
- Transparently escalates ambiguous tickets to Gear 2 neural cross-encoders with adaptive probability fusion:

$$
p_{\text{fused}} = (1 - \alpha) p_0 + \alpha p_1
$$

### 4. Interactive Studio & Cloud Sync
- Interactive Gradio Decision Studio supporting live switching between Fast, MiniLM, mDeBERTa, and Auto cascade modes.
- Seamless synchronization workflow to Hugging Face Spaces (`jmrenouard/microllm-decision-studio`).

---

## 📦 Getting Started

### 1. Installation

Install via pip:

```bash
pip install micro-jev
```

Or install in development mode from source:

```bash
git clone https://github.com/jmrenouard/MicroLLM.git
cd MicroLLM
pip install -e ".[dev]"
```

### 2. Python SDK Example

```python
from microllm import MicroJevClient

client = MicroJevClient()

# Evaluate multilingual support state with Fast Strategy (Sub-millisecond)
response = client.choice(
    state="Client mécontent: la facture #9823 comporte un prélèvement en double.",
    question_id="ticket_category",
    question_text="Quelle est la catégorie de cette demande?",
    options=[
        ("billing", "Problème de facturation ou paiement"),
        ("shipping", "Expédition ou livraison de colis"),
        ("tech_support", "Assistance technique ou panne"),
    ],
    multilingual_strategy="fast",
)

print(f"Decision: {response.best_option}")
print(f"Confidence: {response.confidence:.4f}")
print(f"Abstain: {response.x_abstain}")
```

### 3. REST API & CLI Serving

Launch the local decision engine server:

```bash
microllm serve --host 0.0.0.0 --port 8000 --strategy auto --neural-model minilm
```

Send a request via cURL:

```bash
curl -X POST http://localhost:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jev-latest",
    "state": "Mi pedido no ha llegado y el número de seguimiento no funciona.",
    "multilingual_strategy": "auto",
    "neural_model": "minilm",
    "questions": [
      {
        "id": "departamento",
        "type": "choice",
        "question": "¿A qué departamento corresponde esta consulta?",
        "options": [
          {"name": "facturacion", "description": "Consultas sobre facturas o pagos"},
          {"name": "envios", "description": "Seguimiento y entrega de paquetes"},
          {"name": "soporte", "description": "Problemas técnicos con el servicio"}
        ]
      }
    ]
  }'
```

---

## 📋 Full Milestone Summary (M1–M10)

| Milestone | Scope | Key Deliverables | PR |
| :--- | :--- | :--- | :--- |
| **M1** | Strategy 1: Lightweight Multilingual Engine | `src/microllm/lexicons/` (`en`, `fr`, `es`, `pt`), normalizer, concept clusters, negation | PR #154 |
| **M2** | Strategy 2: Neural Multilingual Models | ONNX entailment fix, localized hypothesis templates, dual model profiles (`minilm`, `mdeberta`) | PR #156 |
| **M3** | Strategy Selection Options | Runtime configuration in `create_app`, `SystemOneRequest`, `MicroJevClient`, and CLI | PR #158 |
| **M4** | Multilingual Benchmark Translation | Translated probe tickets in `microllm_multilingual_corpus.jsonl`, 4-language runner | PR #160 |
| **M5** | Hugging Face Publishing Utilities | Dataset upload (`microllm-multilingual-benchmarks`), model profiles, KeePass token resolution | PR #162 |
| **M6** | Isolated Benchmark Execution | Automated benchmark runner, `multilingual_benchmarks_results.json`, `BENCHMARKS.md §11` | PR #164 |
| **M7** | Gradio Application & HF Space Sync | Strategy selection & multilingual ticket testing in `gradio_app.py`, Space sync automation | PR #166 |
| **M8** | Documentation & Critiques | Comparative critique, worst-case analysis in `docs/FAILURE_MODES_AND_WORST_CASES.md` | PR #168 |
| **M9** | Full E2E Pass & Hardening | 194 opaque-box and adversarial tests passed (100%), latency SLA hardening, typing fixes | PR #170 |
| **M10** | Production Release Publication | v0.3.0 version bump across 8 files, release notes, changelog, git tag, and GitHub Release | PR #228 |
