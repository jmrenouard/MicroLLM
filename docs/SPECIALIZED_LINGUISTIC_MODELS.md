# Specialized Linguistic Models, Multilingual Tokenizers & Decision Gating

This document provides a technical reference for the specialized linguistic components, multilingual tokenizers, semantic concept clusters, cross-lingual NLI hypothesis framing engines, and System-1 vs. System-2 decision mechanisms in the **MicroLLM** framework.

---

## 1. Architectural Architecture & Processing Pipeline

The MicroLLM linguistic subsystem processes multi-lingual textual input through deterministic normalization, concept extraction, localized hypothesis synthesis, and Value-of-Computation (VoC) uncertainty gating:

```
[Raw Multilingual Text Ingress]
              │
              ▼
   [detect_language_fast] (< 0.02 ms: EN, FR, ES, PT, DE, KO)
              │
              ▼
    [tokenize_multilingual] (Unicode range \u0080-\uffff + \w)
              │
              ├──────────────────────────────┐
              ▼                              ▼
  [strip_accents] (NFKD)            [_token_hash_contribution]
  [light_stem] (Suffix guards)      (BLAKE2b word +/-1.6, 3-gram +0.45)
              │                              │
              ▼                              ▼
  [10 Concept Clusters (0..9)]       [256-D Categorical Slot Embedding]
  (Strict word boundary protection)  (Slots 16..255, LRU cache)
              │                              │
              └──────────────┬───────────────┘
                             ▼
     [format_nli_hypothesis] & Multi-Template Consensus (HYP-011)
                             │
                             ▼
  [Clause-Salient Max-Pooling (HYP-008)] (States > 650 chars / Connectors)
                             │
                             ▼
   [Dual-Gear System-1 Decision Gate & Value-of-Computation Gating]
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
     [Gear 1 Direct]                 [Gear 2 Escalation]
     Fast Path (< 0.20 ms)           Neural ONNX / Causal Prefill
     (89.58% volume)                 (10.42% volume)
```

---

## 2. Multilingual Tokenization & Heuristic Normalizers

### 2.1 Standard Library Unicode Tokenizer (`tokenize_multilingual`)
Standard tokenizers in lightweight Python runtimes often drop non-ASCII diacritics or break on multi-byte UTF-8 sequences. MicroLLM uses a regex tokenizer that spans full Unicode alphanumeric and diacritic code blocks:

```python
def tokenize_multilingual(text: str) -> list[str]:
    text_str = text if isinstance(text, str) else str(text)
    lowered = text_str.lower()
    return re.findall(r"[\w\u0080-\uffff]+", lowered)
```

This guarantees non-destructive token extraction for:
- French vowels with circumflex, grave, and acute accents (`é`, `è`, `ê`, `ë`, `à`, `ç`).
- Spanish inverted punctuation and tildes (`ñ`, `¿`, `¡`, `á`, `í`, `ó`, `ú`).
- Portuguese nasal vowels and cedillas (`ã`, `õ`, `ç`, `ê`, `á`).

### 2.2 Instantaneous Language Detection (`detect_language_fast`)
Operating in $< 0.02\text{ ms}$ on CPU with zero external dependencies, `detect_language_fast` uses functional unigram frequencies and exclusive character discriminators:

```
Exclusive Discriminators:
- Portuguese (`PT`): Presence of "ã", "õ", or distinctive digraphs ("lh", "nh").
- Spanish (`ES`): Presence of "ñ", inverted question marks "¿", or exclamations "¡".
- French (`FR`): Functional prepositions ("d'", "l'", "qu'", "pour", "avec", "dans").
- English (`EN`): Common functional markers ("the", "is", "for", "with", "this").
```

### 2.3 Accents Stripping (`strip_accents`) & Light Stemmer (`light_stem`)
- **`strip_accents`**: Employs Unicode Normalization Form KD (NFKD) decomposition via `unicodedata.normalize("NFKD", text)` and filters combining diacritical marks (`Mn`), executing in $0.065\ \mu\text{s}$ per word with LRU caching.
- **`light_stem`**: Length-guarded suffix stripper addressing grammatical inflections across romance languages without third-party stemmers:
  - Nominal plurals (`-s`, `-es`, `-res`, `-des`).
  - Gerunds and participles (`-ing`, `-ment`, `-ando`, `-iendo`, `-ant`, `-ado`, `-ido`).
  - Adverbs (`-mente`, `-ment`, `-ly`).

---

## 3. Semantic Grounding: 10 Multilingual Concept Clusters

Categorical slots $0\text{--}9$ in the MicroLLM 256-dimensional embedding space are reserved for grounded semantic concept clusters:

| Slot | Concept Cluster Name | Dialectal Coverage & Representative Keywords | Strict Boundary Protected |
| :---: | :--- | :--- | :---: |
| `0` | `billing_finance` | invoice, facture, factura, fatura, refund, remboursement, reembolso, charge, subscription | No |
| `1` | `technical_support` | bug, crash, error, panne, erreur, falló, erro, broken, timeout, exception, stacktrace | No |
| `2` | `shipping_logistics` | parcel, package, colis, tracking, paquet, envio, rastreo, entrega, livraison, carrier | No |
| `3` | `urgency_high` | urgent, asap, immediately, immédiatement, secours, critico, emergencia, critical, outage | No |
| `4` | `urgency_low` | whenever, someday, info, pas_presse, quando_puder, informational, minor | No |
| `5` | `affirmative` | yes, oui, si, sim, correct, exact, agree, daccord, confirmo, approved | No |
| `6` | `negative` | no, non, nao, never, jamais, nunca, cancel, refuse, decline, dispute | No |
| `7` | `security_destructive` | rm -rf, drop table, delete from, wipe, truncate, format c:, shred, mkfs | **Yes** |
| `8` | `security_injection` | ignore previous instructions, system prompt, jailbreak, DAN, override system | **Yes** |
| `9` | `security_pii` | ssn, credit card, api key, bearer token, password, private key, seed phrase | **Yes** |

### Strict Word Boundary Protection
To prevent false-positive security escalations on innocent subwords (e.g. matching "format" inside "information" or "table" inside "comfortable"), sensitive security terms enforce regex word boundaries:

```python
STRICT_WORD_BOUNDARY_SECURITY_KEYWORDS = frozenset({
    "format", "table", "system", "credit", "private"
})
```

---

## 4. Cross-Lingual NLI Hypothesis Framing & Consensus

### 4.1 Localized Hypothesis Synthesis (`format_nli_hypothesis`)
Candidate options are dynamically rendered into natural language hypotheses matching the prompt language across three distinct grammatical styles:

$$
h_{i} = \text{HypothesisTemplate}_{\text{lang, style}}(\text{option}_i)
$$

| Language | Default Style | About Style | Inquiry Style |
| :---: | :--- | :--- | :--- |
| **English (`EN`)** | `"This text is about {opt}."` | `"The issue relates to {opt}."` | `"The user is inquiring about {opt}."` |
| **French (`FR`)** | `"Ce texte concerne {opt}."` | `"Il s'agit de {opt}."` | `"Le client pose une question sur {opt}."` |
| **Spanish (`ES`)** | `"Este texto trata de {opt}."` | `"El asunto se refiere a {opt}."` | `"El cliente consulta sobre {opt}."` |
| **Portuguese (`PT`)** | `"Este texto é sobre {opt}."` | `"O assunto refere-se a {opt}."` | `"O cliente tem uma dúvida sobre {opt}."` |

### 4.2 Multi-Template Consensus (`HYP-011`)
When the margin between the top two candidate options is narrow:

$$
\Delta p = p_{(1)} - p_{(2)} \le 0.12
$$

The encoder activates Multi-Template Consensus, evaluating the premise against alternative grammatical frames (`_CONSENSUS_TEMPLATES`) and averaging their logits:

$$
l_{\text{consensus}}(x, o_i) = \frac{1}{|T|} \sum_{t \in T} l(x, t(o_i))
$$

This eliminates framing bias caused by minor lexical variations in the hypothesis sentence.

### 4.3 Clause-Salient Max-Pooling (`HYP-008` / `APP-011`)
In long customer service tickets ($> 650$ characters) containing boilerplate or distractor background clauses, standard global mean pooling dilutes the critical failure sentence. MicroLLM applies clause-salient max-pooling:

$$
l_{\text{salient}}(x, o_i) = 0.25 \cdot l_{\text{global}}(x, o_i) + 0.75 \cdot \max_{c \in \text{clauses}(x)} l(c, o_i)
$$

This isolates the operative sentence (e.g., *"My card was charged twice today"*) while preserving context awareness.

---

## 5. System-1 vs. System-2 Decision Mechanisms & Gating

MicroLLM implements the Kahneman-inspired dual-system architecture formalized in the *StartLux-Decision* framework:

### 5.1 Dual-Gear Routing Mechanics
- **System-1 (Gear 1)**: Instantaneous, non-autoregressive associative evaluation ($< 0.20\text{ ms}$, FastEncoder).
- **System-2 (Gear 2)**: Deliberative, deep cross-encoder or causal prefill rotation ($15\text{--}35\text{ ms}$, DeBERTa-v3-large / AnyJev).

```
Routing Gate Decision Rule:
If (Δp > 0.12) AND (H_norm <= 0.88) AND (Confidence >= 0.60):
    Commit System-1 decision immediately (Gear 1 Fast Path)
Else:
    Escalate to System-2 (Gear 2 Neural / Causal)
    Output Convex Fusion: p_fused = 0.25 * p_fast + 0.75 * p_neural
```

### 5.2 Value-of-Computation (VoC) Uncertainty Regimes (`evaluate_voc_gate_regime`)
The gate assigns queries to one of four mathematically rigorous uncertainty regimes:

1. **`confident_direct`**: High confidence, clear margin, low entropy. System-1 decision committed directly.
2. **`aleatoric_tie_direct`**: Two semantically equivalent candidates tied in probability under irreducible data noise. Direct tie resolution without expensive neural escalation.
3. **`epistemic_deliberative`**: High model uncertainty due to syntactic ambiguity or conflicting clauses. Escalates to System-2.
4. **`open_set_abstain`**: Query falls outside candidate support (conformal prediction set is empty). Conformal abstention triggered.

### 5.3 AnyJev Causal Prefill Readout ($L_0$ Circular Rotation)
When Gear 2 routes to an external causal LLM (e.g. Qwen, Gemma, Llama), standard autoregressive generation suffers from slot primacy bias. AnyJev evaluates circular candidate permutations scheduled via the Van der Corput quasi-random sequence:

$$
\pi_k = \text{VanDerCorputOrder}(k)
$$

Dynamic early-stopping halts evaluation when log-odds margin $\Delta \mathcal{L}$ exceeds threshold $\tau(K, \alpha)$:

$$
\Delta \mathcal{L} \ge \tau(K, \alpha) \iff \text{Early Stop Triggered}
$$

Guaranteed Order Invariance:
Across 114,018 benchmark decisions, MicroLLM maintains:

$$
\text{Flip Rate} = 0.000000
$$

---

## 6. Implementation References

- `src/microllm/backends/base.py`: Tokenizer (`tokenize_multilingual`), token hashing, entailment index resolution.
- `src/microllm/lexicons/normalizer.py`: Accent stripping (`strip_accents`), fast language detection (`detect_language_fast`), stemming (`light_stem`).
- `src/microllm/lexicons/concepts.py`: 10 semantic concept clusters and strict boundary keyword protection.
- `src/microllm/backends/fast_encoder.py`: NLI hypothesis framing (`format_nli_hypothesis`), consensus averaging, context budget truncation.
- `src/microllm/backends/cascade.py`: Dual-gear cascade routing (`MultilingualCascadeBackend`), Value-of-Computation gating (`evaluate_voc_gate_regime`), probability fusion.
- `src/microllm/backends/prefill_llm.py`: AnyJev circular permutation scheduling, Answer Mass Gate, prior debiasing.

---

## 7. Verification Commands

```powershell
# 1. Verify linguistic normalizer and stdlib-only isolation
pytest tests/test_strategy1_lightweight_multilingual.py -v

# 2. Verify NLI hypothesis formatting and dynamic index resolution
pytest tests/test_strategy2_neural_multilingual.py -v

# 3. Verify multilingual decision benchmarks
pytest tests/test_multilingual_benchmarks.py tests/test_multilingual_performance_benchmarks.py -v

# 4. Verify mathematical formatting in all Markdown docs
pytest tests/test_docs_math_syntax.py -v
```
