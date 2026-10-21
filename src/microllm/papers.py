"""Structured Academic Literature Catalog & Search Engine for Jev, RLCD, System-1 & Calibrated LLMs (Issue #42)."""

from __future__ import annotations

from typing import Any

LITERATURE_PILLARS: list[dict[str, str]] = [
    {
        "id": "pillar_1_core_architectures",
        "title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "description": "Single-pass typed decision architectures, AnyJev prefill readout, OpenJev-RLCD, StartLux AutoThink, Verdict-MQ encode-once-decide-many, Jeva null-aware encoders, and looped System-1/2 models.",
    },
    {
        "id": "pillar_2_calibration_coherence",
        "title": "Pillar 2 — Probability Calibration, Coherence, Scoring Rules & Uncertainty",
        "description": "Kolmogorov probability axioms (additivity, complementation, partition invariance), Sys1Cal epistemic heads, Probability Contracts, ordinal-scale bias, prevalence shift, PRISM counterfactual probes, and strictly proper Brier/RLCD scoring rules.",
    },
    {
        "id": "pillar_3_benchmarks_audits",
        "title": "Pillar 3 — Benchmarks, Empirical Audits & Structural Failure Modes",
        "description": "37-dataset Jev evaluations, 14-item Typed Decision Checklist, DecisionPivot contrast pairs, System1-Arena, TypedBench, Option Name vs. Definition bias, Independence of Irrelevant Alternatives (IIA), and open-set rejection policy transfer.",
    },
    {
        "id": "pillar_4_cascades_agents",
        "title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "description": "JEV-as-a-Judge accept/escalate cascades, REFLEX selective control, JevAgentHarness, Jev-LiteMAS, SOMA continuous System-2 monitoring, P(Yes) test-time scaling, source-judged memory compaction, and LoRA decision composition.",
    },
    {
        "id": "pillar_5_safety_adversarial",
        "title": "Pillar 5 — AI Safety, Guardrails, Alignment Detection & Adversarial Robustness",
        "description": "Zero-shot alignment failure detection (Just Ask Jev / RLCDAlignBench), JevAdvBench black-box attacks, Jev-Guard policy trees, Decision Hijacking, DICE interface attacks, JevOut context flips, and calibrated prompt-injection deferral.",
    },
    {
        "id": "pillar_6_multimodal_domain",
        "title": "Pillar 6 — Multimodal, Numeric, Retrieval & Domain-Specific System-One Models",
        "description": "NumericJev multiway decision trees, Visual Jev, UniJev with Cora calibration, Qwen-Audio-Jev, TimesJev, SAG2 BM25+Jev multi-hop retrieval, clinical triage calibration, 6G Open RAN RIC intent routing, and CovNav conformal adaptive inference.",
    },
]

ACADEMIC_PAPERS_CATALOG: list[dict[str, Any]] = [
    {
        "arxiv_id": "2609.38850",
        "versioned_id": "2609.38850v1",
        "title": "OpenJev-RLCD: A Working RLCD Implementation",
        "authors": [
            "Zhimin Gao",
            "Pichao Wang"
        ],
        "published": "2026-09-30",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_1_core_architectures",
        "pillar_title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "abs_url": "https://arxiv.org/abs/2609.38850",
        "pdf_url": "https://arxiv.org/pdf/2609.38850.pdf",
        "summary": "Decision models such as Jev answer questions with probabilities, which are only useful if they are calibrated. Open-source reproductions rely on supervised fine-tuning plus temperature scaling, while reinforcement learning from verifiable rewards (RLVR) makes reasoning models overconfident. We present a working implementation of reinforcement learning...",
        "microllm_relevance": "Core foundation of src/microllm/rlcd.py: Proposition 1 group-mean variance identity, shifted Brier reward r(u, y*), Stage 1 pathwise calibration + Stage 2 REINFORCE LOO-KL."
    },
    {
        "arxiv_id": "2610.00831",
        "versioned_id": "2610.00831v1",
        "title": "AnyJev Technical Report",
        "authors": [
            "Jiamu Zhang",
            "Tianze Yang",
            "Yucheng Shi",
            "Evan Chen",
            "Zixiang Nie",
            "Kelly Wan"
        ],
        "published": "2026-09-30",
        "primary_category": "cs.LG",
        "pillar_id": "pillar_1_core_architectures",
        "pillar_title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "abs_url": "https://arxiv.org/abs/2610.00831",
        "pdf_url": "https://arxiv.org/pdf/2610.00831.pdf",
        "summary": "A typed decision is a choice among a fixed set of options, returned as a probability rather than as text. Systems that need typed decisions today use models trained for that purpose. This report describes AnyJev, which reads a typed decision from one prefill of a pretrained instruction-tuned language model. The readout restricts the next-token...",
        "microllm_relevance": "Core foundation of src/microllm/backends/prefill_llm.py: AnyJev L0 circular rotation readout (Theorem 1 exact slot-bias cancellation), Van der Corput early stopping, and prior debiasing."
    },
    {
        "arxiv_id": "2610.17309",
        "versioned_id": "2610.17309v2",
        "title": "StartLux-Decision: AutoThink Decision Models with Reinforcement Learning for Calibrated Deliberative Decisions",
        "authors": [
            "Xialie Zhuang",
            "Zunhai Su",
            "Jia Pan",
            "Ngai Wong",
            "Guanhua Chen",
            "Chuan-Wei Kuo"
        ],
        "published": "2026-10-12",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_1_core_architectures",
        "pillar_title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "abs_url": "https://arxiv.org/abs/2610.17309",
        "pdf_url": "https://arxiv.org/pdf/2610.17309.pdf",
        "summary": "We introduce StartLux-Decision, decision models from 0.8B to 27B parameters that combine efficient single-pass prediction with adaptive, confidence-guided reasoning. The Instant models use bidirectional attention in full-attention layers to improve decision accuracy while keeping single-pass inference. The AutoThink models invoke explicit reasoning for...",
        "microllm_relevance": "Core foundation of src/microllm/backends/cascade.py and src/microllm/rlcd.py: Instant (Gear 1) vs. AutoThink (Gear 2) dual-gear cascade, RLCD^2 Beta(8,2) weighting, and DecisionPivot benchmark."
    },
    {
        "arxiv_id": "2610.18773",
        "versioned_id": "2610.18773v2",
        "title": "Jeva: System One Models from First Principles",
        "authors": [
            "Qi Li",
            "Xinchao Wang"
        ],
        "published": "2026-10-13",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_1_core_architectures",
        "pillar_title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "abs_url": "https://arxiv.org/abs/2610.18773",
        "pdf_url": "https://arxiv.org/pdf/2610.18773.pdf",
        "summary": "System one models like Jev enable direct, low-latency decision-making. Yet recent open-source practice of predicting an answer from serialized text does not, by itself, respect the structure of a decision problem. We argue that current designs underemphasize four important structural requirements. (i) null-aware: Models should distinguish answer absence...",
        "microllm_relevance": "Maps to MicroLLM's BLAKE2b cross-request option embedding cache (IDEA-012), permutation-invariant scoring, and answer-mass null-awareness (IDEA-005)."
    },
    {
        "arxiv_id": "2610.17205",
        "versioned_id": "2610.17205v1",
        "title": "Releasing the Inner Jev: A Simple Recipe for Open Decision Models",
        "authors": [
            "Jonathan Geuter",
            "Andrew Cheng",
            "David Alvarez-Melis"
        ],
        "published": "2026-10-12",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_1_core_architectures",
        "pillar_title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "abs_url": "https://arxiv.org/abs/2610.17205",
        "pdf_url": "https://arxiv.org/pdf/2610.17205.pdf",
        "summary": "LLM-based decision-making systems, also called System One models, have recently gained popularity, most notably through Jev (Almeida, 2026). Given a typed input state, a question, and a set of options, they return a probability for each option in a single forward pass, without text generation. We show that existing open instruction-tuned language models...",
        "microllm_relevance": "Validates single-pass logit readout and LoRA decision-head initialization; informs OpenDecisions capability scoring (accuracy + calibration)."
    },
    {
        "arxiv_id": "2610.02076",
        "versioned_id": "2610.02076v3",
        "title": "LLM-as-Jev: LLMs Are Already Jev-Style Decision Models -- When and How to Fine-Tune Them",
        "authors": [
            "Yinheng Li",
            "Justin Wagle"
        ],
        "published": "2026-10-01",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_1_core_architectures",
        "pillar_title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "abs_url": "https://arxiv.org/abs/2610.02076",
        "pdf_url": "https://arxiv.org/pdf/2610.02076.pdf",
        "summary": "Jev-style decision models have gained surging traction by making decision-making dramatically faster and more efficient, returning categorical probability distributions over predefined options without generating free-form text. However, because Jev's architecture and training methodologies remain proprietary, how to replicate these capabilities...",
        "microllm_relevance": "Validates KL-anchored fine-tuning (beta=0.04 in RLCDConfig) and budgeted System-1 to System-2 escalation."
    },
    {
        "arxiv_id": "2610.20778",
        "versioned_id": "2610.20778v1",
        "title": "Verdict: Encode Once, Decide Many",
        "authors": [
            "Avdhesh Singh Chouhan",
            "Jack Perry"
        ],
        "published": "2026-10-14",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_1_core_architectures",
        "pillar_title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "abs_url": "https://arxiv.org/abs/2610.20778",
        "pdf_url": "https://arxiv.org/pdf/2610.20778.pdf",
        "summary": "Automating organizational work means answering many small typed questions about one piece of state (does this email need a reply, how urgent is it, who owns it) with probabilities good enough to act on without a human. The open typed-decision model we compare against, like a standard cross-encoder, answers each question with a separate pass over the...",
        "microllm_relevance": "Directly motivates Verdict-MQ shared state encoding across batched questions in POST /v1/systemone and content-addressable state caching (IDEA-012, IDEA-022)."
    },
    {
        "arxiv_id": "2610.20951",
        "versioned_id": "2610.20951v1",
        "title": "LEV: Thinking Fast and Deep with Looped Decision Models",
        "authors": [
            "Dingjie Song",
            "Zhiling Yan",
            "Yuxuan Zhang",
            "Lichao Sun"
        ],
        "published": "2026-10-14",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_1_core_architectures",
        "pillar_title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "abs_url": "https://arxiv.org/abs/2610.20951",
        "pdf_url": "https://arxiv.org/pdf/2610.20951.pdf",
        "summary": "Decision models answer typed questions about an input with one probability per option, computed in a single forward pass without generating text. Like the fast System 1 of dual-process theory, they answer at once, but their depth is fixed at training time, so they spend the same computation on easy and hard questions. Models that reason in generated text...",
        "microllm_relevance": "Informs iterative latent refinement between fast single-pass Gear 1 and deeper Gear 2 deliberation via dynamic stopping criteria."
    },
    {
        "arxiv_id": "2610.25553",
        "versioned_id": "2610.25553v1",
        "title": "Narrow Decision Models: A System-1 Architecture for Calibrated, Typed Decisions",
        "authors": [
            "Carlos André Antunes",
            "Aldry Albuquerque",
            "Gabriel de Melo Schmith",
            "Natã Batista",
            "Pedro Oziel Escobar da Silva",
            "Stefano Benko Rossi"
        ],
        "published": "2026-10-18",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_1_core_architectures",
        "pillar_title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "abs_url": "https://arxiv.org/abs/2610.25553",
        "pdf_url": "https://arxiv.org/pdf/2610.25553.pdf",
        "summary": "System One models, introduced with TypeSafe's Jev, answer typed questions about a state with calibrated probabilities instead of generated text: choosing among options described in natural language (Choice), placing the state on an ordered scale (Score), or judging a yes/no statement (Noul). Jev is general-purpose: one set of weights, served as an API...",
        "microllm_relevance": "Supports MicroLLM's sub-10ms specialized CPU profiles (fast-cpu-int8, multilingual, guardrail) in src/microllm/registry.py."
    },
    {
        "arxiv_id": "2610.25456",
        "versioned_id": "2610.25456v1",
        "title": "System X: An LLM Decision Model That Replies from Its Own KV Cache",
        "authors": [
            "Syed Usama Khalid Bukhari",
            "Syed Safwan Khalid",
            "Muhammad Aamir"
        ],
        "published": "2026-10-18",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_1_core_architectures",
        "pillar_title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "abs_url": "https://arxiv.org/abs/2610.25456",
        "pdf_url": "https://arxiv.org/pdf/2610.25456.pdf",
        "summary": "Decision models return probabilities over supplied options, but a deployed agent may also need to reply in words. The present study introduces System X, a large language model (LLM) decision model that adds a text-reply option to the LLM-as-Jev readout. When this option wins, or confidence is low, the same model continues from the decision's key-value...",
        "microllm_relevance": "Inspires zero-generation KV-cache readout for multi-question probing on long state contexts."
    },
    {
        "arxiv_id": "2610.22131",
        "versioned_id": "2610.22131v1",
        "title": "Intern-Decision: Efficient Multimodal Decision Making with Autoregressive Masked Language Modeling",
        "authors": [
            "Tong Zhu",
            "Zhe Xu",
            "Zefeng He",
            "Yunkai Chai",
            "Xuyang Hu",
            "Dehong Zheng"
        ],
        "published": "2026-10-15",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_1_core_architectures",
        "pillar_title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "abs_url": "https://arxiv.org/abs/2610.22131",
        "pdf_url": "https://arxiv.org/pdf/2610.22131.pdf",
        "summary": "Interactive agents repeatedly choose among a finite set of actions, labels, or ordinal outcomes. Generating a free-form response for each choice adds latency and complicates the interface between a model and its environment. We present Intern-Decision, a family of 0.8B, 2B, and 4B multimodal decision models that return structured choices and candidate...",
        "microllm_relevance": "Provides bilingual EN/ZH multi-task typed decision benchmarks and cross-attention pooling patterns."
    },
    {
        "arxiv_id": "2610.23170",
        "versioned_id": "2610.23170v1",
        "title": "Chinese Decision: A Family of Language Models for Single-Pass Probabilistic Decision-Making",
        "authors": [
            "Qiang Yan"
        ],
        "published": "2026-10-16",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_1_core_architectures",
        "pillar_title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "abs_url": "https://arxiv.org/abs/2610.23170",
        "pdf_url": "https://arxiv.org/pdf/2610.23170.pdf",
        "summary": "Many language-based applications require probabilities over a finite set of decisions rather than open-ended responses. We present Chinese Decision, a family of Chinese-language models for single-pass probabilistic decision-making. Given context, an instruction, and candidate descriptions, the model extracts final-position candidate-letter logits in one...",
        "microllm_relevance": "Informs multilingual tokenization (tokenize_multilingual in src/microllm/backends/base.py) for non-Latin scripts."
    },
    {
        "arxiv_id": "2610.17175",
        "versioned_id": "2610.17175v1",
        "title": "FunDM: Cheap & Label-Free Decision Models from Frozen Language Models",
        "authors": [
            "Ujjawal K. Panchal"
        ],
        "published": "2026-10-12",
        "primary_category": "cs.LG",
        "pillar_id": "pillar_1_core_architectures",
        "pillar_title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "abs_url": "https://arxiv.org/abs/2610.17175",
        "pdf_url": "https://arxiv.org/pdf/2610.17175.pdf",
        "summary": "Decision models such as Jev score candidate answers or actions. A small decision head trained label-free can recover about 99% of a frozen language model's multiple-choice accuracy while using only two-thirds of its transformer layers. It learns the full model's answer probabilities with KL divergence, without correct-answer labels for training. One...",
        "microllm_relevance": "Complements OnlinePriorTracker (IDEA-006) with label-free self-supervised decision head distillation."
    },
    {
        "arxiv_id": "2610.11379",
        "versioned_id": "2610.11379v1",
        "title": "FastJEV: Understanding Redundancy for Compact JEV Inference",
        "authors": [
            "Jie Ma",
            "Jie Gao",
            "Yihang Liu",
            "Zhike Qiu",
            "Junle Li",
            "Chongyi Zhuang"
        ],
        "published": "2026-10-08",
        "primary_category": "cs.CV",
        "pillar_id": "pillar_1_core_architectures",
        "pillar_title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "abs_url": "https://arxiv.org/abs/2610.11379",
        "pdf_url": "https://arxiv.org/pdf/2610.11379.pdf",
        "summary": "JEV models make multimodal decisions by directly scoring candidates. Although the common context is encoded once, candidate evaluation can still repeat matching token histories, duplicate inference states, and execute the full backbone. In this paper, we study these sources of redundancy and present FastJEV for compact candidate evaluation. We jointly...",
        "microllm_relevance": "Reinforces sub-2ms CPU execution via symmetric INT8 quantization (quantize_matrix_int8 in src/microllm/registry.py)."
    },
    {
        "arxiv_id": "2610.02486",
        "versioned_id": "2610.02486v1",
        "title": "From Retrieval to Typed Decisions: Calibrated System One Models from Biomedical Sentence Encoders",
        "authors": [
            "Pritam Deka"
        ],
        "published": "2026-10-01",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_1_core_architectures",
        "pillar_title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "abs_url": "https://arxiv.org/abs/2610.02486",
        "pdf_url": "https://arxiv.org/pdf/2610.02486.pdf",
        "summary": "Typed decision models answer schema-constrained questions about a text in one forward pass and return probabilities meant to be thresholded. We ask whether biomedical sentence encoders trained for retrieval are good starting points for such models. We present SBERT2S1, which converts Sentence-Transformers encoders into bi-encoder, cross-head (C) and...",
        "microllm_relevance": "Validates sentence-encoder to System-1 NLI hypothesis formatting (format_nli_hypothesis in src/microllm/backends/fast_encoder.py)."
    },
    {
        "arxiv_id": "2609.23886",
        "versioned_id": "2609.23886v1",
        "title": "this-that-model-1.0: A typed decision model that decides in 30 ms, for a millionth of a cent",
        "authors": [
            "Zehua Cheng",
            "Wei Dai",
            "Jiahao Sun"
        ],
        "published": "2026-09-20",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_1_core_architectures",
        "pillar_title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "abs_url": "https://arxiv.org/abs/2609.23886",
        "pdf_url": "https://arxiv.org/pdf/2609.23886.pdf",
        "summary": "Software delegates more of its branches to models every year: which queue a ticket enters, whether a command is safe to run, whether a claim clears without a person. What the program needs back is not prose. It is one of n declared options and a number it can threshold. Today that costs a round trip to a frontier model -- hundreds of milliseconds, a...",
        "microllm_relevance": "Early open-weight binary/choice encoder baseline compared in StartLux-Decision and OpenJev-RLCD."
    },
    {
        "arxiv_id": "2609.36115",
        "versioned_id": "2609.36115v2",
        "title": "Koa-decision: Fast and Consistent Structured Decision Making with Generative LLMs",
        "authors": [
            "Shenghong Dai",
            "Shiva Kumar Pentyala",
            "Yingchi Liu",
            "Shubham Mehrotra",
            "Suman Banerjee",
            "James Zhu"
        ],
        "published": "2026-09-28",
        "primary_category": "cs.LG",
        "pillar_id": "pillar_1_core_architectures",
        "pillar_title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "abs_url": "https://arxiv.org/abs/2609.36115",
        "pdf_url": "https://arxiv.org/pdf/2609.36115.pdf",
        "summary": "Industry applications often demand low-latency classification, yet current large language model (LLM) approaches remain poorly suited for latency-critical applications. Existing prompting and constrained decoding produce verbose, multi-token outputs that require expensive token-by-token generation, while encoder-based models achieve faster inference but...",
        "microllm_relevance": "Compact bidirectional encoder baseline for JevCompat Choice, Noul, and Score primitives."
    },
    {
        "arxiv_id": "2609.36965",
        "versioned_id": "2609.36965v1",
        "title": "Chinese-Jev: Bringing System One Model to Chinese-Language Tasks",
        "authors": [
            "Zexiao Wang",
            "Zihao Zhang",
            "Xudong Wang",
            "Pan Wang",
            "Ziyi Ye",
            "Haoyu Zhao"
        ],
        "published": "2026-09-29",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_1_core_architectures",
        "pillar_title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "abs_url": "https://arxiv.org/abs/2609.36965",
        "pdf_url": "https://arxiv.org/pdf/2609.36965.pdf",
        "summary": "System One models such as Jev offer an efficient alternative to generative language models for tasks that require decisions rather than open-ended responses. However, existing Jev models exhibit limited Chinese-language decision accuracy, restricting their utility in both general and specialized settings. In this paper, we introduce Chinese-Jev, a System...",
        "microllm_relevance": "Demonstrates cross-lingual transfer of JevCompat primitives across English and Chinese."
    },
    {
        "arxiv_id": "2609.33843",
        "versioned_id": "2609.33843v1",
        "title": "Laya as a Typed Probabilistic Assessor: An Independent Reproduction and a Preregistered Study of Calibration and Selective Escalation",
        "authors": [
            "Gowthamkumar Nandakishore"
        ],
        "published": "2026-09-27",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_1_core_architectures",
        "pillar_title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "abs_url": "https://arxiv.org/abs/2609.33843",
        "pdf_url": "https://arxiv.org/pdf/2609.33843.pdf",
        "summary": "The shipped Laya Typed-Decisions checkpoint, a 421M-parameter ModernBERT-large assessor that answers typed choice/noul/score questions over workflow state, is uniformly under-confident. The signed confidence-accuracy gap is -0.214, every occupied reliability bin's accuracy exceeds its confidence, and that sign uniformity collapses every binned ECE...",
        "microllm_relevance": "Evaluates Laya-421M as a typed probabilistic assessor and motivates Per-K temperature scaling."
    },
    {
        "arxiv_id": "2609.33209",
        "versioned_id": "2609.33209v1",
        "title": "Beyond Calibration: Do a Typed-Decision Model's Probabilities Obey the Probability Axioms?",
        "authors": [
            "Keyi Li",
            "Yihao He",
            "Quanyi Li"
        ],
        "published": "2026-09-27",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_2_calibration_coherence",
        "pillar_title": "Pillar 2 — Probability Calibration, Coherence, Scoring Rules & Uncertainty",
        "abs_url": "https://arxiv.org/abs/2609.33209",
        "pdf_url": "https://arxiv.org/pdf/2609.33209.pdf",
        "summary": "Typed-decision models such as TypeSafe's Jev answer a declared yes/no or multiple-choice question about a state with a probability instead of text, and their evaluations report accuracy and calibration. Neither requires that the probabilities a model gives to logically related questions fit together item by item, which is what a system that acts on those...",
        "microllm_relevance": "Identifies violations of Kolmogorov additivity, complementation (P(yes)+P(no)!=1), and partition invariance in hosted decision models; directly addressed by MicroLLM's normalized simplex and IDEA-023 coherence audit."
    },
    {
        "arxiv_id": "2609.35342",
        "versioned_id": "2609.35342v1",
        "title": "Jev thinks \"I don't know'', but doesn't say it: Introducing Sys1Cal-v1 Dataset for Probability Calibration",
        "authors": [
            "Riccardo Porcedda"
        ],
        "published": "2026-09-28",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_2_calibration_coherence",
        "pillar_title": "Pillar 2 — Probability Calibration, Coherence, Scoring Rules & Uncertainty",
        "abs_url": "https://arxiv.org/abs/2609.35342",
        "pdf_url": "https://arxiv.org/pdf/2609.35342.pdf",
        "summary": "The appearance of Jev marked the era of System One Models, foundation models that return structured decisions with probability distributions rather than text. Aside from low cost and great speed, Jev's central promise is that these probabilities are calibrated: such claim is not backed by any public test and available external benchmarks evaluate...",
        "microllm_relevance": "Demonstrates that standard confidence conflates aleatoric tie uncertainty with epistemic ignorance; motivates MicroLLM's SplitConformalCalibrator abstention flag (x_abstain) and answer-mass gate (IDEA-005)."
    },
    {
        "arxiv_id": "2609.37470",
        "versioned_id": "2609.37470v2",
        "title": "Probability Contracts: Accuracy, Coherence, and Decisions Across LLM Interfaces",
        "authors": [
            "Han Chen",
            "Yingrui Li"
        ],
        "published": "2026-09-27",
        "primary_category": "cs.LG",
        "pillar_id": "pillar_2_calibration_coherence",
        "pillar_title": "Pillar 2 — Probability Calibration, Coherence, Scoring Rules & Uncertainty",
        "abs_url": "https://arxiv.org/abs/2609.37470",
        "pdf_url": "https://arxiv.org/pdf/2609.37470.pdf",
        "summary": "Equivalent probability requests can lead to different decisions even when both reports are valid. We introduce probability contracts, a benchmark that connects exact finite-world posteriors, validated event alignment, interface coherence, and failure-aware decision evaluation. Across four model-interface configurations on 1,000 worlds, Kev has lower...",
        "microllm_relevance": "Directly implemented in src/microllm/calibration.py via clopper_pearson_threshold finite-sample high-probability risk certification."
    },
    {
        "arxiv_id": "2609.33971",
        "versioned_id": "2609.33971v1",
        "title": "Do System One Decisions Add Up? A Study of Probabilistic Coherence",
        "authors": [
            "Saman Sarker Joy"
        ],
        "published": "2026-09-27",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_2_calibration_coherence",
        "pillar_title": "Pillar 2 — Probability Calibration, Coherence, Scoring Rules & Uncertainty",
        "abs_url": "https://arxiv.org/abs/2609.33971",
        "pdf_url": "https://arxiv.org/pdf/2609.33971.pdf",
        "summary": "A decision model can give probabilities that sum to one for every question yet disagree with itself when the same decision is broken into smaller steps. We study this form of probabilistic coherence in Jev and the English Laya checkpoint, using 2,500 matched examples per system across TREC, CLINC150, and MASSIVE. Across 72,000 classification questions,...",
        "microllm_relevance": "Audits sub-additivity and super-additivity when splitting or merging Choice options; motivates hierarchical tournament normalization in src/microllm/confidence.py."
    },
    {
        "arxiv_id": "2609.38827",
        "versioned_id": "2609.38827v2",
        "title": "More Choices, Fewer Decisions: Ordinal-Scale Bias in JEV-like Direct-Decision Models",
        "authors": [
            "Tianxiang Gao",
            "Jinzhe Li",
            "Zhiyuan Li",
            "Yi Chang",
            "Yuan Wu"
        ],
        "published": "2026-09-30",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_2_calibration_coherence",
        "pillar_title": "Pillar 2 — Probability Calibration, Coherence, Scoring Rules & Uncertainty",
        "abs_url": "https://arxiv.org/abs/2609.38827",
        "pdf_url": "https://arxiv.org/pdf/2609.38827.pdf",
        "summary": "Direct-decision models enable efficient classification and evaluation by selecting from predefined candidates, but their reliability depends on more than prediction accuracy. Across 40 datasets and nine models from the JEV, KEV, and StartLux Decision families, we identify a systematic tendency to underuse ordinal rating scales. We call this phenomenon...",
        "microllm_relevance": "Identifies ordinal-scale central-tendency and boundary bias in Score questions; addressed by unimodal_ordinal_smooth and PerKTemperatureScaler (IDEA-008)."
    },
    {
        "arxiv_id": "2610.26325",
        "versioned_id": "2610.26325v1",
        "title": "Typed Decision Models under Prevalence Shift: Decision Training, Calibration, and Implicit Priors",
        "authors": [
            "Abdul Rahman Iqbal",
            "Muhammad Fawad Nasim",
            "Amir Saleem"
        ],
        "published": "2026-10-19",
        "primary_category": "cs.LG",
        "pillar_id": "pillar_2_calibration_coherence",
        "pillar_title": "Pillar 2 — Probability Calibration, Coherence, Scoring Rules & Uncertainty",
        "abs_url": "https://arxiv.org/abs/2610.26325",
        "pdf_url": "https://arxiv.org/pdf/2610.26325.pdf",
        "summary": "Typed decision models (TDMs) return a probability over caller-defined options, and Jev is described by its vendor as calibrated. But in deployment class frequencies change, so calibration measured on a fixed test set need not hold where thresholds are applied. Existing evaluations leave two gaps: the gains of vendor decision training have not been...",
        "microllm_relevance": "Analyzes prior/prevalence shift in deployed decision endpoints; directly addressed by OnlinePriorTracker (IDEA-006) and VectorScalingCalibrator (IDEA-004)."
    },
    {
        "arxiv_id": "2610.26797",
        "versioned_id": "2610.26797v1",
        "title": "Teacher Supervision, Fixed Representations and Calibration: Exact Risk Calculations for Candidate-Scoring Decision Models",
        "authors": [
            "Agus Sudjianto",
            "Denis Burakov"
        ],
        "published": "2026-10-19",
        "primary_category": "stat.ML",
        "pillar_id": "pillar_2_calibration_coherence",
        "pillar_title": "Pillar 2 — Probability Calibration, Coherence, Scoring Rules & Uncertainty",
        "abs_url": "https://arxiv.org/abs/2610.26797",
        "pdf_url": "https://arxiv.org/pdf/2610.26797.pdf",
        "summary": "Candidate-scoring decision models return a probability distribution over a presented set of answers. Public documentation of one such system, Jev, and of an open counterpart, JevAny, describes three training ingredients that determine what these probabilities estimate: soft targets from larger models, a frozen pretrained backbone and temperature...",
        "microllm_relevance": "Provides exact risk asymptotics for fixed-representation decision heads trained against soft teacher probabilities vs. hard labels."
    },
    {
        "arxiv_id": "2610.18488",
        "versioned_id": "2610.18488v1",
        "title": "Recalibrating Many Questions with Few Labels: Minimax Rates and Identification Limits",
        "authors": [
            "Xiaoyu Li",
            "Chiwun Yang"
        ],
        "published": "2026-10-13",
        "primary_category": "cs.LG",
        "pillar_id": "pillar_2_calibration_coherence",
        "pillar_title": "Pillar 2 — Probability Calibration, Coherence, Scoring Rules & Uncertainty",
        "abs_url": "https://arxiv.org/abs/2610.18488",
        "pdf_url": "https://arxiv.org/pdf/2610.18488.pdf",
        "summary": "Many related prediction problems, which we call questions, come with few labels each. What information can be shared across them, and what is actually necessary for good decisions? We answer both for recalibration, motivated by decision models such as Jev (TypeSafe AI, 2026) that return a probability for any yes/no question a developer writes. A...",
        "microllm_relevance": "Supports few-shot recalibration via POST /v1/calibrate and RLCDReplayBuffer (IDEA-009)."
    },
    {
        "arxiv_id": "2610.18389",
        "versioned_id": "2610.18389v1",
        "title": "PRISM: Calibrating Black-Box Judges with Counterfactual Probes",
        "authors": [
            "Kakei Yamamoto",
            "So Kuroki"
        ],
        "published": "2026-10-13",
        "primary_category": "cs.LG",
        "pillar_id": "pillar_2_calibration_coherence",
        "pillar_title": "Pillar 2 — Probability Calibration, Coherence, Scoring Rules & Uncertainty",
        "abs_url": "https://arxiv.org/abs/2610.18389",
        "pdf_url": "https://arxiv.org/pdf/2610.18389.pdf",
        "summary": "Large language models are increasingly used as judges to reduce reliance on costly human evaluation, yet their scores can systematically disagree with human judgments. We ask whether responses to an explicitly imposed evaluative bias can help correct a judge's implicit scoring biases. We introduce PRISM (Probe-based Residual Inference of Scoring...",
        "microllm_relevance": "Inspires counterfactual probing in explain_counterfactual (src/microllm/backends/base.py, IDEA-019)."
    },
    {
        "arxiv_id": "2610.17016",
        "versioned_id": "2610.17016v1",
        "title": "ProbabilityBench: Model Evaluation and Calibration with Exact Ground-Truth Uncertainty",
        "authors": [
            "Yuyang Dai",
            "Yuting Tang",
            "Futoshi Futami",
            "Takashi Ishida"
        ],
        "published": "2026-10-12",
        "primary_category": "cs.LG",
        "pillar_id": "pillar_2_calibration_coherence",
        "pillar_title": "Pillar 2 — Probability Calibration, Coherence, Scoring Rules & Uncertainty",
        "abs_url": "https://arxiv.org/abs/2610.17016",
        "pdf_url": "https://arxiv.org/pdf/2610.17016.pdf",
        "summary": "Evaluating model uncertainty on real-world data is challenging because each instance provides only a single observed label, leaving true conditional class probabilities unknown. To address this, we introduce ProbabilityBench, a benchmark that provides exact, instance-level class probabilities across three simulated text-based domains: peer review, covert...",
        "microllm_relevance": "Provides reference proper scoring rule evaluation protocols (Brier, NLL, ECE) implemented in src/microllm/calibration.py."
    },
    {
        "arxiv_id": "2610.01006",
        "versioned_id": "2610.01006v1",
        "title": "Beyond Answer Confidence: A Controlled Audit of Self-Knowledge in a Black-Box Decision Model",
        "authors": [
            "Sharath M Shankaranarayana",
            "Davor Runje",
            "Jan Jannink"
        ],
        "published": "2026-10-01",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_2_calibration_coherence",
        "pillar_title": "Pillar 2 — Probability Calibration, Coherence, Scoring Rules & Uncertainty",
        "abs_url": "https://arxiv.org/abs/2610.01006",
        "pdf_url": "https://arxiv.org/pdf/2610.01006.pdf",
        "summary": "Decision models return probabilities intended for routing, abstention and automated action. Calibration makes those probabilities useful on average, but does not establish whether low confidence reflects chance or missing knowledge, nor whether confidence falls when a model moves beyond what it knows. We audit this distinction in Jev, a decision model,...",
        "microllm_relevance": "Separates option-competition entropy from state-sufficiency confidence in selective prediction."
    },
    {
        "arxiv_id": "2609.39496",
        "versioned_id": "2609.39496v1",
        "title": "When the Right Answer Is Missing: An Arithmetic-Dependent Rejection Bottleneck in Jev",
        "authors": [
            "Jike Zhong",
            "Ming Li",
            "Yuxiang Lai"
        ],
        "published": "2026-09-30",
        "primary_category": "cs.LG",
        "pillar_id": "pillar_2_calibration_coherence",
        "pillar_title": "Pillar 2 — Probability Calibration, Coherence, Scoring Rules & Uncertainty",
        "abs_url": "https://arxiv.org/abs/2609.39496",
        "pdf_url": "https://arxiv.org/pdf/2609.39496.pdf",
        "summary": "Typed decision models such as Jev offer an efficient alternative to generative LLMs in decision-making workflows by selecting directly from predefined options. When candidate sets contain no valid answer, TypeSafe recommends including an \"other\" or \"none-of-the-above\" option to enable rejection. In this report, however, we identify an...",
        "microllm_relevance": "Analyzes open-set / missing-answer scenarios; mitigated in MicroLLM by SplitConformalCalibrator empty-set abstention and answer_mass_gate_passed."
    },
    {
        "arxiv_id": "2507.16806",
        "versioned_id": "2507.16806v2",
        "title": "Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty",
        "authors": [
            "Mehul Damani",
            "Isha Puri",
            "Stewart Slocum",
            "Idan Shenfeld",
            "Leshem Choshen",
            "Yoon Kim"
        ],
        "published": "2025-07-22",
        "primary_category": "cs.LG",
        "pillar_id": "pillar_2_calibration_coherence",
        "pillar_title": "Pillar 2 — Probability Calibration, Coherence, Scoring Rules & Uncertainty",
        "abs_url": "https://arxiv.org/abs/2507.16806",
        "pdf_url": "https://arxiv.org/pdf/2507.16806.pdf",
        "summary": "When language models (LMs) are trained via reinforcement learning (RL) to generate natural language \"reasoning chains\", their performance improves on a variety of difficult question answering tasks. Today, almost all successful applications of RL for reasoning use binary reward functions that evaluate the correctness of LM outputs. Because such reward...",
        "microllm_relevance": "Foundational derivation of Reinforcement Learning with Calibration Rewards (RLCR) using proper scoring rules, extended to RLCD in OpenJev-RLCD."
    },
    {
        "arxiv_id": "2508.15050",
        "versioned_id": "2508.15050v1",
        "title": "Don't Think Twice! Over-Reasoning Impairs Confidence Calibration",
        "authors": [
            "Romain Lacombe",
            "Kerrie Wu",
            "Eddie Dilworth"
        ],
        "published": "2025-08-20",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_2_calibration_coherence",
        "pillar_title": "Pillar 2 — Probability Calibration, Coherence, Scoring Rules & Uncertainty",
        "abs_url": "https://arxiv.org/abs/2508.15050",
        "pdf_url": "https://arxiv.org/pdf/2508.15050.pdf",
        "summary": "Large Language Models deployed as question answering tools require robust calibration to avoid overconfidence. We systematically evaluate how reasoning capabilities and budget affect confidence assessment accuracy, using the ClimateX dataset (Lacombe et al., 2023) and expanding it to human and planetary health. Our key finding challenges the \"test-time...",
        "microllm_relevance": "Proves that uncalibrated System-2 over-reasoning degrades ECE, motivating StartLux RLCD^2 length penalties and MicroLLM's System-1 fast path."
    },
    {
        "arxiv_id": "2609.37647",
        "versioned_id": "2609.37647v1",
        "title": "Evaluating and Benchmarking the System One Model Jev",
        "authors": [
            "Tobias Deußer",
            "Lorenz Sparrenberg",
            "Rafet Sifa"
        ],
        "published": "2026-09-29",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_3_benchmarks_audits",
        "pillar_title": "Pillar 3 — Benchmarks, Empirical Audits & Structural Failure Modes",
        "abs_url": "https://arxiv.org/abs/2609.37647",
        "pdf_url": "https://arxiv.org/pdf/2609.37647.pdf",
        "summary": "Jev is a commercial System One model from TypeSafe AI that does not generate text: given a state and typed questions, it returns a choice from fixed options, a position on a rubric, or the probability that a statement is true, with probabilities the vendor describes as calibrated. Such models target small decisions in information access pipelines, such...",
        "microllm_relevance": "Establishes the 37-dataset Jev evaluation taxonomy mirrored in MicroLLM's JevBench-Core track (src/microllm/benchmarks.py)."
    },
    {
        "arxiv_id": "2609.32160",
        "versioned_id": "2609.32160v1",
        "title": "Typed Decision Models: An Early Evidence Audit and Evaluation Checklist",
        "authors": [
            "Lijuan Tang",
            "Yuemeng Zheng"
        ],
        "published": "2026-09-26",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_3_benchmarks_audits",
        "pillar_title": "Pillar 3 — Benchmarks, Empirical Audits & Structural Failure Modes",
        "abs_url": "https://arxiv.org/abs/2609.32160",
        "pdf_url": "https://arxiv.org/pdf/2609.32160.pdf",
        "summary": "Typed decision models (TDMs) return probability distributions over caller-defined options without generating text. TypeSafe released Jev, a commercial typed decision model, on 15 September 2026, and a small body of evaluation and replication work appeared within days. We review 28 papers posted between 19 and 24 September and relate their findings to...",
        "microllm_relevance": "Defines the 14-item evaluation checklist for typed decision models (order invariance, calibration, latency, abstention) enforced in docs/TEST_PLAN.md."
    },
    {
        "arxiv_id": "2610.28522",
        "versioned_id": "2610.28522v1",
        "title": "DecisionPivot: A Diagnostic Benchmark for Decision Models in LLM Systems",
        "authors": [
            "Xialie Zhuang",
            "Mingcong Lei",
            "Zunhai Su",
            "Ri Liu",
            "Jianjin Li",
            "Peixian Ma"
        ],
        "published": "2026-10-20",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_3_benchmarks_audits",
        "pillar_title": "Pillar 3 — Benchmarks, Empirical Audits & Structural Failure Modes",
        "abs_url": "https://arxiv.org/abs/2610.28522",
        "pdf_url": "https://arxiv.org/pdf/2610.28522.pdf",
        "summary": "Decision models support routing, guardrails, and output verification in LLM systems. However, item accuracy alone cannot establish whether models make correct decisions across controlled variations or assign reliable confidence to their predictions. We introduce DecisionPivot, a diagnostic benchmark comprising 48 tasks and 40,033 items across seven...",
        "microllm_relevance": "Standalone specification of the DecisionPivot contrastive pair benchmark implemented in Track 2 of src/microllm/benchmarks.py (100% pair accuracy)."
    },
    {
        "arxiv_id": "2610.21244",
        "versioned_id": "2610.21244v1",
        "title": "System1-Arena: Which Decisions Can Be Left to System 1?",
        "authors": [
            "Tianyu Chen",
            "Xinyang Liu",
            "Ruyi An",
            "Zhendong Wang",
            "Mingyuan Zhou"
        ],
        "published": "2026-10-15",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_3_benchmarks_audits",
        "pillar_title": "Pillar 3 — Benchmarks, Empirical Audits & Structural Failure Modes",
        "abs_url": "https://arxiv.org/abs/2610.21244",
        "pdf_url": "https://arxiv.org/pdf/2610.21244.pdf",
        "summary": "Large language models (LLMs) often make discrete decisions, such as flagging phishing or escalating to a human. System-1 models, such as Jev and the OpenAI Decisions API, return probabilities over options in a single forward pass, while System-2 approaches generate reasoning steps. Which decisions can System-1 models handle, and which require generative...",
        "microllm_relevance": "Characterizes which decision families can be handled by System-1 Gear 1 alone vs. requiring Gear 2 escalation."
    },
    {
        "arxiv_id": "2610.11392",
        "versioned_id": "2610.11392v2",
        "title": "TypedBench: A Benchmark for Calibration, Framing Sensitivity, and Cost in System One Decision Models",
        "authors": [
            "Rahul Sharma",
            "Andrew B. Duncan",
            "Gaétan Marceau Caron",
            "Sebastian J. Vollmer"
        ],
        "published": "2026-10-08",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_3_benchmarks_audits",
        "pillar_title": "Pillar 3 — Benchmarks, Empirical Audits & Structural Failure Modes",
        "abs_url": "https://arxiv.org/abs/2610.11392",
        "pdf_url": "https://arxiv.org/pdf/2610.11392.pdf",
        "summary": "System One models output calibrated probabilities over typed answers such as categorical choices, ordinal levels, or binary outcomes, via a non-generative interface. Software can act on these probabilities through thresholds, cost-weighted choices, and escalation rules. Consequently, if these probabilities are miscalibrated or wording-sensitive, the...",
        "microllm_relevance": "Multi-domain benchmark for Noul, Choice, Score, and Multi primitives under strict schema constraints."
    },
    {
        "arxiv_id": "2610.03935",
        "versioned_id": "2610.03935v2",
        "title": "General Decision Models: Benchmarking and Insights Beyond Jev",
        "authors": [
            "Feiyu Duan",
            "Jiayu Lin",
            "Jia Wang",
            "Jun Xiang",
            "Jialiang Wu",
            "Xinnong Zhang"
        ],
        "published": "2026-10-02",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_3_benchmarks_audits",
        "pillar_title": "Pillar 3 — Benchmarks, Empirical Audits & Structural Failure Modes",
        "abs_url": "https://arxiv.org/abs/2610.03935",
        "pdf_url": "https://arxiv.org/pdf/2610.03935.pdf",
        "summary": "General decision models, such as Jev, have recently emerged as efficient alternatives to LLMs for structured judgment and selection. But what kinds of decisions can these models reliably make, and how does their behavior change when individual decisions are composed into larger systems? To study this, we introduce JEVal, a bilingual benchmark comprising...",
        "microllm_relevance": "Compares dedicated System-1 encoders against general LLMs across latency, cost, and calibration axes."
    },
    {
        "arxiv_id": "2610.11978",
        "versioned_id": "2610.11978v1",
        "title": "Specialized Decision Models vs. General-Purpose LLMs: Benchmarking Jev Across Knowledge, Reasoning, and Multilingual Tasks",
        "authors": [
            "Xing Li",
            "Qingcheng Chang",
            "Jinzhong Ning",
            "Changfeng Xu",
            "Shenlong Zhang",
            "Yijia Zhang"
        ],
        "published": "2026-10-08",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_3_benchmarks_audits",
        "pillar_title": "Pillar 3 — Benchmarks, Empirical Audits & Structural Failure Modes",
        "abs_url": "https://arxiv.org/abs/2610.11978",
        "pdf_url": "https://arxiv.org/pdf/2610.11978.pdf",
        "summary": "Jev is a \"System One\" model that returns a choice among given options instead of generating text. We study how such a specialized decision model compares with general-purpose large language models (LLMs). We evaluate Jev on 13 multiple-choice benchmarks covering knowledge, reasoning, and multilingual understanding, and compare it with 19 LLMs in three...",
        "microllm_relevance": "Demonstrates that calibrated small decision models match or beat 10x larger general LLMs on structured routing and rubric scoring."
    },
    {
        "arxiv_id": "2610.24929",
        "versioned_id": "2610.24929v1",
        "title": "What Can't Jev Decide? Probing the Limits of One-Pass Decision Models",
        "authors": [
            "Wang Yang",
            "Xinpeng Li",
            "Yanyan Zhang",
            "Chuang Ma",
            "Shouren Wang",
            "Jianzhe Wei"
        ],
        "published": "2026-10-18",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_3_benchmarks_audits",
        "pillar_title": "Pillar 3 — Benchmarks, Empirical Audits & Structural Failure Modes",
        "abs_url": "https://arxiv.org/abs/2610.24929",
        "pdf_url": "https://arxiv.org/pdf/2610.24929.pdf",
        "summary": "Jev and its open replicas choose among the options a caller supplies with a single forward pass, fast enough to decide inside an agent loop. We ask what such a model cannot decide, and find a simple limit: Jev cannot make a decision that an LLM makes only by writing out intermediate steps. On a benchmark that varies one factor at a time, Jev finds and...",
        "microllm_relevance": "Identifies compositional and multi-hop limits of 1-pass encoders, justifying MicroLLM's DualGearCascadeBackend escalation."
    },
    {
        "arxiv_id": "2609.26758",
        "versioned_id": "2609.26758v3",
        "title": "Type-Safe Is Not Error-Free: Typed Decision Models Follow the Option Name, Not the Definition Bound to It",
        "authors": [
            "Yu Sun",
            "Junhao Xu",
            "Jiajia Shi",
            "Zijin Yang"
        ],
        "published": "2026-09-22",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_3_benchmarks_audits",
        "pillar_title": "Pillar 3 — Benchmarks, Empirical Audits & Structural Failure Modes",
        "abs_url": "https://arxiv.org/abs/2609.26758",
        "pdf_url": "https://arxiv.org/pdf/2609.26758.pdf",
        "summary": "Typed decision models return structured results, but output-type correctness alone does not ensure that decisions follow explicit option definitions. Each option pairs a name with a definition that defines its intended meaning; the name, however, can provide a competing semantic cue. We study this conflict in Jev and two open-weight models by changing...",
        "microllm_relevance": "Discovers that contradictory option keys vs. definitions degrade accuracy by up to 50pp; mitigated in MicroLLM by format_nli_hypothesis weighting both key and definition (IDEA-024)."
    },
    {
        "arxiv_id": "2610.02586",
        "versioned_id": "2610.02586v1",
        "title": "Labels Override Definitions in Jev-Style Typed Decision Models",
        "authors": [
            "Seyedarmin Azizi",
            "Erfan Baghaei Potraghloo",
            "Massoud Pedram"
        ],
        "published": "2026-10-01",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_3_benchmarks_audits",
        "pillar_title": "Pillar 3 — Benchmarks, Empirical Audits & Structural Failure Modes",
        "abs_url": "https://arxiv.org/abs/2610.02586",
        "pdf_url": "https://arxiv.org/pdf/2610.02586.pdf",
        "summary": "A typed decision model answers a fixed question about an input by returning a probability for each of several caller-defined options. Each option carries a short label and a written definition, which is where a developer states the rule the model should apply. Jev introduced this interface for routing, moderation and triage, open implementations...",
        "microllm_relevance": "Quantifies label-override bias when option names conflict with their criteria descriptions; motivates definition-dominant scoring mode."
    },
    {
        "arxiv_id": "2610.16655",
        "versioned_id": "2610.16655v1",
        "title": "Does a Hosted Decision Model Respect Irrelevant Alternatives? A Pre-Registered Test of Option-Set Dependence in Jev",
        "authors": [
            "Dushyant Rajput",
            "Nirdesh Chauhan",
            "Siddharth Kosaraju"
        ],
        "published": "2026-10-12",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_3_benchmarks_audits",
        "pillar_title": "Pillar 3 — Benchmarks, Empirical Audits & Structural Failure Modes",
        "abs_url": "https://arxiv.org/abs/2610.16655",
        "pdf_url": "https://arxiv.org/pdf/2610.16655.pdf",
        "summary": "Decision APIs that return a probability for each option are routinely used as if those probabilities were properties of the options themselves. That reading is valid only if the model satisfies independence of irrelevant alternatives (IIA): adding an option that is neither the right answer nor its strongest rival should leave the odds between those two...",
        "microllm_relevance": "Audits Independence of Irrelevant Alternatives (IIA) when adding decoy options; informs pairwise log-ratio stability in AnyJev L0."
    },
    {
        "arxiv_id": "2610.16191",
        "versioned_id": "2610.16191v1",
        "title": "Correct Choices, Failed Sequences: Choice2Seq Benchmarks Correctness Retention Across Output Interfaces",
        "authors": [
            "Tianyou Lai",
            "Wentao Yue",
            "Zhiying Deng",
            "Qilei Li"
        ],
        "published": "2026-10-12",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_3_benchmarks_audits",
        "pillar_title": "Pillar 3 — Benchmarks, Empirical Audits & Structural Failure Modes",
        "abs_url": "https://arxiv.org/abs/2610.16191",
        "pdf_url": "https://arxiv.org/pdf/2610.16191.pdf",
        "summary": "A model may select the correct answer yet fail to construct it through a sequence of local decisions. We introduce Choice2Seq, a diagnostic benchmark that measures correctness retention across complete-answer selection and sequential construction in independent runs on the same problems and candidate sets. Sequence Composition Rate (SCR) measures the...",
        "microllm_relevance": "Analyzes sequential vs. parallel option scoring for high-cardinality choice sets (K > 25)."
    },
    {
        "arxiv_id": "2610.00346",
        "versioned_id": "2610.00346v2",
        "title": "Benchmarking System One decision models against trained classifiers and language models for automated decision gates",
        "authors": [
            "Amir Rafe",
            "Subasish Das"
        ],
        "published": "2026-09-29",
        "primary_category": "cs.LG",
        "pillar_id": "pillar_3_benchmarks_audits",
        "pillar_title": "Pillar 3 — Benchmarks, Empirical Audits & Structural Failure Modes",
        "abs_url": "https://arxiv.org/abs/2610.00346",
        "pdf_url": "https://arxiv.org/pdf/2610.00346.pdf",
        "summary": "Software that hands branching decisions to a model needs a declared option and a probability it can threshold. Typed decision models, also called System One models, return such probabilities without generating text, while supervised classifiers and generative language models are the established alternatives. One harness sends eight decision-model...",
        "microllm_relevance": "Benchmarks zero-shot System-1 models against supervised domain classifiers across tabular and text tasks."
    },
    {
        "arxiv_id": "2610.03387",
        "versioned_id": "2610.03387v2",
        "title": "Benchmarking candidate coverage and rejection policy transfer in typed decision models",
        "authors": [
            "Jiawen Lu",
            "Tongtong Wu"
        ],
        "published": "2026-10-02",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_3_benchmarks_audits",
        "pillar_title": "Pillar 3 — Benchmarks, Empirical Audits & Structural Failure Modes",
        "abs_url": "https://arxiv.org/abs/2610.03387",
        "pdf_url": "https://arxiv.org/pdf/2610.03387.pdf",
        "summary": "Rejection policies must remain useful as candidate sets and tasks change. We compare Laya, Jev and Qwen2.5-7B-Instruct using public reference labels, testing Laya/Jev policy transfer at equal calibration budgets and all three models on artificial omission, natural retrieval misses and public out-of-scope queries. Source calibration often fails to...",
        "microllm_relevance": "Evaluates transferability of conformal and threshold rejection policies across domain shifts."
    },
    {
        "arxiv_id": "2609.30216",
        "versioned_id": "2609.30216v1",
        "title": "Jev in the Wild: A Data-Driven Analysis of the Jev Model's Functionality, Applications and Ecosystem",
        "authors": [
            "Guoming Ling",
            "Muen Xue",
            "Zijian Ye"
        ],
        "published": "2026-09-24",
        "primary_category": "cs.SE",
        "pillar_id": "pillar_3_benchmarks_audits",
        "pillar_title": "Pillar 3 — Benchmarks, Empirical Audits & Structural Failure Modes",
        "abs_url": "https://arxiv.org/abs/2609.30216",
        "pdf_url": "https://arxiv.org/pdf/2609.30216.pdf",
        "summary": "Jev is a fast, low-cost decision model that answers natural-language questions with choices, binary judgments, and scores. As its public ecosystem grows rapidly, it remains unclear how Jev is used across applications and how public attention relates to project distribution. To answer these questions, we conduct a large-scale, data-driven analysis of...",
        "microllm_relevance": "Field study of production JevCompat deployments measuring real-world latency, token savings, and failure modes."
    },
    {
        "arxiv_id": "2609.26550",
        "versioned_id": "2609.26550v5",
        "title": "JEV-as-a-Judge: Accept When Confident, Escalate When Unsure",
        "authors": [
            "Yubo Li",
            "Yidi Miao",
            "Ramayya Krishnan",
            "Rema Padman"
        ],
        "published": "2026-09-22",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2609.26550",
        "pdf_url": "https://arxiv.org/pdf/2609.26550.pdf",
        "summary": "LLM-as-a-judge scales evaluation, but reasoning judges are slow and costly. We study JEV-as-a-Judge: evaluation with JEV, a decision-only judge that returns label probabilities instead of text, and whose confidence decides whether to accept its verdict or escalate to a reasoning judge. Against sixteen generative and reward-model judges, with blinded...",
        "microllm_relevance": "Direct architectural blueprint for DualGearCascadeBackend: accept System-1 decision when confidence >= gate_threshold, escalate when unsure."
    },
    {
        "arxiv_id": "2609.29769",
        "versioned_id": "2609.29769v3",
        "title": "Decision Models vs. LLMs as Rubric Judges: Cheaper, Faster, and Wrong in the Same Places",
        "authors": [
            "Delip Rao",
            "Chris Callison-Burch"
        ],
        "published": "2026-09-24",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2609.29769",
        "pdf_url": "https://arxiv.org/pdf/2609.29769.pdf",
        "summary": "LLM judges now score outputs against rubrics, in benchmarks and as training rewards. Decision models return probabilities over permitted answers with a confidence score but no free text. Two such models are Jev and an open reproduction, Perplexity's Decider. We compare each with three flash-tier LLM judges on nine panels from seven benchmarks with human...",
        "microllm_relevance": "Shows System-1 rubric judges achieve 10x–50x lower cost/latency than LLM judges with correlated error profiles."
    },
    {
        "arxiv_id": "2609.26532",
        "versioned_id": "2609.26532v1",
        "title": "REFLEX with Jev for Efficient Selective Control in LLM Agents",
        "authors": [
            "Tiantong Wu",
            "Wei Yang Bryan Lim"
        ],
        "published": "2026-09-22",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2609.26532",
        "pdf_url": "https://arxiv.org/pdf/2609.26532.pdf",
        "summary": "LLM agents often use generative models for bounded decisions, raising the question of when these decisions can be handled more efficiently without reducing task success. We study REFLEX, an agent architecture that uses Jev as a fast, typed decision layer and calls a strong LLM when confidence is low, or generation is required. On a frozen 100-task...",
        "microllm_relevance": "Formulates REFLEX risk-controlled selective escalation, complementing Clopper-Pearson thresholds in src/microllm/calibration.py."
    },
    {
        "arxiv_id": "2610.21412",
        "versioned_id": "2610.21412v1",
        "title": "JevAgentHarness: Benchmarking, Deploying and Training Decision Models That Share Control with LLMs",
        "authors": [
            "Kunlun Zhu",
            "Tao Feng",
            "Xu Han",
            "Heng Wang",
            "Ling Yang",
            "Xiangru Tang"
        ],
        "published": "2026-10-15",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2610.21412",
        "pdf_url": "https://arxiv.org/pdf/2610.21412.pdf",
        "summary": "Large language model (LLM) agents now fix software, operate computers and serve customers, but their harnesses, the programs that run them, call a large LLM at every step, which makes long tasks slow and costly. Most of these calls end in a multiple-choice decision, such as which tool to call next. Decision models such as Jev answer these questions...",
        "microllm_relevance": "Demonstrates replacing generative agent routing loops with typed System-1 calls, mirrored by @system_one_router in src/microllm/client.py."
    },
    {
        "arxiv_id": "2610.21411",
        "versioned_id": "2610.21411v2",
        "title": "Jev-LiteMAS: Separating Decisions from Generation in Multi-Agent Workflows",
        "authors": [
            "Kaixiang Wang",
            "Zhaijiacheng Zhou",
            "Jiong Lou",
            "Jie Li"
        ],
        "published": "2026-10-15",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2610.21411",
        "pdf_url": "https://arxiv.org/pdf/2610.21411.pdf",
        "summary": "Multi-agent LLM systems rely on generative models for expert execution, workflow planning, and artifact review. Yet many coordination operations require only bounded decisions, such as selecting the next expert or judging artifact quality. Profiling three representative MAS frameworks, we find that routing and review account for 31.7-43.5% of...",
        "microllm_relevance": "Uses lightweight System-1 probability consensus for multi-agent task allocation and voting."
    },
    {
        "arxiv_id": "2610.26793",
        "versioned_id": "2610.26793v1",
        "title": "SOMA: Continuous Monitoring and Intervention for System Two Reasoning with System One Models",
        "authors": [
            "Ruizhi Qian",
            "Lei Shi",
            "Tianyu Hu",
            "Bo Yan",
            "Song Wang",
            "Yu Tian"
        ],
        "published": "2026-10-19",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2610.26793",
        "pdf_url": "https://arxiv.org/pdf/2610.26793.pdf",
        "summary": "Reasoning models often run into problems without noticing them, even when symptoms are visible in their own text. In humans, metacognition separates monitoring from control: fast, intuitive signals keep watch while slow, deliberate reasoning unfolds and help decide whether to continue, change course, or stop. System One (S1) decision models, such as Jev,...",
        "microllm_relevance": "Uses continuous System-1 Noul/Score probes to monitor and halt diverging System-2 reasoning chains."
    },
    {
        "arxiv_id": "2610.25768",
        "versioned_id": "2610.25768v1",
        "title": "Truncated Reasoning Under a Budget: Evaluating System-1 Models as Decision-makers",
        "authors": [
            "Zhongyin Wang",
            "Zhaoqian Xue",
            "Junting Mu",
            "Haozheng Liu",
            "Ye Yuan",
            "Haolun Wu"
        ],
        "published": "2026-10-19",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2610.25768",
        "pdf_url": "https://arxiv.org/pdf/2610.25768.pdf",
        "summary": "Large language models (LLMs) often generate more reasoning tokens than needed. Limiting the reasoning budget reduces cost, but forcing an answer from an incomplete trace can hurt accuracy. System-1 decision models select among candidate options without producing a reasoning response. We evaluate Jev as a decision reader of truncated reasoning across...",
        "microllm_relevance": "Combines truncated reasoning budgets with calibrated System-1 readout under strict latency SLAs."
    },
    {
        "arxiv_id": "2610.25474",
        "versioned_id": "2610.25474v1",
        "title": "P(Yes): 15 Improvements via Test-Time Scaling of Decision Models",
        "authors": [
            "Lev Stambler"
        ],
        "published": "2026-10-18",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2610.25474",
        "pdf_url": "https://arxiv.org/pdf/2610.25474.pdf",
        "summary": "Decision models answer typed questions, such as multiple-choice or yes/no questions, with a probability for each option. We add test-time search to frozen decision models, with no training: we show the model a complete proposed answer, read off its probability of saying \"Yes\" to the proposal's correctness, and search for answers that it rates higher. The...",
        "microllm_relevance": "Shows test-time aggregation of multiple System-1 probes yields up to 15x error reduction on hard verification tasks."
    },
    {
        "arxiv_id": "2610.20927",
        "versioned_id": "2610.20927v1",
        "title": "Prune the Summary, Not the History: Source-Judged Compaction for Long-Horizon Agents",
        "authors": [
            "Guanghui Min",
            "Chen Chen"
        ],
        "published": "2026-10-14",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2610.20927",
        "pdf_url": "https://arxiv.org/pdf/2610.20927.pdf",
        "summary": "Long-horizon LLM agents rely on context compaction: when the interaction history outgrows the context window, a compressor replaces it with a natural-language summary. We show that these summaries contain more than a record of what happened. Compaction prompts ask for decisions and next steps, so each summary mixes facts established by the environment...",
        "microllm_relevance": "Applies calibrated System-1 Noul decisions to prune redundant agent memory without losing source facts."
    },
    {
        "arxiv_id": "2610.23804",
        "versioned_id": "2610.23804v1",
        "title": "Jarvis: Coordination Deserves a Brain of Its Own in Agentic Interactive Models",
        "authors": [
            "Yuxiang Wang",
            "Shengbo Cai",
            "Zhixian Zhao",
            "Yingda Shen",
            "Yuancheng Wang",
            "Xueyao Zhang"
        ],
        "published": "2026-10-16",
        "primary_category": "cs.SD",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2610.23804",
        "pdf_url": "https://arxiv.org/pdf/2610.23804.pdf",
        "summary": "Interactive models increasingly hand complex reasoning and tool use to a slow brain in the background, leaving a realtime front model free to sustain full-duplex dialogue. Yet the user may refine, cancel or replace a request while its task runs, so a plan or result that made sense a moment ago goes stale. Each new utterance must therefore be read against...",
        "microllm_relevance": "Separates agent orchestration (System-1 decision router) from worker execution (System-2 generators)."
    },
    {
        "arxiv_id": "2610.24649",
        "versioned_id": "2610.24649v1",
        "title": "Good Judgment, Bad Advice: Decision-Model Controllers Raise Agent Cost Without Improving Success",
        "authors": [
            "Naman Vats"
        ],
        "published": "2026-10-17",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2610.24649",
        "pdf_url": "https://arxiv.org/pdf/2610.24649.pdf",
        "summary": "Agent scaffolds increasingly delegate small decisions (whether to stop, verify, or change approach) to cheaper models. We ask whether a typed decision model, Jev 1.13, consulted only at deterministic decision points, makes a Claude Sonnet 5.5 terminal agent cheaper without making it worse. In a pre-specified study on Terminal-Bench 2.1, we hold one agent...",
        "microllm_relevance": "Shows that decoupled System-1 verification outperforms asking a generator LLM to critique its own advice."
    },
    {
        "arxiv_id": "2610.13092",
        "versioned_id": "2610.13092v1",
        "title": "When Faster Decisions Do Not Make Better Agents: An Empirical Study of Jev in Agent Workflows",
        "authors": [
            "Yuchen Ling",
            "Shengcheng Yu",
            "Zhenyu Chen",
            "Chunrong Fang"
        ],
        "published": "2026-10-09",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2610.13092",
        "pdf_url": "https://arxiv.org/pdf/2610.13092.pdf",
        "summary": "AI agents depend on repeated decisions to accomplish tasks, with large language models (LLMs) commonly serving as their core decision makers. The recent emergence of Jev offers an opportunity to make these decisions faster and at lower cost. However, it remains unclear whether such local savings can be translated into efficient and reliable task...",
        "microllm_relevance": "Demonstrates that uncalibrated fast routing causes cascading agent errors unless paired with conformal abstention."
    },
    {
        "arxiv_id": "2610.19596",
        "versioned_id": "2610.19596v1",
        "title": "JevCoder: When Does Decision Delegation Pay Off in Code Repair?",
        "authors": [
            "Jiahao Zhang",
            "Yifan Zhang",
            "Yu Huang"
        ],
        "published": "2026-10-14",
        "primary_category": "cs.SE",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2610.19596",
        "pdf_url": "https://arxiv.org/pdf/2610.19596.pdf",
        "summary": "A coding agent repeatedly makes decisions as it repairs code. These decisions shape repair trajectories and affect the token cost of successive model requests. We ask which repair decisions can be delegated to a low-cost probabilistic judge to reduce token use while maintaining repair quality. The judges answer questions with probabilities over...",
        "microllm_relevance": "Uses System-1 probability gates to select candidate code patches and prune unit-test execution trees."
    },
    {
        "arxiv_id": "2609.28919",
        "versioned_id": "2609.28919v2",
        "title": "Harness Tokenomics: A Router for the Enterprise Agentic Control Plane",
        "authors": [
            "Ted Kwartler",
            "Alan Aqrawi",
            "Arian Abbasi"
        ],
        "published": "2026-09-24",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2609.28919",
        "pdf_url": "https://arxiv.org/pdf/2609.28919.pdf",
        "summary": "Harnesses, the products that run AI coding agents, are multiplying, and enterprises are rolling them out to their employees. What started as pilots with a few hundred seats is now scaling to tens of thousands. Enterprises rarely build these harnesses and usually adopt the ones model vendors bundle with their seats and APIs, such as Anthropic's Claude...",
        "microllm_relevance": "Quantifies the 90%+ token and cost reduction of replacing JSON-mode LLM calls with JevCompat /v1/systemone."
    },
    {
        "arxiv_id": "2609.23986",
        "versioned_id": "2609.23986v1",
        "title": "Jev-Mem: System-One-Controlled Agentic Memory for Efficient AI Agents",
        "authors": [
            "Dongming Jiang",
            "Yi Li",
            "Bingzhe Li"
        ],
        "published": "2026-09-21",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2609.23986",
        "pdf_url": "https://arxiv.org/pdf/2609.23986.pdf",
        "summary": "Agentic memory is becoming essential for long-horizon AI agents, yet many existing systems rely on autoregressive LLMs to control how memories are organized, retrieved, and used, placing expensive generation on the critical path of memory operations. We introduce , a new agentic memory architecture inspired by System-One/System-Two cognition. System One...",
        "microllm_relevance": "Uses typed Noul and Score gates to control episodic memory write/evict policies in long-horizon agents."
    },
    {
        "arxiv_id": "2610.00437",
        "versioned_id": "2610.00437v1",
        "title": "JevSpawn: Adaptive Agentic Inference through Compositional Action Spaces",
        "authors": [
            "Haoyang Su",
            "Weiran Huang"
        ],
        "published": "2026-09-30",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2610.00437",
        "pdf_url": "https://arxiv.org/pdf/2610.00437.pdf",
        "summary": "LLM agents generate intermediate reasoning and actions token by token, making extended interactions slow and computationally expensive. Jev-style models offer fast probabilistic predictions over finite fields, but require those fields to be specified in advance. This requirement limits autonomous task solving, where the available actions must be derived...",
        "microllm_relevance": "Governs dynamic subagent spawning via calibrated value-of-computation probabilities."
    },
    {
        "arxiv_id": "2609.30922",
        "versioned_id": "2609.30922v1",
        "title": "JevSoup: System-One Routing for Training-Free LoRA Composition",
        "authors": [
            "Xiuying Wang",
            "Jiahua Cheng",
            "Shuotian Li",
            "Yufan Cheng",
            "Bowen Deng",
            "Zhexuan Bai"
        ],
        "published": "2026-09-25",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2609.30922",
        "pdf_url": "https://arxiv.org/pdf/2609.30922.pdf",
        "summary": "Building adaptable AI systems requires effective coordination of specialized capabilities across diverse tasks. Low-rank adaptation (LoRA) enables modular expertise, but existing routing approaches may require auxiliary data, additional training, or autoregressive decoding. We propose JevSoup, a training-free framework separating System One expert...",
        "microllm_relevance": "Explores weight-space interpolation (Model Soups / LoRA composition) of specialized RLCD decision heads."
    },
    {
        "arxiv_id": "2609.30706",
        "versioned_id": "2609.30706v1",
        "title": "LAVOIR: Teaching a Single-Pass Decision Encoder When and What to Ask with Amortized Value of Information",
        "authors": [
            "Furkan Yilmaz",
            "Habibe Aleyna Tasdemir",
            "Muhammed Faruk Gozay"
        ],
        "published": "2026-09-25",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2609.30706",
        "pdf_url": "https://arxiv.org/pdf/2609.30706.pdf",
        "summary": "\"System One\" decision models such as TypeSafe's Jev and its open counterpart Laya answer typed questions about a text in a single forward pass with calibrated probabilities, but they cannot ask for missing information: when a first message does not say what separates two departments, they guess. We present LAVOIR (Laya with Value-Of-Information Routing),...",
        "microllm_relevance": "Estimates the expected Value of Information (VoI) before invoking expensive external tools or Gear 2."
    },
    {
        "arxiv_id": "2610.20634",
        "versioned_id": "2610.20634v1",
        "title": "Jev-as-a-Judge for RAG Claim Verification: Confidence-Gated Cascading Between a Non-Generative Verifier and a Frontier LLM",
        "authors": [
            "Alden Do Rosario",
            "Kirollos Atef"
        ],
        "published": "2026-09-30",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2610.20634",
        "pdf_url": "https://arxiv.org/pdf/2610.20634.pdf",
        "summary": "Retrieval-augmented generation asks one question at volume: does this evidence support this claim? The answer is one bit, yet most pipelines obtain it from a text generator and pay generation prices for the prose wrapped around it. We compare a non-generative decision model, which returns a typed value from a declared enumeration together with a native...",
        "microllm_relevance": "Directly informs the 4th rubric (hallucination_risk / groundedness_score) in POST /v1/guardrail."
    },
    {
        "arxiv_id": "2609.29429",
        "versioned_id": "2609.29429v1",
        "title": "Just Ask Jev: Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of AI Alignment Failures",
        "authors": [
            "Ruoqi Guo",
            "Yi Liu",
            "Gelei Deng",
            "Yuekang Li",
            "Lida Zhao",
            "Yutao Wu"
        ],
        "published": "2026-09-24",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_5_safety_adversarial",
        "pillar_title": "Pillar 5 — AI Safety, Guardrails, Alignment Detection & Adversarial Robustness",
        "abs_url": "https://arxiv.org/abs/2609.29429",
        "pdf_url": "https://arxiv.org/pdf/2609.29429.pdf",
        "summary": "Detectors of alignment failures screen deployed language models and score alignment benchmarks. Most are generative judges that spend a decoding pass on every criterion, and classifiers that read token probabilities, such as Llama Guard, still score one fixed label per call. Jev, a model trained with reinforcement learning for calibrated decisions...",
        "microllm_relevance": "Direct foundation of POST /v1/guardrail (src/microllm/server.py): zero-shot detection of prompt injection, destructive ops, PII leaks, and hallucinations."
    },
    {
        "arxiv_id": "2609.31142",
        "versioned_id": "2609.31142v1",
        "title": "JevAdvBench: A Benchmark and Black-Box Attacks for Reinforcement Learning for Calibrated Decisions Models",
        "authors": [
            "Jianyi Hu",
            "Hangtao Zhang",
            "Yi Liu",
            "Yeqi Zeng",
            "Li Zeng",
            "Xianlong Wang"
        ],
        "published": "2026-09-25",
        "primary_category": "cs.CR",
        "pillar_id": "pillar_5_safety_adversarial",
        "pillar_title": "Pillar 5 — AI Safety, Guardrails, Alignment Detection & Adversarial Robustness",
        "abs_url": "https://arxiv.org/abs/2609.31142",
        "pdf_url": "https://arxiv.org/pdf/2609.31142.pdf",
        "summary": "Models trained with reinforcement learning for calibrated decisions (RLCD), such as Jev, answer a typed question about an input, the state, with a probability, a choice, or a score, and software acts on the answer without a person reading it. Their robustness has not been measured: adversarial benchmarks score what a model generates or executes, whereas...",
        "microllm_relevance": "Introduces black-box adversarial attacks against RLCD models; informs Track 4 (guardrail_safety) and Track 6 adversarial tests in quality_suite.py."
    },
    {
        "arxiv_id": "2610.18326",
        "versioned_id": "2610.18326v1",
        "title": "Jev-Guard: Fast Routing and Auditing Agent Actions over Editable Policy Trees",
        "authors": [
            "Wenbin Hu",
            "Huihao Jing",
            "Haochen Shi",
            "Yuxuan Liu",
            "Changxuan Fan",
            "Sirui Zhang"
        ],
        "published": "2026-10-13",
        "primary_category": "cs.CR",
        "pillar_id": "pillar_5_safety_adversarial",
        "pillar_title": "Pillar 5 — AI Safety, Guardrails, Alignment Detection & Adversarial Robustness",
        "abs_url": "https://arxiv.org/abs/2610.18326",
        "pdf_url": "https://arxiv.org/pdf/2610.18326.pdf",
        "summary": "Agent harnesses require a low-latency guardrail model together with customizable audit policies. However, deploying an auto-regressive guardrail model remains challenging: such models incur high inference latency and are often poorly aligned with customized policies. To address these problems, we present Jev-Guard, a unified model for safety...",
        "microllm_relevance": "Maps editable safety policy trees to batched Noul/Choice questions, mirrored by custom_rubrics in POST /v1/guardrail (IDEA-018)."
    },
    {
        "arxiv_id": "2610.20907",
        "versioned_id": "2610.20907v1",
        "title": "From Detection to Decision: Calibration and Deferral for Prompt-Injection Detection",
        "authors": [
            "Qianlong Lan",
            "Anuj Kaul",
            "Vinothini Pandurangan",
            "Indranil Sanyal"
        ],
        "published": "2026-10-14",
        "primary_category": "cs.CR",
        "pillar_id": "pillar_5_safety_adversarial",
        "pillar_title": "Pillar 5 — AI Safety, Guardrails, Alignment Detection & Adversarial Robustness",
        "abs_url": "https://arxiv.org/abs/2610.20907",
        "pdf_url": "https://arxiv.org/pdf/2610.20907.pdf",
        "summary": "Prompt-injection defenses are commonly evaluated as binary detectors, although deployed LLM applications must make operational decisions about whether to allow, block, or escalate potentially malicious inputs. Recent probabilistic decision models aim to support such decisions through calibrated confidence estimates, but their reliability under...",
        "microllm_relevance": "Combines calibrated prompt-injection probabilities with three-way action routing (allow / review / block) implemented in POST /v1/guardrail."
    },
    {
        "arxiv_id": "2609.28613",
        "versioned_id": "2609.28613v1",
        "title": "Decision Hijacking: Prompt Injection Attacks on Jev's Typed Probabilistic Decisions",
        "authors": [
            "Tiantong Wu",
            "Wei Yang Bryan Lim"
        ],
        "published": "2026-09-23",
        "primary_category": "cs.CR",
        "pillar_id": "pillar_5_safety_adversarial",
        "pillar_title": "Pillar 5 — AI Safety, Guardrails, Alignment Detection & Adversarial Robustness",
        "abs_url": "https://arxiv.org/abs/2609.28613",
        "pdf_url": "https://arxiv.org/pdf/2609.28613.pdf",
        "summary": "Most studies of prompt injection focus on generative agents, leaving their effects on models with schema-defined outputs unclear. We examine these effects in Jev, a non-generative decision model, using 510 reconstructed InjecAgent cases. Malicious content shifts action probabilities but rarely causes Jev to select the attacker's target. Override markers...",
        "microllm_relevance": "Analyzes adversarial state suffixes that hijack Choice probabilities; mitigated by token attribution & counterfactual auditing in Studio."
    },
    {
        "arxiv_id": "2610.18612",
        "versioned_id": "2610.18612v1",
        "title": "Nudge the Judge: Uncovering Program Decision Hijacking in Real-World Jev Applications",
        "authors": [
            "Yang Luo",
            "Tiantian Ji",
            "Zifeng Kang",
            "Yucheng Wang",
            "Yong Liu",
            "Zhiyi Yin"
        ],
        "published": "2026-10-13",
        "primary_category": "cs.CR",
        "pillar_id": "pillar_5_safety_adversarial",
        "pillar_title": "Pillar 5 — AI Safety, Guardrails, Alignment Detection & Adversarial Robustness",
        "abs_url": "https://arxiv.org/abs/2610.18612",
        "pdf_url": "https://arxiv.org/pdf/2610.18612.pdf",
        "summary": "System One models such as Jev change how software makes decisions by mapping open-ended semantic contexts into structured judgments that directly drive program execution. However, this paradigm introduces a new security risk, which we term Jev Program Decision Hijacking (JPDH): attacker-controlled contexts can redirect program decisions through...",
        "microllm_relevance": "Studies subtle rubric-level prompt injections targeting LLM/Jev judges."
    },
    {
        "arxiv_id": "2610.14404",
        "versioned_id": "2610.14404v2",
        "title": "DICE: Tilting the Odds through Interface Attacks on System One Model",
        "authors": [
            "Ben Dong",
            "Tianyi Lu",
            "Zhonghao Guo",
            "Qian Wang"
        ],
        "published": "2026-10-09",
        "primary_category": "cs.CR",
        "pillar_id": "pillar_5_safety_adversarial",
        "pillar_title": "Pillar 5 — AI Safety, Guardrails, Alignment Detection & Adversarial Robustness",
        "abs_url": "https://arxiv.org/abs/2610.14404",
        "pdf_url": "https://arxiv.org/pdf/2610.14404.pdf",
        "summary": "System One models offer typed judgments that applications can turn into decisions. Their answers, however, are conditional on the state and question supplied by surrounding components, and on how the application uses the response. We study this boundary in a Jev-based pipeline with bounded substitutions at three adapter outputs: state extraction,...",
        "microllm_relevance": "Demonstrates interface-level attacks via option ordering and decoy injection; neutralized by AnyJev L0 canonicalized circular rotations."
    },
    {
        "arxiv_id": "2610.15537",
        "versioned_id": "2610.15537v1",
        "title": "Poisoned into Action: Context Attacks on Typed Decision Workflows",
        "authors": [
            "Chenkai Wu",
            "Xing Chang",
            "Jun Yan"
        ],
        "published": "2026-10-11",
        "primary_category": "cs.CR",
        "pillar_id": "pillar_5_safety_adversarial",
        "pillar_title": "Pillar 5 — AI Safety, Guardrails, Alignment Detection & Adversarial Robustness",
        "abs_url": "https://arxiv.org/abs/2610.15537",
        "pdf_url": "https://arxiv.org/pdf/2610.15537.pdf",
        "summary": "Typed decision interfaces restrict models to predefined actions, yet schema-valid choices can still trigger policy-violating state changes. Evaluating context poisoning thus requires connecting model decisions to execution outcomes while accounting for legitimate automation retained by defenses. We introduce an execution-aware protocol tracing poisoned...",
        "microllm_relevance": "Examines indirect prompt injection in agent tool outputs; caught by MicroLLM's context-aware guardrail endpoint."
    },
    {
        "arxiv_id": "2610.17820",
        "versioned_id": "2610.17820v1",
        "title": "Same Attack, Different Judge: A First Controlled Adversarial Comparison of Jev, LLM Judges, and Purpose-Built Safety Classifiers",
        "authors": [
            "Xiang Li",
            "Aichun Huang",
            "Mingming Zhang",
            "Zheli Liu"
        ],
        "published": "2026-10-13",
        "primary_category": "cs.CR",
        "pillar_id": "pillar_5_safety_adversarial",
        "pillar_title": "Pillar 5 — AI Safety, Guardrails, Alignment Detection & Adversarial Robustness",
        "abs_url": "https://arxiv.org/abs/2610.17820",
        "pdf_url": "https://arxiv.org/pdf/2610.17820.pdf",
        "summary": "TypeSafe AI's Jev is a new kind of safety judge. It generates no text and no label: it answers questions the developer defines in advance and returns a probability for each answer. The vendor admits that adversarial input can change these answers but gives no numbers. In this paper, we present the first controlled adversarial evaluation of Jev. We build...",
        "microllm_relevance": "Compares transferability of adversarial perturbations across System-1 encoders and System-2 LLM judges."
    },
    {
        "arxiv_id": "2610.19118",
        "versioned_id": "2610.19118v1",
        "title": "DecisionRobust: Benchmarking Robustness for JEV Style Decision Models Across Text and Vision",
        "authors": [
            "Yifan Lan",
            "Zhongye Liu",
            "Yurui Chang",
            "Hanyu Wang",
            "Yongkang Du",
            "Lu Lin"
        ],
        "published": "2026-10-13",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_5_safety_adversarial",
        "pillar_title": "Pillar 5 — AI Safety, Guardrails, Alignment Detection & Adversarial Robustness",
        "abs_url": "https://arxiv.org/abs/2610.19118",
        "pdf_url": "https://arxiv.org/pdf/2610.19118.pdf",
        "summary": "Emerging decision models such as JEV provide typed interfaces that map task instructions and observations directly to choices, judgments, and scores. As these models are increasingly used for automated decision-making, understanding their reliability under adversarial manipulation becomes critical. We introduce DecisionRobust, a benchmark for...",
        "microllm_relevance": "Proposes adversarial training and randomized smoothing for typed decision heads."
    },
    {
        "arxiv_id": "2609.30243",
        "versioned_id": "2609.30243v2",
        "title": "JevOut: Natural Context Can Flip Decision Models",
        "authors": [
            "Zixiang Xu",
            "Zirui Song",
            "Chiyu Zhang",
            "Xiuying Chen",
            "Xi Liu",
            "Xiyang Hu"
        ],
        "published": "2026-09-24",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_5_safety_adversarial",
        "pillar_title": "Pillar 5 — AI Safety, Guardrails, Alignment Detection & Adversarial Robustness",
        "abs_url": "https://arxiv.org/abs/2609.30243",
        "pdf_url": "https://arxiv.org/pdf/2609.30243.pdf",
        "summary": "An ordinary-looking background detail can turn a correct model decision into a confident mistake. We demonstrate this fragility in four decision systems, including Jev, across seven datasets covering knowledge, reasoning, and tool routing. Within 64 accepted target evaluations per decision, we uncover short context additions that redirect 61.4%-73.2% of...",
        "microllm_relevance": "Shows natural context distractors can flip uncalibrated decision models; motivates DecisionPivot contrast testing."
    },
    {
        "arxiv_id": "2609.33401",
        "versioned_id": "2609.33401v3",
        "title": "Evaluating System One Models for Agent Security Decisions: Reliability, Calibration, and Selective Automation",
        "authors": [
            "Yixuan Liu"
        ],
        "published": "2026-09-27",
        "primary_category": "cs.CR",
        "pillar_id": "pillar_5_safety_adversarial",
        "pillar_title": "Pillar 5 — AI Safety, Guardrails, Alignment Detection & Adversarial Robustness",
        "abs_url": "https://arxiv.org/abs/2609.33401",
        "pdf_url": "https://arxiv.org/pdf/2609.33401.pdf",
        "summary": "Agentic software connects language models to tools that modify files and interact with external services. Developers use model-based security checks to screen external content and user requests before agents act. System One models answer developer-defined questions with probabilities over predefined answers such as safe or unsafe, but classification...",
        "microllm_relevance": "Benchmarks System-1 models on real-time tool-call authorization and shell command sandboxing."
    },
    {
        "arxiv_id": "2610.03073",
        "versioned_id": "2610.03073v2",
        "title": "SecJev: Bringing Security Expertise to System One Decision Models",
        "authors": [
            "Zheng Chen",
            "Fei Yu",
            "Haohao Huang",
            "Yang Li",
            "Anlong Chen",
            "Lei Chen"
        ],
        "published": "2026-10-02",
        "primary_category": "cs.CR",
        "pillar_id": "pillar_5_safety_adversarial",
        "pillar_title": "Pillar 5 — AI Safety, Guardrails, Alignment Detection & Adversarial Robustness",
        "abs_url": "https://arxiv.org/abs/2610.03073",
        "pdf_url": "https://arxiv.org/pdf/2610.03073.pdf",
        "summary": "Security workflows need models that turn complex observations and explicit policies into decisions. System One models introduced by Jev return typed predictions and probabilities; security specialization supplies the domain expertise behind those predictions. We introduce SecJev, to our knowledge the first family of Jev-style decision models specialized...",
        "microllm_relevance": "Specialized security classification profile inspiring the 'guardrail' hardware profile in src/microllm/registry.py."
    },
    {
        "arxiv_id": "2610.04985",
        "versioned_id": "2610.04985v1",
        "title": "Hidden Risks of Jev: An Empirical Study of Security, Privacy, and Dual Use",
        "authors": [
            "Shang Wang",
            "Tianqing Zhu",
            "Huajie Chen",
            "Jiayang Li",
            "Meng Yang",
            "Bo Liu"
        ],
        "published": "2026-10-04",
        "primary_category": "cs.CR",
        "pillar_id": "pillar_5_safety_adversarial",
        "pillar_title": "Pillar 5 — AI Safety, Guardrails, Alignment Detection & Adversarial Robustness",
        "abs_url": "https://arxiv.org/abs/2610.04985",
        "pdf_url": "https://arxiv.org/pdf/2610.04985.pdf",
        "summary": "Jev turns natural-language questions into typed answers and probabilities with low latency and cost, enabling applications to route requests and select tools. While this interface allows Jev to integrate naturally into application workflows as a decision layer, the security and privacy implications of this emerging use remain largely unexplored. To...",
        "microllm_relevance": "Audits over-trust and automation bias when downstream systems consume uncalibrated System-1 probabilities."
    },
    {
        "arxiv_id": "2609.34862",
        "versioned_id": "2609.34862v1",
        "title": "JEV as a Judge for Agent Trace Security: An Empirical Comparison with Generative LLM Judges",
        "authors": [
            "Zhiqiang Wang",
            "Yichao Gao"
        ],
        "published": "2026-09-28",
        "primary_category": "cs.CR",
        "pillar_id": "pillar_5_safety_adversarial",
        "pillar_title": "Pillar 5 — AI Safety, Guardrails, Alignment Detection & Adversarial Robustness",
        "abs_url": "https://arxiv.org/abs/2609.34862",
        "pdf_url": "https://arxiv.org/pdf/2609.34862.pdf",
        "summary": "Security evaluation of tool-using agents requires judging actions in context, yet generative judges add latency, explanation overhead, and output-validation failures. We study whether JEV, a typed decision model, offers a useful alternative for retrospective trace classification. We evaluate JEV and four generative judges on four benchmark collections...",
        "microllm_relevance": "Applies batched System-1 checks across multi-step agent execution traces for anomaly detection."
    },
    {
        "arxiv_id": "2609.28940",
        "versioned_id": "2609.28940v1",
        "title": "Calibrated Decision Models for Autonomous Penetration-Testing Harnesses: JEV and Laya as System One Decision Layers for LLM-Driven Pentest Agents",
        "authors": [
            "Joas Antonio dos Santos Barbosa"
        ],
        "published": "2026-09-24",
        "primary_category": "cs.CR",
        "pillar_id": "pillar_5_safety_adversarial",
        "pillar_title": "Pillar 5 — AI Safety, Guardrails, Alignment Detection & Adversarial Robustness",
        "abs_url": "https://arxiv.org/abs/2609.28940",
        "pdf_url": "https://arxiv.org/pdf/2609.28940.pdf",
        "summary": "Autonomous penetration-testing harnesses use large language models (LLMs) for reconnaissance, exploitation, and reporting, but often rely on those same models to confirm findings, grade severity, and select agents. This can lead to false positives, inflated severity, and wasted compute. We examine how System One decision models, lightweight...",
        "microllm_relevance": "Uses calibrated System-1 risk scores to bound autonomous security testing actions."
    },
    {
        "arxiv_id": "2609.28587",
        "versioned_id": "2609.28587v1",
        "title": "NumericJev: Jev-like LLM Numerical Decoding with Multiway Decision Trees",
        "authors": [
            "Weiwei Ye",
            "Hangchen Liu",
            "Renhe Jiang"
        ],
        "published": "2026-09-23",
        "primary_category": "stat.ML",
        "pillar_id": "pillar_6_multimodal_domain",
        "pillar_title": "Pillar 6 — Multimodal, Numeric, Retrieval & Domain-Specific System-One Models",
        "abs_url": "https://arxiv.org/abs/2609.28587",
        "pdf_url": "https://arxiv.org/pdf/2609.28587.pdf",
        "summary": "Large language models can interpret natural lan- guage, yet robust decisions remain challenging. Jev-like models expose structured choices, but these interfaces do not directly provide numeri- cal values at a requested precision. We propose NUMERICJEV, a training-free numerical decod- ing algorithm that enables numerical output from any LLM with a...",
        "microllm_relevance": "Extends binary Noul and K-way Choice into balanced multiway decision trees for continuous numeric regression/quantiles (IDEA-025)."
    },
    {
        "arxiv_id": "2609.25845",
        "versioned_id": "2609.25845v1",
        "title": "Visual Jev: Accurate and Efficient Decisions from Shared Visual Context",
        "authors": [
            "Guanxu Yu",
            "Yuhang Yao"
        ],
        "published": "2026-09-22",
        "primary_category": "cs.CV",
        "pillar_id": "pillar_6_multimodal_domain",
        "pillar_title": "Pillar 6 — Multimodal, Numeric, Retrieval & Domain-Specific System-One Models",
        "abs_url": "https://arxiv.org/abs/2609.25845",
        "pdf_url": "https://arxiv.org/pdf/2609.25845.pdf",
        "summary": "Many vision applications ask several independent, forced-choice questions about the same image. Visual Jev encodes the image and public context once, executes isolated question suffixes as a batch, and reads candidate probabilities from the backbone's language-model head. Across four benchmarks, answer-supervised post-training raises equal-weight macro...",
        "microllm_relevance": "Extends single-pass JevCompat primitives to vision-language inputs."
    },
    {
        "arxiv_id": "2610.25784",
        "versioned_id": "2610.25784v1",
        "title": "UniJev: Separating Accuracy from Calibration in Trimodal System One Decision Models",
        "authors": [
            "Jianing Wang",
            "Youwen Jian",
            "Derong Wang",
            "Jinyang Yu",
            "Qingyi Si",
            "Chuanyu Qin"
        ],
        "published": "2026-10-19",
        "primary_category": "cs.CV",
        "pillar_id": "pillar_6_multimodal_domain",
        "pillar_title": "Pillar 6 — Multimodal, Numeric, Retrieval & Domain-Specific System-One Models",
        "abs_url": "https://arxiv.org/abs/2610.25784",
        "pdf_url": "https://arxiv.org/pdf/2610.25784.pdf",
        "summary": "System One models produce probabilistic judgments that software can consume directly, making reliable confidence as important as answer accuracy. Yet joint training across text, images, and video remains underexplored, and the contributions of supervised fine-tuning (SFT) and reinforcement learning for calibrated decisions (RLCD) are difficult to...",
        "microllm_relevance": "Introduces Cora post-hoc calibration across unified text, image, and structured inputs."
    },
    {
        "arxiv_id": "2610.21614",
        "versioned_id": "2610.21614v1",
        "title": "Qwen-Audio-Jev: A Unified Interface for Audio Decisions",
        "authors": [
            "Ying Fang",
            "Zhifu Gao",
            "Qing Gu",
            "Jiajun He",
            "Kangqi Jing",
            "Xiangang Li"
        ],
        "published": "2026-10-15",
        "primary_category": "cs.SD",
        "pillar_id": "pillar_6_multimodal_domain",
        "pillar_title": "Pillar 6 — Multimodal, Numeric, Retrieval & Domain-Specific System-One Models",
        "abs_url": "https://arxiv.org/abs/2610.21614",
        "pdf_url": "https://arxiv.org/pdf/2610.21614.pdf",
        "summary": "Large audio language models (LALMs) have made substantial progress in audio understanding. However, in high-throughput applications requiring frequent selection, judgment, and scoring, autoregressive text generation introduces additional computational overhead and response latency. To address this limitation, we introduce Qwen-Audio-Jev by augmenting...",
        "microllm_relevance": "Applies single-pass typed decision readouts to audio and speech representations."
    },
    {
        "arxiv_id": "2610.26002",
        "versioned_id": "2610.26002v1",
        "title": "TimesJev: Thinking Fast on Time Series with System 1 Sufficiency",
        "authors": [
            "Chenghao Liu",
            "Wei Fan",
            "Jindong Han",
            "Dongjie Wang",
            "Aniket Mahanti",
            "Pengyang Wang"
        ],
        "published": "2026-10-19",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_6_multimodal_domain",
        "pillar_title": "Pillar 6 — Multimodal, Numeric, Retrieval & Domain-Specific System-One Models",
        "abs_url": "https://arxiv.org/abs/2610.26002",
        "pdf_url": "https://arxiv.org/pdf/2610.26002.pdf",
        "summary": "Large language models (LLMs) have recently shown strong potential for time series reasoning, while models that reason over time series are usually with long chains of thought. However, this process often consumes thousands of tokens, directly leading to substantial latency during inference. In contrast, System 1 models such as Jev make decisions...",
        "microllm_relevance": "Adapts typed decision heads to time-series anomaly and regime classification."
    },
    {
        "arxiv_id": "2610.28435",
        "versioned_id": "2610.28435v1",
        "title": "Multimodal Reranking Is a System One Decision",
        "authors": [
            "Zhuchenyang Liu",
            "Yao Zhang",
            "Yu Xiao"
        ],
        "published": "2026-10-20",
        "primary_category": "cs.IR",
        "pillar_id": "pillar_6_multimodal_domain",
        "pillar_title": "Pillar 6 — Multimodal, Numeric, Retrieval & Domain-Specific System-One Models",
        "abs_url": "https://arxiv.org/abs/2610.28435",
        "pdf_url": "https://arxiv.org/pdf/2610.28435.pdf",
        "summary": "Multimodal rerankers are usually dedicated models: a vision-language model (VLM) is trained on retrieval data to score the candidates of a query. Scoring options is not specific to reranking. System One decision models such as Jev map a state and a typed question to one probability per option without generating text, and general models can make such...",
        "microllm_relevance": "Formulates listwise document/multimodal reranking as a calibrated Choice/Multi System-1 call."
    },
    {
        "arxiv_id": "2610.22588",
        "versioned_id": "2610.22588v1",
        "title": "SAG2: Embedding-Free Multi-Hop Retrieval with BM25 and JEV",
        "authors": [
            "Yuchao Wu"
        ],
        "published": "2026-10-15",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_6_multimodal_domain",
        "pillar_title": "Pillar 6 — Multimodal, Numeric, Retrieval & Domain-Specific System-One Models",
        "abs_url": "https://arxiv.org/abs/2610.22588",
        "pdf_url": "https://arxiv.org/pdf/2610.22588.pdf",
        "summary": "Strong multi-hop retrieval should be practical to deploy without expensive corpus preparation or complex database infrastructure. We present SAG2, an embedding-free retrieval architecture built around a single SQLite database, BM25, and contextual evidence decisions from JEV. SAG2 follows the query-time association principle of SAG (SQL-Retrieval...",
        "microllm_relevance": "Combines fast lexical retrieval (BM25) with calibrated Jev Noul/Score verification for embedding-free multi-hop RAG."
    },
    {
        "arxiv_id": "2610.23346",
        "versioned_id": "2610.23346v1",
        "title": "Jev-as-a-Judge for Brazilian Portuguese Information Retrieval: Relevance Assessment and Reranking",
        "authors": [
            "Leandro Carísio Fernandes",
            "Jayr Pereira"
        ],
        "published": "2026-10-16",
        "primary_category": "cs.IR",
        "pillar_id": "pillar_6_multimodal_domain",
        "pillar_title": "Pillar 6 — Multimodal, Numeric, Retrieval & Domain-Specific System-One Models",
        "abs_url": "https://arxiv.org/abs/2610.23346",
        "pdf_url": "https://arxiv.org/pdf/2610.23346.pdf",
        "summary": "This study investigates Jev, a System One model designed for fast, structured decision-making, in information retrieval. We use Jev-as-a-judge to generate relevance judgments for four Brazilian Portuguese IR collections covering different domains and compare them with the official relevance judgments of each collection. We also evaluate Jev as a...",
        "microllm_relevance": "Uses calibrated Noul and Score probabilities for low-latency passage reranking."
    },
    {
        "arxiv_id": "2610.18529",
        "versioned_id": "2610.18529v2",
        "title": "How Reliable Are System-One Decisions in Medicine? A Comparative Study of Jev, Jev-like Models, and LLMs",
        "authors": [
            "Yanrui Du",
            "Danyang Zhao",
            "Sendong Zhao",
            "Wenhao Li",
            "Yibo Gao",
            "Haochun Wang"
        ],
        "published": "2026-10-13",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_6_multimodal_domain",
        "pillar_title": "Pillar 6 — Multimodal, Numeric, Retrieval & Domain-Specific System-One Models",
        "abs_url": "https://arxiv.org/abs/2610.18529",
        "pdf_url": "https://arxiv.org/pdf/2610.18529.pdf",
        "summary": "Jev has attracted considerable attention for its rapid, generalizable decision-making, yet its medical reliability remains underexplored. To address this gap, our study compares various types of system-one decision models across 15 medical datasets comprising 8,742 instances under a unified protocol. Beyond accuracy, we assess decision stability and the...",
        "microllm_relevance": "Audits calibration, prevalence sensitivity, and abstention safety of System-1 models on clinical vignettes."
    },
    {
        "arxiv_id": "2610.24692",
        "versioned_id": "2610.24692v1",
        "title": "Native Probability Outputs for Clinical Triage: An Evaluation of Calibration and Decision Accuracy",
        "authors": [
            "Liang You",
            "Zhiyin Zhou",
            "Dawei Tang",
            "Chuchu Wu",
            "Jingzhuo Hu"
        ],
        "published": "2026-10-17",
        "primary_category": "cs.LG",
        "pillar_id": "pillar_6_multimodal_domain",
        "pillar_title": "Pillar 6 — Multimodal, Numeric, Retrieval & Domain-Specific System-One Models",
        "abs_url": "https://arxiv.org/abs/2610.24692",
        "pdf_url": "https://arxiv.org/pdf/2610.24692.pdf",
        "summary": "Patient-facing triage requires an appropriate care recommendation and recognition of when information is insufficient. We evaluate whether Jev, a decision model that returns probabilities over predefined actions, supports both judgments. Using CARE-Bench, we compare Jev with four general-purpose language models and three input-independent baselines...",
        "microllm_relevance": "Demonstrates Clopper-Pearson and conformal risk bounds for medical triage routing."
    },
    {
        "arxiv_id": "2609.23136",
        "versioned_id": "2609.23136v2",
        "title": "Intent Interpretation at RIC Timescales: Jev Decision Models versus Large Language Models in 6G Open RAN",
        "authors": [
            "Delong Li",
            "Xu Wang",
            "Haochen Gong",
            "Rui Lang",
            "Guangsheng Yu"
        ],
        "published": "2026-09-19",
        "primary_category": "cs.NI",
        "pillar_id": "pillar_6_multimodal_domain",
        "pillar_title": "Pillar 6 — Multimodal, Numeric, Retrieval & Domain-Specific System-One Models",
        "abs_url": "https://arxiv.org/abs/2609.23136",
        "pdf_url": "https://arxiv.org/pdf/2609.23136.pdf",
        "summary": "Intent-based Open RAN needs an interpreter that turns intents into A1 policies within the loop of the RAN intelligent controller (RIC). Decision models such as Jev-1.13.0 return typed policy fields, whereas generative large language models (LLMs) produce the policy token by token. We ask whether the extra delay of LLMs costs control deadlines, RIC...",
        "microllm_relevance": "Deploys sub-10ms AnyJev/System-1 intent classifiers inside 6G Open RAN Near-RT RIC loops, validating MicroLLM's edge latency target."
    },
    {
        "arxiv_id": "2610.26035",
        "versioned_id": "2610.26035v1",
        "title": "Decision4Rec: A Jev-Inspired Framework of Decision Models on Knowledge and Active Graphs for Multi-Agent Recommender Systems",
        "authors": [
            "Marco Polignano",
            "Het Darshan Mehta",
            "Pasquale Lops",
            "Ernesto William De Luca",
            "Giovanni Semeraro"
        ],
        "published": "2026-10-19",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_6_multimodal_domain",
        "pillar_title": "Pillar 6 — Multimodal, Numeric, Retrieval & Domain-Specific System-One Models",
        "abs_url": "https://arxiv.org/abs/2610.26035",
        "pdf_url": "https://arxiv.org/pdf/2610.26035.pdf",
        "summary": "Decision models, such as the commercial Jev, are language models that return one probability for each option of a question instead of generating text. We present Decision4Rec, an open and modular framework that makes decision models useful for recommendation. A knowledge graph writes the logs of each recommendation scenario as short statements on a...",
        "microllm_relevance": "Applies calibrated Choice and Score primitives to low-latency sequential recommendation."
    },
    {
        "arxiv_id": "2610.27677",
        "versioned_id": "2610.27677v1",
        "title": "CovNav: Conformal Prediction-Guided Adaptive Inference for Object Goal Navigation",
        "authors": [
            "Pengying Wu",
            "Zhejia Wen",
            "Kang Ding",
            "Kangjie Zhou",
            "Chunxuan Jiao",
            "Wei Pan"
        ],
        "published": "2026-10-20",
        "primary_category": "cs.RO",
        "pillar_id": "pillar_6_multimodal_domain",
        "pillar_title": "Pillar 6 — Multimodal, Numeric, Retrieval & Domain-Specific System-One Models",
        "abs_url": "https://arxiv.org/abs/2610.27677",
        "pdf_url": "https://arxiv.org/pdf/2610.27677.pdf",
        "summary": "In object goal navigation (ObjectNav), an embodied agent must find a target object (e.g., \"find a bed\") in a previously unseen environment, relying on spatial commonsense such as object co-occurrence. Large Language Models (LLMs) encode this commonsense, but deploying them on robots exposes a trade-off between reasoning quality and efficiency:...",
        "microllm_relevance": "Directly validates combining Split Conformal Prediction sets with adaptive System-1/System-2 escalation (src/microllm/calibration.py)."
    },
    {
        "arxiv_id": "2610.24358",
        "versioned_id": "2610.24358v1",
        "title": "DKnown-Jev: Fine-Tuning and Evaluation of a JEV-Style Model for Tasks Involving Normative Knowledge",
        "authors": [
            "Qi Li",
            "Jianjun Xu",
            "Pingtao Wei"
        ],
        "published": "2026-10-17",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_6_multimodal_domain",
        "pillar_title": "Pillar 6 — Multimodal, Numeric, Retrieval & Domain-Specific System-One Models",
        "abs_url": "https://arxiv.org/abs/2610.24358",
        "pdf_url": "https://arxiv.org/pdf/2610.24358.pdf",
        "summary": "Tasks involving normative knowledge include concept interpretation, eligibility determination, procedure explanation, material requirements, and service-outcome assessment, all of which require rules and facts published, maintained, or recorded by authorized entities. Whether a candidate paragraph can answer a question therefore depends on applicability...",
        "microllm_relevance": "Injects domain-knowledge constraints into System-1 probability readouts."
    },
    {
        "arxiv_id": "2610.09188",
        "versioned_id": "2610.09188v1",
        "title": "From Probabilities to Decisions: Search and Multi-Teacher Distillation with Jev",
        "authors": [
            "Mohamad Yazan Sadoun",
            "Sarah Sharif",
            "Yaser Mike Banad"
        ],
        "published": "2026-10-06",
        "primary_category": "cs.AI",
        "pillar_id": "pillar_6_multimodal_domain",
        "pillar_title": "Pillar 6 — Multimodal, Numeric, Retrieval & Domain-Specific System-One Models",
        "abs_url": "https://arxiv.org/abs/2610.09188",
        "pdf_url": "https://arxiv.org/pdf/2610.09188.pdf",
        "summary": "Probability-only models, which TypeSafe calls System One models, return calibrated probabilities for fixed choices in milliseconds and generate no text. We study one such model, Jev, through two tasks that require decisions under tight constraints. In bullet chess, a bot that places Jev's judgment inside Stockfish search alongside an opening book and...",
        "microllm_relevance": "Distills multi-teacher calibrated probability distributions into compact student encoders (fast-cpu-int8 profile)."
    },
    {
        "arxiv_id": "1706.04599",
        "title": "On Calibration of Modern Neural Networks",
        "authors": [
            "Chuan Guo",
            "Geoff Pleiss",
            "Yu Sun",
            "Kilian Q. Weinberger"
        ],
        "published": "2017-06-14",
        "primary_category": "cs.LG",
        "pillar_id": "pillar_2_calibration_coherence",
        "pillar_title": "Pillar 2 — Probability Calibration, Coherence, Scoring Rules & Uncertainty",
        "abs_url": "https://arxiv.org/abs/1706.04599",
        "pdf_url": "https://arxiv.org/pdf/1706.04599.pdf",
        "summary": "Foundational formulation of Expected Calibration Error (ECE) and post-hoc Temperature Scaling on logits prior to softmax.",
        "microllm_relevance": "Directly implemented in src/microllm/calibration.py (PerKTemperatureScaler and expected_calibration_error).",
        "versioned_id": "1706.04599v1"
    },
    {
        "arxiv_id": "2107.07511",
        "title": "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification",
        "authors": [
            "Anastasios N. Angelopoulos",
            "Stephen Bates"
        ],
        "published": "2021-07-15",
        "primary_category": "cs.LG",
        "pillar_id": "pillar_2_calibration_coherence",
        "pillar_title": "Pillar 2 — Probability Calibration, Coherence, Scoring Rules & Uncertainty",
        "abs_url": "https://arxiv.org/abs/2107.07511",
        "pdf_url": "https://arxiv.org/pdf/2107.07511.pdf",
        "summary": "Comprehensive mathematical foundation for distribution-free finite-sample Split Conformal Prediction sets and Mondrian class-conditional conformal guarantees.",
        "microllm_relevance": "Directly implemented in src/microllm/calibration.py (SplitConformalCalibrator.fit, predict_set, and fit_mondrian).",
        "versioned_id": "2107.07511v1"
    },
    {
        "arxiv_id": "2207.05221",
        "title": "Language Models (Mostly) Know What They Know",
        "authors": [
            "Saurav Kadavath",
            "Tom Conerly",
            "Amanda Askell",
            "Tom Henighan",
            "Dawn Drain",
            "Ethan Perez",
            "Nicholas Schiefer",
            "Zac Hatfield-Dodds",
            "Nova DasSarma",
            "Eli Tran-Johnson",
            "Scott Johnston",
            "Sheer El-Showk",
            "Andy Jones",
            "Nelson Elhage",
            "Tristan Hume",
            "Anna Chen",
            "Yuntao Bai",
            "Sam Bowman",
            "Stanislav Fort",
            "Deep Ganguli",
            "Danny Hernandez",
            "Josh Jacobson",
            "Jackson Kernion",
            "Shauna Kravec",
            "Liane Lovitt",
            "Kamal Ndousse",
            "Catherine Olsson",
            "Sam Ringer",
            "Dario Amodei",
            "Tom Brown",
            "Jack Clark",
            "Nicholas Joseph",
            "Ben Mann",
            "Sam McCandlish",
            "Chris Olah",
            "Jared Kaplan"
        ],
        "published": "2022-07-11",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_2_calibration_coherence",
        "pillar_title": "Pillar 2 — Probability Calibration, Coherence, Scoring Rules & Uncertainty",
        "abs_url": "https://arxiv.org/abs/2207.05221",
        "pdf_url": "https://arxiv.org/pdf/2207.05221.pdf",
        "summary": "Introduces P(True) and P(IK) epistemic self-evaluation probes for language models and analyzes calibration degradations under RLHF.",
        "microllm_relevance": "Underpins the noul (P(True)) primitive and answer-mass epistemic gating in src/microllm/backends/prefill_llm.py.",
        "versioned_id": "2207.05221v1"
    },
    {
        "arxiv_id": "2305.05176",
        "title": "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance",
        "authors": [
            "Lingjiao Chen",
            "Matei Zaharia",
            "James Zou"
        ],
        "published": "2023-05-09",
        "primary_category": "cs.LG",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2305.05176",
        "pdf_url": "https://arxiv.org/pdf/2305.05176.pdf",
        "summary": "Pioneers confidence-gated LLM cascades that route queries sequentially from lightweight models to heavier models only when confidence falls below a learned threshold.",
        "microllm_relevance": "Directly inspires DualGearCascadeBackend (src/microllm/backends/cascade.py) and tune_cascade_pareto (src/microllm/calibration.py).",
        "versioned_id": "2305.05176v1"
    },
    {
        "arxiv_id": "2309.00667",
        "title": "Large Language Models Are Not Robust Multiple Choice Selectors",
        "authors": [
            "Chujie Zheng",
            "Hao Zhou",
            "Fandong Meng",
            "Jie Zhou",
            "Minlie Huang"
        ],
        "published": "2023-09-01",
        "primary_category": "cs.CL",
        "pillar_id": "pillar_1_core_architectures",
        "pillar_title": "Pillar 1 — Core Jev, System-One & Typed Decision Architectures",
        "abs_url": "https://arxiv.org/abs/2309.00667",
        "pdf_url": "https://arxiv.org/pdf/2309.00667.pdf",
        "summary": "Diagnoses token bias and selection bias in LLM multiple-choice readouts and introduces PriDe (Prior Debiasing) and cyclic permutation averaging.",
        "microllm_relevance": "Direct theoretical precursor to AnyJev L0 circular rotation readout and OnlinePriorTracker in src/microllm/backends/prefill_llm.py.",
        "versioned_id": "2309.00667v1"
    },
    {
        "arxiv_id": "2406.18665",
        "title": "RouteLLM: Learning to Route LLMs with Preference Data",
        "authors": [
            "Isaac Ong",
            "Amjad Almahairi",
            "Vincent Wu",
            "Wei-Lin Chiang",
            "Tianhao Wu",
            "Joseph E. Gonzalez",
            "M Waleed Kadous",
            "Ion Stoica"
        ],
        "published": "2024-06-26",
        "primary_category": "cs.LG",
        "pillar_id": "pillar_4_cascades_agents",
        "pillar_title": "Pillar 4 — Cascades, Selective Control, Agent Harnesses & System-1/2 Orchestration",
        "abs_url": "https://arxiv.org/abs/2406.18665",
        "pdf_url": "https://arxiv.org/pdf/2406.18665.pdf",
        "summary": "Formulates cost-quality routing between strong and weak models using calibrated win-probability classifiers.",
        "microllm_relevance": "Mapped to @system_one_router in src/microllm/client.py and DualGearCascadeBackend in src/microllm/backends/cascade.py.",
        "versioned_id": "2406.18665v1"
    }
]


def list_academic_papers(
    pillar: str | None = None,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """Return academic papers filtered by thematic pillar ID/substring and/or keyword query."""
    results: list[dict[str, Any]] = []
    pillar_norm = pillar.strip().lower() if pillar else None
    query_norm = query.strip().lower() if query else None

    for entry in ACADEMIC_PAPERS_CATALOG:
        if pillar_norm:
            if (
                pillar_norm not in entry["pillar_id"].lower()
                and pillar_norm not in entry["pillar_title"].lower()
            ):
                continue
        if query_norm:
            haystack = " ".join(
                [
                    entry["arxiv_id"],
                    entry["title"],
                    " ".join(entry["authors"]),
                    entry["summary"],
                    entry["microllm_relevance"],
                    entry["pillar_title"],
                ]
            ).lower()
            if query_norm not in haystack:
                continue
        results.append(dict(entry))
    return results


def get_academic_paper(arxiv_id: str) -> dict[str, Any] | None:
    """Look up a specific academic paper by its arXiv identifier (e.g. '2609.38850' or '2609.38850v1')."""
    clean_id = arxiv_id.strip().replace("arXiv:", "").split("v")[0]
    for entry in ACADEMIC_PAPERS_CATALOG:
        if entry["arxiv_id"] == clean_id or entry["versioned_id"] == arxiv_id.strip():
            return dict(entry)
    return None


def get_literature_taxonomy() -> dict[str, Any]:
    """Return the 6-pillar academic taxonomy with paper counts and foundational highlights."""
    pillars_summary: list[dict[str, Any]] = []
    for p in LITERATURE_PILLARS:
        papers_in_pillar = [
            e for e in ACADEMIC_PAPERS_CATALOG if e["pillar_id"] == p["id"]
        ]
        pillars_summary.append(
            {
                "id": p["id"],
                "title": p["title"],
                "description": p["description"],
                "count": len(papers_in_pillar),
                "arxiv_ids": [e["arxiv_id"] for e in papers_in_pillar],
            }
        )
    return {
        "total_papers": len(ACADEMIC_PAPERS_CATALOG),
        "total_pillars": len(LITERATURE_PILLARS),
        "foundational_core_ids": ["2609.38850", "2610.00831", "2610.17309"],
        "pillars": pillars_summary,
    }


def format_papers_markdown(
    pillar: str | None = None,
    query: str | None = None,
) -> str:
    """Format the filtered academic paper catalog as a GitHub-Flavored Markdown table."""
    papers = list_academic_papers(pillar=pillar, query=query)
    lines = [
        f"# MicroLLM Academic Literature Catalog ({len(papers)} papers)",
        "",
        "| arXiv ID | Title | Authors | Published | Pillar | MicroLLM Architectural Mapping |",
        "| :--- | :--- | :--- | :---: | :--- | :--- |",
    ]
    for p in papers:
        authors_short = ", ".join(p["authors"][:2]) + (" et al." if len(p["authors"]) > 2 else "")
        lines.append(
            f"| [`{p['arxiv_id']}`]({p['abs_url']}) ([PDF]({p['pdf_url']})) "
            f"| **{p['title']}** | {authors_short} | `{p['published']}` "
            f"| {p['pillar_title'].split('—')[0].strip()} | {p['microllm_relevance']} |"
        )
    return "\n".join(lines)
