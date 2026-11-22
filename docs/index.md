---
title: MicroLLM Documentation
description: Calibrated sub-5ms CPU System-1 probabilistic decision engine (jevcompat 0.1, AnyJev L0, StartLux Cascade, OpenJev-RLCD2)
hide:
  - navigation
  - toc
---

<div class="microllm-hero">
  <img src="assets/header-banner.svg" alt="MicroLLM — Local System-One Probabilistic Decision Engine" />
</div>

# MicroLLM — Calibrated Sub-5ms CPU System-1 Decision Engine

MicroLLM (`micro-jev`) is a local probabilistic decision engine that replaces slow generative System-2 prompts with deterministic, calibrated CPU inference across `noul`, `choice`, `score`, and `multi` primitives.

<div class="microllm-metrics">
  <div class="microllm-metric">
    <span class="microllm-metric-value">32/32 MUST</span>
    <span class="microllm-metric-label">JevCompat 0.1 Wire Contract</span>
  </div>
  <div class="microllm-metric">
    <span class="microllm-metric-value">&lt; 1 ms CPU</span>
    <span class="microllm-metric-label">0.013 ms–0.367 ms p50 · 0% GPU</span>
  </div>
  <div class="microllm-metric">
    <span class="microllm-metric-value">0.000000</span>
    <span class="microllm-metric-label">AnyJev L₀ Option-Order Drift</span>
  </div>
  <div class="microllm-metric">
    <span class="microllm-metric-value">99.75% Acc</span>
    <span class="microllm-metric-label">11,000 Structured Cases · Conformal x_abstain</span>
  </div>
</div>

<div class="microllm-actions" markdown>
[Architecture & Quickstart](README.md){ .md-button .md-button--primary }
[Interactive HTML Simulator](simulator.html){ .md-button }
[LightPath 12-Slide Deck](slides.html){ .md-button }
[Live Hugging Face Space](https://huggingface.co/spaces/jmrenouard/microllm-decision-studio){ .md-button }
</div>

---

## Explore Documentation by Track

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    Install `micro-jev`, pull quantized INT8 profiles, run the Python SDK (`MicroJevClient`), and explore the curriculum.

    - [Architecture & Quickstart (`docs/README.md`)](README.md)
    - [Product Documentation](PRODUCT_DOCUMENTATION.md)
    - [Engineering & Scientific Syllabus](SYLLABUS.md)

-   :material-cog-outline:{ .lg .middle } **Architecture & Reference**

    ---

    Dive into `jevcompat 0.1` wire contracts, Gear 1 FastEncoder, Gear 2 AnyJev $L_0$, StartLux Cascade, and failure limits.

    - [Technical Specifications](TECHNICAL_SPECIFICATIONS.md)
    - [Algorithmic Reference Manual](ALGORITHMS.md)
    - [Failure Modes & Worst Cases](FAILURE_MODES_AND_WORST_CASES.md)

-   :material-chart-line:{ .lg .middle } **Benchmarks & Quality**

    ---

    Inspect reproducible CPU benchmarks across 114,018 public dataset rows, 11,000 structured cases, and SLA test gates.

    - [Full Benchmark Report (`BENCHMARKS.md`)](BENCHMARKS.md)
    - [External Datasets Benchmarks](EXTERNAL_DATASETS_BENCHMARKS.md)
    - [Quality & Performance Test Plan](TEST_PLAN.md)

-   :material-book-open-variant:{ .lg .middle } **Research & Kits**

    ---

    Review the synthesis of OpenJev-RLCD, AnyJev, and StartLux-Decision plus executive communication templates.

    - [Academic Literature Review](LITERATURE_REVIEW.md)
    - [Curated ArXiv Papers Catalog](papers/README.md)
    - [Results Communication Kit](RESULTS_COMMUNICATION_KIT.md)

-   :material-history:{ .lg .middle } **Releases & Governance**

    ---

    Track release notes across versions alongside living engineering logs, root-cause analyses, and roadmap ideas.

    - [Action Logbook (`ACTION_LOGBOOK.md`)](ACTION_LOGBOOK.md)
    - [RCA & Transparency Register (`RCA_REGISTER.md`)](RCA_REGISTER.md)

-   :material-play-box-outline:{ .lg .middle } **Interactive Portals & Demos**

    ---

    Launch browser-native interactive simulators, widescreen executive decks, and hosted Gradio playgrounds.

    - [Interactive HTML Simulator (`simulator.html`)](simulator.html)
    - [LightPath 12-Slide Deck (`slides.html`)](slides.html)
    - [Live Hugging Face Decision Studio](https://huggingface.co/spaces/jmrenouard/microllm-decision-studio)

</div>
