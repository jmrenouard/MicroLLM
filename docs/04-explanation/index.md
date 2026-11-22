# Architecture & Design Explanations

This quadrant explains how MicroLLM works internally, why key architectural trade-offs were chosen,
and how its mathematical guarantees hold under CPU latency constraints.

## Contents

| Document | Focus | Primary Topics |
| :--- | :--- | :--- |
| [Dual-Gear System-1 Topology](architecture-overview/dual-gear-system1-topology.md) | System Architecture | Request isolation, Gear 1 `FastEncoderBackend`, Gear 2 `AnyJevPrefillBackend`, StartLux Cascade, and `MicroLLMGuard` dataflow |
| [ADR-001: Dual-Gear Conformal Cascade](design-decisions/adr-001-dual-gear-conformal-cascade.md) | Design Decision | Why MicroLLM routes confident traffic through Gear 1 and escalates uncertain or deliberative cases to Gear 2 |
| [ADR-002: Diátaxis and Versioned MkDocs Publishing](design-decisions/adr-002-diataxis-versioned-mkdocs-publishing.md) | Documentation Decision | Why documentation uses a 4-quadrant layout with multi-version tag overlays on IONOS Apache |
| [Calibration and Order Invariance Deep Dive](deep-dives/calibration-and-order-invariance.md) | Mathematical Internals | Per-$K$ temperature scaling, Split Conformal abstention, Van der Corput $L_0$ rotation, and OpenJev-RLCD² `Beta(8, 2)` weighting |

## Foundational Analysis and Limits

- **Scientific Literature Synthesis**: Read [Academic Literature Review](../LITERATURE_REVIEW.md)
  for the 116-paper taxonomy covering *OpenJev-RLCD*, *AnyJev*, *StartLux-Decision*, and System-1 alignment.
- **Empirical Failure Boundaries**: Read [Failure Modes & Worst Cases](../FAILURE_MODES_AND_WORST_CASES.md)
  for unsampled public-split error regimes (`114,018` rows), structured failure bounds (`27 / 11,000`),
  and System-1 structural limits.

## Sibling Quadrants

- **Tutorials**: [Guided Sandbox Lessons](../01-tutorials/index.md)
- **How-To Guides**: [Operational Runbooks & Procedures](../02-how-to-guides/index.md)
- **Reference**: [Configuration, CLI & Alert Matrices](../03-reference/index.md)
