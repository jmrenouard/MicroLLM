# Reference: Technical Specifications, Configuration, CLI, and Error Matrix

Normative specifications for **MicroLLM** (`micro-jev` `0.4.1`, `jevcompat 0.1`, and `microllm.guard.v1`).

## Reference Specifications

| Document | Scope | Primary Source Modules |
| :--- | :--- | :--- |
| [Environment Variables, Model Profiles, Ports, and Endpoints](configuration/environment-and-model-profiles.md) | Environment variables, ONNX INT8 profiles, aliases, defaults, network ports, and HTTP REST routes. | `server.py`, `registry.py`, `guard.py`, `schemas.py`, `server_models.py` |
| [CLI Commands, Script Flags, and Exit Codes](cli-and-scripts/cli-commands-and-exit-codes.md) | `microllm`, `micro-jev`, and `microllm-guard` subcommands, `build_versioned_docs.py` flags, and exit codes. | `cli.py`, `guard_cli.py`, `scripts/build_versioned_docs.py` |
| [Error, Threat, and SLA Alert Matrix](troubleshooting-codes/error-and-alert-matrix.md) | HTTP status shapes, `microllm.guard.v1` threats, JSON-RPC errors, SLA gates `G1`–`G8`, and runbooks. | `server.py`, `guard.py`, `guard_cli.py`, `quality_suite.py` |

## Normative Engineering & Algorithmic Manuals

| Document | Scope | Focus |
| :--- | :--- | :--- |
| [Complete Technical Specifications](../TECHNICAL_SPECIFICATIONS.md) | `jevcompat 0.1` (`32/32 MUST`), Pydantic v2 schemas, REST contracts, and runtime NFRs. | Protocol & Architecture |
| [Algorithmic Reference Manual](../ALGORITHMS.md) | Mathematical formulations for FastEncoder, AnyJev $L_0$, DecisionPivot, Conformal Safety, and RLCD². | Equations & Calibration |
| [Quality & Performance Test Plan](../TEST_PLAN.md) | Validation tracks (`750`–`1,000` cases), hypotheses `VAL-HYP-001`–`VAL-HYP-012`, and SLA gates. | Verification Criteria |

## Related Documentation Quadrants

| Quadrant | Purpose | Entrypoint |
| :--- | :--- | :--- |
| **Tutorials** | Isolated local sandbox onboarding lessons. | [Tutorials Index](../01-tutorials/index.md) |
| **How-To Guides** | Goal-oriented operational procedures and incident runbooks. | [How-To Guides Index](../02-how-to-guides/index.md) |
| **Explanation** | Architectural topology, design rationales, and ADRs. | [Explanation Index](../04-explanation/index.md) |
