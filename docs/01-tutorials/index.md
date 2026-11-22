# Tutorials

> [!NOTE]
> **Learning-Oriented Quadrant (`01-tutorials/`):** Follow these guided, self-contained lessons in an isolated local
> sandbox to get hands-on experience with MicroLLM on standard CPU hardware.

Each tutorial starts from a clean local state, provides exact commands and expected outputs, and ends with a clean
teardown step. Zero GPUs or external cloud API keys are required.

---

## Guided Learning Path

| Lesson | What You Will Practice | Key Commands & Endpoints |
| :--- | :--- | :--- |
| [1. Local Server Quickstart](01-local-server-quickstart.md) | Start `microllm serve`, verify liveness, and execute a typed `choice` + `noul` inference request | `microllm serve`, `GET /healthz`, `GET /v1/models`, `POST /v1/systemone` |
| [2. Decision Studio and Calibration Lab](02-decision-studio-and-calibration-lab.md) | Explore the web playground, run Gear 1 vs. Gear 2 order-flip diagnostics, run RLCD² calibration, and verify 15 RCAs | `POST /v1/studio/analyze`, `POST /v1/calibrate`, `POST /v1/studio/rlcd-demo`, `microllm rca --all` |
| [3. Guard Middleware Sandbox](03-guard-middleware-sandbox.md) | Inspect the `microllm.guard.v1` contract and screen safe, destructive, and policy-bound tool calls via CLI and `stdin` | `microllm-guard --contract`, `microllm-guard --policy-file`, exit codes `0`, `2`, `3` |

---

## Structured Course Curriculum

For a full 6-module graduate and senior ML engineering curriculum covering mathematical derivations, conformal safety
proofs, and capstone deliverables, see the [Engineering & Scientific Syllabus](../SYLLABUS.md).

---

## Next Steps

- Execute production deployment, benchmark, and security tasks: [How-To Guides](../02-how-to-guides/index.md)
- Look up exhaustive CLI flags, environment variables, and error matrices: [Reference](../03-reference/index.md)
- Study Dual-Gear System-1 topology, AnyJev $L_0$ invariance, and ADRs: [Explanation](../04-explanation/index.md)
