---
trigger: always_on
description: Mandatory MicroLLM project governance — systematic update of the Action Logbook (docs/JOURNAL_DE_BORD.md) and RCA Register (docs/REGISTRE_RCA.md) on every request, change, and result.
---

# Project Rule: Systematic Update of Action Logbook and RCA Register

For **every user request, every technical/documentation change, and every test or benchmark result** (*à chaque demande, changement et résultat*) on the **MicroLLM** project, you **MUST MANDATORILY** update the following two living registers before closing the task:

## 1. Action Logbook (`docs/JOURNAL_DE_BORD.md`)

Must be updated on **every request (`chaque demande`)**, **every change (`changement`)**, and **every result (`résultat`)**:
- **Ongoing Actions (`Actions en Cours`)**: Record every new request as soon as work begins (`ACT-XXX`, date/time, Issue `#<id>`, branch, **Request / `Demande`**).
- **Completed Actions (`Actions Terminées`)**: Once implementation is verified, record:
  - **Request (`Demande`)**: Initial objective requested by the user.
  - **Changes (`Changements`)**: Created or modified files, impacted architecture or rules.
  - **Result (`Résultat`)**: Real test suite output (`pytest`), code coverage, benchmark metrics if applicable, and Git/GitHub references (Issue `#<id>`, PR `#<pr>`, commit).

## 2. RCA & Technical Transparency Register (`docs/REGISTRE_RCA.md`)

Must be updated on **every request (`chaque demande`)**, **every change (`changement`)**, and **every result (`résultat`)** across its 4 sections:
1. **Errors & Bugs Encountered (`RCA-XXX`)**: Symptom, Root Cause Analysis (RCA), Applied Fix, Non-regression Test, and Result.
2. **Technical Dead-Ends (`IMP-XXX`)**: Explored and abandoned paths with the technical or mathematical justification preventing re-attempts.
3. **Current Approximations (`APP-XXX`)**: Simplifications, heuristics, or mocks present in the code (`src/microllm/`), accuracy/latency/dependency trade-offs, and removal plan.
4. **Unimplemented Ideas (`IDEA-XXX`)**: Features from the *OpenJev-RLCD*, *AnyJev*, or *StartLux-Decision* papers and engineering improvements identified for the roadmap.

> **Strict Prohibition**: Never declare a request finished or merge a Pull Request without updating `docs/JOURNAL_DE_BORD.md` (and `docs/REGISTRE_RCA.md` whenever an error, dead-end, approximation, or idea is identified during the change).
