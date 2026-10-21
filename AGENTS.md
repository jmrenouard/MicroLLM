# MicroLLM Project Rules (AGENTS.md)

These rules apply **without exception** to every session and every agent working on the **MicroLLM** repository.

---

## 1. Mandatory GitHub Workflow (Required for Every Request)

For **every user request** (new feature, bug fix, CI/CD, documentation, or refactoring), the agent **MUST systematically** execute the complete GitHub lifecycle below:

1. **Open a GitHub Issue**:
   - Create a detailed issue using `gh issue create --title "<type>: <title>" --body "<context, objectives, and acceptance criteria>"` before or at the start of processing the request.
   - Retrieve the issue number (`#<id>`).

2. **Create a Dedicated Branch**:
   - Create a dedicated branch from `main`: `<type>/issue-<id>-<slug>` (e.g., `ci/issue-1-unit-tests-actions`, `feat/issue-2-...`, `fix/issue-3-...`).
   - When parallel work exists in the main working directory, use `git worktree` to cleanly isolate the branch without disturbing uncommitted files from other tasks.

3. **TDD Implementation & Unit Tests**:
   - Write or update unit tests in `tests/`.
   - Verify that the full test suite passes locally (`pytest --cov=microllm --cov-report=term-missing -v`).

4. **Update Governance Registers (`docs/ACTION_LOGBOOK.md` & `docs/RCA_REGISTER.md`)**:
   - Systematically update the Action Logbook (`docs/ACTION_LOGBOOK.md`) and the RCA Register (`docs/RCA_REGISTER.md`) before committing (see Sections 2 and 3 below).

5. **Commit & Push**:
   - Make atomic commits following the Conventional Commits format (`feat:`, `fix:`, `ci:`, `docs:`, `test:`, `chore:`) referencing the issue (`Refs #<id>` or `Closes #<id>`).
   - Push the branch to the remote repository: `git push -u origin <branch>`.

6. **Create a Pull Request (PR)**:
   - Open a Pull Request targeting `main` via `gh pr create --base main --head <branch> --title "<type>: <title> (#<id>)" --body "Closes #<id> ..."` including a summary of changes and the test plan.

7. **Merge the Pull Request & Close**:
   - Merge the Pull Request into `main` via `gh pr merge <pr-id> --merge --delete-branch`.
   - Verify that the linked issue is closed and synchronize the local `main` branch (`git pull --ff-only origin main`).

---

## 2. Ongoing / Completed Action Logbook (`docs/ACTION_LOGBOOK.md`)

The agent **MUST systematically update** [`docs/ACTION_LOGBOOK.md`](docs/ACTION_LOGBOOK.md) **on every request, every change, and every result**:

1. **Upon receiving a request (`IN PROGRESS` / `Ongoing Actions`)**:
   - Add an entry in the ongoing actions section with the identifier (`ACT-XXX`), ISO timestamp, GitHub Issue reference (`#<id>`), and exact formulation of the **Request**.
2. **During and after implementation (`CHANGES & RESULT`)**:
   - Document the exact list of created/modified files (**Changes**) as well as verification metrics (**Result**: number of passing unit tests, `pytest-cov` coverage, benchmark metrics, PR number, and merge commit).
3. **Upon completion (`COMPLETED` / `Completed Actions`)**:
   - Move the action to the completed actions section once tests and the Pull Request are validated.

---

## 3. RCA Register of Errors, Dead-Ends, Approximations, and Unimplemented Ideas (`docs/RCA_REGISTER.md`)

The agent **MUST systematically update** [`docs/RCA_REGISTER.md`](docs/RCA_REGISTER.md) **on every request, every change, and every result** whenever any of the following four events occurs:

1. **Errors & Bugs Encountered (`RCA-XXX`)**:
   - Every runtime error, test failure, benchmark regression, typing issue, or CI/CD conflict must be logged with: *Observed Symptom*, *Root Cause Analysis (5 Whys)*, *Applied Change / Fix*, and *Result / Non-Regression Test*.
2. **Technical Dead-Ends & Abandoned Leads (`IMP-XXX`)**:
   - Every explored and rejected approach (mathematical bias, incompatible dependency, latency or accuracy degradation) must be recorded so no subsequent session repeats the same dead-end.
3. **Current Approximations & Simplifying Assumptions (`APP-XXX`)**:
   - Every simplification compared to the reference research papers (*OpenJev-RLCD*, *AnyJev*, *StartLux-Decision*) or production environments must be explicitly declared with its measured impact and removal condition.
4. **Unimplemented Ideas & Future Roadmap (`IDEA-XXX`)**:
   - Every feature, optimization, or extension identified during a request or review that is outside the immediate scope must be logged with its priority and target module.
