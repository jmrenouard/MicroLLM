---
trigger: always_on
description: Mandatory GitHub workflow: Issue -> Branch -> Tests -> Pull Request -> Merge for every request.
---

# Project Rule: Systematic GitHub Workflow (Issue + Branch + PR + Merge)

For every request made on the **MicroLLM** project, you MUST systematically:

1. **Open a GitHub Issue** (`gh issue create`) describing the request and its acceptance criteria.
2. **Create a dedicated branch** (`<type>/issue-<id>-<slug>`) from `main` (use `git worktree` if other files are being modified in the workspace).
3. **Implement and run unit tests** (`pytest`) until full validation.
4. **Commit (Conventional Commits) and push the branch** to `origin`.
5. **Create a Pull Request** (`gh pr create`) referencing `Closes #<id>`.
6. **Merge the Pull Request** (`gh pr merge --merge --delete-branch`) and update `main`.
