---
description: Mandatory Git and GitHub workflow (Issue -> Branch -> Pull Request -> Merge) for every request in the MicroLLM project.
trigger: always_on
---

# Project Rule: Systematic Issue + Branch + Pull Request + Merge

For every user request on this project, you **MUST MANDATORILY** execute the complete GitHub workflow:

1. **Create a GitHub Issue** with `gh issue create --title "..." --body "..."`.
2. **Create a dedicated branch** from `main`: `git checkout -b <type>/issue-<number>-<slug>` (use `git worktree` if other uncommitted changes exist in the workspace).
3. **Implement and test** all changes with `pytest` (100% passing tests).
4. **Commit and push** the branch (`git push -u origin <branch>`) with a Conventional Commits message referencing `Closes #<number>`.
5. **Create a Pull Request** with `gh pr create --base main --head <branch> --title "..." --body "Closes #<number>..."`.
6. **Merge the Pull Request** with `gh pr merge --merge --delete-branch` and update `main`.
7. **Close the Issue with a Comment**: Post a completion comment on the GitHub Issue detailing the changes, verification results, and merge commit (`gh issue comment <number> --body "..."`), then close the issue (`gh issue close <number>`).
