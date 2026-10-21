---
trigger: always_on
description: Workflow GitHub obligatoire : Issue -> Branche -> Tests -> Pull Request -> Merge pour chaque demande.
---

# Règle Projet : Workflow GitHub Systématique (Issue + Branche + PR + Merge)

Pour chaque demande formulée sur le projet **MicroLLM**, tu DOIS systématiquement :

1. **Ouvrir une Issue GitHub** (`gh issue create`) décrivant la demande et ses critères de validation.
2. **Créer une branche dédiée** (`<type>/issue-<id>-<slug>`) depuis `main` (utiliser `git worktree` si d'autres fichiers sont en cours de modification dans l'espace de travail).
3. **Implémenter et exécuter les tests unitaires** (`pytest`) jusqu'à validation complète.
4. **Commiter (Conventional Commits) et pousser la branche** sur `origin`.
5. **Créer une Pull Request** (`gh pr create`) référençant `Closes #<id>`.
6. **Merger la Pull Request** (`gh pr merge --merge --delete-branch`) et mettre à jour `main`.
