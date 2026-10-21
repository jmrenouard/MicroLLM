# Règles du Projet MicroLLM (AGENTS.md)

## 1. Workflow GitHub Systématique (Obligatoire pour chaque demande)

Pour **toute demande utilisateur** (nouvelle fonctionnalité, correction de bug, CI/CD, documentation, refactoring), l'agent **DOIT systématiquement** exécuter le cycle complet suivant sur GitHub :

1. **Ouverture d'une Issue GitHub** :
   - Créer une issue détaillée via `gh issue create --title "<type>: <titre>" --body "<contexte, objectifs et critères d'acceptation>"` avant ou au début du traitement de la demande.
   - Récupérer le numéro de l'issue (`#<id>`).

2. **Création d'une Branche Dédiée** :
   - Créer une branche spécifique à la demande à partir de `main` : `<type>/issue-<id>-<slug>` (ex. `ci/issue-1-unit-tests-actions`, `feat/issue-2-...`, `fix/issue-3-...`).
   - En cas de travaux parallèles dans le répertoire principal, utiliser `git worktree` pour isoler proprement la branche sans perturber les fichiers non commités d'autres tâches.

3. **Implémentation TDD & Tests Unitaires** :
   - Écrire ou mettre à jour les tests unitaires dans `tests/`.
   - Vérifier que l'ensemble de la suite passe localement (`pytest --cov=microllm --cov-report=term-missing -v`).

4. **Commit & Push** :
   - Effectuer des commits atomiques au format Conventional Commits (`feat:`, `fix:`, `ci:`, `docs:`, `test:`, `chore:`) en référençant l'issue (`Refs #<id>` ou `Closes #<id>`).
   - Pousser la branche sur le dépôt distant : `git push -u origin <branche>`.

5. **Création de la Pull Request (PR)** :
   - Ouvrir une Pull Request vers `main` via `gh pr create --base main --head <branche> --title "<type>: <titre> (#<id>)" --body "Closes #<id> ..."` avec le résumé des changements et le plan de test.

6. **Merge de la Pull Request & Clôture** :
   - Fusionner la Pull Request dans `main` via `gh pr merge <pr-id> --merge --delete-branch`.
   - Vérifier que l'issue associée est bien fermée et synchroniser la branche locale `main` (`git pull --ff-only origin main`).
