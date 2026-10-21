# Journal de Bord des Actions — Projet MicroLLM

> **Règle de Gouvernance (`AGENTS.md` §2 & `.agents/rules/project-governance.md`)** :
> Ce journal de bord **DOIT être alimenté systématiquement à chaque demande, à chaque changement et à chaque résultat**.
> - Toute nouvelle demande est inscrite dans **« Actions en Cours »** dès l'ouverture de son Issue GitHub.
> - Dès la validation par les tests (`pytest`) et la création/fusion de la Pull Request, l'entrée est complétée avec les **Changements** et le **Résultat** chiffré, puis archivée dans **« Actions Terminées »**.

---

## 1. Actions en Cours (`EN COURS`)

| ID Action | Date / Heure (ISO) | Issue / Branche | Statut | Demande, Changements en cours & Résultats attendus |
| :--- | :--- | :--- | :--- | :--- |
| **ACT-006** | `2026-10-21T21:51` | Issue `#10` | `EN COURS` | **Demande** : Suite de benchmarks reproductible (540 items) et 20 améliorations architecturales.<br/>**Changements** : `src/microllm/benchmarks.py`, `benchmarks/microllm_eval_corpus.jsonl`, `README.md`, `docs/README.md`.<br/>**Résultat intermédiaire** : 75+ tests unitaires passants localement (`pytest`), corpus 540 items évalué à 98.1% sur `JevBench-Core`. |

---

## 2. Actions Terminées (`TERMINÉ`)

### ACT-011 — Correction de l'Échec du Workflow GitHub Actions Conda (Issue `#7` / PR `#9`)
- **Date** : `2026-10-21T22:07+02:00`
- **Statut** : `TERMINÉ`
- **Issue / PR** : Issue `#7` · PR `#9` (`fix/issue-7-conda-ci-workflow`, mergée dans `main`, commit `78a8d04`)
- **Demande** :
  - Corriger l'échec du job GitHub Actions [`Python Package using Conda` (Run `39659905478`, job `119429893101`, step 5:26)](https://github.com/jmrenouard/MicroLLM/actions/runs/39659905478/job/119429893101#step:5:26).
- **Changements** :
  - Création de [`environment.yml`](../environment.yml) à la racine du dépôt configurant l'environnement Conda `base` avec `pip`, `-e .[dev]` et `flake8>=7.0.0`.
  - Mise à jour de [`.github/workflows/python-package-conda.yml`](../.github/workflows/python-package-conda.yml) : passage de `actions/setup-python@v3` à `actions/setup-python@v5` et suppression des appels interactifs redondants `conda install flake8` et `conda install pytest` sans `--yes`.
  - Ajout des tests unitaires de non-régression [`tests/test_conda_workflow.py`](../tests/test_conda_workflow.py).
- **Résultat** :
  - `pytest tests/test_conda_workflow.py -v` : **2 / 2 tests passés (100%)**.
  - Job GitHub Actions [`Python Package using Conda` (Run `39667185608`, job `119455213016`)](https://github.com/jmrenouard/MicroLLM/actions/runs/39667185608/job/119455213016) : **100% vert en 1m34s** (`Install dependencies`, `Lint with flake8`, `Test with pytest`).

---

### ACT-010 — Ajout des GitHub Actions de Publication PyPI / TestPyPI et Métadonnées Packaging (Issue `#3`)
- **Date** : `2026-10-21T22:05+02:00`
- **Statut** : `TERMINÉ`
- **Issue / Branche** : Issue `#3` (`ci/issue-3-pypi-github-actions`)
- **Demande** :
  - Ajouter les GitHub Actions pour publier automatiquement le projet sous PyPI (et TestPyPI) et inscrire la règle de workflow Git/GitHub systématique (Issue $\rightarrow$ Branche $\rightarrow$ PR $\rightarrow$ Merge).
- **Changements** :
  - Création de [`.github/workflows/publish-pypi.yml`](../.github/workflows/publish-pypi.yml) :
    - Déclenchement sur publication de GitHub Release (`release: types: [published]`), tags `v*` (`push: tags: ["v*"]`) et exécution manuelle (`workflow_dispatch` avec choix `pypi` ou `testpypi`).
    - Job `test` sur matrice Python `3.10`, `3.11`, `3.12` (`pytest --cov=microllm --cov-report=term-missing`).
    - Job `build` (`python -m build` produisant `sdist` et `bdist_wheel` + validation `twine check --strict dist/*`).
    - Jobs `publish-pypi` et `publish-testpypi` utilisant l'action officielle PyPA `pypa/gh-action-pypi-publish@release/v1` avec **Trusted Publishing OIDC** (`id-token: write`).
  - Création de [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) validant les tests et le build du package sur chaque `push` / `pull_request` vers `main`.
  - Enrichissement de [`pyproject.toml`](../pyproject.toml) avec `keywords`, `classifiers` PyPI et `[project.urls]` (`Homepage`, `Repository`, `Issues`).
  - Ajout de la suite de tests unitaires [`tests/test_packaging_ci.py`](../tests/test_packaging_ci.py) vérifiant les métadonnées PyPI, le workflow de publication OIDC et la présence des règles de gouvernance Git/GitHub.
- **Résultat** :
  - `pytest tests/test_packaging_ci.py -v` : **3 / 3 tests passés (100%)**.
  - Build `python -m build` + `twine check --strict dist/*` : **100% valide** (`microllm-0.2.0.tar.gz` et `microllm-0.2.0-py3-none-any.whl`).
  - Pull Request ouverte, validée et fusionnée dans `main` via SSH ; Issue `#3` clôturée automatiquement.

---

### ACT-009 — Implémentation de la Suite Complète des GitHub Actions Python (Issue `#4`)
- **Date** : `2026-10-21T22:02+02:00`
- **Statut** : `TERMINÉ`
- **Issue / Branche** : Issue `#4` (`ci/issue-4-python-github-actions`)
- **Demande** :
  - Implémenter toutes les GitHub Actions nécessaires pour un projet Python (`ci.yml`, `unit-tests.yml`, `lint.yml`, `security.yml`, `publish-pypi.yml`, `python-package-conda.yml`, `release.yml`, et `.github/dependabot.yml`).
- **Changements** :
  - Création de [`.github/workflows/lint.yml`](../.github/workflows/lint.yml) (Ruff, Flake8 et Mypy sur Python 3.12).
  - Création de [`.github/workflows/security.yml`](../.github/workflows/security.yml) (analyse statique Bandit SAST + audit des vulnérabilités `pip-audit`).
  - Création de [`.github/workflows/release.yml`](../.github/workflows/release.yml) (build `sdist`/`wheel`, validation `twine check --strict`, création de GitHub Release sur tags `v*` via `softprops/action-gh-release@v2`).
  - Création de [`.github/dependabot.yml`](../.github/dependabot.yml) (mises à jour hebdomadaires pour les écosystèmes `pip` et `github-actions`).
  - Consolidation de [`.github/workflows/ci.yml`](../.github/workflows/ci.yml), [`.github/workflows/unit-tests.yml`](../.github/workflows/unit-tests.yml), [`.github/workflows/publish-pypi.yml`](../.github/workflows/publish-pypi.yml), [`.github/workflows/python-package-conda.yml`](../.github/workflows/python-package-conda.yml) et [`environment.yml`](../environment.yml).
  - Ajout de la suite TDD [`tests/test_python_github_actions.py`](../tests/test_python_github_actions.py) et sécurisation CI de [`tests/test_cascade_and_sdk.py`](../tests/test_cascade_and_sdk.py) pour les PDF gitignorés.
- **Résultat** :
  - `pytest tests/test_python_github_actions.py -v` : **5 / 5 tests passés (100%)**.
  - Suite complète : **72+ tests passés (100%), couverture globale 94%**.

---

### ACT-008 — Ajout du Workflow GitHub Actions des Tests Unitaires (Issue `#2` / PR `#8`)
- **Date** : `2026-10-21T21:58+02:00`
- **Statut** : `TERMINÉ`
- **Issue / PR** : Issue `#2` · PR `#8` (`ci/issue-2-unit-tests-github-actions`, mergée dans `main`, commit `aa76d8c`)
- **Demande** :
  - Ajouter les GitHub Actions permettant de réaliser automatiquement les tests unitaires à chaque push et Pull Request.
- **Changements** :
  - Création de [`.github/workflows/unit-tests.yml`](../.github/workflows/unit-tests.yml) (matrice Python `3.10`, `3.11`, `3.12` sur `ubuntu-latest`, déclenchement sur `push`, `pull_request` et `workflow_dispatch`, exécution de `pytest --cov=microllm --cov-report=term-missing --cov-report=xml --junitxml=pytest-results.xml -v`, upload des artefacts `coverage.xml` et `pytest-results.xml`).
  - Ajout de la suite de tests unitaires [`tests/test_unit_tests_workflow.py`](../tests/test_unit_tests_workflow.py) validant le workflow CI et l'inscription des règles GitHub.
  - Bascule du remote `origin` sur SSH (`git@github.com:jmrenouard/MicroLLM.git`) pour autoriser le push des fichiers sous `.github/workflows/`.
- **Résultat** :
  - `pytest tests/test_unit_tests_workflow.py -v` : **2 / 2 tests passés (100%)**.
  - Suite complète : **70 passed in 27.14s**.
  - PR `#8` ouverte, validée et fusionnée dans `main` ; Issue `#2` clôturée automatiquement.

---

### ACT-007 — Inscription des Règles de Gouvernance : Journal de Bord & Registre RCA (Issue `#5` / PR `#11`)
- **Date** : `2026-10-21T22:04+02:00`
- **Statut** : `TERMINÉ`
- **Issue / PR** : Issue `#5` · PR `#11` (`docs/issue-5-journal-de-bord-registre-rca`, mergée dans `main`, commit `4bca57f`)
- **Demande** :
  1. Ajouter dans les règles du projet l'obligation de tenir un **Journal de bord des actions en cours / terminées** (`docs/JOURNAL_DE_BORD.md`).
  2. Ajouter un **Registre RCA des erreurs, impasses, approximations et idées non implémentées** (`docs/REGISTRE_RCA.md`).
  3. Rendre obligatoire l'alimentation de ces registres **à chaque demande, à chaque changement et à chaque résultat**.
- **Changements** :
  - Mise à jour de [`AGENTS.md`](../AGENTS.md) (ajout des sections §2 *Journal de Bord* et §3 *Registre RCA*).
  - Création de la règle Antigravity toujours active [`.agents/rules/project-governance.md`](../.agents/rules/project-governance.md) (`trigger: always_on`).
  - Création et alimentation initiale exhaustive de [`docs/JOURNAL_DE_BORD.md`](JOURNAL_DE_BORD.md) (historique complet des actions `ACT-001` à `ACT-007`).
  - Création et alimentation initiale exhaustive de [`docs/REGISTRE_RCA.md`](REGISTRE_RCA.md) (`RCA-001`..`RCA-005`, `IMP-001`..`IMP-004`, `APP-001`..`APP-006`, `IDEA-001`..`IDEA-006`).
  - Ajout de la suite de tests unitaires [`tests/test_project_governance_rules.py`](../tests/test_project_governance_rules.py) vérifiant l'existence et le contenu normatif des règles et des deux registres.
- **Résultat** :
  - `pytest tests/test_project_governance_rules.py -v` : **3 / 3 tests passés (100%)**.
  - Suite globale `pytest` : **100% des tests passants**, couverture globale maintenue à **94%**.

---

### ACT-005 — Inscription du Workflow Git / GitHub Systématique (Issue `#1` / PR `#6`)
- **Date** : `2026-10-21T21:50+02:00`
- **Statut** : `TERMINÉ`
- **Issue / PR** : Issue `#1` · PR `#6` (`chore/issue-1-github-workflow-rules`, mergée dans `main`)
- **Demande** :
  - Inscrire dans les règles du projet l'obligation d'ouvrir systématiquement une Issue GitHub, de créer une branche dédiée, de soumettre une Pull Request et de la merger pour chaque demande utilisateur.
- **Changements** :
  - Création de [`AGENTS.md`](../AGENTS.md) à la racine du dépôt détaillant le cycle en 6 étapes (`gh issue create` $\rightarrow$ `git checkout -b` / `git worktree` $\rightarrow$ `pytest` $\rightarrow$ `git push` $\rightarrow$ `gh pr create` $\rightarrow$ `gh pr merge`).
  - Création des règles Antigravity [`.agents/rules/git-workflow.md`](../.agents/rules/git-workflow.md) et [`.agents/rules/github-workflow.md`](../.agents/rules/github-workflow.md) avec `trigger: always_on`.
  - Ajout du test de vérification `test_project_git_workflow_rules_registered` dans [`tests/test_packaging_ci.py`](../tests/test_packaging_ci.py).
- **Résultat** :
  - PR `#6` ouverte, validée et fusionnée dans `main` ; Issue `#1` clôturée automatiquement.

---

### ACT-004 — Ajout des Workflows GitHub Actions Conda & Jekyll (Commits `8cb51d4`, `9b00eea`)
- **Date** : `2026-10-21T21:14+02:00`
- **Statut** : `TERMINÉ`
- **Issue / Commit** : Commits `8cb51d4b` & `9b00eea8` sur `main`
- **Demande** :
  - Ajouter les workflows d'intégration continue GitHub Actions pour l'environnement Conda (`.github/workflows/python-package-conda.yml`) et la publication du site documentaire (`.github/workflows/jekyll-docker.yml`).
- **Changements** :
  - Création de [`.github/workflows/python-package-conda.yml`](../.github/workflows/python-package-conda.yml) et [`.github/workflows/jekyll-docker.yml`](../.github/workflows/jekyll-docker.yml).
  - Ajout de [`environment.yml`](../environment.yml) et des tests de validation [`tests/test_conda_workflow.py`](../tests/test_conda_workflow.py).
- **Résultat** :
  - Validation syntaxique et contractuelle du workflow Conda par tests unitaires (`tests/test_conda_workflow.py`).

---

### ACT-003 — Documentation Technique Markdown Complète & Enrichissement du README (Commits `89850ff`, `41c59e5`)
- **Date** : `2026-10-21T21:00+02:00`
- **Statut** : `TERMINÉ`
- **Issue / Commit** : Commits `89850ff1` & `41c59e51` sur `main`
- **Demande** :
  - Reproduire l'intégralité de la documentation technique interactive HTML (`docs/index.html`) dans un fichier Markdown lisible directement sur GitHub (`docs/README.md`) et enrichir le `README.md` principal avec les badges, les 4 primitives (`noul`, `choice`, `score`, `multi`), le pack Guardrail et le CLI `microllm pull`.
- **Changements** :
  - Création de [`docs/README.md`](README.md) (362 lignes : diagramme d'architecture Mermaid, formules mathématiques LaTeX, matrice de conformité `32/32 MUST`, détails Gear 1 / Gear 2 / Cascade / RLCD², SDK Python et table des endpoints HTTP).
  - Mise à jour de [`README.md`](../README.md) avec tableau comparatif System-1 vs System-2 et guide Quickstart.
- **Résultat** :
  - Documentation 100% synchronisée entre `README.md`, `docs/README.md` et `docs/index.html`.

---

### ACT-002 — Correction de l'Invariance à l'Ordre (AnyJev $L_0$), de la Négation Politique (`noul`) et de l'Attribution Token
- **Date** : `2026-10-21T20:58+02:00`
- **Statut** : `TERMINÉ`
- **Demande** :
  - Atteindre `0.000000` d'Order-Flip Rate sur `AnyJev-Order` même avec arrêt précoce ($\mu(\bar{z}) \ge \tau$), corriger les inversions sur les règles de politique comportant des clauses d'exemption (`unless`, `exempt`) dans `evaluate_noul`, et stabiliser `explain_state_tokens` sur des tokens isolés.
- **Changements** :
  - [`src/microllm/backends/prefill_llm.py`](../src/microllm/backends/prefill_llm.py) : tri canonique préalable des clés pour `choice` avant la séquence de Van der Corput + gestion explicite des marqueurs `exemption_markers` et `policy_false_markers`.
  - [`src/microllm/backends/fast_encoder.py`](../src/microllm/backends/fast_encoder.py) : alignement de la logique de vérification de politique (`is_policy_check`, `has_exemption`, `has_negation`) dans `evaluate_noul`.
  - [`src/microllm/backends/base.py`](../src/microllm/backends/base.py) : calcul direct d'attribution token par produit scalaire d'embeddings + chevauchement lexical + boost conceptuel dans `explain_state_tokens`.
- **Résultat** :
  - Exactitude `JevBench-Core` : **98.1%** (`choice` 100%, `score` 100%, `multi` 100%, `noul` 92%).
  - `DecisionPivot` : **100.0%** d'exactitude par paire contrastive (40/40 paires).
  - `AnyJev-Order` : **0.000000 Flip Rate** et **0.000000 Max Drift** avec **78.3% de rotations économisées**.

---

### ACT-001 — Conception & Implémentation Initiale du Moteur Décisionnel Local MicroLLM (Commit `ca776c0`)
- **Date** : `2026-10-21T20:56+02:00`
- **Statut** : `TERMINÉ`
- **Issue / Commit** : Commit initial `ca776c0d` (`feat: initial release of MicroLLM local System-One decision engine`)
- **Demande** :
  - Construire un moteur de décision probabiliste System-1 local (<15 ms sur CPU), 100% conforme aux 32 exigences `MUST` de la spécification `jevcompat 0.1`, synthétisant les 3 articles de recherche *OpenJev-RLCD* (`arXiv:2609.38850`), *AnyJev* (`arXiv:2610.00831`) et *StartLux-Decision* (`arXiv:2610.17309`).
- **Changements** :
  - Implémentation complète de `src/microllm/` (`schemas.py`, `confidence.py`, `calibration.py`, `rlcd.py`, `registry.py`, `client.py`, `server.py`, `backends/base.py`, `backends/fast_encoder.py`, `backends/prefill_llm.py`, `backends/cascade.py`).
  - Interfaces web interactives : **Decision Studio** (`src/microllm/static/index.html`) et documentation interactive (`docs/index.html`).
  - Suite de tests contractuels et scientifiques (`tests/test_jevcompat_contract.py`, `tests/test_anyjev_rotations.py`, `tests/test_calibration_conformal.py`, `tests/test_rlcd_trainer.py`, `tests/test_cascade_and_sdk.py`).
- **Résultat** :
  - **32 / 32 `MUST` `jevcompat 0.1` validés**, latence CPU p50 de `1.5 ms` (Gear 1), 60 tests initiaux passants à 100%.
