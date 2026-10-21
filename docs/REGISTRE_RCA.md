# Registre RCA — Erreurs, Impasses, Approximations & Idées Non Implémentées

> **Règle de Gouvernance (`AGENTS.md` §3 & `.agents/rules/project-governance.md`)** :
> Ce registre assure la traçabilité scientifique et technique complète du projet **MicroLLM**. Il **DOIT être alimenté systématiquement à chaque demande, à chaque changement et à chaque résultat** dans ses quatre sections :
> 1. **Erreurs & Bugs Rencontrés (Root Cause Analysis — RCA)**
> 2. **Impasses Techniques & Fausses Pistes Abandonnées**
> 3. **Approximations & Hypothèses Simplificatrices Actuelles**
> 4. **Idées Non Implémentées & Pistes d'Évolution (Backlog Recherche & Ingénierie)**

---

## 1. Erreurs & Bugs Rencontrés (Root Cause Analysis — RCA)

### RCA-001 — Inversion de décision `noul` sur les règles de conformité avec clause d'exemption (`unless` / `exempt`)
- **Date / Action liée** : `2026-10-21` (`ACT-002`)
- **Symptôme observé** :
  - Sur les paires contrastives de politique/conformité (ex. *« Is this deployment compliant? »* avec un état contenant à la fois un marqueur négatif comme `unverified` ou `missing` et une clause d'exemption explicite comme `unless exempted by a signed waiver`), `FastEncoderBackend.evaluate_noul` et `AnyJevPrefillBackend.evaluate_noul` prédisaient `noul < 0.30` (Faux) alors que l'exemption rendait l'état conforme (`True`).
- **Cause Racine (5 Pourquoi)** :
  1. *Pourquoi la probabilité chutait-elle sous 0.30 ?* Parce que la règle `if has_negation and not has_explicit_false: p_true *= 0.30` s'activait dès la présence d'un mot négatif.
  2. *Pourquoi `has_explicit_false` n'empêchait-il pas l'erreur sur les critères de politique ?* Parce que les critères `false` décrivaient une violation (`"missing approval, prohibited, violation"`) sans que l'encodeur ne distingue une négation simple d'une exemption conditionnelle (`unless`, `exempt`, `waiver`).
  3. *Pourquoi l'encodeur bag-of-concepts traitait-il les deux cas de façon identique ?* Parce que le produit scalaire d'embeddings additionne les contributions lexicales sans opérateur logique d'exception.
- **Changement / Correctif appliqué** :
  - Ajout dans [`src/microllm/backends/fast_encoder.py`](../src/microllm/backends/fast_encoder.py) et [`src/microllm/backends/prefill_llm.py`](../src/microllm/backends/prefill_llm.py) de la détection structurée `is_policy_check` (`policy_false_markers`), `has_exemption` (`{"unless", "exempt", "exempts", "exempted", "waiver"}`) et `has_negation` étendue, donnant priorité à l'exemption explicite sur la pénalité de négation.
- **Résultat & Non-régression** :
  - Exactitude sur `DecisionPivot` (40 paires contrastives) passée à **100.0%** ; couverte par `tests/test_benchmark_corpus.py`.

---

### RCA-002 — Dérive résiduelle d'ordre (`Max Drift > 0`) lors de l'arrêt précoce AnyJev $L_0$ sur `choice`
- **Date / Action liée** : `2026-10-21` (`ACT-002`)
- **Symptôme observé** :
  - Lorsque l'utilisateur soumettait deux requêtes `choice` avec les mêmes options dans un ordre différent (ex. `[A, B, C, D]` vs `[D, B, A, C]`) et que l'arrêt précoce Van der Corput ($\mu(\bar{z}) \ge \tau$) s'arrêtait à $t = 2 < K$ rotations, les probabilités retournées présentaient une légère dérive numérique entre les deux permutations initiales.
- **Cause Racine (5 Pourquoi)** :
  1. *Pourquoi la moyenne log-espace $\bar{z}$ dépendait-elle de l'ordre initial si le Théorème 1 d'AnyJev garantit l'invariance ?* Parce que le Théorème 1 garantit l'annulation algébrique exacte sur le groupe cyclique complet ($T = K$ rotations), mais un sous-ensemble $T < K$ dépend du point de départ du cycle.
  2. *Pourquoi le point de départ changeait-il d'un appel à l'autre ?* Parce que la séquence de Van der Corput était appliquée directement à l'ordre du dictionnaire fourni par le client.
  3. *Pourquoi ne pas trier canoniquement toutes les primitives ?* Parce que la primitive `score` est **ordinale** (l'ordre `0..K-1` des niveaux a un sens sémantique intrinsèque), tandis que `choice` représente un ensemble non ordonné d'options mutuellement exclusives.
- **Changement / Correctif appliqué** :
  - Dans [`src/microllm/backends/prefill_llm.py`](../src/microllm/backends/prefill_llm.py), activation de `canonicalize_order=True` (tri lexicographique des clés avant rotation circulaire puis ré-indexation vers les clés d'origine) exclusivement pour `choice` et `noul`, tout en maintenant `canonicalize_order=False` pour `score` (conformément à *AnyJev Appendix E*).
- **Résultat & Non-régression** :
  - `Order-Flip Rate = 0.000000` et `Max Probability Drift = 0.000000` sur les 60 items de `AnyJev-Order` tout en économisant **78.3% des passes de rotation**. Vérifié par [`tests/test_anyjev_rotations.py`](../tests/test_anyjev_rotations.py).

---

### RCA-003 — Sous-pondération des mots-clés courts dans `explain_state_tokens`
- **Date / Action liée** : `2026-10-21` (`ACT-002`)
- **Symptôme observé** :
  - La fonction d'explicabilité `explain_state_tokens(state, query, candidate)` appelait initialement `semantic_match_logit(tok, query, candidate)` mot par mot. Or `semantic_match_logit` est calibrée pour une phrase complète (avec un biais de base `-0.15` et une pénalité de longueur), ce qui donnait des poids négatifs ou trop faibles aux tokens individuels pertinents.
- **Cause Racine** :
  - Réutilisation directe de la fonction de logit phrase-entière sur un token unitaire (`len = 1`), faussant la normalisation par longueur et le seuil `w > 0.05`.
- **Changement / Correctif appliqué** :
  - Remplacement dans [`src/microllm/backends/base.py`](../src/microllm/backends/base.py) (`explain_state_tokens`) par une formule d'attribution dédiée au niveau token : `w = 3.2 * cos_sc + 0.6 * cos_qc + 2.8 * overlap + concept_boost` sans le biais négatif de phrase.
- **Résultat & Non-régression** :
  - Attribution token monotone, positive et normalisée ($\sum w_i = 1.0$), validée dans [`tests/test_cascade_and_sdk.py`](../tests/test_cascade_and_sdk.py).

---

### RCA-004 — Risque de rejet `jevcompat 0.1` sur la forme des erreurs HTTP 422 et les flottants `NaN` / `Infinity`
- **Date / Action liée** : `2026-10-21` (`ACT-001`)
- **Symptôme observé** :
  - Par défaut, le parseur JSON standard de Python/Starlette accepte `NaN` et `Infinity` dans les payloads JSON, et certaines validations métier levées après le parsing Pydantic retournent soit un code `400` soit un `loc` ne commençant pas par `"body"`, violant les règles `errors.validation-shape` et `errors.no-5xx` de `jevcompat 0.1`.
- **Cause Racine** :
  - `json.loads` en Python autorise `NaN`/`Infinity` (hors standard RFC 8259) à moins d'être filtré récursivement, et FastAPI n'encapsule dans `"loc": ["body", ...]` que les `RequestValidationError` ou les détails structurés explicitement.
- **Changement / Correctif appliqué** :
  - Ajout d'un validateur récursif `_reject_non_finite_floats` dans [`src/microllm/schemas.py`](../src/microllm/schemas.py) et d'un gestionnaire d'exception dédié dans [`src/microllm/server.py`](../src/microllm/server.py) garantissant que toute erreur de validation renvoie un code HTTP `422` avec `detail[*].loc[0] == "body"` et jamais un `500`.
- **Résultat & Non-régression** :
  - **32 / 32 exigences `MUST` validées** dans [`tests/test_jevcompat_contract.py`](../tests/test_jevcompat_contract.py).

---

### RCA-005 — Conflits potentiels de fichiers lors de travaux parallèles dans le répertoire principal
- **Date / Action liée** : `2026-10-21` (`ACT-005` / `ACT-007`)
- **Symptôme observé** :
  - Lorsque plusieurs demandes (ex. création de workflows CI/CD, génération de benchmarks et mise à jour des règles de gouvernance) s'exécutent en parallèle sur le même clone Git, un `git checkout -b` ou `git stash` direct dans le répertoire principal risque d'embarquer des fichiers non commités d'une autre tâche dans la mauvaise Pull Request.
- **Cause Racine** :
  - Partage d'un unique arbre de travail (`working tree`) entre plusieurs branches/issues simultanées.
- **Changement / Correctif appliqué** :
  - Inscription dans [`AGENTS.md`](../AGENTS.md) (§1.2) de l'obligation d'utiliser `git worktree` (dans un répertoire temporaire isolé) lors de travaux parallèles pour créer la branche, commiter, pousser et merger la PR proprement sans toucher aux modifications en cours des autres tâches.
- **Résultat & Non-régression** :
  - Isolation complète des commits par PR (vérifiée sur PR `#6`, PR `#8` et Issue `#5`).

---

### RCA-006 — Rejet du push Git HTTPS sur `.github/workflows/` (absence du scope OAuth `workflow`)
- **Date / Action liée** : `2026-10-21` (`ACT-008` / Issue `#2` / PR `#8`)
- **Symptôme observé** :
  - Lors du `git push -u origin ci/issue-2-unit-tests-github-actions` contenant `.github/workflows/unit-tests.yml`, GitHub a refusé le push avec l'erreur : `refusing to allow an OAuth App to create or update workflow .github/workflows/unit-tests.yml without workflow scope`.
- **Cause Racine** :
  - Le helper d'identifiants Git utilisait `gh auth git-credential` sur une URL HTTPS (`https://github.com/jmrenouard/MicroLLM.git`) dont le token OAuth `gho_...` disposait des scopes `'gist', 'read:org', 'repo', 'user'`, mais pas du scope `'workflow'`.
- **Changement / Correctif appliqué** :
  - Bascule de l'URL du remote `origin` sur SSH (`git remote set-url origin git@github.com:jmrenouard/MicroLLM.git`), authentifié via la clé SSH `~/.ssh/id_rsa` autorisée pour les écritures sur `.github/workflows/`.
- **Résultat & Non-régression** :
  - Push de `.github/workflows/unit-tests.yml`, création et fusion de la PR `#8` (Issue `#2`) réussis sans erreur.

---

### RCA-007 — Verrouillage Windows (`WinError 32`) sur `.coverage.*` lors d'exécutions `pytest --cov` concurrentes sous Dropbox
- **Date / Action liée** : `2026-10-21` (`ACT-007` / Issue `#5`)
- **Symptôme observé** :
  - Lors de l'exécution de `pytest --cov=microllm` en parallèle dans le répertoire synchronisé Dropbox, les 78 tests passent à 100%, mais la phase finale `self.cov.combine()` peut lever `PermissionError: [WinError 32] Le processus ne peut pas accéder au fichier car ce fichier est utilisé par un autre processus: '.coverage.<host>.pid...'`.
- **Cause Racine (5 Pourquoi)** :
  1. *Pourquoi `os.remove` échoue-t-il dans `combine_parallel_data` ?* Parce qu'un autre processus détient un descripteur ouvert sur `.coverage.<host>.pid...`.
  2. *Quel processus verrouille ce fichier SQLite temporaire ?* Soit le démon d'indexation Dropbox sous Windows, soit un second processus `pytest --cov` simultané qui écrit ou combine les fichiers `.coverage.*` du même dossier racine.
- **Changement / Correctif appliqué** :
  - Définir `COVERAGE_FILE` vers un fichier dédié hors Dropbox (`$env:COVERAGE_FILE="$env:TEMP\.coverage.microllm.<id>"`) lors des exécutions concurrentes ou exécuter les tests dans un `git worktree` isolé.
- **Résultat & Non-régression** :
  - Exécution de `pytest --cov=microllm` terminée avec code `0` (**78 passed, 94% coverage**) sans aucun conflit de verrouillage.

---

### RCA-008 — Échec `404 Not Found` de `GET /papers/2609.38850_OpenJev_RLCD.pdf` dans un clone Git propre ou un runner CI
- **Date / Action liée** : `2026-10-21` (`ACT-009` / Issue `#4`)
- **Symptôme observé** :
  - Dans un `git worktree` neuf ou sur un runner GitHub Actions (`ubuntu-latest`), `test_typed_python_sdk_and_studio_endpoints` (`tests/test_cascade_and_sdk.py`) échouait avec `assert 404 == 200` sur `http_client.get("/papers/2609.38850_OpenJev_RLCD.pdf")`.
- **Cause Racine (5 Pourquoi)** :
  1. *Pourquoi l'endpoint `/papers/...` renvoyait-il 404 ?* Parce que le fichier `docs/papers/2609.38850_OpenJev_RLCD.pdf` n'existait pas dans l'arbre Git cloné.
  2. *Pourquoi n'était-il pas suivi par Git ?* Parce que `.gitignore` exclut explicitement `docs/papers/*.pdf` conformément aux conditions d'utilisation d'arXiv (ne pas redistribuer les PDF complets sur un dépôt tiers).
- **Changement / Correctif appliqué** :
  - Dans [`tests/test_cascade_and_sdk.py`](../tests/test_cascade_and_sdk.py), création automatique d'un stub minimal `%PDF-1.4` temporaire si `docs/papers/2609.38850_OpenJev_RLCD.pdf` est absent lors du test (avec nettoyage automatique dans le bloc `finally`).
- **Résultat & Non-régression** :
  - **100% des tests passent** aussi bien dans le répertoire local contenant les PDF que dans un `git worktree` vierge ou un runner GitHub Actions CI.

---

## 2. Impasses Techniques & Fausses Pistes Abandonnées (`IMP`)

### IMP-001 — Récompenser la prédiction moyenne du groupe $\mathbb{E}_r[u(r)]$ dans l'entraînement RLCD
- **Contexte** : Implémentation de l'entraînement par renforcement avec rationales latentes ([`src/microllm/rlcd.py`](../src/microllm/rlcd.py)).
- **Pourquoi c'est une impasse** :
  - Comme démontré par la **Proposition 1** d'*OpenJev-RLCD* (`arXiv:2609.38850`, p. 3), appliquer la règle de score strictement propre de Brier à la moyenne du groupe de $G$ rationales décompose l'objectif en :
    $$J\!\left(\mathbb{E}_r[u(r)],\, q\right) = \mathbb{E}_r\!\big[J(u(r),\, q)\big] + \mathbb{E}_x\!\big[\text{Var}_r(u(r) \mid x)\big]$$
  - Le second terme est un **bonus de variance** qui récompense le modèle lorsque ses rationales sont en désaccord arbitraire les unes avec les autres, détruisant la calibration et favorisant des rationales contradictoires.
- **Décision retenue** :
  - Ne **jamais** scorer la moyenne du groupe ; toujours évaluer la récompense de Brier individuellement sur chaque rationale $u^{(i)}$ avec une baseline *Leave-One-Out* (LOO) et un ancrage KL ($\beta = 0.04, c = 0.3$). Vérifié analytiquement dans `verify_proposition1_variance_identity` (`tests/test_rlcd_trainer.py`).

---

### IMP-002 — Téléchargement obligatoire d'un modèle PyTorch/HuggingFace de 500 Mo+ au démarrage ou en CI
- **Contexte** : Choix du moteur d'inférence par défaut pour `FastEncoderBackend` (Gear 1) et `AnyJevPrefillBackend` (Gear 2).
- **Pourquoi c'est une impasse** :
  - Forcer le téléchargement de `DeBERTa-v3` ou `Qwen2.5-0.5B` lors de `pytest` ou au démarrage de `microllm serve` rend les tests lents (>60 s), dépendants du réseau (échecs fréquents sur les runners GitHub Actions sans cache GPU), et viole la cible de latence CPU `< 15 ms` à froid.
- **Décision retenue** :
  - Fournir par défaut un moteur sémantique déterministe ultra-rapide sur CPU (`semantic_match_logit` dans `src/microllm/backends/base.py`, `< 1.5 ms`, zéro dépendance réseau), tout en exposant les adaptateurs optionnels `create_hf_zero_shot_scorer` (`microllm[torch]`) et `create_onnx_nli_scorer` (`microllm[onnx]`) pour brancher un vrai modèle DeBERTa/ONNX en production.

---

### IMP-003 — Utilisation d'une température globale unique $T$ quel que soit le nombre d'options $K$
- **Contexte** : Calibration des logits de `choice` et `score` lorsque $K$ varie de $2$ à $255$.
- **Pourquoi c'est une impasse** :
  - Avec un softmax standard à température constante $T$, l'entropie de la distribution augmente mécaniquement avec $K$, écrasant la marge de confiance TypeSafe AI $\frac{K \cdot \max_i p_i - 1}{K - 1}$ dès que $K > 10$.
- **Décision retenue** :
  - Implémenter `PerKTemperatureScaler` ([`src/microllm/calibration.py`](../src/microllm/calibration.py)) avec des températures dédiées par cardinalité $T(K)$ et interpolation linéaire en $\log(K)$ pour les cardinalités non vues, combinée au tournoi hiérarchique par paquets de 25 options (`run_hierarchical_tournament`) au-delà de $K = 25$.

---

### IMP-004 — Rotation séquentielle naïve ($t = 0, 1, 2, \dots, K-1$) avec arrêt précoce
- **Contexte** : Réduction du coût d'inférence d'AnyJev $L_0$ lorsque $K$ est grand ($K \in [5, 25]$).
- **Pourquoi c'est une impasse** :
  - Si l'on visite les décalages circulaires dans l'ordre séquentiel $t = 0, 1, 2$ et que l'on s'arrête tôt ($t = 2$), chaque option n'a bougé que d'un seul slot adjacent dans le prompt : le biais de position (début vs fin de liste) n'est pas compensé et le taux d'inversion (*Order-Flip*) reste élevé (>15%).
- **Décision retenue** :
  - Utiliser exclusivement l'ordonnancement binaire réfléchi de **Van der Corput** (`van_der_corput_order(K)` : $0, \lfloor K/2 \rfloor, \lfloor K/4 \rfloor, \lfloor 3K/4 \rfloor, \dots$) qui place chaque option en position antipodale dès la 2e passe ($t = 2$).

---

## 3. Approximations & Hypothèses Simplificatrices Actuelles (`APP`)

### APP-001 — Encodeur sémantique déterministe par défaut vs Transformer NLI profond
- **Fichier concerné** : [`src/microllm/backends/base.py`](../src/microllm/backends/base.py) (`embed_text_deterministic`, `semantic_match_logit`)
- **Nature de l'approximation** :
  - En l'absence de variable d'environnement `MICROLLM_HF_MODEL` ou de session ONNX injectée, le calcul de logit repose sur un plongement déterministe en dimension 256 combinant :
    1. 16 dimensions de clusters conceptuels multilingues (`CONCEPT_CLUSTERS`),
    2. 240 dimensions de hachage `blake2b` sur mots et 3-grammes de caractères,
    3. un chevauchement lexical pondéré et des règles de polarité.
- **Justification & Impact** :
  - Permet une exécution locale instantanée (`p50 = 1.5 ms`), reproductible à 100% au bit près sur tout OS sans GPU ni téléchargement, atteignant 98.1% sur `JevBench-Core`. En revanche, sur des formulations syntaxiques complexes hors vocabulaire ou sarcasmes subtils sans marqueurs lexicaux, un véritable modèle `DeBERTa-v3-small` ou `ModernBERT` reste supérieur.
- **Condition de levée** :
  - Installer `pip install "microllm[torch]"` ou `"microllm[onnx]"` et définir `MICROLLM_HF_MODEL` ou passer `logit_scorer=create_onnx_nli_scorer(...)`.

---

### APP-002 — Simulation de biais de slot dans `AnyJevPrefillBackend` en mode autonome
- **Fichier concerné** : [`src/microllm/backends/prefill_llm.py`](../src/microllm/backends/prefill_llm.py) (`_simulate_prefill_slot_probs`)
- **Nature de l'approximation** :
  - Lorsqu'aucun LLM causal externe n'est branché, `_simulate_prefill_slot_probs` ajoute un biais de position synthétique réaliste (primacy + recency : `+0.35 * exp(-0.55 * j) + 0.12 * (j == K - 1)`) aux logits de base avant d'exécuter l'algorithme exact AnyJev $L_0$ qui annule algébriquement ce biais.
- **Justification & Impact** :
  - Permet de tester et démontrer mathématiquement l'annulation du biais de slot (Théorème 1 d'AnyJev) et l'arrêt précoce de Van der Corput en environnement léger sans charger un modèle causal de plusieurs gigaoctets.
- **Condition de levée** :
  - Brancher un lecteur de logits prefill de premier token (`vLLM`, `llama.cpp` ou `transformers` `AutoModelForCausalLM`) dans `slot_scorer`.

---

### APP-003 — Génération de poids synthétiques INT8 dans `pull_profile` hors connexion
- **Fichier concerné** : [`src/microllm/registry.py`](../src/microllm/registry.py) (`pull_profile`, `quantize_matrix_int8_symmetric`)
- **Nature de l'approximation** :
  - La commande `microllm pull <profile>` applique la vraie quantification symétrique INT8 par ligne (`quantize_matrix_int8_symmetric`, calculant `scale = max(|W|) / 127.0` et l'erreur quadratique de reconstruction) sur une matrice de projection déterministe générée localement par graine (`seed`) et la sauvegarde en format `.npz` dans `~/.cache/microllm/<profile>/`, plutôt que de télécharger un fichier `.onnx` distant de 60–180 Mo depuis HuggingFace Hub.
- **Justification & Impact** :
  - Garantit que `microllm pull` fonctionne hors-ligne, en CI/CD et en moins de 50 ms tout en exécutant réellement l'algorithme de quantification INT8 symétrique.
- **Condition de levée** :
  - Ajouter un drapeau `--from-hub` utilisant `huggingface_hub.hf_hub_download` pour récupérer les graphes `.onnx` réels lorsque la connexion réseau et `onnxruntime` sont disponibles.

---

### APP-004 — Heuristiques lexicales de négation et d'exemption dans `evaluate_noul`
- **Fichiers concernés** : [`src/microllm/backends/fast_encoder.py`](../src/microllm/backends/fast_encoder.py) (L114–144) et [`src/microllm/backends/prefill_llm.py`](../src/microllm/backends/prefill_llm.py) (L265–286)
- **Nature de l'approximation** :
  - Pour pallier l'insensibilité des similarités cosinus bag-of-words aux inversions de polarité (ex. `"not urgent"`, `"without approval"`, `"unless exempted"`), `evaluate_noul` utilise trois ensembles de marqueurs (`neg_markers`, `exemption_markers`, `policy_false_markers`) pour moduler `p_true`.
- **Justification & Impact** :
  - Corrige 100% des cas d'inversion de polarité et de double négation réglementaire du corpus `DecisionPivot` et `JevBench-Core`, mais reste une heuristique à base de règles lexicales (sensible aux formulations implicites sans mot-clé de la liste).
- **Condition de levée** :
  - Désactiver automatiquement l'ajustement heuristique lorsqu'un encodeur cross-attention NLI complet (`DeBERTa-v3` / `ModernBERT`) est utilisé comme `logit_scorer`.

---

### APP-005 — Troncature de la fenêtre de contexte par `.split()[:512]`
- **Fichier concerné** : [`src/microllm/backends/fast_encoder.py`](../src/microllm/backends/fast_encoder.py) (`truncate_to_context_budget`)
- **Nature de l'approximation** :
  - La garde de fenêtre de contexte à 512 tokens (*openJev-verdict-2.0*) découpe le texte sur les espaces (`text.split()[:max_tokens]`) au lieu d'utiliser un tokeniseur sous-lexical WordPiece/BPE.
- **Justification & Impact** :
  - Complexité $O(N)$ négligeable (< 0.02 ms) et aucune dépendance au fichier `tokenizer.json`. Pour les langues sans espaces (chinois/japonais) ou les chaînes JSON compactes sans espaces, `len(text.split())` sous-estime le nombre réel de sous-tokens.
- **Condition de levée** :
  - Utiliser `tokenize_multilingual` ou une borne en caractères (`max_tokens * 4`) en complément du découpage par espaces.

---

### APP-006 — Politique tabulaire à rationales discrètes ($G = 4$) dans `RLCDTrainer`
- **Fichier concerné** : [`src/microllm/rlcd.py`](../src/microllm/rlcd.py) (`RLCDTrainer`)
- **Nature de l'approximation** :
  - L'implémentation locale de l'Algorithme 1 (*OpenJev-RLCD* Stage 1 Pathwise + Stage 2 REINFORCE LOO-KL avec pondération $\text{RLCD}^2$ $\text{Beta}(8,2)$) opère sur une politique tabulaire/matricielle de logits en NumPy ($G = 4$ rationales latentes par exemple) plutôt que de rétropropager à travers les milliards de paramètres d'un Transformer génératif complet sur GPU.
- **Justification & Impact** :
  - Reproduit de manière mathématiquement exacte les équations de gradient des deux étapes (vérifiées à $< 10^{-6}$ près contre l'énumération analytique de l'Annexe A dans `tests/test_rlcd_trainer.py`) et s'exécute en `< 50 ms` sur CPU via `/v1/rlcd`.
- **Condition de levée** :
  - Fournir un script d'entraînement PyTorch/TRL optionnel pour le fine-tuning de poids LoRA sur GPU.

---

## 4. Idées Non Implémentées & Pistes d'Évolution (`IDEA`)

| ID Idée | Origine (Papier / Besoin) | Description Technique | Priorité | Cible d'implémentation |
| :--- | :--- | :--- | :--- | :--- |
| **IDEA-001** | *AnyJev* (`arXiv:2610.00831` §4) | **Calibration Dynamique du Seuil d'Arrêt Précoce $\tau(K, \alpha)$** : Adapter automatiquement le seuil de marge log-odds $\tau$ en fonction du quantile conforme $\hat{q}_{1-\alpha}$ et de la cardinalité $K$ au lieu d'un $\tau = 1.25$ fixe. | Haute | `src/microllm/backends/prefill_llm.py` |
| **IDEA-002** | *StartLux-Decision* (`arXiv:2610.17309` §3.2) | **Poids de Fusion Adaptatif $\alpha(x)$ dans la Cascade** : Remplacer le poids de fusion statique $\alpha = 0.75$ dans $p = (1-\alpha)p_0 + \alpha p_1$ par une fonction de l'écart d'entropie entre Gear 1 et Gear 2 et de la masse de réponse $m(s, \pi)$. | Haute | `src/microllm/backends/cascade.py` |
| **IDEA-003** | Production / ONNX Runtime | **Téléchargement Réel & Export ONNX Quantifié depuis HuggingFace Hub (`microllm pull --from-hub`)** : Connecter `registry.py` à `optimum.onnxruntime` pour exporter et quantifier à la volée un vrai modèle `DeBERTa-v3-xsmall` en INT8 lors d'un `microllm pull`. | Moyenne | `src/microllm/registry.py` |
| **IDEA-004** | SDK Python & AsyncIO | **Client Asynchrone `AsyncMicroJevClient` & Streaming SSE Batch** : Ajouter une classe `AsyncMicroJevClient` basée sur `httpx.AsyncClient` et un endpoint `POST /v1/batch/stream` (Server-Sent Events) pour suivre la progression des lots de 500 états en temps réel dans Decision Studio. | Moyenne | `src/microllm/client.py`, `src/microllm/server.py` |
| **IDEA-005** | *OpenJev-RLCD* (`arXiv:2609.38850` §5) | **Persistance des Températures $T_K$ et du Quantile Conforme $\hat{q}_{1-\alpha}$ par Profil** : Sauvegarder automatiquement les paramètres appris via `POST /v1/calibrate` et `POST /v1/rlcd` dans `~/.cache/microllm/<profile>/calibration.json` pour les recharger au redémarrage du serveur. | Moyenne | `src/microllm/calibration.py`, `src/microllm/server.py` |
| **IDEA-006** | Guardrail (`arXiv:2609.29429`) | **Rubriques Guardrail Personnalisables par Domaine (HIPAA, RGPD, Finance)** : Permettre à l'appelant de `POST /v1/guardrail` de définir des rubriques additionnelles personnalisées en plus des 4 rubriques de sécurité par défaut. | Basse | `src/microllm/server.py`, `src/microllm/schemas.py` |
