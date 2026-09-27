# Modern Backend Stand-by — Basket Corp

## DO NOT ENABLE AUTOMATICALLY

**Fondation préparée, testée et figée ; elle n’est pas activée comme système principal de production. Aucune activation n’est autorisée par ce document.**

Le fonctionnement actuel validé de Basket Corp continue de reposer sur le système legacy existant tant qu’une décision explicite de migration n’a pas été prise. L’existence des commits, des migrations, de tables modernes éventuellement installées ou des flags **ne signifie pas que le moderne doit être activé**.

Ce document décrit le stand-by décidé le 27 septembre 2026. Il ne constate ni l’état actuel de la DB de production, ni ses variables d’environnement, ni le code effectivement déployé : aucun accès production n’a été effectué pour le rédiger. Un checkpoint GitHub publié n’est pas une preuve de déploiement ou d’activation.

## Pourquoi différer l’activation

Le jeu fonctionne actuellement. Les fondations modernes préparent et fiabilisent les besoins futurs de multi-carrières, multi-appareils, lifecycle, suppression/restauration, wallet serveur, ledger économique, idempotence, concurrence et paiements multi-provider. Elles ne constituent pas encore une intégration complète de ces fonctions avec les clients.

Ces fondations ne sont pas nécessaires pour publier Steam immédiatement. Leur activation est volontairement différée jusqu’à un besoin réel : croissance significative du nombre de joueurs, incidents multi-device, conflits de carrières, incohérences de tokens, besoin d’un wallet serveur central, de plusieurs providers de paiement ou d’un Cloud moderne plus strict. Aucun seuil numérique n’est fixé.

## OFFICIAL RECOVERY POINT

| Référence | Valeur officielle |
|---|---|
| Repository | `Tizona76/bm-online-api` |
| Local repo | `/Users/isidroetannebosch/Dev/bm-online-api_render` |
| Branch at freeze | `main` |
| CODE FREEZE TAG | `modern-backend-standby-p04-20260927` |
| CODE FREEZE COMMIT | `9d61eea81598e79196e7bfb1b95136a6da1e067e` |
| origin/main au contrôle du freeze code | `9d61eea81598e79196e7bfb1b95136a6da1e067e` |
| FULL RUNBOOK TAG | `modern-backend-standby-p04-runbook-20260927` |
| Commit du futur tag runbook | À connaître après le commit documentaire ; aucun SHA fixé ici |

Le premier tag fige exactement le code fonctionnel P02-B/P03/P04. Il ne contient pas ce document, créé après le freeze code. Il reste la référence technique immuable du code P04 : **NE JAMAIS déplacer ni remplacer `modern-backend-standby-p04-20260927`**, même si main évolue.

Le second tag sera créé après le commit de ce document, sur le commit documentaire. Une fois créé et disponible sur origin, `modern-backend-standby-p04-runbook-20260927` sera le point préféré pour une reprise humaine plug-and-play : il contiendra le code moderne ET ce runbook. Le présent ajustement ne crée aucun tag.

L’ancien commit pré-rebase `76bf59042d559de3fc46526fbc16122b463fa6ce` n’est plus la référence officielle ; il peut subsister sous `modern-backend-pre-rebase-p04-20260927`, uniquement comme archive.

Historique officiel, dans l’ordre chronologique :

| Commit | Sujet |
|---|---|
| `284fe68` | Stripe UX V1.1 payment success landing page |
| `c38bbb5` | Allow tournament funnel events |
| `12bd8e3` | feat(api): add dormant v2 capabilities endpoint |
| `638bcec` | feat(api): add modern careers schema |
| `7892bbe` | feat(api): add dormant modern career lifecycle |
| `9d61eea` | feat(api): add modern wallet and economic ledger |

`c38bbb5` a été conservé avant les quatre commits modernes lors du rebase final. Les anciens SHA locaux P02/P03/P04 ne remplacent pas ce checkpoint officiel.

## P02-B — Capability dormante

Route existante : `GET /v2/capabilities` dans `main.py`.

```json
{"contract_version": 1, "modern": false}
```

La capability reste volontairement `modern=false`, indépendamment des tests locaux des primitives. Ne pas la changer pendant le stand-by. Elle ne certifie pas à elle seule que les flags sont OFF : les deux contrôles sont distincts.

## P03-A — Schéma carrières

Migration : `migrations/001_modern_careers.sql`. Table : `public.modern_careers`.

La table porte l’identité `career_id` UUID, le propriétaire `user_id`, le contexte `profile_uuid`, la `generation`, l’état `ACTIVE` / `DELETED`, `created_at`, `updated_at` et `deleted_at`. Elle conserve la ligne/tombstone et la génération au fil du lifecycle ; ce n’est pas un journal exhaustif de toutes les transitions.

La migration est additive et explicite, jamais exécutée au startup ni via route HTTP. Elle ne modifie aucune table legacy et ne crée aucune FK legacy. Son contrôle de schéma refuse les objets incompatibles ; un rejeu n’est pas une réparation automatique.

## P03-B — Lifecycle moderne dormant

Fonctions préparées : create, read, logical delete, restore, ownership, génération, idempotence et concurrence, avec contrôle du contexte profil/career.

| Route | Contrat minimal |
|---|---|
| `POST /v2/profiles/{profile_uuid}/careers` | Body `career_id` UUID client ; 201 initial, 200 retry ACTIVE ; refuse une carrière DELETED |
| `GET /v2/profiles/{profile_uuid}/careers/{career_id}` | 200 avec état/génération, y compris DELETED ; aucune mutation |
| `DELETE /v2/profiles/{profile_uuid}/careers/{career_id}` | Body `expected_generation` ; ACTIVE N → DELETED N ; retry sans mutation |
| `POST /v2/profiles/{profile_uuid}/careers/{career_id}/restore` | Body `expected_generation` ; DELETED N → ACTIVE N+1 |

Le propriétaire vient du Bearer validé. Un autre propriétaire/profil ou une ligne inexistante reçoit 404 sans détail d’existence. Les UUID et générations sont validés. DELETE est logique, ne supprime jamais la ligne et ne change pas la génération. RESTORE incrémente celle-ci exactement une fois ; un retry avec l’ancienne génération retourne `409 GENERATION_MISMATCH` sans nouvelle mutation. GET permet de retrouver l’état courant. Aucun CREATE ne ressuscite implicitement une carrière.

Flag exact lu dans `main.py` : `ENABLE_MODERN_CAREER_LIFECYCLE`. **OFF par défaut ; seule la chaîne exacte `"true"` active les routes.** Absent, `"false"`, `"1"` ou toute autre valeur : OFF. La valeur est lue à l’import du processus. OFF produit `503 MODERN_DISABLED` avant accès DB. Ne jamais activer automatiquement.

## P04 — Wallet moderne au niveau compte

Migration : `migrations/002_modern_wallets.sql`. Table : `public.modern_wallets`.

- Propriétaire unique : `user_id`, pas le profil ou la carrière.
- `balance BIGINT`, `CHECK balance >= 0` ; aucun float.
- `version BIGINT` monotone, +1 par opération économique committée ; aucun incrément sur retry.
- Lecture d’un wallet absent : solde/version 0 sans écriture.
- Premier mouvement réussi : création du wallet dans la même transaction ; échec → rollback de cette création.
- Aucune reprise automatique du wallet legacy, aucun solde issu d’un blob Cloud.
- La suppression d’une carrière ne supprime pas le wallet compte.

## P04 — Ledger économique autoritaire

Migration : `migrations/003_modern_economic_operations.sql`. Table : `public.modern_economic_operations`. Module : `modern_economy.py`.

Primitives internes : `read_wallet`, `credit`, `debit`. Aucun endpoint économique, test-credit ou test-debit n’est livré ; les helpers ont été testés directement. Le code serveur appelant doit autoriser l’événement métier et fournir un utilisateur de confiance. Ce module n’est pas une API client d’attribution libre de tokens.

Le ledger append-only conserve `operation_id` unique, `request_hash`, montant signé, `balance_before`, `balance_after`, `wallet_version`, provenance `source` / `source_ref`, contexte carrière optionnel, `reverses_operation_id` et `committed_at`. Chaque ligne visible représente une opération COMMITTED ; le ledger ne sert pas de machine à états d’un paiement asynchrone.

Un trigger d’immutabilité refuse UPDATE, DELETE et TRUNCATE. Une correction est une nouvelle opération, jamais une réécriture. Crédit/débit écrivent ledger et wallet atomiquement dans une transaction. Le verrou d’opération protège les retries ; le verrou wallet sérialise les mutations du compte. Solde insuffisant ou dépassement des bornes : rejet sans mutation.

Invariant : pour chaque compte, solde = somme des montants committés ; before + amount = after ; la version décrit l’ordre des opérations du compte. Les timestamps transactionnels ne remplacent pas cet ordre.

Une nouvelle opération career-scoped vérifie propriétaire/profil, carrière ACTIVE et génération courante, sous verrou partagé conservé jusqu’au commit. Les opérations purement compte-level n’exigent pas de carrière. Aucun Cloud, Stripe, gameplay ou droit futur n’est appelé. Un futur droit économique devra être écrit dans la même transaction que le ledger et le wallet, pas après un helper déjà committé.

## Compensation — décision sémantique figée

Une compensation :

- agit sur le même compte `user_id` et porte le montant exactement opposé à l’opération référencée ;
- possède son propre `operation_id` et référence l’originale via `reverses_operation_id` ;
- possède son propre contexte et sa propre provenance `source` / `source_ref` ;
- n’a pas à recopier `profile_uuid`, `career_id` ou `career_generation` de l’originale ;
- laisse l’originale immuable et consultable via ce lien ;
- respecte les préconditions lifecycle si elle fournit un nouveau contexte carrière ;
- ne restaure aucun droit gameplay automatiquement.

Une reversal de reversal est autorisée : elle annule économiquement la compensation précédente. Le lien unique impose **une seule compensation directe maximum par opération**, pas une seule compensation pour toute la chaîne. Exemple : +25, puis −25 lié au crédit, puis +25 lié à la compensation. Chaque écriture conserve son identité et sa traçabilité.

Le support doit distinguer le contexte de l’originale et celui de la compensation ; ne pas interpréter un changement de contexte comme un transfert automatique de droit entre carrières.

## Idempotence technique P04

Pour une opération committée, `operation_id + request_hash` garantit :

- même operation_id + même contenu canonique : même reçu logique historique, aucun second effet ;
- même operation_id + contenu différent : `IDEMPOTENCY_CONFLICT`.

Le hash est calculé côté serveur sur un payload canonique, pas sur un timestamp ou un ordre arbitraire JSON. Un retry peut restituer un reçu ancien après d’autres mouvements ou un changement de lifecycle ; le solde courant se lit séparément. Un refus/rollback ne réserve pas l’ID : P04 n’est pas un journal persistant des tentatives échouées.

**source_ref n’est volontairement PAS unique.** Deux operation_id différents avec les mêmes user_id, source, source_ref et montant peuvent produire deux effets. Ce choix est volontaire : plusieurs effets distincts peuvent partager une référence. Ne pas confondre référence de provenance et déduplication métier.

## Idempotence métier future

L’unicité métier devra vivre dans les futures tables métier : paiement, mission, tournoi, saison, récompense ou achat de droit. Ne pas ajouter une contrainte globale `UNIQUE(user_id, source, source_ref)` au ledger qui empêcherait compensations, effets distincts ou workflows multi-étapes.

Approche recommandée : événement métier unique + bénéficiaire + type d’effet + operation_id stable + request_hash. Les contraintes UNIQUE métier doivent porter sur la véritable identité serveur de l’effet. Un operation_id déterministe peut être utile mais ne remplace ni la validation de la preuve, ni l’autorisation métier, ni la détermination du bénéficiaire.

## Flag économie — ne pas activer

Flag exact vérifié dans `modern_economy.py` : `ENABLE_MODERN_ECONOMY`.

**OFF par défaut ; seule la chaîne exacte `"true"` active les helpers.** Toute autre valeur reste OFF. La valeur est lue à l’import du module. OFF produit `ECONOMY_DISABLED` avant accès DB. Le flag économie est indépendant du flag lifecycle : lifecycle ON ne signifie pas économie ON.

Ne jamais activer le module simplement parce que les migrations sont installées. Même ON, ce module n’ajoute aucune route HTTP ni aucun branchement gameplay. La capability reste `modern=false`.

## NOT IMPLEMENTED / NOT ENABLED

Les éléments suivants ne sont pas fournis comme intégration active par ce checkpoint :

- branchement réel du jeu sur le lifecycle moderne ;
- vraie migration des carrières actuelles ;
- vrais gains gameplay branchés sur modern_economy ;
- vraies dépenses gameplay branchées sur modern_economy ;
- Cloud moderne ;
- migration automatique des sauvegardes ;
- migration automatique des anciens wallets ;
- import automatique des tokens legacy ;
- migration des anciens paiements ;
- paiements Steam modernes ;
- Apple IAP moderne ;
- Google Play Billing moderne ;
- Stripe moderne ;
- droits économiques gameplay modernes ;
- `modern=true` ;
- déploiement production du système moderne comme système principal actif.

La publication des commits sur GitHub ne valide aucune de ces activations.

## Legacy à préserver et validations acquises

Le legacy reste la référence de production jusqu’à décision explicite de migration. Ne pas modifier automatiquement `cloud_saves_v2`, `cloud_save`, `club_token_wallets`, `payments`, Stripe legacy, leaderboard, funnel ou les routes `/v1`.

Le golden legacy a servi d’oracle : P02-A, golden complet validé ; P03, legacy inchangé ; P04, **666 scénarios legacy identiques dans six configurations**. Les suites P04 M01–M10, W01–W12, L01–L12 et C01–C06 sont connues PASS. Les vérifications ciblées de compensation ont confirmé changement de contexte, reversal de reversal, une seule compensation concurrente et rejet d’un autre utilisateur.

Ces résultats sont des validations historiques, pas un nouveau test du déploiement final ou de la DB réelle. Ce freeze documentaire ne rejoue aucune suite. À la reprise, retrouver les preuves et le harness figé, vérifier leur correspondance avec le checkpoint puis rejouer sur la cible contrôlée ; ne pas transférer automatiquement une validation pré-rebase à tout état futur de main.

Pistes d’archives locales connues : `/private/tmp/basket-p02a-Wsa4KMJg` (golden/harness), `/private/tmp/basket-p03b-sOxcw4F4` (P03-B), `/private/tmp/basket-p04-diiuci77` (P04 et contrôle sémantique). **Ces dossiers /tmp sont volatils et ne font pas partie du checkpoint Git officiel.** Leur disponibilité n’est pas garantie. Si les preuves/harness manquent, les récupérer depuis une archive vérifiée avant de déclarer un nouveau gate PASS. Le tag restaure le code, pas la DB, les secrets ou les archives de tests.

## Steam et paiements — chantier différé

Stripe moderne reste en stand-by ; il n’est pas requis pour publier Steam. Le futur contrat devra être provider-agnostic :

```text
Steam / Apple / Google Play / Stripe-Web éventuel / autre provider
→ preuve d’achat provider
→ contrat serveur commun
→ effet économique unique
→ operation_id stable
→ P04 ledger
→ modern_wallets
```

Aucun flow provider, achat ou webhook supplémentaire n’est implémenté par cette procédure documentaire. La vérification provider, l’identité du bénéficiaire et l’unicité métier devront être conçues et validées avant toute intégration.

## Conditions de reprise

- [ ] besoin réel démontré
- [ ] audit production récent, autorisé dans le futur chantier
- [ ] état des clients actifs connu
- [ ] état DB réel connu
- [ ] backup DB vérifié
- [ ] restauration DB vérifiée
- [ ] rollback backend possible
- [ ] migrations revues
- [ ] backend compatible legacy + moderne testé
- [ ] golden legacy PASS
- [ ] mixed deployment PASS
- [ ] lifecycle ciblé PASS
- [ ] economy ciblée PASS
- [ ] flags toujours OFF avant gate final
- [ ] décision explicite d’activation

## Procédure de reprise — 1. Retrouver le checkpoint

Procédure future, à ne pas exécuter pendant le stand-by. Vérifier d’abord le dépôt, la branche, le worktree et l’index ; STOP si changements non compris. Pour une reprise normale, récupérer les tags puis vérifier d’abord le tag runbook et la présence du document :

```sh
cd /Users/isidroetannebosch/Dev/bm-online-api_render
git fetch origin --tags
git rev-list -n 1 modern-backend-standby-p04-runbook-20260927
git show modern-backend-standby-p04-runbook-20260927:docs/MODERN_BACKEND_STANDBY.md
git rev-list -n 1 modern-backend-standby-p04-20260927
git rev-parse HEAD
git status -sb
```

Le tag `modern-backend-standby-p04-runbook-20260927` contient le code moderne et le runbook une fois le commit documentaire et la création/publication du tag réalisés. Son SHA sera connu à ce moment-là seulement. S’il manque, STOP et faire vérifier cette publication ; ne pas prétendre que le tag code contient le document.

Le tag `modern-backend-standby-p04-20260927` reste la référence exacte du code P04 original et doit toujours pointer sur `9d61eea81598e79196e7bfb1b95136a6da1e067e`. Vérifier la provenance du tag runbook et que son ajout documentaire n’a pas modifié ce code.

Si main a avancé, **NE PAS reset automatiquement et NE PAS déplacer les tags** : créer une branche/worktree de reprise depuis le tag runbook ou comparer main avec lui. Ne pas remplacer les évolutions légitimes de main par l’ancien état. Ne jamais exécuter `git reset --hard` sans audit explicite. Si une référence ne correspond pas à son rôle, STOP et résoudre la provenance avant toute suite.

## Procédure de reprise — 2. Migrations explicites

Ordre obligatoire :

1. `migrations/001_modern_careers.sql`
2. `migrations/002_modern_wallets.sql`
3. `migrations/003_modern_economic_operations.sql`

Avant exécution : environnement contrôlé identifié, état DB connu, backup et restauration vérifiés, migrations revues, connexion cible vérifiée, flags OFF. Un opérateur exécute explicitement chaque fichier avec arrêt sur erreur, jamais au startup ni via endpoint. Vérifier après chaque migration les objets, contraintes et index attendus, le trigger ledger, ainsi que l’absence de changement legacy. **STOP au premier mismatch**, sans réparation improvisée ni poursuite de la migration suivante.

Les migrations ont leurs transactions et verrous advisory. Elles sont additives et refusent un schéma incompatible. Leur contrat de catalogue a été validé sur PostgreSQL 18.6 ; vérifier la compatibilité de la version cible lors de la reprise. L’installation du schéma n’autorise aucune activation.

## Procédure de reprise — 3. Compatibilité avant activation

Après autorisation du futur chantier, déployer d’abord un backend compatible legacy avec tables modernes présentes, **flags modernes OFF** et capability `modern=false`.

Valider health, auth, Cloud legacy, wallet legacy, leaderboard, funnel, paiements legacy et absence de changement de comportement utilisateur. Rejouer le golden, le mixed deployment et les tests ciblés sur l’artefact retenu. Un simple health vert n’est pas un gate économique ou lifecycle.

## Procédure de reprise — 4. Activation progressive

Ne jamais activer tout d’un coup. Ordre recommandé :

1. environnement de test ;
2. lifecycle moderne ciblé ;
3. lecture moderne ciblée ;
4. économie moderne sur flux contrôlé ;
5. clients compatibles ;
6. monitoring ;
7. extension progressive ;
8. Cloud moderne plus tard ;
9. paiements provider-specific plus tard.

Chaque étape exige son gate, ses tests, une possibilité de rollback et une décision explicite. Le monitoring doit être prêt avant tout flux utilisateur réel ; l’étape dédiée consolide son observation. Les fonctions encore absentes doivent être développées et validées séparément : cette liste n’implique pas qu’un flag suffise à les livrer. Aucun changement de capability sans décision cohérente avec les clients effectivement compatibles.

## Mixed deployment à supporter et tester

| État | Attendu |
|---|---|
| A. Ancien backend + nouvelles tables présentes | Legacy stable, tables/données conservées |
| B. Backend P03 + economy absente | Lifecycle selon son flag, legacy stable |
| C. Backend moderne + flags OFF | Moderne dormant, capability false, legacy stable |
| D. Backend moderne + lifecycle ON uniquement | Lifecycle ciblé fonctionnel ; économie OFF, aucune mutation économique |
| E. Backend moderne + economy ON en environnement contrôlé uniquement | Helpers économiques testés ; pas d’activation client implicite |
| F. Backend moderne + schéma incomplet | Fail closed attendu sur les accès modernes concernés, sans DDL implicite ; legacy stable |

Les flags OFF doivent rejeter avant DB. Flag ON, l’absence d’une table/colonne requise doit produire une erreur de readiness plutôt qu’une création automatique. Tester les schémas partiels réellement envisagés ; ne pas généraliser les tests historiques d’absence de table à tout type d’altération.

## Rollback conservateur

Un rollback backend peut revenir à une version compatible legacy conservatrice, après vérification de compatibilité. Les tables modernes peuvent rester présentes. Désactiver un flag n’efface pas les effets économiques déjà committés.

Ne pas DROP `modern_careers`, `modern_wallets` ou `modern_economic_operations`. Ne pas effacer le ledger, réécrire l’historique, fusionner automatiquement legacy et moderne ou recréditer un ancien wallet. Si des opérations modernes réelles existent, les conserver même flag OFF ; toute correction reste explicite, traçable et autorisée. Pas de down migration destructive.

## NEVER DO THIS

- Ne pas activer `modern=true` par défaut.
- Ne pas activer `ENABLE_MODERN_ECONOMY` sans gate.
- Ne pas activer le lifecycle moderne sans audit.
- Ne pas migrer automatiquement les balances legacy.
- Ne pas prendre les tokens du Cloud comme source de vérité.
- Ne pas doubler gains legacy + modernes.
- Ne pas utiliser source_ref comme seule sécurité métier.
- Ne pas force-push pour remettre le tag sur main ; ne pas déplacer le tag officiel.
- Ne pas DROP les tables modernes pendant un rollback.
- Ne pas lancer les migrations automatiquement au boot.

## Portée de ce freeze documentaire

Seul `docs/MODERN_BACKEND_STANDBY.md` est ajouté. Aucun code fonctionnel, migration existante, flag ou client n’est modifié. Aucun commit, push, tag, déploiement ou accès production n’est réalisé par ce chantier. Le fetch Git demandé vérifie les références GitHub ; il n’est pas un accès à la production applicative.

Prochaine action : **valider humainement ce document avant commit documentaire**. La validation du document ne vaut pas décision d’activation du backend moderne.
