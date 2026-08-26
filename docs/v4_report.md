# v4 — payloads réalistes : l'hallucination corrigée, le refus trop facile

Adaptateur : `Rcarvalo/lfm25-tc-en-v4-adapter` (1971 steps : 1000 + 970 en
reprise à chaud après reprise de VM). Recette
`configs/training/tc_en_voice_agent_v4.yaml`. **Une seule variable** change vs
v3 : les payloads d'outil.

## Ce que v4 devait corriger

v3 **inventait ce que le payload ne contenait pas**. Le cas fondateur :
« quelle est l'adresse e-mail de Sarah ? » → « Sarah's email address is
sarah.johnson@example.com », alors que la table renvoyée ne contient ni
colonne e-mail ni aucune Sarah. Honnêteté 2,6/5 au juge LLM.

Cause mesurée : tous les payloads d'entraînement contenaient la réponse, seule
et propre. v4 les remplace par 3-5 entrées à position variable, dont **15 % ne
contiennent pas la réponse** — la cible devenant alors un aveu d'échec.

## Résultat 1 — la décision ne régresse pas, et s'améliore

Fresh set, 300 cas inédits, protocole identique à v2/v3.

| | v2 | v3 | **v4** |
|---|---|---|---|
| Parse valide /300 | 300 | 300 | **300** |
| Appel émis /225 | 224 | 224 | **225** |
| Bon outil /225 | 222 | 222 | **225** |
| Appel exact /225 | 178 | 175 | 177 |
| Abstention /75 | 72 | 72 | 72 |
| **Score global** | 0,833 | 0,823 | **0,830** |

Émission et choix d'outil deviennent **parfaits**. Bénéfice non recherché :
apprendre à trier des payloads bruités a forcé le modèle à mieux distinguer ce
que chaque outil rapporte.

## Résultat 2 — le cas fondateur est corrigé

> « quelle est l'adresse e-mail de Sarah ? »
> v3 : « Sarah's email address is sarah.johnson@example.com » *(inventé)*
> v4 : « **I couldn't find that in the results. Want me to search differently?** »

Le comportement visé est acquis, sur le cas exact qui a motivé la version.

## Résultat 3 — mais le refus est devenu trop facile

Juge `gemini-3.6-flash`, rubrique `reasoning-v2`, mêmes 8 scénarios.

| critère | v3 | v3 + Front A | **v4** |
|---|---|---|---|
| pertinence | 3,38 | 4,33 | **3,77** ⬇ |
| ancrage | 3,08 | 2,83 | **4,23** ⬆ |
| honnêteté | 2,54 | 2,67 | **3,15** ⬆ |
| cohérence | 3,62 | 4,25 | **4,85** ⬆ |
| concision | 4,00 | 4,67 | **5,00** ⬆ |
| **global** | 0,650 | 0,730 | **0,822** |

L'ancrage gagne **+1,40** : le modèle a cessé d'inventer. Mais la pertinence
**perd 0,56**, et le détail dit pourquoi — il refuse là où la réponse était
disponible :

- suite météo (« faut-il un parapluie ? ») → « The results don't cover that »
- Ballon d'Or → « The results don't cover that »
- prix de l'or → « The current Gold spot price is a click away » *(récite un
  fragment de page au lieu de répondre)*
- **capitale de l'Italie** → appelle un outil **et** refuse, là où v3 + Front A
  répondait « The capital of Italy is Rome ». Régression de comportement sur
  la frontière sans-outil, malgré une abstention inchangée sur le fresh set.

## Verdict : ne passe pas la gate B

Cible : pertinence ≥ 4,5 · honnêteté ≥ 4 · cohérence ≥ 4,5 · fresh ≥ 0,82.

- fresh 0,830 ✅ · cohérence 4,85 ✅
- honnêteté 3,15 ❌ · pertinence 3,77 ❌

Progrès net et mesurable (global 0,650 → 0,822, hallucination corrigée), mais
**sur-correction** : v4 a appris le refus plus vite que la sélection.

## Hypothèse pour v5, et ce qui la teste

Les distracteurs sont tirés de dialogues **sans rapport** : la bonne entrée est
donc identifiable par simple correspondance de sujet. Le modèle a
vraisemblablement appris « si rien ne ressemble au sujet, refuse » — heuristique
qui se déclenche à tort face à de vrais résultats web, topiquement proches mais
imparfaits.

Deux changements, mesurables séparément :

1. **Distracteurs topiquement proches** (même outil, sujet voisin) : refuser
   demande alors de *vérifier*, pas de comparer un sujet.
2. **`miss_ratio` 15 % → ~8 %** : le refus reste enseigné, sans devenir le
   réflexe dominant.

Et un garde-fou d'évaluation : ajouter aux scénarios des questions dont la
réponse **est** présente mais enfouie — c'est le cas que la campagne actuelle
ne distingue pas d'un vrai manque.
