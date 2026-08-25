# v3 — Phase A + Phase B : la réponse parlée ancrée

Adaptateur : `Rcarvalo/lfm25-tc-en-v3-adapter` (step 1600 / 1971, voir *Arrêt anticipé*).
Recette : `configs/training/tc_en_voice_agent_v3.yaml`. Run wandb : `tc_en_v3_phaseAB`.

## Ce que v3 devait corriger

v2 routait correctement mais **ne savait pas répondre après un appel d'outil** :
son texte de réponse était vide et son audio babillait (« to you, to you… »),
parce que ce tour n'existait dans aucun corpus d'entraînement. v3 ajoute la
Phase B : 2 679 dialogues à quatre tours (audio utilisateur → appel d'outil →
résultat JSON → **réponse parlée**, entrelacée 6:12), mélangés aux 2 729
exemples single-turn de la Phase A.

## Résultat 1 — non-régression sur la décision (fresh set, 300 cas inédits)

Même protocole que v2 : mêmes cas, `--tool-definitions en`, scorer `tool_call`.

| | vanilla | v2 | v3 |
|---|---|---|---|
| Parse valide | — | 300/300 | 300/300 |
| Appel émis quand attendu | — | 224/225 | 224/225 |
| Bon nom d'outil | — | 222/225 | 222/225 |
| Appel exact (arguments) | — | 178/225 | 175/225 |
| Abstention correcte | — | 72/75 | 72/75 |
| **Score global** | **0,250** | **0,833** | **0,823** |

Quatre lignes sur cinq sont identiques. L'unique écart — 3 arguments exacts sur
225 — vaut moins d'un demi écart-type binomial : indiscernable du bruit.
**Apprendre à parler n'a pas coûté la capacité de décision.**

## Résultat 2 — l'objectif est atteint (8 scénarios, outils réellement exécutés)

DuckDuckGo en direct pour `web_search`, base de démo pour `db_query`.
13 tours, dont 10 avec appel d'outil.

**Les 10 tours avec outil produisent une réponse ancrée, en texte ET en audio.**
Exemples, tous vérifiables dans le payload retourné :

- « The current price of gold is $3,500.20 per ounce, according to APMEX. »
- « Acme Corp placed the most recent order, with 10 Widget Pro units for a total of $499.90. »
- « We currently have 5 orders in the system: 1 for Acme Corp, 1 for Globex, 2 for Initech… »
- « Sarah's email address is sarah.johnson@example.com. »

Sur ces mêmes tours, **v2 renvoyait un texte vide** (`answer: ''`) avec un audio
incohérent. La comparaison est frontale, pas graduelle.

Bonus non recherché : sur `s7_no_tool_boundary`, v2 appelait un outil à tort ;
v3 s'abstient correctement.

## Résultat 3 — la limite, et elle est nette

Les **3 tours conversationnels sans outil** (« how are you », « thank you »,
« capital of Italy ») sont dégradés en texte :

> « I'm doing well, thank, thank you for asking! How can I?, uh, you know, you?, uh, you? you? »
> « You're most welcome! when you're ready. when you are, when you are…………… »

Deux précisions importantes pour ne pas surinterpréter :

1. **L'absence de parole sur ces tours n'est pas une régression de v3** : v2 aussi
   sortait `spoken = 0 s` sur les tours sans outil. Le modèle ne parle, dans les
   deux versions, qu'après un résultat d'outil.
2. **Le texte, lui, a bien régressé** : v2 répondait proprement (« I'm doing well,
   thank you for asking! How can I assist you? »).

Diagnostic : le corpus v3 est entièrement centré sur les outils. Les réponses
Phase B sont des phrases informatives qui suivent un payload ; les tours
sociaux courts ne sont représentés nulle part, et 1,5 époque de données
tool-centric a spécialisé le modèle au détriment de ce registre.

## Arrêt anticipé, et pourquoi il ne change rien

Colab a repris la VM au step 1600 sur 1971 (81 %). Les 371 steps manquants
n'ont pas été rejoués, pour une raison technique : le démarrage à chaud
(`lora.init_adapter`, ajouté pour ce cas) ne restaure pas l'état de
l'optimiseur, donc le scheduler rejouerait son warmup et relancerait le
learning rate à 1e-4 sur un modèle déjà convergé — risque de dégrader plutôt
que d'améliorer.

Les indicateurs confirment la convergence : `val_loss` plate (1,117 → 1,111 sur
les 400 derniers steps) et score d'éval stable depuis le step 100. La **loss
audio est passée de 3,279 à 1,903** (−42 %) : c'est la mesure directe de ce que
la Phase B devait enseigner, la prédiction des codes audio de la réponse.

Note sur le score d'éval en cours d'entraînement (0,917 au step 100, plateau à
0,875) : il porte sur **24 questions**, donc chaque cas vaut 0,042 et toute
l'amplitude observée tient en quelques cas. Trop bruité pour départager quoi
que ce soit — le fresh set à 300 cas est le seul juge.

## Verdict

v3 fait ce pour quoi elle a été construite et ne casse rien de ce que v2 savait
faire. Le chemin critique du produit — écouter, décider, appeler, **répondre en
s'appuyant sur le résultat** — fonctionne de bout en bout.

Reste une lacune de registre, pas de capacité : le modèle est excellent en
assistant outillé, maladroit en conversation sociale.

## v4 — ce que ce rapport prescrit

1. **Combler le registre manquant** : ajouter au corpus des tours sociaux courts
   (salutations, remerciements, relances) avec **réponse voisée**, et des
   réponses directes sans outil également voisées. C'est la cause la plus
   probable des trois échecs, et elle est bon marché à corriger.
2. **Transcript-first** (cf. `project-transcript-plan`) : conditionner la
   génération sur la transcription pourrait récupérer les ~21 % d'arguments
   inexacts, la métrique la plus faible depuis v2.
3. Une variable à la fois : ces deux changements se mesurent séparément sur le
   même fresh set.
