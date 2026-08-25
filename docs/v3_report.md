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

## Résultat 3 — deux défauts, tous deux diagnostiqués

### 3a. La dégénérescence hors-outil est un bug d'orchestrateur, pas du modèle

Sur les tours sans appel d'outil, la réponse part en boucle jusqu'à la limite
de tokens (884 caractères sur « How are you? », 2 094 sur « capitale de
l'Italie »), avec un motif de disfluences — « uh, you know, you? » — qui est la
signature de nos tours **utilisateur**.

Cause : `agent.py` décide le mode par `speaking = tool_ran or not hybrid`. Le
tour 0 est **toujours** généré en séquentiel texte-seul, délibérément, pour que
l'entrelacement ne déchiquette pas le span `<|tool_call_start|>…` (leçon v1).
Or v3 a appris ses réponses en **entrelacé** (Phase B, 6:12). Sans appel
d'outil, elle n'atteint jamais le mode parole et doit répondre dans un mode que
sa distribution ne connaît plus.

Vérifié expérimentalement (mêmes audios, `hybrid` basculé) :

| | `hybrid=True` | `hybrid=False` |
|---|---|---|
| « How are you? » | 884 car. de boucle, 0 s | « I'm doing well, thank you for asking! How can I assist you? » — 3,7 s |
| « Capitale de l'Italie » | 2 094 car. de boucle, 0 s | « The capital of Italy is Rome. » — 1,7 s |
| Contrôle avec outil | e-mail correct, 3,7 s | **texte vide, appel cassé** |

v3 sait donc parfaitement tenir une conversation ; l'orchestrateur l'en empêche.
Mais la dernière ligne interdit de simplement basculer en entrelacé.
**Correctif (sans réentraînement)** : décider en séquentiel, puis, si aucun
appel d'outil n'est émis, régénérer la réponse en mode parole.

### 3b. Ancrage ≠ pertinence : nos payloads d'entraînement ne ressemblent pas au réel

Question : « qui a gagné le dernier Ballon d'Or ? »
Réponse : « In 2022, France Football modified the rules for the Ballon d'Or… »

Le modèle **récite le snippet** au lieu de répondre. Il est correctement ancré
dans le payload — mais le payload ne contient pas la réponse, et il ne le
détecte pas.

Mesure sur `data/phase_b_train.jsonl` : les payloads d'outil de la Phase B sont
**toujours une réponse propre et unique**.

- `{"results": "The Antarctic Polar Desert is considered the largest desert in the world."}`
- quand c'est une liste : **418 cas sur 473 ont exactement un élément** ;
- clés hétérogènes (`results`, `result`, `snippet`, `answer`) ;
- **aucun payload ne contient de bruit, ni ne manque la réponse.**

DuckDuckGo réel renvoie 5 résultats hétérogènes `{title, url, snippet}`, parfois
sans la réponse. Le modèle a appris « le payload contient la réponse, reformule-la » —
il n'a jamais appris à **sélectionner** parmi plusieurs résultats, ni à dire
**« je n'ai pas trouvé »**. Face au réel, il fait ce qu'on lui a enseigné.

C'est la limite que la validation « 10/10 tours ancrés » masquait : je vérifiais
que la réponse venait du payload, pas qu'elle répondait à la question.

## Détail des tours conversationnels

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

## Ce que ce rapport prescrit

**Priorité 1 — orchestrateur, aucun réentraînement** (§3a). Basculer en mode
parole quand le tour 0 n'émet pas d'appel d'outil. Corrige la dégénérescence
conversationnelle immédiatement, à coût nul.

**Priorité 2 — réalisme des payloads, v4** (§3b). Reconstruire les tours `tool`
de la Phase B à l'image du réel : plusieurs résultats hétérogènes
`{title, url, snippet}`, réponse parfois en position 3, **et une fraction de cas
où la réponse est absente** — la bonne réponse étant alors « je n'ai pas
trouvé ». C'est le défaut le plus visible à l'usage.

**Priorité 3 — transcript-first** (cf. `project-transcript-plan`) : conditionner
la génération sur la transcription pourrait récupérer les ~21 % d'arguments
inexacts, métrique la plus faible depuis v2.

Une variable à la fois, mesurée sur le même fresh set — et désormais aussi sur
un jeu de scénarios enrichi de questions dont la réponse **n'est pas** dans les
résultats, puisque c'est le cas que la validation actuelle ne couvrait pas.
