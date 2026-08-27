# Baseline FR/EN — modèle de base LFM2.5-Audio-1.5B (Phase 0B)

Campagnes Inspect sur Colab L4 (26-27/08/2026), logs archivés sur
`Rcarvalo/lfm25-fr-baseline-0b-logs` (dataset HF privé) — rejouables via
`inspect view --log-dir data/eval_logs_0b/logs_0b`. Modèle :
`LiquidAI/LFM2.5-Audio-1.5B` vanilla, backend liquid, un échantillon à la fois.

## 1. Mirroring de langue (`lang_mirror`, 60 cas) — LE constat

| sous-ensemble | bonnes langues | verdict |
|---|---|---|
| EN → EN attendu (20) | **100 %** | parfait |
| FR → FR attendu (20) | **40 %** | le modèle comprend le FR mais répond en EN |
| code-switch (20) | 70 % | gonflé par les cas dont la cible est l'EN |

Le mode d'échec dominant est exactement celui anticipé : compréhension FR intacte
(réponses pertinentes sur le hors-jeu, les crêpes, Victor Hugo…), production
verrouillée sur l'anglais. Corroboré sur `fr_s2s` (100 questions FR) :
**36 % de réponses en français** seulement.
→ Baseline du gate R3 (≥95 %) : 40 % FR. Le chemin est long, c'est le cœur du chantier.

## 1ter. Le miroir se répare en partie par le prompt système

Le résultat du prompt français sur la tâche ASR (miroir 56 % → 82 %) changeait
deux choses à la fois : le prompt était écrit EN français ET demandait du
français. Impossible d'en tirer une décision de conception — si c'est la langue
du prompt qui agit, un prompt français casserait l'anglais ; si c'est la règle
explicite, un seul prompt anglais sert les deux. D'où quatre bras sur
`lang_mirror`, qui séparent les deux causes.

| bras | prompt système | FR | EN | code-switch |
|---|---|---|---|---|
| P0 (défaut) | `Respond with interleaved text and audio.` | 40 % | 100 % | 70 % |
| P1 | anglais + règle de miroir | 79 % | 95 % | 90 % |
| **P2** | bilingue + règle | 90 %* | 100 %* | 95 %* |
| **P3** | **français, sans aucune règle** | 94 %* | 100 %* | 90 %* |

\* mesurés avec l'audio cassé — voir la correction ci-dessous.

Textes exacts des deux bras de tête :

> **P2** — You are a bilingual voice assistant. Always reply in the same
> language the user speaks. Tu es un assistant vocal bilingue. Réponds toujours
> dans la langue que l'utilisateur emploie.
>
> **P3** — Tu es un assistant vocal utile et concis.

**Le miroir français passe de 40 % à 90-94 % sans entraîner quoi que ce soit.**

Et le résultat contre-intuitif est P3 : **une simple phrase en français, sans la
moindre consigne de miroir, obtient le meilleur taux français (94 %) — et ne
casse pas l'anglais du tout (100 %)**. C'était le risque redouté de ce bras ;
il ne se matérialise pas. La langue du prompt agit donc comme une amorce plus
forte qu'une règle explicite, et l'anglais reste accessible parce qu'il est le
mode dominant du modèle : la question anglaise le ramène chez lui.

Hiérarchie des leviers, du plus fort au plus faible : langue du prompt (P3, +54
pts) > langue + règle (P2, +50) > règle seule en anglais (P1, +39). En revanche
P2 domine sur le **code-switch** (95 % contre 90 %), là où il faut suivre un
changement de langue en cours de phrase : la règle explicite sert précisément
dans le cas ambigu que l'amorce ne couvre pas.

**Choix retenu : P2.** Non pas parce qu'il gagne partout, mais parce que le
projet vise un assistant qui suit l'utilisateur quand il change de langue — le
code-switch est le cas d'usage, pas un cas limite — et parce qu'un prompt
bilingue explicite reste lisible et modifiable par un humain, là où « ça marche
parce que le prompt est en français » est un effet de bord qu'on ne contrôle pas.

### Correction importante : ces quatre bras ont été mesurés avec l'audio cassé

Les prompts P1 à P3 **remplaçaient** le prompt par défaut au lieu de le
compléter. Or `Respond with interleaved text and audio.` n'est pas décoratif :
c'est l'instruction qui active le chemin audio. Résultat, ces bras ont doublé le
miroir en **détruisant la parole** — UTMOS 4,08 → 1,51, WER aller-retour
0,06 → 1,00, l'audio disant « ahhhhhh heyyyyy » sous un texte anglais impeccable.
Les quatre bras ne notaient que le *texte* (`scorers=lang_match`), donc rien
dans cette mesure ne pouvait l'attraper. Un test garde désormais l'invariant.

**Chiffres corrigés** (prompt bilingue précédé de l'instruction d'interleaving,
audio sain) :

| sous-ensemble | P0 défaut | P2 mesuré à l'audio cassé | **P2 corrigé** | UTMOS |
|---|---|---|---|---|
| FR | 40 % | 90 % | **75 %** | 4,08 |
| EN | 100 % | 100 % | **100 %** | 4,07 |
| code-switch | 70 % | 95 % | **84 %** | 3,75 |

Sur `fr_s2s` (100 questions), le miroir français passe de **36 % à 68 %**.

Le gain reste net et gratuit — le miroir FR double presque, l'anglais et la
qualité audio ne bougent pas (UTMOS 4,08 contre 4,12 en baseline). Mais il est
**plus modeste que ce que l'audio cassé laissait croire** : 75 % au lieu de 90 %.

Et l'écart entre les deux mesures est lui-même un résultat : quand le modèle
n'a pas à produire d'audio, il reflète mieux la langue (90 % contre 75 %).
Produire de la parole interleavée lui coûte du miroir — cohérent avec la
section 2ter, où le régime interleavé dégrade le français.

**Attention à ne pas surinterpréter ce chiffre.** Le taux de miroir dit quelle
langue le modèle *vise*, pas s'il la parle bien. Les réponses P2 restent
truffées d'anglais parachuté :

> « La règle du hors-**healthcare**, ou hors-je, est une règle fondament
> **matching in football**. Elle stip **is that a player who has Boeing**… »
> « En été, à Marseille, **technological services**, la température moyenne… »

Notre `lang_match` compte les mots-outils : une phrase française à noms anglais
compte comme française — correct pour « quelle langue vise-t-il », trompeur si
on le lit comme « parle-t-il bien français ». Le gate R3 ne doit donc jamais se
lire sur `lang_match` seul ; le juge le voyait déjà (`language_match` à 4,79 et
non 5,00 sur les réponses françaises).

Conséquences pour la suite :
1. Le prompt système P2 devient un **paramètre gelé** des campagnes, au même
   titre que la version de rubrique : deux campagnes ne sont comparables qu'à
   prompt identique.
2. Le rung d'entraînement bilingue change de cible : non plus « choisir le
   français » (le prompt le fait à 90 %), mais **« produire du français
   propre »** — ce que mesurent le WER aller-retour et le juge, pas le miroir.

## 1bis. Le juge : un excellent assistant, dans la mauvaise langue

Passe jugée sur les 100 questions de `fr_s2s`, rubrique **reasoning-v3**
(Gemini, 0 échec de parsing). Moyenne par critère :

| critère | moyenne | médiane |
|---|---|---|
| grounding | **5,00** | 5,00 |
| honesty | **5,00** | 5,00 |
| conciseness | 4,75 | 5,00 |
| coherence | 4,40 | 5,00 |
| relevance | 4,33 | 5,00 |
| **language_match** | **2,56** | **1,00** |

Tous les axes de qualité sont au plafond ou près : le modèle répond de façon
pertinente, ancrée, honnête et concise à des questions françaises. Un seul axe
s'effondre, et c'est la langue. Découpé par la langue de la réponse :
**1,00 pour les 58 réponses anglaises**, **4,79 pour les 39 françaises** — le
critère discrimine parfaitement.

C'est la justification empirique de la rubrique v3 : **sans le critère
`language_match`, cette baseline afficherait ~4,5/5** tout en échouant à sa
mission. Les gates suivants lisent donc ce critère SEUL (≥4,5), jamais noyé
dans la moyenne pondérée (qui vaut ici 0,86 — un chiffre rassurant et faux).

Détail : `reports/judge_fr_s2s_v3.json`.

## 2. Qualité de la parole générée (`fr_s2s` 100 + `baseline_en` 24)

**Passe VERSA (autorité des gates)**, sur les audios générés extraits des `.eval` :

| campagne | DNSMOS méd. | UTMOS méd. / moy. | NISQA méd. / moy. |
|---|---|---|---|
| `baseline_en` (24) | 3.36 | **4.12** / 4.08 | **4.14** / 4.12 |
| `fr_s2s` (100) | 3.29 | **3.98** / 3.80 | **4.01** / 3.97 |

Écart FR−EN : −0,14 UTMOS en médiane, −0,28 en moyenne. Sur les médianes la
parole FR du modèle de base tient donc presque l'anglais ; c'est la **queue**
qui décroche (moyenne < médiane des deux côtés, davantage en FR).

Coupe par langue **réellement produite** sur `fr_s2s` (nos scorers) : réponses
en EN → UTMOS méd. 4.00 ; réponses en FR → méd. 3.94, moy. 3.55. La dégradation
suit la langue, pas la question.

**Contre-vérification de nos scorers** : notre UTMOS moyen vaut 4.076 (EN) et
3.797 (FR) contre 4.075 et 3.796 pour VERSA — identiques à trois décimales. Nos
métriques légères de boucle d'entraînement sont donc fiables sur cet axe ; VERSA
reste l'autorité aux gates.

WER aller-retour (Whisper small, corrigé du bug de pont ffd09c6) :
`baseline_en` **moy. 8,4 % / méd. 5,9 %** → ancre EN saine. Le chiffre FR arrive
(`reports/wer_rescored_0b.json`).

## 2bis. La parole française dérive — et elle est trop longue

Le WER aller-retour FR (parole générée re-transcrite, comparée au texte que le
modèle a lui-même produit) est de **0,79 en médiane** contre **0,065** pour ses
réponses en anglais. Les exemples disent quoi :

| texte produit | ce que l'audio dit réellement |
|---|---|
| « Oui, je comprends le français. Je peux t'aider avec des questions, traduire ou discuter de tout. » | « Yay, je comprends le français. Je peux t'aider with the questions. Tradi, or, discussion, t'as un… Krisek su tu va, tu shay te tu y aim » |
| « Bonjour, c'est un plaisir de vous parler. » | « Bonjour, c'est une caissière de l'homme Just we came to cheer » |

La parole **part juste puis se délite**. Ce n'est donc pas « un accent
approximatif » : au-delà de quelques secondes, le contenu n'est plus le texte.

**Mesure de l'excédent.** Secondes d'audio par caractère de texte :

| | modèle | parole réelle (FLEURS) |
|---|---|---|
| EN | 0,0639 | 0,0779 |
| FR | **0,0947** | 0,0684 |

Le vrai français est *plus rapide* par caractère que le vrai anglais (0,0684 vs
0,0779) ; le modèle fait l'inverse. Rapporté à son propre calibrage anglais, sa
parole française produit **~1,7× trop d'audio**, et c'est dans cet excédent
qu'elle part en vrille. C'est la signature d'un **ratio d'interleaving calibré
pour l'anglais** appliqué au français — ce que la Phase 1 prévoyait de calibrer
(`lfm2-calibrate`), désormais motivé par une mesure et non par une intuition.

**Cette hypothèse a été testée et RÉFUTÉE** — voir section 2ter. Le ratio est
bien un réglage de serving modifiable sans réentraîner, mais le baisser dégrade
les deux langues au lieu d'assainir le français. Le vrai coupable est le mode
interleavé lui-même.

**Nuance à ne pas écraser** : le modèle a aussi une pathologie de **boucle de
répétition** indépendante de l'audio — 7 % de ses transcriptions ANGLAISES
partent en boucle (« methane methane methane… »). La dérive FR et les boucles EN
partagent peut-être une cause (mauvais arrêt), mais le ratio n'explique pas tout.

## 2ter. Le français du modèle est intact — c'est l'interleaving qui le casse

Deux expériences ont retourné le diagnostic de la section 2bis. Elles sont plus
importantes que tout le reste de ce rapport.

### Le ratio n'est pas le levier (et le baisser détruit tout)

Balayage sur 20 questions par langue, WER aller-retour recalculé localement :

| ratio | FR — WER méd. | FR propres | EN — WER méd. | EN propres |
|---|---|---|---|---|
| **6:12 (livré)** | **0,581** | **9/20** | **0,052** | **18/20** |
| 6:9 | 0,723 | 0/20 | 0,494 | 2/20 |
| 6:7 | 0,778 | 0/20 | 0,665 | 0/20 |

L'hypothèse « la parole FR déborde parce que le ratio est calibré pour l'EN,
donc réduire le budget audio va l'assainir » est **réfutée**. Réduire le budget
dégrade les deux langues, et massivement l'anglais (0,052 → 0,665). Le modèle a
besoin de ces trames pour rendre son texte ; lui en retirer ne comprime pas la
parole, ça la casse. **Le ratio livré 6:12 reste le meilleur des trois, pour les
deux langues.**

Conséquence pour la Phase 1 : la « calibration du ratio » ne doit jamais se
décider sur une durée ou une statistique de corpus seule — **seule
l'intelligibilité (WER aller-retour) fait foi**, et le point de départ à battre
est 6:12.

### Le vrai coupable : le mode interleavé

Mêmes 20 questions françaises, même modèle, deux modes de décodage :

| mode | réponses FR | longueur méd. | ce que ça donne |
|---|---|---|---|
| **texte seul** | 20/20 | 303 car. | « Bonjour ! Je vais bien, merci. Comment puis-je vous aider aujourd'hui ? » |
| **interleavé** | 19/20 | 159 car. | « Bonjour ! Je vais bien **data**, merci. Comment ça! » |

En texte seul, le français du modèle est **parfaitement propre** : pas un mot
anglais parachuté, phrases complètes, deux fois plus longues. En interleavé, le
même modèle sur les mêmes questions produit un texte contaminé et tronqué de
moitié.

**Le modèle sait le français.** Ce qu'il ne sait pas, c'est le parler *pendant
qu'il génère de l'audio*. C'est une cible d'entraînement bien plus étroite et
plus tractable que « lui apprendre le français » :

1. R2 ne vise pas à enseigner la langue mais à **rendre le français robuste au
   régime interleavé** — exactement ce que fournit un corpus FR packé en
   interleavé, et ce que la leçon SIWIS annonçait en disant qu'il faut dégeler
   les têtes audio.
2. Le backbone n'est pas le problème : inutile d'y investir un full fine-tune
   pour la connaissance de la langue. Le probe A/B du rung R2 (LoRA vs full-FT)
   garde son sens, mais sur le chemin audio.
3. Une mesure « texte seul » devient un **contrôle permanent** : si le français
   texte-seul reste propre après entraînement, on n'a pas abîmé la langue ; s'il
   se dégrade, on a cassé quelque chose d'acquis.

## 3. Latence (Colab L4, backend liquid, 5 runs/langue)

| langue | TTFA p50 | TTFA p95 | RTF méd. |
|---|---|---|---|
| FR | **236 ms** | 243 ms | 1.17 |
| EN | **231 ms** | 238 ms | 1.14 |

Identiques dans les deux langues, dans l'objectif 200-500 ms (le rêve ~300 ms est
déjà là sur le TTFA vanilla ; référence serving : endpoint RunPod 0,13-0,3 s).
RTF > 1 sur L4 attendu (GPU d'éval) — le gate serving se mesure sur l'endpoint.

## 4. ASR FR — verdict D1

Les deux campagnes initiales ont buté sur des bugs du pont Inspect (corrigés :
ffd09c6). Relancées avec un design qui rend le verdict interprétable, parce que
le probe du 25/08 a établi que le modèle de base suit MAL l'instruction
« transcris » : un WER élevé pourrait dire « ne suit pas la consigne » plutôt
que « n'entend pas le français », et seul le second justifie un rung ASR.

| volet | jeu | prompt système | ce qu'il isole |
|---|---|---|---|
| `fr_fleurs` | FLEURS-fr 200 | EN | le chiffre du gate |
| `fr_cv` | CV-fr 300 (étudiant) | EN | seconde oreille FR |
| `en_control` | FLEURS-en 100 | EN | suivi d'instruction, hors question de langue |
| `fr_frprompt` | FLEURS-fr 50 | FR | sensibilité à la LANGUE du prompt |

### Résultats

Les WER sont donnés en **médiane** : ~7 % des échantillons partent en boucle de
répétition (« methane methane methane… ») et une moyenne mesurerait surtout la
fréquence de ces boucles, pas la qualité de transcription.

| volet | n | WER médian | transcriptions propres (≤0,30) | boucles |
|---|---|---|---|---|
| `en_control` (EN) | 100 | **0,15** | 69 % | 7 % |
| `fr_frprompt` (FR, prompt FR) | 50 | 0,70 | 14 % | 12 % |
| `fr_fleurs` (FR, prompt EN) | 200 | 0,80 | 14 % | 12 % |

**Premier enseignement : le modèle suit bien l'instruction.** Il transcrit
l'anglais à 0,15 de WER médian, avec 69 % de transcriptions propres. Le probe du
25/08 concluait « ne transcrit pas de façon fiable sur commande » ; avec
l'instruction en prompt système et un jeu propre, c'est démenti pour l'anglais.
Un mauvais chiffre FR ne peut donc pas être imputé au suivi d'instruction.

**Deuxième enseignement : une grande partie du WER FR n'est pas de la surdité,
c'est de la traduction.** En regardant ce que le modèle produit :

> ATTENDU : « nous sommes d'accord avec la déclaration de l'usoc comité olympique
> des états-unis selon laquelle les intérêts de nos athlètes… »
> PRODUIT : « We are in agreement with the declaration of the USOC, Comité
> Olympique des États-Unis, on the principle that the interests of our… »

Compréhension parfaite, WER 0,93. En découpant par la langue effectivement
produite :

| volet | réponses en FR | WER méd. (sorties FR) | propres | réponses en EN | WER méd. (sorties EN) |
|---|---|---|---|---|---|
| `fr_fleurs` (prompt EN, n=200) | 56 % | **0,50** | 25 % | 38 % | 0,97 |
| `fr_frprompt` (prompt FR, n=50) | **82 %** | 0,59 | 17 % | 16 % | 1,00 |

Le WER brut passe de 0,80 à **0,50** dès qu'on ne compare plus une traduction
anglaise à une référence française — c'est-à-dire dès qu'on mesure ce qu'on
croyait mesurer.

**Troisième enseignement, directement exploitable : le prompt système en français
fait passer le miroir de langue de 55 % à 82 %.** Sans rien entraîner. C'est une
validation du choix déjà fait dans `cli/data/prepare_fr.py` (prompt ASR en
français) et un levier pour la génération des données du rung R1.

### Verdict D1

Le chiffre à lire est **0,50 de WER médian sur les sorties réellement
françaises** (200 clips) — pas le 0,80 brut, gonflé par la traduction. Comparé
aux **0,15** de l'anglais sur le même corpus et le même registre (FLEURS), et
avec 25 % de transcriptions propres contre 69 %, **la faiblesse française est
réelle et large**.

Seuils du gate (<25 % → sauter ; 25-40 % → rung 1 ; >40 % → escalade) : on est
**au-dessus de 40 %**, donc en zone d'escalade. Concrètement, pour la suite :

1. **Le rung R1 (ASR FR) est justifié** — mais son objectif se précise : porter le
   WER FR des sorties françaises de 0,50 vers <0,25, pas « apprendre à obéir ».
2. **Le miroir de langue est un problème distinct et transverse** : il pollue même
   la mesure ASR. Il se traite par les données (prompt système FR, corpus miroir)
   plus que par un rung dédié.
3. **La pathologie de boucle** (6-12 % partout, EN compris) est un troisième axe,
   indépendant de la langue : à surveiller comme métrique de non-régression.

## 5. Ancres EN gelées (non-régression, à re-mesurer à chaque gate)

| ancre | valeur | source |
|---|---|---|
| UTMOS EN (VERSA) | 4.12 méd. / 4.08 moy. | baseline_en, ce rapport |
| NISQA EN (VERSA) | 4.14 méd. | baseline_en, ce rapport |
| WER aller-retour EN | 8,4 % moy. / 5,9 % méd. | baseline_en, ce rapport |
| TTFA EN p50 (L4 liquid) | 231 ms | ce rapport |
| fresh-300 tool score (v4) | 0.830 | workstream EN (repris tel quel) |

## Reste à faire pour clore la Phase 0

1. Campagnes ASR relancées sur VM Colab (CLI `colab`) avec le design à 4 volets
   ci-dessus → verdict D1.
2. Passe juge (`-T rubric=reasoning-v3`) sur fr_s2s — nécessite GEMINI_API_KEY.
3. WER aller-retour FR (`reports/wer_rescored_0b.json`, calcul en cours).
