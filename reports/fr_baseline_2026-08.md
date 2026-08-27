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

**Conséquence pratique** : le ratio vit dans le `config.json` du checkpoint, donc
il est testable **sans réentraîner**. Une expérience à quelques euros (réécrire
le ratio, rejouer 30 questions FR, mesurer le WER aller-retour) doit précéder le
rung R2 : si l'intelligibilité FR remonte nettement, une partie du problème est
un réglage de serving, pas un manque d'entraînement.

**Nuance à ne pas écraser** : le modèle a aussi une pathologie de **boucle de
répétition** indépendante de l'audio — 7 % de ses transcriptions ANGLAISES
partent en boucle (« methane methane methane… »). La dérive FR et les boucles EN
partagent peut-être une cause (mauvais arrêt), mais le ratio n'explique pas tout.

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
