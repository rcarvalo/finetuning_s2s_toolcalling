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

## 3. Latence (Colab L4, backend liquid, 5 runs/langue)

| langue | TTFA p50 | TTFA p95 | RTF méd. |
|---|---|---|---|
| FR | **236 ms** | 243 ms | 1.17 |
| EN | **231 ms** | 238 ms | 1.14 |

Identiques dans les deux langues, dans l'objectif 200-500 ms (le rêve ~300 ms est
déjà là sur le TTFA vanilla ; référence serving : endpoint RunPod 0,13-0,3 s).
RTF > 1 sur L4 attendu (GPU d'éval) — le gate serving se mesure sur l'endpoint.

## 4. ASR FR (gate D1) — EN COURS, design à 4 volets

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

Lecture prévue : si `en_control` échoue autant que `fr_fleurs`, le problème est
le suivi d'instruction et le rung ASR est le mauvais remède (la piste devient
(a) inclure de l'ASR dans le mélange d'entraînement). Si l'EN passe et le FR
non, la faiblesse FR est réelle. Gate : WER FLEURS-fr **<25 % → sauter le rung
ASR ; 25-40 % → rung 1 ; >40 % → escalade**.

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
