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

| mesure | réponses EN | réponses FR |
|---|---|---|
| UTMOS (nos scorers, sur audio généré) | méd. 4.00 / moy. 3.98 | méd. 3.94 / moy. 3.55 |

La médiane FR tient presque l'EN, mais la moyenne décroche : ~queue de réponses FR
à l'audio dégradé. L'ancre EN gelée : **UTMOS EN = 4.08** (baseline_en).
Passe VERSA d'autorité (DNSMOS/UTMOS/NISQA) : voir `reports/versa_0b.json`.
WER aller-retour corrigé : voir `reports/wer_rescored_0b.json` (le premier run
comparait au repr de l'objet Target — bug du pont, corrigé en ffd09c6).

## 3. Latence (Colab L4, backend liquid, 5 runs/langue)

| langue | TTFA p50 | TTFA p95 | RTF méd. |
|---|---|---|---|
| FR | **236 ms** | 243 ms | 1.17 |
| EN | **231 ms** | 238 ms | 1.14 |

Identiques dans les deux langues, dans l'objectif 200-500 ms (le rêve ~300 ms est
déjà là sur le TTFA vanilla ; référence serving : endpoint RunPod 0,13-0,3 s).
RTF > 1 sur L4 attendu (GPU d'éval) — le gate serving se mesure sur l'endpoint.

## 4. ASR FR (gate D1) — EN ATTENTE

Les deux campagnes (`fleurs_fr_asr` 200, `cv_fr_asr` 300) ont buté sur des bugs
du pont Inspect (corrigés : ffd09c6) et doivent être relancées sur Colab.
Rappel du gate : WER FLEURS-fr **<25 % → sauter le rung ASR ; 25-40 % → rung 1 ;
>40 % → escalade**. NB : le probe du 25/08 a montré que le modèle de base suit
mal l'instruction « transcris » — un WER élevé dira « ne suit pas la consigne »
autant que « n'entend pas le FR » ; le verdict D1 en tiendra compte.

## 5. Ancres EN gelées (non-régression, à re-mesurer à chaque gate)

| ancre | valeur | source |
|---|---|---|
| UTMOS EN | 4.08 | baseline_en, ce rapport |
| TTFA EN p50 (L4 liquid) | 231 ms | ce rapport |
| fresh-300 tool score (v4) | 0.830 | workstream EN (repris tel quel) |
| WER aller-retour EN | `wer_rescored_0b.json` | ce rapport |

## Reste à faire pour clore la Phase 0

1. Relancer FLEURS + CV ASR sur Colab (runtime à reconnecter) → verdict D1.
2. Passe juge (`-T rubric=reasoning-v3`) sur fr_s2s — bloquée sur GEMINI_API_KEY
   dans les secrets Colab.
3. Intégrer versa_0b.json + wer_rescored_0b.json (jobs en cours) à ce rapport.
