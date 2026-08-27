# Audit comparatif des sources FR

Échantillon de 100 clips par source (streaming, premiers clips valides).
Métriques audio via VERSA (autorité des gates) ; « WER labels » = écart entre le
transcript fourni et une ré-écoute faster-whisper small fr (proxy de propreté des
labels, pas un gate). `lfm2-bilingual-pilot-125h` n'est pas ré-audité : pré-packé et
déjà validé par le pilote (val_loss 2.02). `emilia_yodas_fr` ne distribue que des
codes codec : métadonnées seules, décoder pour juger le corpus jugerait le codec.

| source | registre | clips | locuteurs | durée méd. (s) | DNSMOS | UTMOS | NISQA | WER labels |
|---|---|---|---|---|---|---|---|---|
| student_cv_fr | lu (Common Voice, phrases lues) | 100 | 36 | 4.3 | 2.92 | 2.56 | 3.47 | 0.13 |
| dialogue_tts_1000h | dialogue (TTS synthétisé) | 100 | 1 | 4.4 | 3.22 | 3.73 | 4.38 | 0.00 |
| emilia_yodas_fr (métadonnées seules) | parole spontanée (Emilia/YODAS) | 100 | 43 | 9.8 | — | — | — | — |

## Recoupement inter-sources

Les sources Rcarvalo et le dataset étudiant se recoupent (Common Voice FR).
L'exclusion du hold-out (benchmark/*/speakers.txt + source_ids.txt) doit être
appliquée à TOUTES les sources au moment du mix, pas seulement à la source d'origine.

## Verdict — qui fournit quoi

L'écart de qualité audio entre les deux sources mesurables est net et il ne dit
pas « une bonne source et une mauvaise » : il dit **deux rôles différents**.

| | `student_cv_fr` | `dialogue_tts_1000h` |
|---|---|---|
| UTMOS | 2,56 | **3,73** |
| DNSMOS | 2,92 | **3,22** |
| NISQA | 3,47 | **4,38** |
| WER labels | 0,13 | **0,00** |
| locuteurs / 100 clips | **36** | 1 |
| registre | lu, micros grand public | dialogue, TTS propre |

**Entrée (ce que le modèle doit ENTENDRE) → `student_cv_fr`.** Sa faiblesse
mesurée est sa qualité d'enregistrement, et c'est exactement ce qui en fait un
bon corpus d'entrée : 36 locuteurs pour 100 clips, micros hétérogènes, accents
variés — la distribution qu'un assistant vocal rencontre réellement. Le WER
labels de 0,13 reste acceptable pour de l'ASR (il mélange bruit d'étiquetage et
difficulté acoustique réelle).

**Sortie (ce que le modèle doit DIRE) → `dialogue_tts_1000h` + voix Qwen3-TTS.**
Entraîner la sortie sur du 2,56 apprendrait au modèle à produire cette
qualité-là, alors que l'ancre EN gelée est à **UTMOS 4,08** (cf.
`reports/fr_baseline_2026-08.md`) : le gate R2 « UTMOS FR à ≤0,5 de l'UTMOS EN »
serait perdu d'avance. La contrepartie du corpus dialogue est sa mono-voix
(1 locuteur) — sans importance côté sortie, où l'on veut précisément UNE
identité vocale, et c'est le plan (voix Qwen3-TTS unique FR+EN).

**`emilia_yodas_fr`** (43 locuteurs/100, clips longs ~9,8 s, parole spontanée) :
non mesurable ici (codes codec seulement). Réserve pour diversifier l'entrée si
`student_cv_fr` ne suffit pas en volume ; son registre spontané est le plus
proche de l'usage réel, mais il faudra décoder un échantillon pour le noter.

**`lfm2-bilingual-pilot-125h`** : non ré-audité (pré-packé, déjà validé par le
pilote, val_loss 2.02). Sert de warm start au rung R2.

### Conséquence sur les ratios

Le plan prévoyait de décider la part du dataset étudiant « par comparaison de
qualité mesurée ». Elle est mesurée : part **majoritaire côté ASR/entrée**,
**nulle côté cible de parole**. Le mélange 80/20 FR/EN reste inchangé ; ce qui
se précise, c'est la composition interne du FR selon le rung :

- R1 (ASR FR) : `student_cv_fr` majoritaire, hold-out appliqué à toutes les sources.
- R2/R3 (parole FR en sortie) : `dialogue_tts_1000h` + synthèses Qwen3-TTS,
  `student_cv_fr` uniquement comme audio d'ENTRÉE des tours utilisateur.
