# Corpus FR — inventaire et cible

Établi le 27/08/2026, après la baseline 0B (`reports/fr_baseline_2026-08.md`).
Complète l'audit qualité (`docs/fr_data_audit.md`), qui comparait trois sources ;
ici on inventorie **tout** ce qui existe et on décide ce qu'on construit.

## Ce que la baseline impose au corpus

Le besoin en données a changé de nature le 27/08. Le modèle **sait le français** :
en génération texte seul, ses réponses françaises sont propres et complètes
(20/20, 303 caractères médians). C'est en **régime interleavé** qu'il se dégrade
— texte contaminé d'anglais, tronqué de moitié, parole qui dérive.

Le corpus ne doit donc pas « apprendre le français au modèle ». Il doit lui
apprendre à **tenir le français pendant qu'il génère de l'audio**. Conséquence
directe : ce qui compte est la **paire (texte, parole) alignée en français**,
pas le volume brut d'audio français.

## Inventaire réel

### Ce qui existe et sert

| source | volume | format | rôle |
|---|---|---|---|
| `Rcarvalo/french-dialogue-tts-1000h` | 15 164 clips | wav + `metadata.jsonl` | **cible de sortie** : UTMOS 3,73, NISQA 4,38, WER labels 0,00, registre dialogue, mono-voix |
| `baptistefrancois1/s2s-fr-finetuning` | 238 374 clips CV | wav + `distillmos` + `speaker_id` | **entrée utilisateur** : UTMOS 2,56 mais 36 locuteurs/100 clips — la diversité réelle qu'un assistant entend |
| `Rcarvalo/lfm2-bilingual-pilot-125h` | packé | tenseurs prêts | **warm start** validé (val_loss 2,02) |
| `Rcarvalo/emilia-yodas-fr-filtered` | — | `audio_codes` | parole spontanée, 43 locuteurs/100, 9,8 s médian — réserve d'entrée |

### Ce qui existe et ne sert pas (vérifié, pas supposé)

| source | volume | pourquoi ça ne sert pas |
|---|---|---|
| `Rcarvalo/audio_dataset` | **2 235 936** clips | fragments de **~1 seconde** (1,001-1,045 s mesuré), textes de 2-3 mots (« un seul », « Madeleine »). Volume trompeur : inutilisable pour du dialogue, aucune prosodie, aucun contexte |
| `Rcarvalo/audioFRv1` | 535 851 | `audio_codes` d'un codec tiers — LFM2 utilise Mimi ; inexploitable sans décodage complet |
| `Rcarvalo/omnivoice-fr-100h` | 7 144 | idem, `audio_codes` |
| `Rcarvalo/kanitts2-fr-nanocodec` | 1 255 shards | idem, nanocodec |
| `Rcarvalo/voxtral-french-1000h` | **vide** | dépôt créé, jamais rempli — c'est un projet, pas un actif |

**Le piège de l'inventaire** : additionner les lignes donne « ~2,8 M clips FR ».
En réalité, une fois retirés les fragments d'une seconde et les corpus enfermés
dans un codec étranger, ce qui est directement exploitable tient dans deux
sources — le dialogue TTS et le Common Voice de l'étudiant.

## Où Voxtral entre, et où il ne peut pas

| modèle | licence | usage |
|---|---|---|
| `Voxtral-Mini-3B-2507` / `Small-24B-2507` | **Apache 2.0** | ASR FR — nettoyage et étiquetage |
| `Voxtral-Mini-4B-Realtime-2602` | **Apache 2.0** | ASR temps réel, le plus téléchargé |
| `Voxtral-4B-TTS-2603` | CC-BY-NC-4.0 | candidat voix, **non retenu par défaut** (voir ci-dessous) |

Les modèles ASR sont Apache 2.0, sans réserve : Voxtral est notre **moteur
d'étiquetage**, et c'est là qu'il apporte le plus.

## La voix de l'assistant : SIWIS clonée par Qwen3-TTS

Décision du 27/08, prise sur deux critères qui pointent dans le même sens.

| composant | licence | ce que ça donne |
|---|---|---|
| SIWIS French Speech Synthesis Database | **CC-BY-4.0** | voix française **native, studio, professionnelle** |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | **Apache 2.0** | clonage de cette voix, texte illimité |
| (alternative) `Voxtral-4B-TTS-2603` | CC-BY-NC-4.0 | voix multilingue, contrainte non commerciale |

**Qualité** : SIWIS est de la parole française enregistrée en studio par une
locutrice professionnelle — la prosodie et l'accent sont nativement français,
là où la voix française d'un modèle multilingue reste une approximation. Le
clonage transporte cette identité sur du texte arbitraire, ce qui est exactement
le besoin de la brique A : une seule voix, sur nos dialogues.

**Licence** : Apache 2.0 + CC-BY-4.0 (attribution) — aucune restriction
commerciale. Voxtral-TTS, lui, aurait fait hériter au corpus **et au modèle
entraîné** une clause non commerciale ; sur un robot d'accueil d'entreprise
c'est une dette qu'on aurait payée plus tard.

Le projet a déjà de l'expérience SIWIS (`Rcarvalo/vibevoice-lora-siwis-samples`,
Kokoro `ff_siwis` côté utilisateur), donc la référence est disponible et connue.

**Mais le choix se tranche en mesurant.** « Que de la qualité » ne se décide pas
par réputation : le banc d'essai (`infra/jobs/voice_bakeoff.py`) synthétise les
**mêmes phrases françaises** avec chaque candidat et les note sur les métriques
des gates — UTMOS/NISQA via VERSA pour la naturalité, WER Voxtral pour la
fidélité au texte, plus une écoute. Candidats : SIWIS clonée (Qwen CustomVoice),
voix Qwen de base, Voxtral-TTS, et **la voix actuelle de
`french-dialogue-tts-1000h` comme tenante du titre** (UTMOS 3,73 mesuré). Le
gagnant prend la brique A ; s'il ne bat pas le tenant, on garde le tenant.

Trois usages concrets d'un ASR Voxtral, par valeur décroissante :

1. **Filtrer sur la qualité des labels.** L'audit a mesuré 13 % de WER entre le
   transcript fourni et une ré-écoute whisper-small sur le Common Voice
   étudiant. Une partie est du bruit d'étiquetage, une partie de la difficulté
   acoustique — whisper-small ne permet pas de trancher. Voxtral, lui, permet de
   garder les clips dont le transcript est *vraiment* juste : c'est le levier
   n°1 sur la qualité du corpus, et il s'applique à 238 k clips.
2. **Étiqueter ce qui n'a pas de transcript** — `audio-import` (6 118 opus) et
   tout audio FR brut récupéré ensuite.
3. **Servir d'oreille de référence aux gates**, en complément de whisper : deux
   ASR indépendants qui divergent signalent un audio ambigu plutôt qu'un vrai
   taux d'erreur (c'est déjà la doctrine « WER multi-oreilles » du plan VERSA).

## Le corpus qu'on veut

Quatre briques, dans l'ordre de ce que la baseline dit être critique.

### A. Parole assistant en français — LA brique (cible ~100-150 h)

Dossier `A_assistant_speech`.

Ce que le modèle doit apprendre à produire. Exigences : **une seule identité
vocale**, registre parlé, et surtout **texte parfaitement aligné** puisque c'est
l'alignement texte/parole en interleavé qui est cassé.

- base : `french-dialogue-tts-1000h` (déjà le meilleur mesuré)
- complément : synthèse des dialogues générés avec la **voix SIWIS clonée par
  Qwen3-TTS CustomVoice** (Apache 2.0 + CC-BY-4.0), sous réserve du banc d'essai
- contrôle : passe Voxtral sur chaque clip synthétisé ; tout écart au texte
  source écarte le clip. Un TTS qui dérape produit un exemple qui apprend à
  déraper.

### B. Parole utilisateur en français (cible ~50-100 h)

Dossier `B_user_speech`.

Ce que le modèle doit apprendre à entendre. Exigences inverses de A : **diversité
maximale** de locuteurs, d'accents, de conditions.

- base : Common Voice étudiant, filtré `distillmos ≥ 3,5` **et** validé par
  Voxtral (accord transcript/ré-écoute)
- hold-out appliqué à **toutes** les sources (`data_prep/holdout.py`)
- réserve : `emilia-yodas-fr` pour du spontané, si le décodage vaut le coût

### C. Contenu dialogal (à générer)

Dossier `C_dialogues`.

- ~500 dialogues FR (pipeline Gemini existant)
- **~200 dialogues code-switch** — priorité relevée : c'est le seul axe où le
  prompt système plafonne (84 % contre 100 % en anglais pur), donc le seul que
  l'entraînement doit vraiment gagner
- tool calling FR : plus tard, au rung R4

### D. Préservation de l'anglais (20 % du mélange)

Dossier `D_english`.

Les ancres EN sont non négociables (WER 0,080, UTMOS 4,047). Sources : partie EN
du pilote 125h + `tc-en-voice-agent-v1`.

## « Si peu de données, est-ce que ça suffit ? »

Oui, très probablement — et la raison est le résultat central de la baseline.

On n'apprend pas le français au modèle : **il le possède déjà**. En génération
texte seul, son français est propre et complet. Ce qu'on lui apprend, c'est à le
tenir pendant qu'il produit de l'audio. C'est une **adaptation de comportement**,
pas une acquisition de langue, et les deux n'ont pas les mêmes ordres de
grandeur : acquérir une langue demande des milliers d'heures, déplacer un
comportement déjà présent en demande des dizaines.

Trois appuis empiriques, tous internes au projet :

1. le **pilote à 125 h** (100 FR + 25 EN) a donné val_loss 2,02 avec l'anglais
   préservé — c'est déjà une preuve à l'échelle visée ;
2. les rungs de tool calling v3/v4 ont produit des changements de comportement
   massifs avec des corpus bien plus petits ;
3. la cible est étroite (aligner texte et parole en français), pas large.

**Le risque n'est donc pas le volume, c'est la qualité et la couverture.** Deux
points de vigilance qui méritent l'effort qu'on aurait mis à collecter des
heures supplémentaires :

- un clip TTS qui dérape enseigne à déraper — d'où le contrôle Voxtral
  systématique sur la brique A, clip par clip ;
- 200 dialogues code-switch, c'est peu pour l'axe le plus difficile. Si le gate
  R3 bute uniquement là, la réponse est un top-up ciblé de cette brique, pas un
  corpus plus gros partout.

Doctrine de dimensionnement : **on ne dimensionne pas en heures, on dimensionne
en gates**. On construit la cible ci-dessous, on entraîne, et on n'ajoute des
données que là où un gate le réclame — ce qui suppose de savoir, à chaque gate,
quelle brique est en cause. C'est pourquoi les briques sont séparées dès le
stockage.

## Le dépôt HF : une brique par dossier

Dépôt unique **`Rcarvalo/lfm25-fr-corpus-v1`** (privé), quatre dossiers :

```
A_assistant_speech/   manifest.jsonl + audio/ + README.md
B_user_speech/        idem
C_dialogues/          idem
D_english/            idem
```

La séparation n'est pas du rangement, c'est un **outil de travail** : quand un
gate échoue, la question utile est « quelle brique est courte ? », et un corpus
mis en commun ne sait pas y répondre. Elle permet aussi de compléter une brique
et de la republier sans toucher aux autres.

**Schéma unique**, partagé par les quatre briques, pour que le mixeur ne lise
qu'un seul format :

| champ | rôle |
|---|---|
| `id`, `audio`, `text`, `lang`, `duration_s` | le minimum, validé à l'écriture |
| `role` | `user` ou `assistant` — c'est lui qui décide du côté du tour |
| `speaker`, `source` | traçabilité et hold-out |
| `voxtral_wer` | écart entre `text` et la ré-écoute Voxtral — **le filtre d'entrée** |
| `utmos` | qualité de la parole, pour la brique A |

Publication : `lfm2-corpus-push --brick A --local … --repo-id …`, une brique à la
fois. Le manifeste et l'existence de chaque fichier audio sont **validés avant
tout envoi** — un corpus malformé ne doit jamais atteindre le Hub, parce qu'une
ligne fautive qui y entre est ensuite entraînée en silence.

## Ce qu'il faut mesurer avant de construire

Une expérience à ~1 $ peut supprimer une brique entière. Le verdict D1 (« ASR
français faible : 0,50 contre 0,15 en anglais ») a été mesuré **en mode
interleavé**, comme tout le reste — or on sait maintenant que l'interleaving
casse le français. Si le même benchmark FLEURS-fr en mode `text_only` remonte
vers 0,15, alors **l'écoute française n'a jamais été le problème**, le rung R1
disparaît, et la brique B se réduit à ce qu'il faut pour ne pas régresser.

À faire avant de synthétiser quoi que ce soit.
