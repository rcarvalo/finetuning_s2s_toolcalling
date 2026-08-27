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
| `Voxtral-4B-TTS-2603` | **CC-BY-NC-4.0** | ⚠️ **non commercial** — inutilisable pour une voix d'assistant d'entreprise |

**Voxtral est notre moteur d'étiquetage, pas notre voix.** Les modèles ASR sont
Apache 2.0, donc utilisables sans réserve ; le TTS est non commercial, ce qui
l'exclut pour synthétiser la voix de l'assistant si le produit est commercial
(robot d'accueil). La voix de sortie reste Qwen3-TTS.

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

Ce que le modèle doit apprendre à produire. Exigences : **une seule identité
vocale**, registre parlé, et surtout **texte parfaitement aligné** puisque c'est
l'alignement texte/parole en interleavé qui est cassé.

- base : `french-dialogue-tts-1000h` (déjà le meilleur mesuré)
- complément : synthèse Qwen3-TTS des dialogues qu'on génère
- contrôle : passe Voxtral sur chaque clip synthétisé ; tout écart au texte
  source écarte le clip. Un TTS qui dérape produit un exemple qui apprend à
  déraper.

### B. Parole utilisateur en français (cible ~50-100 h)

Ce que le modèle doit apprendre à entendre. Exigences inverses de A : **diversité
maximale** de locuteurs, d'accents, de conditions.

- base : Common Voice étudiant, filtré `distillmos ≥ 3,5` **et** validé par
  Voxtral (accord transcript/ré-écoute)
- hold-out appliqué à **toutes** les sources (`data_prep/holdout.py`)
- réserve : `emilia-yodas-fr` pour du spontané, si le décodage vaut le coût

### C. Contenu dialogal (à générer)

- ~500 dialogues FR (pipeline Gemini existant)
- **~200 dialogues code-switch** — priorité relevée : c'est le seul axe où le
  prompt système plafonne (84 % contre 100 % en anglais pur), donc le seul que
  l'entraînement doit vraiment gagner
- tool calling FR : plus tard, au rung R4

### D. Préservation de l'anglais (20 % du mélange)

Les ancres EN sont non négociables (WER 0,080, UTMOS 4,047). Sources : partie EN
du pilote 125h + `tc-en-voice-agent-v1`.

## Ce qu'il faut mesurer avant de construire

Une expérience à ~1 $ peut supprimer une brique entière. Le verdict D1 (« ASR
français faible : 0,50 contre 0,15 en anglais ») a été mesuré **en mode
interleavé**, comme tout le reste — or on sait maintenant que l'interleaving
casse le français. Si le même benchmark FLEURS-fr en mode `text_only` remonte
vers 0,15, alors **l'écoute française n'a jamais été le problème**, le rung R1
disparaît, et la brique B se réduit à ce qu'il faut pour ne pas régresser.

À faire avant de synthétiser quoi que ce soit.
