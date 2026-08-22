# lfm2-audio

Fine-tuning, serving vLLM et orchestration tool-calling de **LFM2.5-Audio-1.5B** —
assistant vocal speech-to-speech, embarqué dans Reachy Mini.

Stratégie : **« penser en texte, parler en audio »**. Le modèle émet ses tool calls
dans le flux *texte* de la génération interleaved (`<|tool_call_start|>…<|tool_call_end|>`),
l'orchestrateur exécute l'outil et réinjecte le résultat en rôle `tool`, puis le
modèle produit la réponse **parlée**.

## Inférence en trois lignes

```python
from lfm2_audio import LFM2Audio

model = LFM2Audio.from_pretrained("Rcarvalo/lfm25-tc-en-s2s")
text, audio = model.reply(audio="question.wav")
```

`from_pretrained` accepte **toutes** les formes de checkpoint et matérialise ce
qui manque (fusion LoRA, conversion vers le layout vLLM-Omni) une seule fois, en
cache :

```python
LFM2Audio.from_pretrained("exports/lfm25_audio_fr_omni")  # déjà converti
LFM2Audio.from_pretrained("exports/lfm25_audio_fr")  # layout liquid → converti
LFM2Audio.from_pretrained("Rcarvalo/lfm25-tc-en-s2s-adapter")  # base lue dans l'adaptateur
LFM2Audio.from_pretrained(
    "LiquidAI/LFM2.5-Audio-1.5B",  # base + LoRA → fusionné
    adapter="Rcarvalo/lfm25-tc-en-s2s-adapter",
    interleaved_ratio=(6, 10),
)
LFM2Audio.from_pretrained("exports/lfm25_audio_fr", backend="liquid")  # référence PyTorch

for chunk in model.stream(audio="question.wav"):  # streaming 24 kHz, TTFA ~300 ms
    play(chunk.samples)
```

Deux backends derrière la même API : **vLLM-Omni** (2 stages, streaming basse
latence) et **liquid-audio** (référence PyTorch, batch = 1). `backend="auto"`
prend le premier installé.

## Démarrage

```bash
make install          # .venv + groupe dev (CPU : lint, typecheck, tests, données)
make install-serving  # + vLLM-Omni et liquid-audio (GPU NVIDIA requis)
make hooks            # hooks pre-commit
make check            # lint + typecheck + tests — ce que vérifie la CI
```

Un [devcontainer](.devcontainer/) CPU est fourni : tout ce qui ne demande pas de
GPU y tourne, y compris sur un Mac.

## Arborescence

```
python/lfm2_audio/
├── core/          # abstractions transverses — erreurs, environnement, prompt ChatML
├── ds/            # structures de données — pydantic aux frontières, value objects
│   ├── audio.py         Waveform (signal + fréquence, indissociables)
│   ├── conversation.py  Conversation — garante de « un seul audio par prompt »
│   ├── config.py        EngineConfig / GenerationConfig (pydantic)
│   ├── dialogue.py      schéma JSONL d'entraînement (pydantic)
│   └── reply.py         Reply + TurnMetrics
├── serving/       # chargement du modèle et backends d'inférence
│   ├── model.py         LFM2Audio — ABC + fabrique `from_pretrained`
│   ├── registry.py      catalogue des backends (import paresseux)
│   ├── checkpoint/      sources → détection de layout → préparation
│   └── backends/        vllm_omni.py · liquid.py · omni_engine.py
├── vllm_plugin/   # plugin out-of-tree vLLM-Omni (chargé dans chaque worker)
├── training/      # SFT LoRA, gel encodeur / têtes audio, export de checkpoint
├── data_prep/     # génération, packing et conversion des datasets
├── tools/         # outils métier appelables par le modèle
├── orchestrator/  # boucle agent tool-calling, fillers, transport temps réel
├── rag/           # base de connaissances ChromaDB
├── evaluation/    # scoring BFCL-style des tool calls, métriques de latence
└── cli/          # points d.entrée `lfm2-*` (argparse seulement)

configs/{serving,training,sql}/   tests/   docs/   notebooks/   data/
```

Trois patterns structurent le serving, chacun justifié par une pluralité réelle
d'implémentations (pas par anticipation) :

| Pattern | Où | Ce qu'il permet |
|---|---|---|
| **Fabrique + registre** | `serving/registry.py` | ajouter un backend sans toucher à `LFM2Audio` ; import paresseux pour que `import lfm2_audio` reste léger |
| **Chain of responsibility** | `serving/checkpoint/sources.py` | ajouter un stockage (S3…) sans toucher au résolveur |
| **Strategy** | `serving/checkpoint/preparers.py` | un cas de préparation = une classe (passthrough / conversion / fusion LoRA) |

`LFM2Audio.reply()` est une *template method* : les backends n'implémentent que
`stream()`.

## Commandes

Toutes les CLIs sont des entry points `lfm2-*` (`--help` sur chacune) :

```bash
lfm2-demo --checkpoint exports/lfm25_audio_fr --audio-in question.wav
lfm2-demo --checkpoint exports/lfm25_audio_fr --interactive
lfm2-bench --checkpoint exports/lfm25_audio_fr_omni --runs 5      # TTFA / RTF
lfm2-toolcalling-demo --checkpoint exports/tc_en --adapter <repo> --share
lfm2-smoke --checkpoint exports/lfm25_audio_fr_omni               # plugin vLLM-Omni
```

Données et entraînement :

```bash
lfm2-generate-data --provider gemini --output data/tc_en_train.jsonl --n-total 3000
lfm2-synthesize-audio --engine voxtral --dialogues data/tc_en_train.jsonl ...
lfm2-build-dataset --repo-id <user>/tc-en-audio --train ... --private
lfm2-analyze-dataset --dialogues data/tc_en_train.jsonl --audio-root data/audio_tc_en
lfm2-preprocess-sft --dialogues ... --output datasets/tc_en_train
accelerate launch -m lfm2_audio.cli.train_sft --config configs/training/phase_en_toolcalling.yaml
lfm2-eval-audio --backend vllm --checkpoint exports/tc_en --cases ... --arg-match token_f1
```

## Documentation

- [docs/serving.md](docs/serving.md) — charger un modèle, choisir un backend, régler la latence
- [docs/architecture.md](docs/architecture.md) — découpage du paquet et pourquoi
- [docs/vllm_omni_integration.md](docs/vllm_omni_integration.md) — design du plugin out-of-tree
- [docs/optimization_audit.md](docs/optimization_audit.md) — leviers de latence mesurés
- [docs/audio_in_spec.md](docs/audio_in_spec.md) — chemin audio-in natif (mel, placeholders)

## Critères de validation

| Critère | Seuil | Mesure |
|---|---|---|
| WER FR | < 15 % | Common Voice FR test |
| UTMOS | > 3,7 (chute < 0,3 vs vanilla) | sorties TTS/S2S |
| Tool calling | name > 90 %, relevance > 85 %, arg > 75 %, parse > 98 % | `lfm2-eval-audio` (BFCL-style) |
| TTFA | 200-500 ms | `lfm2-bench` |
| VoiceBench | régression < 5 pts vs vanilla | suite officielle |

## Tests

```bash
make test       # sans GPU
make test-gpu   # parité numérique — checkpoints requis
```
