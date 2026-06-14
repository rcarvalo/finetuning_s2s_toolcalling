# finetuning_s2s_toolcalling

Phases **2 (fine-tuning)** et **3 (tools + RAG)** de l'assistant vocal S2S français
avec tool calling sur **LFM2.5-Audio-1.5B**, embarqué dans **Reachy Mini** —
use case : agent d'accueil d'entreprise (vérification de rendez-vous,
notification du collaborateur, orientation du visiteur, wifi invité, escalade
réceptionniste, requêtes PostgreSQL, base de connaissances RAG).

Stratégie : **« thinking in text, speaking in audio »** — le modèle émet les
tool calls dans le flux **texte** de la génération interleaved (tokens
`<|tool_call_start|>…<|tool_call_end|>` hérités du backbone LFM2.5),
l'orchestrateur exécute l'outil et réinjecte le résultat en rôle `tool`,
puis le modèle génère la réponse audio.

## Arborescence

```
src/s2s_toolcalling/
├── data/            # Phase 2 — format de données
│   ├── chat_format.py        # tokens ChatML LFM2.5 (tool_list/call/response), system prompt
│   ├── dialogue_schema.py    # schéma JSONL des dialogues d'entraînement (validation)
│   ├── liquid_adapter.py     # Dialogue → list[ChatMessage] liquid-audio
│   └── preprocess_sft.py     # CLI : JSONL + wavs → dataset pré-packé (LFM2AudioChatMapper)
├── training/        # Phase 2 — fine-tuning
│   ├── lora.py               # injection LoRA in-place dans le backbone (peft bas niveau)
│   ├── freeze.py             # politiques de gel (encodeur / têtes audio / backbone)
│   └── train_sft.py          # lanceur : réutilise le Trainer OFFICIEL liquid-audio
├── tools/           # Phase 3 — outils métier
│   ├── schemas.py            # définitions JSON (contrat unique entraînement/inférence)
│   ├── registry.py           # dispatch async, validation, timeouts
│   ├── reception.py          # 5 outils accueil + backends InMemory / Postgres
│   └── database.py           # query_database : SELECT-only + session read-only
├── orchestrator/    # Phase 3 — boucle agent
│   ├── tool_parser.py        # parsing streaming des spans <|tool_call_*|> (ast, sans eval)
│   ├── agent.py              # generate_interleaved ↔ exécution ↔ réinjection rôle tool
│   ├── fillers.py            # fillers vocaux (« je vérifie… ») pendant le round-trip
│   ├── events.py             # événements vers le transport
│   └── server.py             # WebSocket minimal (point de branchement Reachy, Phase 4)
├── rag/             # Phase 3 — base de connaissances
│   ├── ingest.py             # chunking + indexation ChromaDB (embeddings multilingues)
│   └── retriever.py          # outil search_knowledge_base
└── evaluation/
    └── eval_toolcalling.py   # scoring BFCL-style FR (call/name/relevance/parse)

configs/             # phase2a (adaptation FR), phase2b (SFT tool calling), orchestrateur
scripts/             # prepare_fr_asr_tts, calibrate_interleaved_ratio, demo_orchestrator
sql/schema.sql       # schéma PostgreSQL + rôle lecture seule + données de démo
data/examples/       # dialogues d'exemple (dont négatif et escalade)
tests/               # 67 tests, sans GPU ni torch
```

## Installation

```bash
pip install -e .                # cœur (Python ≥ 3.11)
pip install -e ".[dev]"         # + tests
pip install -e ".[train]"       # + fine-tuning (Python ≥ 3.12, GPU)
pip install -e ".[serve,rag]"   # + orchestrateur / RAG
```

## Phase 2 — Fine-tuning

### 2a. Adaptation FR (recette du variant JP)

1. Exporter un manifest `{"audio", "text"}` depuis Common Voice FR / MLS-FR (Phase 1).
2. Calibrer le ratio interleaved FR (EN = 6:12, JP = 6:9) :
   ```bash
   python scripts/calibrate_interleaved_ratio.py --manifest cv_fr_stats.jsonl
   ```
3. Générer les dialogues ASR/TTS et packer en mode sequential :
   ```bash
   python scripts/prepare_fr_asr_tts.py --manifest cv_fr.jsonl --output data/fr_asr_tts.jsonl
   python -m s2s_toolcalling.data.preprocess_sft --dialogues data/fr_asr_tts.jsonl \
       --audio-root <racine> --output datasets/fr_asr_tts_train --assistant-audio-mode sequential
   ```
4. Entraîner (encodeur et têtes audio **dégelés** pour la phonétique FR) :
   ```bash
   accelerate launch -m s2s_toolcalling.training.train_sft --config configs/phase2a_fr_adaptation.yaml
   ```

### 2b. SFT interleaved : dialogues FR + tool calls (cœur)

Les dialogues synthétiques (Phase 1) suivent `data/examples/dialogues.sample.jsonl` :
tour assistant *tool call* = **texte seul** (audio supprimé), tour final =
texte + audio (interleaved). Inclure 20–30 % de négatifs sans appel.

```bash
python -m s2s_toolcalling.data.preprocess_sft --dialogues data/dialogues_train.jsonl \
    --audio-root data/audio --output datasets/sft_toolcalling_train \
    --interleaved-text-tokens 6 --interleaved-audio-tokens 10   # ratio calibré
accelerate launch -m s2s_toolcalling.training.train_sft --config configs/phase2b_sft_toolcalling.yaml
```

Le lanceur **réutilise le `Trainer` officiel de liquid-audio** et n'ajoute que
l'injection LoRA (backbone seul) + le gel encodeur/têtes audio + la sauvegarde
de l'adaptateur (`outputs/.../final_adapter/`).

### Critères de validation (méthodologie)

| Critère | Seuil | Mesure |
|---|---|---|
| WER FR | < 15 % | Common Voice FR test |
| UTMOS | > 3,7 (chute < 0,3 vs vanilla, sinon geler têtes audio) | sorties TTS/S2S |
| Tool calling FR | > 75 % | `eval_toolcalling` (BFCL-style) |
| VoiceBench | régression < 5 pts vs vanilla | suite officielle |

```bash
python -m s2s_toolcalling.evaluation.eval_toolcalling --predictions eval_fr.jsonl
```

## Capacité tool calling vocal EN — `web_search` + `db_query` (Phase A, v1)

Donne au **même** modèle LFM2.5-Audio (aucun second modèle) la capacité d'appeler
deux outils **à partir de la parole**, en anglais. v1 = **single-turn** : audio
utilisateur → émission du tool call dans le flux **texte**
(`<|tool_call_start|>[…]<|tool_call_end|>`). Précédent : le cookbook Liquid
`voice-assistant` (OHF-Voice) prouve audio→function-call sur ce modèle
(99 % name / 90 % arg). Réutilise toute la plomberie existante (format, LoRA,
parser, éval) ; `db_query` prend une **question en langage naturel** (NL→SQL côté
backend, hors chemin du modèle).

```bash
pip install -e ".[train,tooldata]"

# 1) Données synthétiques vérifiées (taxonomie + parser/registre + anti-contamination)
export ANTHROPIC_API_KEY=...
python scripts/generate_toolcalling_data.py --output data/tc_en_train.jsonl \
    --n-total 3000 --held-out benchmark/toolcalling_en/cases.sample.jsonl

# 2) TTS des tours user (Kokoro multi-voix, 16 kHz ; voix held-out pour le test)
python scripts/synthesize_user_audio.py --dialogues data/tc_en_train.jsonl \
    --audio-root data/audio_tc_en --out data/tc_en_train.audio.jsonl --split train
python scripts/synthesize_user_audio.py --dialogues benchmark/toolcalling_en/cases.sample.jsonl \
    --audio-root data/audio_tc_en --out data/tc_en_bench.audio.jsonl --split test

# 3) Packer (set d'outils EN) puis entraîner (LoRA backbone, encodeur + têtes audio gelés)
python -c "import json,s2s_toolcalling.tools.schemas as s; open('tools_en.json','w').write(json.dumps(s.TOOLCALLING_EN_TOOL_DEFINITIONS))"
python -m s2s_toolcalling.data.preprocess_sft --dialogues data/tc_en_train.audio.jsonl \
    --audio-root data/audio_tc_en --output datasets/tc_en_train \
    --tool-definitions tools_en.json --assistant-audio-mode sequential
accelerate launch -m s2s_toolcalling.training.train_sft --config configs/phase_en_toolcalling.yaml

# 4) Éval AUDIO → tool-call (harnais : audio → modèle → predicted_text → scoring)
python scripts/eval_audio_toolcalling.py --backend vllm --checkpoint exports/lfm25_tc_en \
    --cases data/tc_en_bench.audio.jsonl --audio-root data/audio_tc_en \
    --out eval_tc_en.jsonl --arg-match token_f1
```

Métriques (BFCL-style + arg tolérant pour le texte libre) : `parse_rate`,
`relevance_accuracy` (appeler / s'abstenir), `name_accuracy`, `call_accuracy`.
Seuils v1 : name > 90 %, relevance > 85 %, arg tolérant > 75 %, parse > 98 %.
**Phase B** (itération) : boucle complète (résultat d'outil réinjecté → réponse
**parlée**) via l'orchestrateur existant + backends live (`ddgs`, NL→SQL sur
`sql/schema_en.sql`).

## Phase 3 — Orchestrateur tools + RAG

Démo du round-trip complet **sans GPU** (parser → registre → réinjection) :

```bash
python scripts/demo_orchestrator.py
```

Mise en service complète (GPU) :

```bash
psql -f sql/schema.sql reception                                  # base + rôle lecture seule
python -m s2s_toolcalling.rag.ingest --docs ./kb_docs --persist-dir ./chroma_db
python -m s2s_toolcalling.orchestrator.server --config configs/orchestrator.yaml
```

Boucle agent (`orchestrator/agent.py`) : audio visiteur → `generate_interleaved`
→ texte décodé token par token dans `StreamingToolCallParser` → au span complet :
arrêt de la génération, **filler vocal** (masque le round-trip, cible < 1,5 s),
exécution via le registre (validation, timeout), réinjection
`<|tool_response_start|>…<|tool_response_end|>` en rôle `tool`, reprise de la
génération pour la réponse audio. Garde-fous : `max_tool_rounds` avec escalade
`notify_receptionist`, erreurs de parsing non fatales, hook barge-in
(`should_stop`) prêt pour la Phase 4.

Sécurité `query_database` : garde syntaxique SELECT-only **et** session
PostgreSQL `default_transaction_read_only=on` + `statement_timeout` **et** rôle
SQL `agent_ro` sans droits d'écriture.

## Inférence optimisée : vLLM / vLLM-Omni

LFM2.5-Audio n'est pas supporté nativement par vLLM-Omni → ce repo fournit le
**plugin out-of-tree `vllm_omni_lfm2_audio`** (`src/vllm_omni_lfm2_audio/`,
entry point `vllm_omni.general_plugins`, pas de fork) : stage 0 AR interleaved
(backbone Lfm2 de vLLM core + depthformer, pattern MiMo-Audio) → stage 1
détokeniseur, avec **prefix caching** (tours et tool round-trips sans
re-prefill du contexte). Design et statut :
[`docs/vllm_omni_integration.md`](docs/vllm_omni_integration.md).

```bash
pip install -e ".[vllm-omni]"
python -m vllm_omni_lfm2_audio.convert_checkpoint \
    --checkpoint exports/lfm25_audio_fr --output exports/lfm25_audio_fr_omni
vllm-omni serve exports/lfm25_audio_fr_omni \
    --stage-config-path configs/vllm_omni_lfm2_audio.yaml
# parité vs liquid-audio (GPU, critère bloquant P2) :
OMNI_CHECKPOINT=exports/lfm25_audio_fr_omni python -m pytest tests/test_omni_parity.py -m gpu
```

Également disponible (phase P0, voie hybride / parité backbone) :

```bash
# merge LoRA → checkpoint complet (ratio interleaved calibré écrit dans config.json)
python -m s2s_toolcalling.training.export_checkpoint \
    --base LiquidAI/LFM2.5-Audio-1.5B --adapter outputs/phase2b_sft/final_adapter \
    --output exports/lfm25_audio_fr --mode full \
    --interleaved-text-tokens 6 --interleaved-audio-tokens 10

# backbone texte seul → servable immédiatement par vLLM (voie hybride / parité P0)
python -m s2s_toolcalling.training.export_checkpoint \
    --base LiquidAI/LFM2.5-Audio-1.5B --adapter outputs/phase2b_sft/final_adapter \
    --output exports/lfm25_backbone_fr --mode backbone
vllm serve exports/lfm25_backbone_fr

# parité backbone (GPU) — prérequis avant d'investir dans les stages vLLM-Omni
EXPORTED_BACKBONE=exports/lfm25_backbone_fr python -m pytest tests/test_backbone_parity.py -m gpu
```

## Risque technique tracé

L'émission d'un tool call en mode interleaved audio n'est **pas documentée par
Liquid AI** : c'est l'hypothèse à dérisquer en premier (entraîner 2b sur un
petit sous-ensemble et vérifier que le modèle émet le span texte sans audio).
En cas d'échec → repli hybride audio-in/text-out + TTS (cf. méthodologie,
seuils de bascule).

## Tests

```bash
python -m pytest tests/ -q   # 67 tests, sans GPU
```
