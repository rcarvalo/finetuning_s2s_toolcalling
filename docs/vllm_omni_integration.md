# Spec : intégration de LFM2.5-Audio (fine-tuné FR + tool calling) dans vLLM-Omni

> Statut : **implémenté comme plugin out-of-tree** dans ce repo
> (`src/vllm_omni_lfm2_audio/`) — pas de fork nécessaire. Vérifié contre le
> package PyPI `vllm-omni==0.22.0` (juin 2026) : le groupe d'entry points
> `vllm_omni.general_plugins` est chargé dans tous les process (engine +
> workers), `register_pipeline()` est documenté « for out-of-tree plugins »,
> et `OmniModelRegistry.register_model()` accepte les architectures externes.
> Reste à valider sur GPU : parité P2/P3 (`tests/test_omni_parity.py`).
>
> ⚠️ **Audit du wheel 0.22.0 (10 juin 2026)** : deux mécanismes utilisés par
> la première version du stage 0 n'existent pas dans le runtime — voir
> §3bis. Le stage 0 doit migrer vers le hook `sample()` /
> `prefer_model_sampler` (idiome cosyvoice3/glm_tts). Itération en cours sur
> Colab (`scripts/colab_smoke_vllm_omni.py`).
>
> ✅ **Jalon engine (10 juin 2026, Colab T4)** : `Omni(model=…)` démarre les
> 2 stages et `generate()` produit du texte (stage 0) + une sortie typée
> audio (stage 1) de bout en bout. 10 écarts runtime corrigés en itérant
> (commits `a9d9336`→`0225145`) : AutoConfig.register, ModelRegistry vLLM
> core, import SharedEmbedding (model.transformer), contrat
> track_weights_loading (noms de paramètres), IsHybrid/HasInnerState sur la
> classe (mamba_block_size), signature forward du runner, détokeniseur
> float32 (torch.polar ∄ en Half) + suivi de device, tolérance dummy-run
> (trim frames partielles, clamp vocab), enforce_eager (capture CUDA graph à
> traiter plus tard à la mimo), partage VRAM entre stages. Manque encore :
> le flux audio réel (sample() + replay + depthformer) — critère P2.

## 1. Pourquoi

Le serveur `liquid-audio` (PyTorch) re-préfille **tout le contexte** à chaque
round d'outil et à chaque tour de conversation. vLLM-Omni apporte :

| Gain | Impact sur notre use case |
|---|---|
| **Automatic prefix caching** | le round-trip d'outil ne préfille que le tour `tool` réinjecté → directement le critère « tool round-trip < 1,5 s » |
| Paged KV cache | sessions longues (hall d'accueil) sans fragmentation mémoire |
| Continuous batching | plusieurs robots / sessions sur un seul GPU |
| Stages désagrégés | encodeur/backbone/détokeniseur sur des devices différents si besoin |

## 2. État des lieux vérifié

- **LFM2.5-Audio absent** de la liste des modèles supportés par vLLM-Omni
  (Qwen3.5/Qwen3-Omni, CosyVoice3, MOSS-TTS, Fish Speech, Voxtral TTS,
  Higgs-Audio v2, MiMo-Audio, …).
- **Le backbone Lfm2 est déjà implémenté dans vLLM core**
  (`vllm/model_executor/models/lfm2.py`, cache conv hybride) → réutilisé tel
  quel via `init_vllm_registered_model(architectures=["Lfm2ForCausalLM"])`.
- **L'out-of-tree est supporté** (vérifié dans le source 0.22.0) : entry
  points `vllm_omni.general_plugins` (`vllm_omni/plugins/__init__.py`),
  `register_pipeline()` (`vllm_omni/config/stage_config.py`) et
  `OmniModelRegistry.register_model()` → **plugin, pas de fork**.
- **Précédent exact suivi : MiMo-Audio**
  (`vllm_omni/model_executor/models/mimo_audio/`) — le seul S2S interleaved
  texte+audio in-tree, en topologie « fused_thinker_talker » mono-stage AR +
  code2wav : placeholder dans le flux d'ids (`<|empty|>` chez MiMo), codes
  produits en interne avec caches par requête, embeddings audio réinjectés au
  step suivant, export par step via `OmniOutput.multimodal_outputs`, chunking
  vers le stage 1 par `SharedMemoryConnector` (async_chunk). Le talker
  Qwen3-Omni (code predictor MTP) confirme le mécanisme multi-codebook.

## 3. Le problème central et sa solution

**Problème** : vLLM échantillonne UN token id par step ; une frame audio
LFM2.5-Audio = **8 codes Mimi** produits par le depthformer, et le flux
interleave texte et audio dans la même séquence (ratio `n_text:n_audio`).

**Solution (pattern MiMo-Audio, transcrit du source in-tree)** :

1. **Chaque step audio apparaît dans le flux d'ids comme un token
   PLACEHOLDER** (MiMo : `<|empty|>` ; nous : deux ids réservés, frame
   normale et frame EOA — deux ids distincts pour que l'état soit rejouable
   depuis la seule séquence). Le modèle force cet id dans son **`sample()`
   custom** (`prefer_model_sampler = True`, cf. §3bis) ; la séquence vLLM
   reste standard → KV paginé, prefix caching, préemption inchangés.
2. **Les 8 codes Mimi de la frame sont produits en interne** par le rollout
   du depthformer sur le hidden state du step (`audio_head.sample_frame`,
   port de `_sample_audio_frame`) et exportés step par step via
   `OmniOutput.multimodal_outputs["codes"]["audio"]` vers le stage 1.
3. **L'embedding d'entrée du step suivant = somme des 8 embeddings de
   codebooks** (`audio_embedding(frame + offsets).sum(0)`), servi par un
   cache par requête (`_frame_emb_by_req`, équivalent du
   `_cached_new_audio_emb_by_req` de MiMo) en décode, ou reconstruit depuis
   `multi_modal_data["audio_out_codes"]` au prefill/recompute.

## 3bis. Écarts vérifiés dans le runtime 0.22.0 (audit du wheel, 10 juin 2026)

Constatés en inspectant `gpu_ar_model_runner.py` / `gpu_model_runner.py` du
wheel PyPI — pas spéculatifs :

1. **`OmniOutput.next_token_id` n'est consommé nulle part** dans le runtime
   (champ déclaré dans le NamedTuple, zéro lecteur). Le forçage du
   placeholder DOIT passer par le hook sampler du modèle :
   `prefer_model_sampler = True` + `sample(logits, sampling_metadata)`,
   routé par `GPUARModelRunner._sample()`. Les modèles in-tree cosyvoice3 et
   glm_tts donnent l'idiome exact : itération par ligne du batch,
   `sampling_metadata.output_token_ids[req_idx]` = historique généré de la
   ligne (reconstruit par `_build_model_sampler_output_token_ids` dans
   l'ordre `input_batch.req_ids`), retour `SamplerOutput(sampled_token_ids=…)`.
2. **`forward()` ne reçoit pas de `request_ids`** — aucun kwarg d'identité de
   requête n'est passé par le runner. Conséquences :
   - le replay de modalité se fait dans `sample()` (l'historique par ligne y
     est fourni), pas dans `forward()` ;
   - le rollout du depthformer migre aussi dans `sample()` ; le hidden gathered
     aux `logits_indices` est stashé par `compute_logits()` (alignement 1:1
     avec les lignes de logits) ;
   - `extract_multimodal_outputs()` (runner) consomme bien
     `OmniOutput.multimodal_outputs` — l'export des codes reste valable, avec
     `meta.req_id` supporté pour les payloads sparses.
3. **RÉSOLU (runtime vivant, 11 juin 2026)** : l'identité de requête vient du
   hook ``has_preprocess`` — le runner appelle ``preprocess_batch(req_ids=…)``
   1×/step puis ``preprocess(input_ids=span, input_embeds=…, request_id=…,
   _omni_is_prefill=…, …)`` PAR requête avant le forward (et le retour
   ``(ids, embeds, update_dict)`` overlaye les embeddings du span). Le cache
   ``{request_id: frame_embedding}`` est servi là ; le mapping ligne→requête
   de ``sample()`` se déduit de l'ordre des appels preprocess.
4. **Async scheduling incompatible avec le replay de modalité (vérifié)** :
   ``_build_model_sampler_output_token_ids`` TRONQUE l'historique quand le
   token in-flight n'est pas encore copié → la décision 6:12 part en retard
   d'un step à la frontière de bloc et le flux émis viole la grammaire
   (garde-fou ``modality.advance`` fatal). ``async_scheduling=False`` requis
   pour l'instant ; support propre à concevoir en phase TTFA.

## 4. Design LFM2.5-Audio (implémenté dans `src/vllm_omni_lfm2_audio/`)

### 4.1 Flux d'ids et placeholders (`constants.py`)

```
ids texte           : vocab LFM2.5 standard (inclut les tokens d'outils)
audio_frame_token_id: placeholder d'une frame normale   (défaut 128, <|audio_start|>,
                                                          jamais émis en interleaved)
audio_eoa_token_id  : placeholder de la frame EOA 2048×8 (défaut 129, configurable)
```

Garde-fou au démarrage : `verify_placeholder_ids(tokenizer, ...)` refuse un
placeholder qui ne serait pas un token spécial du vocabulaire.

### 4.2 Machine à états de modalité (`modality.py` — pur, le cœur dérisqué)

Port fidèle de la boucle `generate_interleaved` (y compris son quirk : une
frame EOA bascule en TEXT **sans** réinitialiser le budget) :

```
état TEXT  : n = interleaved_n_text tokens max, puis bascule AUDIO ;
             <|text_end|> (130) → text_done=True (audio seul ensuite) ;
             <|im_end|>  (7)    → stop (fin des tours de TOOL CALL, texte seul)
état AUDIO : n = interleaved_n_audio frames, puis retour TEXT (si !text_done) ;
             frame EOA → retour TEXT
```

**Verrou tool-call (extension hors-amont, requis pour le port S2S+outils)** :
l'orchestrateur 2-passes (`VllmToolAgent`) regénère depuis un prompt complet, et
la Pass A (appel d'outil) DOIT être texte seul — comme le `generate_sequential`
de liquid-audio — sinon l'interleaving 6:12 hache le span `[fn(arg="…")]`
(token-salad `<|audio_start|>`×12 en plein appel, mesuré sur Colab le 23/06). Un
`interleaved_n_text` global énorme corrige la Pass A mais casse la Pass B parlée
(le modèle, entraîné interleavé, bafouille « uh, uh… » privé de ses frames). La
solution est **basée sur le contenu** (pas de signal par-requête à plomber) : si
`<|tool_call_start|>` est vu sans `<|tool_call_end|>` correspondant, la machine
SUPPRIME la bascule audio périodique → l'appel reste 100 % texte ; à la
fermeture, un budget texte frais est réarmé. Cela traite les 3 cas avec un seul
réglage `6:12` : Pass A (texte), Pass B parlée (interleavé), et réponse directe
sans outil (interleavé). Reste une fonction pure du flux d'ids (rejouable,
prefix-cache safe). Les ids `<|tool_call_start|>`/`<|tool_call_end|>` sont
résolus depuis le tokenizer du checkpoint au démarrage du stage 0 (`None` →
verrou inerte, parité amont stricte).

**L'état est une fonction pure des ids du tour assistant courant** → rejouable
après préemption, compatible prefix caching. Vérifié par tests de propriété
contre une transcription littérale de la boucle amont
(`tests/test_omni_modality.py`, schedules aléatoires, ratios 6:12 et 6:9).
Le ratio est lu dans le `config.json` du checkpoint exporté — **source
unique** entraînement/serving.

État par requête restant (non rejouable, mais reconstructible) :
`{req_id: frame_embedding}` pour l'embedding du step suivant — au prefill il
est reconstruit depuis les codes passés en mm data.

### 4.3 Sortie texte / logits

Les steps texte passent par `compute_logits` du backbone Lfm2 de vLLM core
(embeddings liés). Pendant un step audio, le `sample()` custom du modèle
(`prefer_model_sampler`, cf. §3bis) force le placeholder — la décision
texte/audio est rejouée depuis `sampling_metadata.output_token_ids` (machine
à états pure de `modality.py`). Le sampling **audio** (prosodie :
temperature/top-k du depthformer) est interne au modèle
(`audio_temperature`/`audio_top_k` du config) — le sampling texte reste celui
de l'API (greedy recommandé pour les tool calls).

### 4.4 Entrées audio

- **Micro visiteur** : mel-128 (préprocesseur identique à liquid-audio) →
  ConformerEncoder + `audio_adapter` (modules liquid-audio, mêmes poids) via
  l'interface multimodale standard (pattern Qwen2-Audio/Ultravox).
- **Audio généré aux tours précédents** (rounds d'outils, multi-tours) : les
  placeholders restent dans `prompt_token_ids` ; les frames 8-codebooks
  complètes passent en `multi_modal_data["audio_out_codes"]` pour reconstituer
  les embeddings sommés au prefill. Le mm-hashing rend ces segments compatibles
  prefix cache.

### 4.5 Stages

```
stage 0  Lfm2AudioForConditionalGeneration   (AR interleaved, thinker-talker fusionné)
         in  : tokens texte + mel (mm) + audio_out_codes (mm)
         out : token_ids (texte → API streaming → parser de tool calls)
               multimodal_outputs["audio_codes"] : frames (8, T)

stage 1  Lfm2AudioCode2Wav                   (détokeniseur LFM2, non-AR)
         framework async_chunk : chunked_decode_streaming
         frames Mimi 12,5 Hz → PCM 24 kHz (1920 échantillons/frame)
         chunk_size / left_context_size à calibrer (TTFA vs artefacts de bord)
```

Fichiers du plugin (ce repo, package `vllm-omni-lfm2-audio`) :

```
src/vllm_omni_lfm2_audio/
├── __init__.py               # register() : entry point vllm_omni.general_plugins
├── constants.py              # ids spéciaux, placeholders, garde-fous tokenizer
├── modality.py               # machine à états PURE (rejouable) — testée par
│                             #   propriété contre la transcription littérale de
│                             #   generate_interleaved (tests/test_omni_modality.py)
├── audio_head.py             # depthformer + embeddings de frames (modules liquid-audio)
├── lfm2_audio.py             # architecture Lfm2AudioOmniModel (dispatch model_stage)
├── lfm2_audio_ar.py          # stage 0 AR interleaved
├── lfm2_audio_code2wav.py    # stage 1 détokeniseur
├── stage_input_processors.py # chunking codes → stage 1 (async_chunk)
├── pipeline.py               # PipelineConfig (register_pipeline)
└── convert_checkpoint.py     # export full → layout vLLM-Omni (config.json)
configs/vllm_omni_lfm2_audio.yaml   # déploiement 2 stages, prefix caching ON
```

Mise en service :

```bash
pip install -e ".[vllm-omni]"        # vllm-omni + liquid-audio + le plugin (entry point)
python -m vllm_omni_lfm2_audio.convert_checkpoint \
    --checkpoint exports/lfm25_audio_fr --output exports/lfm25_audio_fr_omni
vllm-omni serve exports/lfm25_audio_fr_omni \
    --stage-config-path configs/vllm_omni_lfm2_audio.yaml
```

### 4.6 Checkpoint

`s2s_toolcalling/training/export_checkpoint.py` (ce repo) produit :
- `--mode full` : LFM2.5-Audio fusionné (LoRA mergé) + `config.json` portant le
  ratio interleaved calibré → entrée de la conversion stage 0/1 (le
  `load_weights` du fork remappe `lfm.*`→backbone, `conformer.*`+
  `audio_adapter.*`→encodeur mm, `depthformer.*`/`depth_*`/`audio_embedding.*`
  →module code-predictor, `audio_detokenizer/`→stage 1) ;
- `--mode backbone` : `Lfm2ForCausalLM` HF standard → `vllm serve` direct
  (test de parité P0 + voie hybride de repli).

### 4.7 Tool calling : aucun changement côté vLLM-Omni

L'orchestrateur de ce repo consomme l'API streaming (deltas texte + chunks
audio) : `StreamingToolCallParser` détecte `<|tool_call_start|>…<|tool_call_end|>`
dans le canal texte, exécute via `ToolRegistry`, re-soumet la conversation avec
le tour `tool` ajouté. Avec le prefix caching, seule la réinjection est
préfillée.

## 5. Plan d'implémentation

| Phase | Contenu | Statut / critère de sortie |
|---|---|---|
| **P0** (code fait) | `export_checkpoint.py` + remapping testé + parité backbone (`tests/test_backbone_parity.py`, marqué `gpu`) | **à exécuter sur GPU** : top-10 logits et greedy identiques liquid-audio ↔ HF/vLLM |
| **P1** (fait) | plugin : registry + pipeline (`register()`, entry point), config conversion, `load_weights` réparti | testé sans GPU (conversion, layout) |
| **P2** (code fait) | stage 0 : placeholders + machine à états + depthformer + mel mm | machine à états **vérifiée par propriété** vs transcription amont ; **parité greedy GPU à exécuter** (`tests/test_omni_parity.py`) — bloquant |
| **P3** (code fait) | stage 1 détokeniseur + chunking async_chunk + `configs/vllm_omni_lfm2_audio.yaml` | parité waveform (test GPU prêt) + TTFA à mesurer |
| **P4** (à faire) | backend `vllm_omni` dans l'orchestrateur + benchs (TTFA, RTF, round-trip ±prefix-cache, multi-sessions) + non-régression `eval_toolcalling` | seuils méthodo : <500 ms perçu, round-trip <1,5 s, FC accuracy = liquid-audio |

## 6. Risques

| Risque | Mitigation |
|---|---|
| Flux mixte texte+codes dans un seul stage | précédent in-tree MiMo-Audio (même topologie) ; le forçage du placeholder via `sample()` custom reste dans le contrat « 1 id/step » (idiome cosyvoice3) |
| Plomberie runtime (mm processor, hooks du runner) à ajuster sur la version exacte de vllm-omni | écarts majeurs déjà identifiés par audit du wheel (§3bis) ; le reste se révèle au smoke Colab (`scripts/colab_smoke_vllm_omni.py`), localisé dans `lfm2_audio_ar.py` |
| Clé d'identité requête pour le cache d'embedding de frame (§3bis.3) | 3 pistes hiérarchisées ; la variante vocab unifié supprime le cache si le canal `additional_information` ne convient pas |
| Embedding de frame perdu en préemption | reconstruit au prefill depuis les codes (mm data), déterministe |
| Divergence ratio entraînement/serving | ratio lu uniquement depuis le config exporté |
| Échec de parité P2 | rester sur liquid-audio en S2S ; la voie hybride vLLM (backbone texte, validée P0) reste le plan B d'optimisation |
| Divergences numériques bf16 (conv cache vLLM vs transformers) | parité en top-k + tolérance, pas en bit-exact ; greedy court comme juge de paix |
