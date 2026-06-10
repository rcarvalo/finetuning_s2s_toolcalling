# Spec : intégration de LFM2.5-Audio (fine-tuné FR + tool calling) dans vLLM-Omni

> Statut : design validé, implémentation à faire dans un fork de
> `vllm-project/vllm-omni` (branche `lfm2-audio`). État vérifié en juin 2026
> (vLLM-Omni v0.22.0). Cette spec sert de référence au fork et à une
> éventuelle PR upstream.

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
  Higgs-Audio v2, …).
- **Le backbone Lfm2 est déjà implémenté dans vLLM core**
  (`vllm/model_executor/models/lfm2.py`, cache conv hybride) → réutilisable.
- **Pas de registre de modèles out-of-tree** documenté pour vllm-omni → fork.
- Guides officiels : `docs/contributing/model/adding_omni_model.md` et
  `adding_tts_model.md` (hook `sample()` custom, framework `async_chunk`).
- **Précédent exact** : le talker de Qwen3-Omni
  (`vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_talker.py` +
  `qwen3_omni_moe_code_predictor_mtp.py`) génère des codes codec
  multi-codebooks dans la boucle AR de vLLM.

## 3. Le problème central et sa solution

**Problème** : vLLM échantillonne UN token id par step ; une frame audio
LFM2.5-Audio = **8 codes Mimi** produits par le depthformer, et le flux
interleave texte et audio dans la même séquence (ratio `n_text:n_audio`).

**Solution (mécanisme du talker Qwen3-Omni, vérifié dans le code)** :

1. **Le code du codebook 0 est LE token échantillonné** — `compute_logits`
   projette le hidden state sur la tête codec (`codec_head`), sampling vLLM
   standard.
2. **Les 7 codes résiduels sont calculés au step suivant**, dans
   `embed_input_ids()` : quand l'id entrant est un code codebook-0, on roule le
   code predictor (notre **depthformer**, 8 steps locaux sans KV paginé) avec
   l'embedding du code 0 + le dernier hidden state → codes résiduels +
   **embedding d'entrée = somme des embeddings des 8 codebooks**
   (`proj_buf[:, 1:, :].sum(dim=1)` chez Qwen3-Omni ≡
   `audio_embedding(frame + offsets).sum(0)` dans liquid-audio).
3. Les codes complets sont exportés au stage suivant via
   `OmniOutput.multimodal_outputs`.

## 4. Design LFM2.5-Audio

### 4.1 Vocabulaire unifié

```
ids texte   : [0 .. 65535]                  (vocab LFM2.5, inclut les tokens d'outils)
ids codec0  : [65536 .. 65536+2048]         (2049 codes Mimi, 2048 = fin d'audio)
```

La modalité d'un id se lit dans sa plage → le flux interleaved tient dans une
séquence vLLM standard (KV cache, prefix caching, préemption inchangés).

### 4.2 Tête de sortie unifiée

`compute_logits` = concat de :
- logits texte : `linear(hidden, embed_tokens.weight)` (embeddings liés, comme
  liquid-audio `generate_interleaved`) ;
- logits codec0 : tête codebook-0 du depthformer (`depth_linear` +
  `depth_embeddings[0]`).

### 4.3 `sample()` : machine à états de modalité

Hook custom prévu par vllm-omni (cf. guide TTS). Masque −inf sur le segment
inactif du vocab unifié selon l'état :

```
état TEXT  : n = interleaved_n_text tokens texte max, puis bascule AUDIO ;
             <|text_end|> (id 130) → text_done=True (audio seul ensuite) ;
             <|im_end|>  (id 7)   → stop (tour terminé — c'est ici que se
             terminent les tours de TOOL CALL, texte seul, jamais d'audio)
état AUDIO : n = interleaved_n_audio frames, puis retour TEXT (si !text_done) ;
             code 2048 (fin d'audio) → retour TEXT
```

**L'état est une fonction pure du suffixe de tokens du tour courant**
(le ratio est déterministe) → robuste à la préemption/recompute du scheduler et
au prefix caching ; pas d'état de session fragile. Le ratio
(`interleaved_n_text/n_audio`) est lu depuis le `config.json` du checkpoint
exporté — **source unique** entraînement/serving (cf. `export_checkpoint.py`).

Seul état par requête : `{request_id: last_hidden_state}` pour le rollout du
depthformer (équivalent `last_talker_hidden` chez Qwen3-Omni). Perdu en
préemption → recompute au prefill (déterministe, coût marginal).

### 4.4 Entrées audio

- **Micro visiteur** : mel-128 (préprocesseur identique à liquid-audio) →
  encodeur FastConformer + `audio_adapter` exposés via `SupportsMultiModal`,
  placeholders dans le prompt — pattern standard vLLM (Qwen2-Audio/Ultravox).
- **Audio généré aux tours précédents** (rounds d'outils, multi-tours) : les
  ids codec0 offsetés restent dans `prompt_token_ids` ; les frames 8-codebooks
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

Fichiers du fork :

```
vllm_omni/model_executor/models/lfm2_audio/
├── __init__.py
├── lfm2_audio.py            # stage 0 (réutilise Lfm2Model de vLLM core)
└── lfm2_audio_code2wav.py   # stage 1
vllm_omni/model_executor/stage_input_processors/lfm2_audio.py
vllm_omni/model_executor/stage_configs/lfm2_audio.yaml
vllm_omni/model_executor/models/registry.py   # + entrée _OMNI_MODELS
```

`lfm2_audio.yaml` : stage 0 (`model_stage: ar_interleaved`,
`final_output: false`) → stage 1 (`engine_input_source: [0]`,
`custom_process_input_func: lfm2_audio_codes_to_code2wav`,
`final_output_type: audio`).

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

| Phase | Contenu | Critère de sortie |
|---|---|---|
| **P0** (ce repo, fait) | `export_checkpoint.py` + remapping testé + test de parité backbone (`tests/test_backbone_parity.py`, marqué `gpu`) | top-10 logits et greedy identiques liquid-audio ↔ HF/vLLM sur prompts outillés |
| **P1** (fork) | squelette package + config class + `load_weights` + registry | modèle chargé, prefill texte OK |
| **P2** (fork) | tête unifiée + `sample()` état de modalité + depthformer dans `embed_input_ids` + mel mm | **parité greedy interleaved** : tokens texte ET codes audio identiques à `liquid_audio.generate_interleaved` sur N dialogues d'éval — bloquant |
| **P3** (fork) | stage 1 async_chunk + yaml | parité waveform (tolérance) + TTFA mesuré |
| **P4** (ce repo) | backend `vllm_omni` dans l'orchestrateur + benchs (TTFA, RTF, round-trip ±prefix-cache, multi-sessions) + non-régression `eval_toolcalling` | seuils méthodo : <500 ms perçu, round-trip <1,5 s, FC accuracy = liquid-audio |

## 6. Risques

| Risque | Mitigation |
|---|---|
| Flux mixte texte+codes dans un seul stage (les talkers existants n'émettent que des codes) | tête unifiée + masquage dans `sample()` restent dans le contrat vLLM « 1 id/step » ; en dernier recours, patch du gpu_model_runner (on est en fork) |
| `last_hidden_state` perdu en préemption | recompute au prefill, déterministe |
| Divergence ratio entraînement/serving | ratio lu uniquement depuis le config exporté |
| Échec de parité P2 | rester sur liquid-audio en S2S ; la voie hybride vLLM (backbone texte, validée P0) reste le plan B d'optimisation |
| Divergences numériques bf16 (conv cache vLLM vs transformers) | parité en top-k + tolérance, pas en bit-exact ; greedy court comme juge de paix |
