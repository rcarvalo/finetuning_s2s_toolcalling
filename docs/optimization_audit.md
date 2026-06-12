# Audit d'optimisation : LFM2.5-Audio × vLLM-Omni, composant par composant

> Audit du 12 juin 2026, après le jalon E2E (commit `1ccecb6`). Sources :
> inspection du wheel `vllm-omni==0.22.0` (configs de déploiement in-tree,
> implémentations MiMo-Audio/Qwen3-TTS/Fish/MOSS), issues/PR vLLM upstream,
> blog PyTorch « Hybrid Models as First-Class Citizens in vLLM », papier
> vLLM-Omni (arXiv 2602.02204). Chaque partie de l'architecture LFM2.5-Audio
> est mise en face de son **exemplaire** dans l'écosystème.

## 0. Verdict d'ensemble

Le constat « vLLM pas plus rapide » a trois causes de natures différentes :

1. **Mesure** : on a benché batch=1, tour unique, historique texte-seul —
   le seul scénario où vLLM ne peut PAS gagner.
2. **Notre intégration** : 6 freins identifiés, tous corrigeables, avec un
   exemplaire in-tree pour chacun (tableau §1).
3. **Limitations upstream réelles** (§2) : le prefix caching des modèles
   hybrides (le backbone Lfm2 a 10 couches ShortConv) est expérimental et
   inopérant sous ~528 tokens de préfixe — une partie du gain promis est
   aujourd'hui conditionnelle. À l'inverse, CUDA graphs hybrides et
   concurrence sont matures.

## 1. Audit par composant

### 1.1 Backbone Lfm2 (10 ShortConv + 6 GQA, 1,2B) — stage 0

| Constat actuel | Exemplaire | Action | Gain attendu |
|---|---|---|---|
| `enforce_eager: true` (contournement smoke, commit `a29b87a`) → chaque step paie l'overhead Python/launch ; à 12,5 steps/s audio + texte c'est LE coût dominant d'un 1,5B | `deploy/fish_qwen3_omni_high_concurrency_single_gpu.yaml` : **`enforce_eager: false` sur le stage 0 AR en prod** ; vLLM upstream : full CUDA graphs **décode par défaut pour les modèles mamba/hybrides** (PR #22594, cap des tailles PR #34571) | Lever `enforce_eager` stage 0. Nos hooks (`preprocess`/`sample`) vivent HORS du graph (runner) → compatibles ; à valider au 1er run, sinon `cudagraph_mode: PIECEWISE` | **×1,5–3 sur le décode** (petit modèle = dominé par l'overhead de lancement) — le levier n°1 |
| `async_scheduling: false` partout (commit `d64d308` : replay tronqué d'un token in-flight) | `deploy/moss_tts_realtime.yaml`, `qwen3_tts*.yaml` : **`async_scheduling: true` AVEC sampler custom** (leur état est positionnel par requête, pas rejoué du texte) | Compenser le token in-flight : pour une ligne en bloc AUDIO le token manquant est le placeholder déjà décidé (déductible de l'état) ; n'activer la compensation que là. Étudier comment MOSS gère son delay-pattern par requête | ~1 step de pipeline masqué par step (5-15 ms/step) |
| Prefill re-tokenise/re-encode tout l'historique à chaque tour | §2 (limitation hybride) + `vllm-omni` issue #1184 (hidden-state I/O multi-round) | Voir §2 — et en attendant : **garder le system prompt + tool list > 528 tokens** pour franchir le 1er bloc cachable | conditionnel |

### 1.2 Encodeur Conformer (audio-in, ~115M) — pas encore branché

| Constat actuel | Exemplaire | Action | Gain attendu |
|---|---|---|---|
| **Audio-in non câblé dans vLLM** → ASR via liquid-audio EN SÉRIE avant chaque génération (~1 s/tour, docstring `s2s_demo.py`) | `mimo_audio_llm.py` : `MimoAudioProcessingInfo` + `DummyInputsBuilder` + `MultiModalProcessor` (hérite des classes Qwen2-Audio de vLLM core) ; placeholders dans le prompt | Enregistrer le processor multimodal et brancher `embed_multimodal` (conformer+adapter déjà chargés et pondérés dans `lfm2_audio_ar.py`) | **−~1 s par tour** (suppression d'une étape série) — le levier n°2 |
| (après câblage) le conformer re-encodera les audios des tours PASSÉS à chaque tour | vLLM V1 : **encoder cache + mm hashing** (réutilisation des embeddings mm d'une requête à l'autre) ; papier vLLM-Omni : « multimodal embedding cache » | S'assurer que les items audio passés gardent un hash mm stable (mêmes bytes) pour des hits d'encoder cache ; sinon ne réinjecter que le texte des vieux tours (déjà le cas dans les démos) | évite ~100-300 ms/tour aux longues sessions |

### 1.3 Depthformer (8 codebooks AR par frame audio) — dans `sample()`

| Constat actuel | Exemplaire | Action | Gain attendu |
|---|---|---|---|
| Rollout eager : 8 passes séquentielles par step audio (batché sur les lignes depuis `1ccecb6`, mais ~16 launchs/frame en Python) | **`mimo_audio/cuda_graph_decoder_wrapper.py`** : capture CUDA graph du décodeur local par buckets de batch (`MIMO_CUDAGRAPH_BATCH_SIZES`, buffers pré-alloués, `MiMoLocalSamplerTensor` = sampling DANS le graph) ; idem `qwen3_tts/cuda_graph_decoder_wrapper.py` | Wrapper équivalent pour `audio_head.sample_frames` : buffers statiques (hidden, temperature/top-k), capture par bucket de batch, replay | 8 launchs → 1 replay par frame : **~5-15 ms/frame** soit jusqu'à ~150 ms/s d'audio généré |
| Sampling audio greedy par défaut (parité) ; MiMo documente pourquoi : un local sampler stochastique déstabilise le cache d'embeddings réinjectés | commentaire in-tree `mimo_audio_llm.py` (« voice diversity must be tackled in the codec/vocoder path ») | Garder greedy par défaut ; si la prosodie FR du fine-tuné l'exige, réactiver temp/top-k APRÈS la parité et mesurer l'EOA (risque de sur-génération déjà observé : 16,9 s pour une phrase courte, commit `89a7934`) | qualité, pas latence |

### 1.4 Détokeniseur LFM2 (stage 1, code2wav)

| Constat actuel | Exemplaire | Action | Gain attendu |
|---|---|---|---|
| **float32 + eager** (contournements `torch.polar`∄Half + dummy-run, commit `7f889e5`/`a29b87a`) | `qwen3_tts` : **`decode_cudagraph_capture_sizes: [25, 73, 97, …]` + `decode_cudagraph_batch_sizes`** (capture aux longueurs de chunk exactes) via son `cuda_graph_decoder_wrapper.py` | Pré-calculer le rotary en float32 puis caster (lever le verrou Half) ; capture CUDA graph aux tailles de chunk fixes (nos chunks sont déterministes : initial + steady) | ×2-4 sur le décodage vocodeur ; libère du GPU pour le stage 0 (co-localisé) |
| `codec_chunk_frames: 10` (800 ms d'accumulation) + `left_context: 13` → 1er chunk décode 23 frames pour en émettre 10 (2,3× de calcul) et **TTFA plancher ≈ 800 ms** | **`initial_codec_chunk_frames: 1`** (qwen3_tts), **`4`** (fish) : 1er chunk émis très tôt puis chunks pleins ; `chunk_size_utils.compute_dynamic_initial_chunk_size` (IC adaptatif à la charge) ; `moss_tts_realtime` : `codec_chunk_frames: 15`, `connector_get_max_wait_first_chunk: 1000` | Implémenter `initial_codec_chunk_frames` dans notre `ar2code2wav_async_chunk` (lecture du même champ connector) : 1er chunk à 3-4 frames, left_context 0 au 1er chunk, steady 10-15 | **TTFA audio : ~800 ms → ~250-350 ms** — le levier n°3 |
| Header magic `LEFT_CONTEXT_HEADER_MAGIC` dans le tenseur (contournement du meta non transmis) | — (bug/manque vllm-omni) | Garder ; ouvrir une issue upstream (meta de `OmniPayloadStruct` non propagé au forward du stage suivant) | robustesse |

### 1.5 Orchestration des stages / connector

| Constat actuel | Exemplaire | Action | Gain attendu |
|---|---|---|---|
| `SharedMemoryConnector` avec `connector_get_sleep_s: 0.001` (polling 1 ms) ; 2 stages co-localisés (0.42/0.42 VRAM) | `deploy/qwen3_omni_moe_multi_replicas.yaml` (`num_replicas: 2`), `fish_qwen3_omni_2gpu.yaml` / `*_high_concurrency_dual_gpu.yaml` (stages sur GPU distincts) ; papier vLLM-Omni : disaggregation EPD | Mono-GPU : rien à changer (polling 1 ms ok). Multi-robots : stage 1 sur 2e GPU ou réplique du stage 0 | scaling, pas latence mono-session |
| `max_num_seqs: 4` (stage 0) | `qwen3_tts_high_concurrency.yaml` : stage 0 à 64, stage 1 à 10 (asymétrique) | Dimensionner après bench concurrence (robots réels) | débit |

### 1.6 Sampler custom / machine à états (Python pur)

| Constat actuel | Exemplaire | Action | Gain attendu |
|---|---|---|---|
| `replay()` re-déroule TOUT le tour à chaque step (O(n²) sur la longueur du tour) — mesurable via `LFM2_DEBUG_TIMING` | les samplers in-tree gardent un état incrémental par requête (MOSS : compteurs positionnels) | Cache incrémental `{req_id: (n_tokens_vus, ModalityState)}` : `advance()` des seuls nouveaux tokens, invalidation si l'historique rétrécit (préemption) → O(1)/step amorti, la pureté de `replay` reste la référence/garde-fou | ~1-5 ms/step sur les longs tours |

### 1.7 Couche serving / client (WebRTC)

| Constat actuel | Exemplaire | Action | Gain attendu |
|---|---|---|---|
| Démos en `Omni()` in-process ; gateway fastrtc OK mais ASR série + audio livré en fin de tour en mode sync (« 45 frames · 3,6 s audio · généré en 4,0 s » = latence perçue 4 s) | `vllm-omni serve` (OpenAI-compatible, SSE streaming — `scripts/streaming_client.py` existe déjà) ; abort de requête pour le barge-in | Gateway WebRTC → `vllm-omni serve` HTTP streaming ; jouer les chunks dès réception (le mode async_chunk + initial chunk court rend ça utile) ; barge-in = abort + flush | latence PERÇUE : c'est elle qque l'utilisateur ressent |

## 2. Limitations upstream réelles (à tracer, pas de notre fait)

1. **Prefix caching hybride (ShortConv/Mamba)** : support « expérimental »
   (V1, unified allocator) avec **granularité de bloc ~528 tokens** — hit = 0
   sous cette taille (issue vllm#40696) ; bug ouvert sur les requêtes
   multimodales incrémentales multi-tours (vllm#43587 — exactement notre
   pattern) ; tracking vllm#26201. Conséquences pour nous :
   - le gain « tool round-trip sans re-prefill » est aujourd'hui PARTIEL
     (l'état conv ne se restaure pas en milieu de bloc) ;
   - mitigations : system prompt + tool list ≥ 528 tokens (1er bloc cachable),
     suivre `mamba_cache_mode align`, re-mesurer à chaque release vLLM ;
   - re-vérifier dans les logs serveur si l'APC est réellement actif pour
     Lfm2 (warning de désactivation silencieuse possible).
2. **`OmniPayloadStruct.meta` non propagé** au forward du stage suivant
   (notre header magic) — issue à ouvrir.
3. **`OmniOutput.next_token_id` non consommé** par le runtime (déjà documenté
   §3bis de la spec).

## 3. Plan d'action priorisé

**Objectif produit : TTFA 200-500 ms (parité ElevenLabs).** Budget visé :
prefill (~30-80 ms) + ~8 steps avant la 1re frame audio (interleave 6:12)
+ 2 frames + détokenisation du chunk initial. En eager ≈ 400-500 ms ; avec
CUDA graphs stage 0 ≈ 250-350 ms. Vérification : `scripts/bench_ttfa.py`
(TTFT/TTFA/RTF, p50/p95, verdict vs objectif).

| # | Action | Effort | Impact latence |
|---|---|---|---|
| 1 | ✅ `initial_codec_chunk_frames: 2` (implémenté dans `ar2code2wav_async_chunk` + YAML, tests `test_chunk_streaming.py`) | S | TTFA −450-550 ms |
| 2 | ✅ `enforce_eager: false` stage 0 dans le YAML (CUDA graphs décode hybride) — à valider au 1er run GPU | S (test) | décode ×1,5-3 |
| 3 | Câbler l'audio-in vLLM (processor mm, pattern MiMo) | M | −~1 s/tour |
| 4 | Détokeniseur fp16/bf16 + CUDA graph aux tailles de chunk | M | stage 1 ×2-4 |
| 5 | Wrapper CUDA graph du depthformer (pattern MiMo) | M | −5-15 ms/frame |
| 6 | Async scheduling avec compensation in-flight | M | −5-15 ms/step |
| 7 | État de modalité incrémental (O(1)/step) | S | longs tours |
| 8 | Bench A/B honnête (froid/chaud/concurrence/historique complet) | S | décision |
| 9 | Vérifier l'APC Lfm2 dans les logs + préfixe ≥ 528 tokens | S | conditionnel |

Ordre recommandé : 1-2 (quick wins mesurables immédiatement), 8 (re-mesurer),
3 (architecture), 4-5-6-7, puis re-bench.

## 4. Ce que vLLM-Omni apporte VRAIMENT (attentes recalibrées)

- ✅ **Concurrence** (continuous batching multi-robots/sessions) — mature.
- ✅ **Streaming serveur** (SSE, chunks audio précoces) + infra (abort,
  métriques, réplicas, disaggregation multi-GPU).
- ✅ **CUDA graphs hybrides** sans travail de notre part (vs liquid-audio
  eager pur).
- ⚠️ **Prefix caching** : partiel sur backbone hybride aujourd'hui (cf. §2) —
  le gain « re-prefill nul » est à re-tester à chaque release upstream.
- ❌ **Décode batch=1 plus rapide qu'une boucle PyTorch serrée** : jamais
  promis par vLLM — sans CUDA graphs c'est même l'inverse (notre mesure).
