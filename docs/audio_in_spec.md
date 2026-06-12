# Spec : audio-in natif vLLM (action 3 de l'audit — le « vrai SOTA »)

> Objectif : supprimer l'ASR liquid en série (~1 s/tour). Le modèle « entend »
> l'audio nativement : latence fin-de-parole → premier son ≈ prefill + 336 ms.

## Contrat établi (source liquid-audio 1.3.0, vérifié)

- `ChatState.add_audio(wave, sr)` : resample 16 kHz →
  `AudioToMelSpectrogramPreprocessor` (config = `config.json["preprocessor"]`
  du checkpoint) → mel `(128, T_mel)`.
- Positions d'embedding LFM occupées par l'audio : `mel2emb_len(T_mel) =
  ceil(T_mel / 8)` (downsampling ×8 du conformer ; min 9 frames mel).
- Routage par `modality_flag = LFMModality.AUDIO_IN` sur ces positions ; le
  modèle remplace ces embeddings par `audio_adapter(conformer(mel))`.
- Notre stage 0 (`lfm2_audio_ar.py`) charge DÉJÀ `self.conformer` +
  `self.audio_adapter` avec les poids du checkpoint.

## Implémentation (pattern in-tree : mimo_audio_llm.py, hérite Qwen2-Audio)

1. `src/vllm_omni_lfm2_audio/multimodal.py` :
   - `Lfm2AudioProcessingInfo` : limites (1+ audio / prompt), nombre de
     placeholders = `ceil(T_mel/8)` ;
   - `Lfm2AudioDummyInputsBuilder` (profiling) ;
   - `Lfm2AudioMultiModalProcessor` : audio brut → mel (préprocesseur mel
     PORTÉ en numpy/torch CPU depuis la config `preprocessor` du checkpoint —
     ne pas dépendre de liquid_audio au runtime serveur) ; insère
     `ceil(T_mel/8)` tokens placeholder dans le prompt.
   - Placeholder : RÉUTILISER `audio_frame_token_id` (128) côté prompt est
     ambigu avec l'audio-out → préférer un id dédié lu du config
     (`audio_in_token_id`, à défaut 128 documenté). Vérifier avec
     `verify_placeholder_ids`.
2. `lfm2_audio.py` (wrapper) : `@MULTIMODAL_REGISTRY.register_processor(...)`,
   interface `SupportsMultiModal`, `get_multimodal_embeddings(**mm_kwargs)` →
   `audio_adapter(conformer(mel, lens))`, `get_input_embeddings(...)` → merge
   standard vLLM aux positions placeholder. Délègue au stage AR.
3. Prefill vs notre `preprocess()` : le merge mm standard arrive AVANT le
   preprocess par requête (pattern MiMo, même runner gpu_ar_model_runner) ;
   notre overlay de frames générées ne touche que les spans décode
   (`span_len == 1`) → pas de conflit attendu, à vérifier par test.
4. Parité : test GPU `tests/test_audio_in_parity.py` — mêmes mel → embeddings
   conformer identiques à liquid (`LFM2AudioModel`), et E2E : WAV question →
   réponse cohérente (vs pipeline liquid ASR).
5. Démos : `s2s_demo.py --backend vllm --audio-in` (lever le
   NotImplementedError) ; démo Gradio Colab micro → vLLM direct (un seul
   engine, plus de liquid co-chargé, ~3 Go de VRAM libérés).

## Points de vigilance

- Le préprocesseur mel (NeMo-style : window 25 ms / stride 10 ms, 128 mels,
  normalize per-feature — lire la config exacte du checkpoint) doit être
  reproduit à l'identique : c'est LE risque de parité.
- `audio_in_lens` : le conformer masque par longueur — transmettre les
  longueurs mel réelles dans mm_kwargs.
- Encoder cache vLLM : hash mm stable par contenu → les tours passés ne sont
  pas ré-encodés (bonus latence multi-tours).
- Chunked prefill : les placeholders d'un même audio doivent rester dans le
  même chunk ou être gérés par l'encoder cache standard (vLLM s'en charge si
  on passe par `SupportsMultiModal` — ne PAS bricoler).
