"""CLI Phase 2 : prépare le dataset SFT au format disque liquid-audio.

Convertit un JSONL de dialogues (cf. ``dialogue_schema``) en dataset HF
pré-packé via ``LFM2AudioChatMapper`` (tokens texte + mels + codes Mimi +
masques de supervision), prêt pour ``LFM2DataLoader``.

Usage :

    python -m lfm2_audio.data_prep.preprocess_sft \\
        --dialogues data/dialogues_train.jsonl \\
        --audio-root data/audio \\
        --output datasets/sft_train \\
        --interleaved-text-tokens 6 --interleaved-audio-tokens 10

Le ratio interleaved FR se calibre avec ``lfm2-calibrate``
(recette du variant japonais : 6:9 ; anglais : 6:12).
"""

from __future__ import annotations
