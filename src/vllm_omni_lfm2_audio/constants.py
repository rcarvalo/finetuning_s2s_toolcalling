"""Constantes du protocole LFM2.5-Audio (alignées sur liquid-audio).

Tokens spéciaux du tokenizer LFM2.5 (ids vérifiés dans
``liquid_audio.model.lfm2_audio.generate_interleaved``) et conventions du
plugin pour représenter les frames audio dans le flux d'ids vLLM.
"""

TEXT_VOCAB_SIZE = 65_536

# Ids texte spéciaux (constantes du code liquid-audio).
IM_END_TOKEN_ID = 7  # <|im_end|> — fin de tour (stop)
AUDIO_START_TOKEN_ID = 128  # <|audio_start|> — utilisé par le mode sequential
TEXT_END_TOKEN_ID = 130  # <|text_end|> — fin du texte en interleaved

# Codes Mimi.
CODEBOOKS = 8
AUDIO_VOCAB_SIZE = 2048 + 1  # +1 pour le code fin-d'audio
END_OF_AUDIO_CODE = 2048  # sur les 8 codebooks
MIMI_FRAME_RATE = 12.5  # frames/s
SAMPLES_PER_FRAME = 1920  # à 24 kHz

# --- Convention du plugin -------------------------------------------------- #
# Dans la séquence vLLM, chaque step audio est représenté par UN id placeholder
# (l'embedding réel — somme des 8 codebooks — est servi par le cache/mm data,
# jamais par embed_tokens). Deux ids distincts pour que la machine à états soit
# rejouable depuis la seule séquence d'ids (préemption / prefix caching) :
# - frame normale  : AUDIO_FRAME_PLACEHOLDER_ID
# - frame EOA      : AUDIO_EOA_PLACEHOLDER_ID (frame 2048×8, bascule → texte)
#
# Défauts : <|audio_start|> (128) et l'id suivant (129) — jamais émis comme
# vrais tokens en mode interleaved (réservés au mode sequential, non servi
# ici). Surchargables via le config (audio_frame_token_id / audio_eoa_token_id)
# si le checkpoint utilise ces positions différemment ; à valider au démarrage
# avec verify_placeholder_ids().
AUDIO_FRAME_PLACEHOLDER_ID = 128
AUDIO_EOA_PLACEHOLDER_ID = 129

# Ratio interleaved par défaut (anglais). Le ratio FR calibré DOIT venir du
# config.json du checkpoint exporté (interleaved_n_text / interleaved_n_audio).
DEFAULT_INTERLEAVED_N_TEXT = 6
DEFAULT_INTERLEAVED_N_AUDIO = 12


def verify_placeholder_ids(tokenizer, frame_id: int, eoa_id: int) -> list[str]:
    """Garde-fous au démarrage du serveur : les placeholders doivent être des
    tokens spéciaux existants (jamais produits par l'encodage de texte normal).

    Retourne la liste des problèmes détectés (vide si OK).
    """
    problems: list[str] = []
    for name, tid in (("audio_frame_token_id", frame_id), ("audio_eoa_token_id", eoa_id)):
        token = tokenizer.convert_ids_to_tokens(tid)
        if token is None:
            problems.append(f"{name}={tid} is not in the tokenizer vocabulary")
        elif not (token.startswith("<|") and token.endswith("|>")):
            problems.append(f"{name}={tid} maps to regular token {token!r} (must be a special token)")
    if frame_id == eoa_id:
        problems.append("audio_frame_token_id and audio_eoa_token_id must differ")
    return problems
