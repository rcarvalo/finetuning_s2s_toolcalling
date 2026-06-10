"""Parité de la machine à états interleaved (plugin vLLM-Omni) contre une
transcription littérale de la boucle ``generate_interleaved`` de liquid-audio.

C'est le dérisquage sans-GPU du cœur de la phase P2 : si ``replay`` diverge de
la référence, les steps audio/texte du serving seraient désalignés vs
l'entraînement.
"""

import random

import pytest

from vllm_omni_lfm2_audio.constants import (
    AUDIO_EOA_PLACEHOLDER_ID,
    AUDIO_FRAME_PLACEHOLDER_ID,
    IM_END_TOKEN_ID,
    TEXT_END_TOKEN_ID,
)
from vllm_omni_lfm2_audio.modality import (
    Modality,
    ModalityConfig,
    expected_modalities,
    next_step_modality,
    replay,
)

TXT = 1000  # id texte ordinaire


def reference_generate(text_script, eoa_steps, *, n_text, n_audio, max_tokens=400):
    """Transcription littérale de la boucle de
    ``liquid_audio.model.lfm2_audio.LFM2AudioModel.generate_interleaved``.

    ``text_script`` : tokens que « le modèle » émettrait aux steps TEXT ;
    ``eoa_steps``   : numéros (1-based) des steps AUDIO produisant une frame EOA.
    Retourne [(modalité, id)] des tokens émis (placeholders pour l'audio).
    """
    current, left, text_done = "text", n_text, False
    emitted, ti, audio_step = [], 0, 0

    for _ in range(max_tokens):
        left -= 1
        if current == "text":
            tok = text_script[ti] if ti < len(text_script) else IM_END_TOKEN_ID
            ti += 1
            if tok == IM_END_TOKEN_ID:
                break
            emitted.append(("text", tok))
            if tok == TEXT_END_TOKEN_ID:
                text_done = True
            if not left or text_done:
                current, left = "audio", n_audio
        else:
            audio_step += 1
            is_eoa = audio_step in eoa_steps
            emitted.append(("audio", AUDIO_EOA_PLACEHOLDER_ID if is_eoa else AUDIO_FRAME_PLACEHOLDER_ID))
            if not left and not text_done:
                current, left = "text", n_text
            if is_eoa:
                current = "text"  # quirk amont : left non réinitialisé
    return emitted


def check_against_reference(text_script, eoa_steps, n_text, n_audio):
    cfg = ModalityConfig(n_text=n_text, n_audio=n_audio)
    emitted = reference_generate(text_script, eoa_steps, n_text=n_text, n_audio=n_audio)
    ids = [tok for _, tok in emitted]
    ref_modalities = [Modality(m) for m, _ in emitted]
    assert expected_modalities(ids, cfg) == ref_modalities
    return ids, cfg, emitted


def test_simple_interleaving_en_ratio():
    # 6 tokens texte → bascule audio → 12 frames → retour texte
    script = [TXT] * 20 + [TEXT_END_TOKEN_ID]
    ids, cfg, emitted = check_against_reference(script, eoa_steps=set(), n_text=6, n_audio=12)
    kinds = [m for m, _ in emitted]
    assert kinds[:6] == ["text"] * 6
    assert kinds[6:18] == ["audio"] * 12
    assert kinds[18:24] == ["text"] * 6


def test_text_end_triggers_audio_early():
    # <|text_end|> au 3e token : l'audio commence aussitôt, sans retour texte
    script = [TXT, TXT, TEXT_END_TOKEN_ID]
    ids, cfg, emitted = check_against_reference(script, eoa_steps=set(), n_text=6, n_audio=12)
    kinds = [m for m, _ in emitted]
    assert kinds[:3] == ["text"] * 3
    assert all(k == "audio" for k in kinds[3:])


def test_eoa_returns_to_text_mid_block():
    # EOA à la 5e frame (avant la fin du bloc de 12) → retour texte immédiat
    script = [TXT] * 6 + [TXT] * 6 + [TEXT_END_TOKEN_ID]
    ids, cfg, emitted = check_against_reference(script, eoa_steps={5}, n_text=6, n_audio=12)
    kinds = [m for m, _ in emitted]
    assert kinds[6:11] == ["audio"] * 5  # 4 frames + EOA
    assert kinds[11] == "text"


def test_jp_ratio_6_9():
    script = [TXT] * 30
    check_against_reference(script, eoa_steps={20}, n_text=6, n_audio=9)


def test_next_step_modality_empty_turn_is_text():
    cfg = ModalityConfig()
    assert next_step_modality([], cfg) is Modality.TEXT


def test_finished_on_im_end():
    cfg = ModalityConfig(n_text=6, n_audio=12)
    state = replay([TXT, TXT, IM_END_TOKEN_ID], cfg)
    assert state.finished


def test_tool_call_turn_stays_text():
    # un tour de tool call : que du texte puis im_end avant épuisement du budget
    cfg = ModalityConfig(n_text=6, n_audio=12)
    ids = [TXT, TXT, TXT, TXT]  # 4 < 6 tokens
    assert next_step_modality(ids, cfg) is Modality.TEXT


def test_inconsistent_stream_raises():
    cfg = ModalityConfig(n_text=6, n_audio=12)
    with pytest.raises(ValueError, match="TEXT step"):
        replay([AUDIO_FRAME_PLACEHOLDER_ID], cfg)  # frame pendant un step texte
    with pytest.raises(ValueError, match="AUDIO step"):
        replay([TXT] * 6 + [TXT], cfg)  # texte pendant un step audio


@pytest.mark.parametrize("seed", range(20))
def test_property_random_schedules(seed):
    rng = random.Random(seed)
    n_text = rng.choice([2, 4, 6, 8])
    n_audio = rng.choice([3, 6, 9, 12])
    script = []
    for _ in range(rng.randint(1, 40)):
        r = rng.random()
        if r < 0.08:
            script.append(TEXT_END_TOKEN_ID)
        elif r < 0.12:
            script.append(IM_END_TOKEN_ID)
        else:
            script.append(rng.randint(200, 60000))
    eoa_steps = {rng.randint(1, 60) for _ in range(rng.randint(0, 3))}
    check_against_reference(script, eoa_steps, n_text, n_audio)


def test_replay_is_pure_prefix_consistent():
    # rejouer un préfixe puis avancer == rejouer le tout (robustesse préemption)
    from vllm_omni_lfm2_audio.modality import advance

    cfg = ModalityConfig(n_text=6, n_audio=12)
    script = [TXT] * 8 + [TEXT_END_TOKEN_ID]
    ids = [tok for _, tok in reference_generate(script, {7}, n_text=6, n_audio=12)]
    for cut in range(len(ids)):
        state = replay(ids[:cut], cfg)
        for tok in ids[cut:]:
            state = advance(state, tok, cfg)
        assert state == replay(ids, cfg)
