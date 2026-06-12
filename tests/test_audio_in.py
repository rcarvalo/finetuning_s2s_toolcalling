"""Audio-in natif vLLM : math des placeholders + extraction mel (parité liquid).

Les imports vllm/liquid_audio ne sont dispo que sur Colab/pod — skip local.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_omni_lfm2_audio.multimodal import (  # noqa: E402
    MIN_MEL_LEN,
    audio_in_token_id,
    extract_mel,
    mel2emb_len,
)


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_mel2emb_len_is_ceil_div_8():
    # contrat liquid utils.mel2emb_len : -(l // -8)
    assert mel2emb_len(8) == 1
    assert mel2emb_len(9) == 2
    assert mel2emb_len(16) == 2
    assert mel2emb_len(17) == 3
    assert mel2emb_len(100) == 13


def test_audio_in_token_id_default_and_override():
    from vllm_omni_lfm2_audio.constants import AUDIO_FRAME_PLACEHOLDER_ID

    assert audio_in_token_id(_Cfg()) == AUDIO_FRAME_PLACEHOLDER_ID
    assert audio_in_token_id(_Cfg(audio_in_token_id=512)) == 512


@pytest.mark.parametrize("seconds,approx_emb", [(1.0, 13), (2.0, 25)])
def test_extract_mel_shapes(seconds, approx_emb):
    liquid_audio = pytest.importorskip("liquid_audio")  # noqa: F841
    # config préprocesseur du checkpoint LFM2.5-Audio (NeMo-style)
    cfg = _Cfg(preprocessor={
        "sample_rate": 16_000, "normalize": "per_feature", "window_size": 0.025,
        "window_stride": 0.01, "window": "hann", "features": 128, "n_fft": 512,
        "log": True, "frame_splicing": 1, "dither": 1e-5, "pad_to": 0, "pad_value": 0.0,
    })
    wave = torch.randn(int(16_000 * seconds))
    mel = extract_mel(cfg, wave)
    assert mel.shape[0] == 128
    # stride 10 ms → ~100 frames/s ; embeddings = ceil(T/8)
    assert abs(mel2emb_len(mel.shape[1]) - approx_emb) <= 1


def test_extract_mel_too_short_raises():
    pytest.importorskip("liquid_audio")
    cfg = _Cfg(preprocessor={
        "sample_rate": 16_000, "normalize": "per_feature", "window_size": 0.025,
        "window_stride": 0.01, "window": "hann", "features": 128, "n_fft": 512,
        "log": True, "frame_splicing": 1, "dither": 1e-5, "pad_to": 0, "pad_value": 0.0,
    })
    with pytest.raises(ValueError, match="trop court"):
        extract_mel(cfg, torch.randn(MIN_MEL_LEN * 40))  # << 9 frames mel


def test_data_parser_hook_resamples_to_16k():
    """Le hook surchargé doit être celui que vLLM consulte réellement.

    vLLM 0.22.1 construit le parser via ``BaseProcessingInfo.get_data_parser``
    (renderer ``info.parse_mm_data`` + ``BaseMultiModalProcessor.__init__``) ;
    un mauvais nom de hook → parser par défaut sans ``target_sr`` →
    RuntimeError « Audio resampling is not supported » au premier audio.
    """
    from vllm.multimodal.processing import BaseProcessingInfo

    from vllm_omni_lfm2_audio.multimodal import (
        AUDIO_IN_SAMPLE_RATE,
        Lfm2AudioProcessingInfo,
    )

    assert hasattr(BaseProcessingInfo, "get_data_parser")
    assert "get_data_parser" in Lfm2AudioProcessingInfo.__dict__

    info = object.__new__(Lfm2AudioProcessingInfo)  # le hook n'utilise pas ctx
    parser = info.get_data_parser()
    assert parser.audio_resampler.target_sr == AUDIO_IN_SAMPLE_RATE
