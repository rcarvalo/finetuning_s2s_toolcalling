"""Parité P2/P3 (GPU) : stage 0 du plugin vLLM-Omni vs liquid-audio de référence.

Critère bloquant du plan (docs/vllm_omni_integration.md) : en greedy texte +
audio déterministe, les tokens texte ET les codes Mimi doivent être identiques
à ``liquid_audio.generate_interleaved`` sur les prompts d'éval ; le waveform du
stage 1 doit coïncider avec le détokeniseur de référence (tolérance bf16).

Prérequis :
    python -m s2s_toolcalling.training.export_checkpoint --mode full ... --output exports/full
    python -m vllm_omni_lfm2_audio.convert_checkpoint --checkpoint exports/full \\
        --output exports/full_omni
    OMNI_CHECKPOINT=exports/full_omni BASE_MODEL=LiquidAI/LFM2.5-Audio-1.5B \\
        python -m pytest tests/test_omni_parity.py -m gpu -q
"""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("liquid_audio")
pytest.importorskip("vllm_omni")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"),
    pytest.mark.skipif("OMNI_CHECKPOINT" not in os.environ, reason="OMNI_CHECKPOINT not set"),
]


@pytest.fixture(scope="module")
def reference():
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

    base = os.environ.get("BASE_MODEL", os.environ["OMNI_CHECKPOINT"])
    model = LFM2AudioModel.from_pretrained(base, device="cuda").eval()
    proc = LFM2AudioProcessor.from_pretrained(base, device="cuda").eval()
    return model, proc


def _reference_interleaved_greedy(model, proc, prompt: str, max_new_tokens: int = 128):
    """Texte greedy + audio greedy via l'API de référence."""
    from liquid_audio import ChatState

    chat = ChatState(proc)
    chat.new_turn("system")
    chat.add_text("Réponds en texte et audio interleaved.")
    chat.end_turn()
    chat.new_turn("user")
    chat.add_text(prompt)
    chat.end_turn()
    chat.new_turn("assistant")

    text_tokens, frames = [], []
    with torch.no_grad():
        for t in model.generate_interleaved(
            **chat, max_new_tokens=max_new_tokens, text_temperature=None, audio_temperature=None
        ):
            if t.numel() == 1:
                text_tokens.append(int(t.item()))
            else:
                frames.append(t.cpu())
    return text_tokens, frames


def test_stage0_greedy_parity(reference):
    """Tokens texte + codes audio identiques entre plugin et référence."""
    from vllm_omni_lfm2_audio.lfm2_audio_ar import Lfm2AudioARForConditionalGeneration  # noqa: F401

    model, proc = reference
    prompts = [
        "Bonjour, je suis Marie Dupont, j'ai rendez-vous avec Claire Martin.",
        "Quel est le wifi invité ?",
    ]
    for prompt in prompts:
        ref_text, ref_frames = _reference_interleaved_greedy(model, proc, prompt)
        # TODO(P2-GPU): instancier le stage 0 via vllm-omni (LLM(model=OMNI_CHECKPOINT,
        # stage_config=...)), soumettre le même contexte, comparer :
        #   - ids texte (placeholders exclus) == ref_text
        #   - frames exportées via multimodal_outputs == ref_frames
        pytest.skip("P2 runtime harness: à brancher sur l'engine vllm-omni (GPU)")


def test_stage1_waveform_parity(reference):
    """Waveform du stage 1 == détokeniseur de référence (tolérance bf16)."""
    model, proc = reference
    _, ref_frames = _reference_interleaved_greedy(model, proc, "Dis bonjour.", max_new_tokens=64)
    if not ref_frames:
        pytest.skip("no audio frames generated")
    frames = torch.stack([f for f in ref_frames if int(f[0]) != 2048], dim=1)
    ref_wav = proc.decode(frames.unsqueeze(0).cuda())

    from vllm_omni_lfm2_audio.lfm2_audio_code2wav import Lfm2AudioCode2Wav

    class _Cfg:
        class model_config:
            hf_config = None

    stage = Lfm2AudioCode2Wav(vllm_config=_Cfg)
    stage.load_weights([], model_path=os.environ["OMNI_CHECKPOINT"])
    stage.detokenizer.cuda()
    out = stage(frames)
    wav = out.multimodal_outputs["model_outputs"]
    assert wav.shape[-1] == ref_wav.shape[-1]
    assert torch.allclose(wav.float().cpu(), ref_wav.float().cpu(), atol=1e-2)
