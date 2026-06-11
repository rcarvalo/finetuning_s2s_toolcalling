"""Parité composant (CPU, sans GPU) : Lfm2AudioHead vs liquid-audio de référence.

Compare, à poids identiques (checkpoint LFM2.5-Audio-1.5B du cache HF) :
- ``sample_frame`` greedy vs ``LFM2AudioModel._sample_audio_frame`` ;
- ``embed_frame`` vs ``audio_embedding(frame + offsets).sum(0)``.

Si ces composants sont exacts, la parité engine (P2, GPU) ne dépend plus que
de la plomberie vLLM — c'est le dérisquage central du port.
"""

import os

import pytest

torch = pytest.importorskip("torch")
liquid_audio = pytest.importorskip("liquid_audio")

from huggingface_hub import try_to_load_from_cache  # noqa: E402

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B"

pytestmark = pytest.mark.skipif(
    not (os.environ.get("HF_HUB_OFFLINE_OK") or try_to_load_from_cache(MODEL_ID, "config.json")),
    reason=f"{MODEL_ID} absent du cache HF local",
)


@pytest.fixture(scope="module")
def reference_model():
    from liquid_audio import LFM2AudioModel

    model = LFM2AudioModel.from_pretrained(MODEL_ID, device="cpu").eval()
    return model.float()  # float32 : déterminisme strict CPU des deux côtés


@pytest.fixture(scope="module")
def audio_head(reference_model):
    from vllm_omni_lfm2_audio.audio_head import Lfm2AudioHead

    m = reference_model
    head = Lfm2AudioHead(
        lfm_hidden_size=m.lfm.config.hidden_size,
        depthformer_layers=m.depthformer_layers,
        depthformer_dim=m.depthformer_dim,
        depthformer_tie=m.depthformer_tie,
        codebooks=m.codebooks,
        audio_tie=bool(getattr(m.conf, "tie_audio_embeddings", False)),
    ).eval()
    head.load_weights(dict(m.state_dict()))
    return head.float()


def _hiddens(model, n: int = 12) -> torch.Tensor:
    g = torch.Generator().manual_seed(42)
    dim = model.lfm.config.hidden_size
    # échelle ~hidden réel (rms-normé en sortie de backbone)
    return torch.randn(n, dim, generator=g)


def test_should_match_reference_greedy_rollout(reference_model, audio_head):
    for hidden in _hiddens(reference_model):
        ref = reference_model._sample_audio_frame(hidden, temperature=None)
        ours = audio_head.sample_frame(hidden, temperature=None)
        assert torch.equal(ours.flatten().cpu(), ref.flatten().cpu())


def test_should_match_reference_frame_embedding(reference_model, audio_head):
    g = torch.Generator().manual_seed(7)
    for _ in range(8):
        frame = torch.randint(0, 2049, (reference_model.codebooks,), generator=g)
        ref = reference_model.audio_embedding(frame + reference_model.codebook_offsets).sum(0)
        ours = audio_head.embed_frame(frame)
        assert torch.allclose(ours, ref, atol=0.0, rtol=0.0)


def test_should_load_all_audio_head_weights(reference_model, audio_head):
    own = {n for n, _ in audio_head.named_parameters()}
    ref = {
        n for n, _ in reference_model.named_parameters()
        if n.split(".")[0] in ("audio_embedding", "depthformer", "depth_linear", "depth_embeddings")
    }
    assert own == ref
