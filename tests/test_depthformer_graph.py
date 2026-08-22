"""Wrapper CUDA graph du depthformer : buckets/fallback (CPU) + parité (GPU).

La parité graph-vs-eager (GPU) est LE test qui valide l'action 5 de l'audit :
mêmes hidden → mêmes codes Mimi, padding de bucket sans effet de bord.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lfm2_audio.vllm_plugin.depthformer_graph import CudaGraphDepthformer  # noqa: E402


class _StubHead:
    """Tête factice : trace les appels eager du wrapper."""

    def __init__(self):
        self.calls: list[tuple[int, float | None, int | None]] = []

    def sample_frames(self, hidden, *, temperature=None, top_k=None):
        self.calls.append((hidden.shape[0], temperature, top_k))
        return torch.zeros(hidden.shape[0], 8, dtype=torch.long)


def test_bucket_selection():
    g = CudaGraphDepthformer(_StubHead(), capture_sizes=(1, 2, 4))
    assert g._bucket_for(1) == 1
    assert g._bucket_for(2) == 2
    assert g._bucket_for(3) == 4
    assert g._bucket_for(4) == 4
    assert g._bucket_for(5) is None  # > bucket max → fallback eager


def test_cpu_falls_back_to_eager_with_sampling_params():
    head = _StubHead()
    g = CudaGraphDepthformer(head, temperature=0.8, top_k=4)
    out = g.sample_frames(torch.randn(3, 16))  # tenseur CPU
    assert out.shape == (3, 8)
    assert head.calls == [(3, 0.8, 4)]
    assert not g._graphs  # aucune capture tentée hors CUDA


def test_oversized_batch_falls_back_to_eager():
    head = _StubHead()
    g = CudaGraphDepthformer(head, capture_sizes=(1, 2))
    g.sample_frames(torch.randn(5, 16))
    assert head.calls == [(5, None, None)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA requis")
def test_graph_matches_eager_greedy_on_gpu():
    pytest.importorskip("liquid_audio")
    from lfm2_audio.vllm_plugin.audio_head import Lfm2AudioHead

    torch.manual_seed(0)
    head = (
        Lfm2AudioHead(
            lfm_hidden_size=64,
            depthformer_layers=2,
            depthformer_dim=32,
            depthformer_tie=False,
            codebooks=8,
            audio_vocab_size=2049,
        )
        .cuda()
        .eval()
    )

    g = CudaGraphDepthformer(head, capture_sizes=(1, 2, 4))
    for batch in (1, 2, 3, 4):  # 3 → bucket 4 : vérifie le padding
        hidden = torch.randn(batch, 64, device="cuda")
        expected = head.sample_frames(hidden)  # greedy eager
        got = g.sample_frames(hidden)  # capture puis replay
        assert torch.equal(got.cpu(), expected.cpu()), f"divergence batch={batch}"
        # 2e appel : replay pur (le graph existe déjà) — toujours identique
        got2 = g.sample_frames(hidden)
        assert torch.equal(got2.cpu(), expected.cpu())
    assert set(g._graphs) == {1, 2, 4}
    assert not g._failed
