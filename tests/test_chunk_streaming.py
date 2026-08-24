"""Chemin async_chunk : seuil initial court (TTFA), régime permanent, drain.

``vllm_omni`` n'est pas installé en CI : on stubbe ``vllm_omni.data_entry_keys``
(seul import du module fait par ``_build_payload``).
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

torch = pytest.importorskip("torch")
# stage_input_processors imports vllm.inputs at module top: torch alone is not
# enough to run these tests (a torch-only env used to unmask this as failures).
pytest.importorskip("vllm")


@dataclass
class _Codes:
    audio: Any = None


@dataclass
class _Meta:
    left_context_size: int = 0
    codec_chunk_frames: int = 0
    codec_left_context_frames: int = 0
    code_flat_numel: int = 0
    finished: Any = None


@dataclass
class _Payload:
    codes: Any = field(default=None)
    meta: Any = field(default=None)


@pytest.fixture(autouse=True)
def _stub_data_entry_keys(monkeypatch):
    mod = types.ModuleType("vllm_omni.data_entry_keys")
    mod.CodesStruct = lambda audio=None: _Codes(audio=audio)
    mod.MetaStruct = _Meta
    mod.OmniPayloadStruct = _Payload
    pkg = types.ModuleType("vllm_omni")
    pkg.data_entry_keys = mod
    monkeypatch.setitem(sys.modules, "vllm_omni", pkg)
    monkeypatch.setitem(sys.modules, "vllm_omni.data_entry_keys", mod)


class _Connector:
    def __init__(self, extra):
        self.config = {"extra": extra}


class _Manager:
    def __init__(self, **extra):
        self.connector = _Connector(extra)


class _Request:
    def __init__(self, req_id="req-0"):
        self.external_req_id = req_id


def _step(mgr, req, frame_codes, finished=False):
    from lfm2_audio.vllm_plugin.stage_input_processors import ar2code2wav_async_chunk

    pooling = None
    if frame_codes is not None:
        pooling = {"codes": {"audio": {req.external_req_id: torch.tensor(frame_codes)}}}
    return ar2code2wav_async_chunk(mgr, pooling, req, is_finished=finished)


FRAME = list(range(8))


def _decode(payload):
    """Relit le header magic : (left_context, n_frames_total_du_payload)."""
    from lfm2_audio.vllm_plugin.constants import LEFT_CONTEXT_HEADER_MAGIC

    flat = payload.codes.audio
    left = int(flat[0].item()) - LEFT_CONTEXT_HEADER_MAGIC
    length = int(flat[1].item()) - LEFT_CONTEXT_HEADER_MAGIC
    assert flat.numel() == 2 + length * 8
    return left, length


def test_initial_chunk_emitted_early():
    mgr, req = _Manager(codec_chunk_frames=10, initial_codec_chunk_frames=2), _Request()
    assert _step(mgr, req, FRAME) is None  # 1 frame < seuil initial
    payload = _step(mgr, req, FRAME)  # 2 frames → premier chunk
    assert payload is not None
    left, length = _decode(payload)
    assert (left, length) == (0, 2)  # pas de contexte gauche au début
    assert payload.meta.codec_chunk_frames == 2


def test_steady_state_after_initial():
    mgr, req = (
        _Manager(codec_chunk_frames=4, initial_codec_chunk_frames=2, codec_left_context_frames=13),
        _Request(),
    )
    _step(mgr, req, FRAME)
    assert _step(mgr, req, FRAME) is not None  # chunk initial (2)
    for _ in range(3):
        assert _step(mgr, req, FRAME) is None  # accumule jusqu'au régime normal
    payload = _step(mgr, req, FRAME)  # 4 nouvelles frames
    assert payload is not None
    left, length = _decode(payload)
    assert payload.meta.codec_chunk_frames == 4
    assert left == 2  # contexte = tout l'historique dispo
    assert length == 6  # 2 (contexte) + 4 (nouvelles)


def test_default_initial_is_two_frames():
    mgr, req = _Manager(codec_chunk_frames=10), _Request()
    assert _step(mgr, req, FRAME) is None
    assert _step(mgr, req, FRAME) is not None  # DEFAULT_INITIAL_CODEC_CHUNK_FRAMES = 2


def test_initial_clamped_to_at_least_one():
    mgr, req = _Manager(codec_chunk_frames=10, initial_codec_chunk_frames=0), _Request()
    assert _step(mgr, req, FRAME) is not None  # clamp à 1 frame


def test_finish_drains_remaining_frames():
    mgr, req = _Manager(codec_chunk_frames=10, initial_codec_chunk_frames=2), _Request()
    _step(mgr, req, FRAME)
    _step(mgr, req, FRAME)  # chunk initial parti
    _step(mgr, req, FRAME)
    payload = _step(mgr, req, FRAME, finished=True)
    assert payload is not None
    assert payload.meta.codec_chunk_frames == 2  # les 2 frames restantes
    assert bool(payload.meta.finished)


def test_backlog_drained_in_one_payload():
    mgr, req = _Manager(codec_chunk_frames=4, initial_codec_chunk_frames=2), _Request()
    _step(mgr, req, FRAME)
    _step(mgr, req, FRAME)  # initial parti (sent=2)
    # arriéré : frames empilées sans émission (stage 1 en retard simulé)
    from lfm2_audio.vllm_plugin.stage_input_processors import _buffers

    _buffers(mgr)[req.external_req_id].extend([FRAME] * 6)
    payload = _step(mgr, req, FRAME)  # 7 frames en attente → tout part
    assert payload is not None
    assert payload.meta.codec_chunk_frames == 7


def test_eoa_frame_not_buffered():
    from lfm2_audio.vllm_plugin.constants import END_OF_AUDIO_CODE

    mgr, req = _Manager(codec_chunk_frames=10, initial_codec_chunk_frames=1), _Request()
    assert _step(mgr, req, [END_OF_AUDIO_CODE] * 8) is None  # EOA ≠ frame audio
    assert _step(mgr, req, FRAME) is not None


def _step_flat(mgr, req, tensor, finished=False):
    """Payload forme APLATIE ({"codes.audio": t}, chemin omni prefix cache)."""
    from lfm2_audio.vllm_plugin.stage_input_processors import ar2code2wav_async_chunk

    return ar2code2wav_async_chunk(mgr, {"codes.audio": tensor}, req, is_finished=finished)


def test_flattened_payload_key():
    mgr, req = _Manager(codec_chunk_frames=10, initial_codec_chunk_frames=2), _Request()
    assert _step_flat(mgr, req, torch.tensor(FRAME)) is None
    payload = _step_flat(mgr, req, torch.tensor(FRAME))
    assert payload is not None
    assert payload.meta.codec_chunk_frames == 2


def test_multiframe_tensor_unrolled():
    from lfm2_audio.vllm_plugin.constants import END_OF_AUDIO_CODE

    mgr, req = _Manager(codec_chunk_frames=10, initial_codec_chunk_frames=3), _Request()
    # (3, 8) : 2 frames audio + 1 EOA → 2 frames bufferisées, pas de send (< 3)
    stacked = torch.tensor([FRAME, FRAME, [END_OF_AUDIO_CODE] * 8])
    assert _step_flat(mgr, req, stacked) is None
    payload = _step_flat(mgr, req, torch.tensor(FRAME))  # 3e frame → seuil initial
    assert payload is not None
    assert payload.meta.codec_chunk_frames == 3
