"""Processors inter-stages : codes Mimi du stage 0 → payloads du stage 1.

Modelés sur ``stage_input_processors/mimo_audio.py``, adaptés à nos frames :
1 step = 1 frame de 8 codes (MiMo : patches 8×4). Accumulation par requête
dans le connector ; le PREMIER payload d'une requête part dès
``initial_codec_chunk_frames`` (TTFA court), les suivants quand
``codec_chunk_frames`` est atteint (ou à la fin), préfixés de
``codec_left_context_frames`` de contexte gauche décodé mais non émis
(frontières propres).

Aplatissement : col-major par frame — frame f, codebook c → index f*8+c —
inversé par ``Lfm2AudioCode2Wav._to_segments``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from vllm.inputs import TextPrompt  # noqa: F401
from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.inputs.data import OmniTokensPrompt

from lfm2_audio.vllm_plugin.constants import (
    CODEBOOKS,
    END_OF_AUDIO_CODE,
    LEFT_CONTEXT_HEADER_MAGIC,
    MIMI_FRAME_RATE,
)

logger = logging.getLogger(__name__)

# 12,5 frames/s : 10 frames = 800 ms par chunk ; contexte gauche 13 ≈ 1 s.
DEFAULT_CODEC_CHUNK_FRAMES = 10
DEFAULT_CODEC_LEFT_CONTEXT_FRAMES = 13
MIN_CODEC_CHUNK_FRAMES = 3
# Premier chunk émis dès N frames (TTFA), puis régime de codec_chunk_frames.
# Même principe que initial_codec_chunk_frames de qwen3_tts/fish in-tree :
# 2 frames = 160 ms d'audio, audible pendant que le chunk suivant s'accumule.
DEFAULT_INITIAL_CODEC_CHUNK_FRAMES = 2


def _frame_from_payload(codes: Any) -> torch.Tensor | None:
    """Normalise la frame du step ((8,) attendu) ; None si invalide/absente."""
    if codes is None:
        return None
    t = codes if isinstance(codes, torch.Tensor) else torch.tensor(codes, dtype=torch.long)
    t = t.reshape(-1).to(torch.long)
    if t.numel() != CODEBOOKS:
        return None
    if int(t[0].item()) == END_OF_AUDIO_CODE:  # frame EOA : marqueur de fin, pas de l'audio
        return None
    return t


def _frames_from_payload(codes: Any) -> list[torch.Tensor]:
    """Toutes les frames d'un payload : (8,), (n, 8) ou plat multiple de 8.

    Le runner peut livrer plusieurs frames d'un coup (export stacké par
    ``_drain_pending_codes``, chemin omni prefix cache) — on les déroule en
    filtrant les frames EOA."""
    if codes is None:
        return []
    t = codes if isinstance(codes, torch.Tensor) else torch.tensor(codes, dtype=torch.long)
    t = t.reshape(-1).to(torch.long)
    if t.numel() == 0 or t.numel() % CODEBOOKS != 0:
        return []
    frames = []
    for row in t.view(-1, CODEBOOKS):
        if int(row[0].item()) != END_OF_AUDIO_CODE:
            frames.append(row)
    return frames


def _connector_cfg(transfer_manager: Any) -> tuple[int, int, int]:
    connector = getattr(transfer_manager, "connector", None)
    raw = getattr(connector, "config", {}) or {}
    cfg = raw.get("extra", raw) if isinstance(raw, dict) else {}
    chunk = int(cfg.get("codec_chunk_frames", DEFAULT_CODEC_CHUNK_FRAMES))
    if chunk < MIN_CODEC_CHUNK_FRAMES:
        logger.warning(
            "codec_chunk_frames=%d < %d, fallback %d",
            chunk,
            MIN_CODEC_CHUNK_FRAMES,
            DEFAULT_CODEC_CHUNK_FRAMES,
        )
        chunk = DEFAULT_CODEC_CHUNK_FRAMES
    left = int(cfg.get("codec_left_context_frames", DEFAULT_CODEC_LEFT_CONTEXT_FRAMES))
    # Le chunk INITIAL peut être < MIN_CODEC_CHUNK_FRAMES (1 frame autorisée) :
    # il borne le TTFA, les frontières restent propres grâce au left_context
    # des chunks suivants.
    initial = int(cfg.get("initial_codec_chunk_frames", DEFAULT_INITIAL_CODEC_CHUNK_FRAMES))
    initial = max(1, min(initial, chunk))
    return chunk, left, initial


def _buffers(transfer_manager: Any) -> dict[str, list[list[int]]]:
    if not hasattr(transfer_manager, "code_prompt_token_ids"):
        transfer_manager.code_prompt_token_ids = {}
    return transfer_manager.code_prompt_token_ids


def _build_payload(frames: list[list[int]], *, new_frames: int, left_context: int, finished: bool):

    end = len(frames)
    start = max(0, end - new_frames - left_context)
    actual_left = end - new_frames - start
    body = torch.tensor(frames[start:end], dtype=torch.long).reshape(-1)
    # Encode actual_left + longueur du body en tête du tenseur : vLLM-Omni ne
    # forwarde pas meta.left_context_size vers additional_information du
    # forward() du stage 1, et sous charge plusieurs payloads d'une même
    # requête sont concaténés avant d'être consommés par le stage 1 — la
    # longueur permet à _to_segments de re-découper chunk par chunk.
    header = torch.tensor(
        [LEFT_CONTEXT_HEADER_MAGIC + actual_left, LEFT_CONTEXT_HEADER_MAGIC + (end - start)],
        dtype=torch.long,
    )
    flat = torch.cat([header, body])
    return OmniPayloadStruct(
        codes=CodesStruct(audio=flat),
        meta=MetaStruct(
            left_context_size=actual_left,
            codec_chunk_frames=new_frames,
            codec_left_context_frames=left_context,
            code_flat_numel=int(body.numel()),
            finished=torch.tensor(finished, dtype=torch.bool),
        ),
    )


def ar2code2wav_async_chunk(transfer_manager: Any, pooling_output: Any, request: Any, is_finished: bool = False):
    """Mode async_chunk : appelé à chaque step émis par le stage 0."""
    request_id = getattr(request, "external_req_id", None)
    if request_id is None:
        return None
    chunk_size, left_context, initial_chunk = _connector_cfg(transfer_manager)
    buffers = _buffers(transfer_manager)
    pending = buffers.setdefault(request_id, [])

    audio = None
    if isinstance(pooling_output, dict):
        codes = pooling_output.get("codes")
        audio = codes.get("audio") if isinstance(codes, dict) else None
        if audio is None:
            # Forme APLATIE : le runner passe les payloads par
            # flatten_payload ({"codes": {"audio": t}} → {"codes.audio": t}),
            # et le chemin omni prefix cache ne livre QUE cette forme.
            audio = pooling_output.get("codes.audio")
        # export sparse {req_id: frames} ; tolère aussi des frames brutes
        if isinstance(audio, dict):
            audio = audio.get(request_id)
        for frame in _frames_from_payload(audio):
            pending.append(frame.tolist())

    sent = getattr(transfer_manager, "_lfm2_sent_frames", None)
    if sent is None:
        sent = transfer_manager._lfm2_sent_frames = {}
    already = sent.get(request_id, 0)
    unsent = len(pending) - already

    if os.environ.get("LFM2_DEBUG_CHUNK"):
        keys = sorted(pooling_output.keys()) if isinstance(pooling_output, dict) else type(pooling_output).__name__
        logger.warning(
            "[chunk] req=%s payload_keys=%s audio_shape=%s pending=%d already=%d unsent=%d finished=%s",
            request_id[:12],
            keys,
            getattr(audio, "shape", None),
            len(pending),
            already,
            unsent,
            is_finished,
        )

    # Premier chunk de la requête : seuil court (TTFA), puis régime normal.
    # On draine tout l'arriéré (new_frames = unsent) : identique au régime
    # permanent (1 frame/step → unsent == seuil), et rattrape le retard si le
    # stage 1 a pris du retard sous charge.
    threshold = initial_chunk if already == 0 else chunk_size
    if unsent >= threshold or (is_finished and unsent > 0):
        new_frames = unsent
        sent[request_id] = already + new_frames
        payload = _build_payload(
            pending[: already + new_frames],
            new_frames=new_frames,
            left_context=left_context,
            finished=is_finished,
        )
        if os.environ.get("LFM2_DEBUG_CHUNK"):
            logger.warning(
                "[chunk-send] req=%s new=%d left_in_payload=%d total_sent=%d finished=%s",
                request_id[:12],
                new_frames,
                int(payload.meta.left_context_size),
                already + new_frames,
                is_finished,
            )
        if is_finished:
            buffers.pop(request_id, None)
            sent.pop(request_id, None)
        return payload

    if is_finished:
        buffers.pop(request_id, None)
        sent.pop(request_id, None)
    return None


def ar2code2wav(source_outputs: list[Any], prompt=None, requires_multimodal_data: bool = False):
    """Mode synchrone (pipeline legacy) : tout l'audio d'un coup, en fin de tour."""

    results: list[OmniTokensPrompt] = []
    for output in source_outputs:
        frames: list[list[int]] = []
        mm = getattr(output.outputs[0], "multimodal_output", None) or {}
        codes = mm.get("codes", {})
        audio = codes.get("audio") if isinstance(codes, dict) else None
        if isinstance(audio, dict):
            for frame_like in audio.values():
                frame = _frame_from_payload(frame_like)
                if frame is not None:
                    frames.append(frame.tolist())
        elif audio is not None:
            tensor = audio if isinstance(audio, torch.Tensor) else torch.tensor(audio, dtype=torch.long)
            for frame_like in tensor.reshape(-1, CODEBOOKS):
                frame = _frame_from_payload(frame_like)
                if frame is not None:
                    frames.append(frame.tolist())

        body = [c for frame in frames for c in frame]
        # Préfixe header avec left_context_size=0 (chemin sync : tout l'audio d'un coup).
        flat = [LEFT_CONTEXT_HEADER_MAGIC, *body]
        results.append(
            OmniTokensPrompt(
                prompt_token_ids=flat,
                additional_information={"left_context_size": 0, "n_frames": len(frames)},
                multi_modal_data=None,
            )
        )
    return results


def ar2code2wav_token_only(source_outputs: list[Any], prompt=None, requires_multimodal_data: bool = False):
    return ar2code2wav(source_outputs, prompt=prompt, requires_multimodal_data=requires_multimodal_data)


def ar2code2wav_full_payload(transfer_manager: Any, pooling_output: Any, request: Any, is_finished: bool = False):
    """Variante sync du chemin connector : accumule à chaque step, n'émet qu'à la fin."""
    request_id = getattr(request, "external_req_id", None)
    if request_id is None:
        return None
    buffers = _buffers(transfer_manager)
    pending = buffers.setdefault(request_id, [])
    if isinstance(pooling_output, dict):
        codes = pooling_output.get("codes", {})
        audio = codes.get("audio") if isinstance(codes, dict) else None
        if isinstance(audio, dict):
            audio = audio.get(request_id)
        frame = _frame_from_payload(audio)
        if frame is not None:
            pending.append(frame.tolist())
    if not is_finished:
        return None
    buffers.pop(request_id, None)
    if not pending:
        return None
    return _build_payload(pending, new_frames=len(pending), left_context=0, finished=True)


def estimate_chunk_latency_ms(chunk_frames: int) -> float:
    """Latence d'accumulation d'un chunk (diagnostic) : frames / 12,5 Hz."""
    return chunk_frames / MIMI_FRAME_RATE * 1000.0
