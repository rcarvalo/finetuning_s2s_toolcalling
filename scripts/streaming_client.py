#!/usr/bin/env python3
"""Client streaming pour vllm-omni serve — conversation avec audio en temps réel.

Lance d'abord le serveur dans un terminal séparé :
    vllm-omni serve /content/lfm25_audio_omni \\
        --stage-config-path configs/vllm_omni_lfm2_audio.yaml \\
        --host 0.0.0.0 --port 8000

Puis dans un 2e terminal :
    python scripts/streaming_client.py

Le client envoie le texte utilisateur via l'API /v1/audio/speech (ou /v1/chat)
et reçoit l'audio en streaming SSE. Chaque chunk PCM est joué via sounddevice
dès réception → latence perçue ≈ 800 ms (1 chunk = 10 frames @ 12.5 fps).

Si sounddevice n'est pas disponible (Colab), les chunks sont sauvegardés
dans --out-dir et concaténés en un WAV final.
"""

from __future__ import annotations

import argparse
import io
import queue
import struct
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 24_000
CHUNK_SAMPLES = int(SAMPLE_RATE * 0.8)  # 800 ms par chunk


# ── lecture audio ─────────────────────────────────────────────────────────────

def _player_thread(q: queue.Queue, rate: int) -> None:
    """Thread de lecture : consomme des ndarray float32 depuis la queue."""
    try:
        import sounddevice as sd

        with sd.OutputStream(samplerate=rate, channels=1, dtype="float32") as stream:
            while True:
                chunk = q.get()
                if chunk is None:
                    break
                stream.write(chunk.reshape(-1, 1))
    except Exception as e:
        print(f"[player] sounddevice indisponible ({e}) — audio sauvegardé uniquement", file=sys.stderr)
        while q.get() is not None:
            pass


def _parse_pcm_chunk(raw: bytes) -> np.ndarray | None:
    """Interprète un chunk SSE brut (int16 LE → float32 normalisé)."""
    if len(raw) % 2 != 0:
        raw = raw[: len(raw) - 1]
    if not raw:
        return None
    arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32_768.0
    return arr


# ── requête HTTP streaming ─────────────────────────────────────────────────────

def stream_turn(
    base_url: str,
    messages: list[dict],
    out_dir: Path,
    turn: int,
    audio_queue: queue.Queue,
    model: str = "lfm2_audio",
) -> str:
    """Envoie un tour de conversation et stream l'audio.

    Retourne le texte de la réponse (reconstitué depuis les événements SSE).
    """
    import httpx

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": 512,
    }

    text_acc = ""
    wav_chunks: list[np.ndarray] = []
    t0 = time.time()
    first_audio = None

    try:
        with httpx.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=120,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            buf = b""
            for raw_chunk in resp.iter_bytes():
                buf += raw_chunk
                while b"\n\n" in buf:
                    event, buf = buf.split(b"\n\n", 1)
                    for line in event.split(b"\n"):
                        line = line.strip()
                        if line.startswith(b"data: "):
                            data = line[6:]
                            if data == b"[DONE]":
                                continue
                            import json as _json
                            try:
                                obj = _json.loads(data)
                            except Exception:
                                # chunk audio brut (PCM16) non-JSON
                                pcm = _parse_pcm_chunk(data)
                                if pcm is not None:
                                    if first_audio is None:
                                        first_audio = time.time() - t0
                                    wav_chunks.append(pcm)
                                    audio_queue.put(pcm)
                                continue
                            # delta texte OpenAI-compatible
                            delta = (
                                obj.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if delta:
                                print(delta, end="", flush=True)
                                text_acc += delta
                            # chunk audio encodé en base64 (format vllm-omni)
                            audio_b64 = obj.get("audio") or obj.get("audio_chunk")
                            if audio_b64:
                                import base64
                                raw = base64.b64decode(audio_b64)
                                pcm = _parse_pcm_chunk(raw)
                                if pcm is not None:
                                    if first_audio is None:
                                        first_audio = time.time() - t0
                                    wav_chunks.append(pcm)
                                    audio_queue.put(pcm)
    except httpx.HTTPStatusError as e:
        print(f"\n[HTTP {e.response.status_code}] {e.response.text[:200]}", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"\n[stream_turn] {e}", file=sys.stderr)
        return ""

    print()  # saut de ligne après le texte streamé
    dt = time.time() - t0
    total_s = sum(c.size for c in wav_chunks) / SAMPLE_RATE if wav_chunks else 0
    print(
        f"[{dt:.1f}s total | TTFA {first_audio:.2f}s | {total_s:.1f}s audio]"
        if first_audio else f"[{dt:.1f}s | pas d'audio]"
    )

    # Sauvegarde WAV
    if wav_chunks:
        full = np.concatenate(wav_chunks)
        wav_path = out_dir / f"reply_{turn:03d}.wav"
        pcm16 = (np.clip(full, -1, 1) * 32_767).astype(np.int16)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm16.tobytes())
        print(f"   → {wav_path}")

    return text_acc


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="lfm2_audio")
    parser.add_argument("--system", default="You are a helpful assistant. Respond with interleaved text and audio.")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/lfm2_stream"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Sanity-check serveur
    import httpx
    try:
        httpx.get(f"{args.url}/health", timeout=5).raise_for_status()
        print(f"[client] serveur {args.url} — OK")
    except Exception as e:
        print(
            f"[client] impossible de joindre {args.url} ({e})\n"
            "Lancer : vllm-omni serve <checkpoint> "
            "--stage-config-path configs/vllm_omni_lfm2_audio.yaml",
            file=sys.stderr,
        )
        return 1

    audio_q: queue.Queue = queue.Queue()
    player = threading.Thread(target=_player_thread, args=(audio_q, SAMPLE_RATE), daemon=True)
    player.start()

    messages = [{"role": "system", "content": args.system}]
    turn = 0

    print("Dialogue streaming — Ctrl+C ou entrée vide pour quitter.\n")
    try:
        while True:
            try:
                u = input("👤 ").strip()
            except EOFError:
                break
            if not u:
                break
            turn += 1
            messages.append({"role": "user", "content": u})
            print("🤖 ", end="", flush=True)
            reply = stream_turn(args.url, messages, args.out_dir, turn, audio_q, model=args.model)
            if reply:
                messages.append({"role": "assistant", "content": reply})
    except KeyboardInterrupt:
        print("\nInterrompu.")
    finally:
        audio_q.put(None)
        player.join(timeout=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
