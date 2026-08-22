"""App Gradio de test de l'endpoint serverless RunPod (style NIM).

Micro ou fichier → ``LiquidAudioClient`` → audio de réponse (streamé chunk par
chunk) + texte + métriques. Un compteur estime le coût GPU cumulé de la session
et **refuse d'envoyer** au-delà du budget (1 $ par défaut) : on peut itérer
sans surveiller la console RunPod.

Lancement : ``make app`` (lit ``.env`` : RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID).

Estimation de coût = temps mur × prix/s (``RUNPOD_COST_PER_S``, défaut
0.00031 $/s ≈ RTX 4090 flex). C'est une borne HAUTE : le temps mur inclut la
file d'attente et le réseau, non facturés — la vérité est dans la console
RunPod.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

import gradio as gr
import numpy as np

from lfm2_audio.ds.audio import Waveform
from lfm2_audio.remote import LiquidAudioClient

GradioAudio = tuple[int, np.ndarray]

_PCM16_MAX = 32_767


class CostMeter:
    """Cumule le coût estimé de la session et fait respecter le budget."""

    def __init__(self, cost_per_s: float, budget_usd: float) -> None:
        self.cost_per_s = cost_per_s
        self.budget_usd = budget_usd
        self.spent_usd = 0.0
        self.turns = 0

    def charge(self, seconds: float) -> None:
        self.spent_usd += seconds * self.cost_per_s
        self.turns += 1

    @property
    def exhausted(self) -> bool:
        return self.spent_usd >= self.budget_usd

    def summary(self) -> str:
        return (
            f"~{self.spent_usd:.4f} $ / budget {self.budget_usd:.2f} $ ({self.turns} tour(s), borne haute au temps mur)"
        )


METER = CostMeter(
    cost_per_s=float(os.environ.get("RUNPOD_COST_PER_S", "0.00031")),
    budget_usd=float(os.environ.get("APP_BUDGET_USD", "1.0")),
)
_CLIENTS: dict[str, LiquidAudioClient] = {}


def _client(endpoint_id: str) -> LiquidAudioClient:
    if endpoint_id not in _CLIENTS:
        _CLIENTS[endpoint_id] = LiquidAudioClient(endpoint_id)
    return _CLIENTS[endpoint_id]


def _to_gradio(waveform: Waveform) -> GradioAudio:
    pcm16 = (np.clip(waveform.samples, -1.0, 1.0) * _PCM16_MAX).astype(np.int16)
    return waveform.sample_rate, pcm16


def _metrics_text(client: LiquidAudioClient, elapsed_s: float) -> str:
    reply = client.last_reply
    lines = [f"temps mur : {elapsed_s:.2f} s → coût estimé ~{elapsed_s * METER.cost_per_s:.4f} $"]
    if reply is not None:
        metrics = reply.metrics
        ttfa = f"{metrics.ttfa_s:.2f} s" if metrics.ttfa_s is not None else "—"
        lines.append(f"worker : TTFA {ttfa}, génération {metrics.total_s:.2f} s, {metrics.audio_frames} frames")
    lines.append(METER.summary())
    return "\n".join(lines)


def run_turn(
    endpoint_id: str,
    audio_in: GradioAudio | None,
    streaming: bool,
    max_tokens: int,
) -> Iterator[tuple[GradioAudio | None, str, str]]:
    """Un tour S2S. Générateur : en mode streaming, chaque yield ajoute un chunk audio."""
    if METER.exhausted:
        message = f"Budget épuisé : {METER.summary()} — relancer l'app (ou monter APP_BUDGET_USD) pour continuer."
        raise gr.Error(message)
    if not endpoint_id.strip():
        message = "Endpoint ID manquant (champ ci-dessus ou RUNPOD_ENDPOINT_ID dans .env)."
        raise gr.Error(message)
    if audio_in is None:
        message = "Enregistre ou charge un audio d'abord."
        raise gr.Error(message)

    sample_rate, samples = audio_in
    waveform = Waveform.from_pcm16(np.asarray(samples), int(sample_rate))
    client = _client(endpoint_id.strip())
    start = time.monotonic()
    try:
        if streaming:
            for chunk in client.invoke_stream(audio=waveform, max_tokens=max_tokens):
                yield _to_gradio(chunk), gr.skip(), gr.skip()
            reply = client.last_reply
            final_audio: GradioAudio | None = None  # les chunks sont déjà dans le composant
        else:
            reply = client.invoke(audio=waveform, max_tokens=max_tokens)
            final_audio = _to_gradio(reply.audio) if reply.audio is not None else None
    except Exception as exc:  # large à dessein : l'UI doit afficher l'erreur, pas mourir
        METER.charge(time.monotonic() - start)
        raise gr.Error(f"{type(exc).__name__} : {exc}") from exc
    elapsed = time.monotonic() - start
    METER.charge(elapsed)
    yield final_audio, (reply.text if reply else ""), _metrics_text(client, elapsed)


def check_health(endpoint_id: str) -> str:
    if not endpoint_id.strip():
        message = "Endpoint ID manquant."
        raise gr.Error(message)
    health: dict[str, Any] = _client(endpoint_id.strip()).health()
    return f"{health}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="LFM2.5-Audio serverless") as demo:
        gr.Markdown(
            "# LFM2.5-Audio — test de l'endpoint serverless\n"
            "Parle (ou charge un WAV), le tour part sur RunPod et la réponse revient en audio + texte. "
            f"Garde-fou budget : **{METER.budget_usd:.2f} $**."
        )
        with gr.Row():
            endpoint_id = gr.Textbox(
                label="Endpoint ID RunPod",
                value=os.environ.get("RUNPOD_ENDPOINT_ID", ""),
                scale=3,
            )
            health_button = gr.Button("Santé de l'endpoint", scale=1)
        health_box = gr.Textbox(label="Santé", interactive=False)

        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Question (parole)")
                streaming = gr.Checkbox(value=True, label="Streaming (le 1er chunk joue pendant la génération)")
                max_tokens = gr.Slider(64, 2048, value=512, step=64, label="max_tokens")
                send = gr.Button("Envoyer", variant="primary")
            with gr.Column():
                audio_out = gr.Audio(label="Réponse (audio)", streaming=True, autoplay=True)
                text_out = gr.Textbox(label="Réponse (texte)", interactive=False)
                metrics_out = gr.Textbox(label="Métriques & coût", interactive=False, lines=3)

        send.click(
            run_turn, inputs=[endpoint_id, audio_in, streaming, max_tokens], outputs=[audio_out, text_out, metrics_out]
        )
        health_button.click(check_health, inputs=[endpoint_id], outputs=[health_box])
    return demo


if __name__ == "__main__":
    build_app().launch()
