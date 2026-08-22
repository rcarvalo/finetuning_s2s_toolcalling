"""Serveur WebSocket minimal pour l'agent d'accueil (Phase 3).

Protocole (JSON sur WebSocket ``/ws``) :

- client → serveur :
    {"type": "audio", "pcm16": "<base64>", "sample_rate": 16000}   # chunks
    {"type": "commit"}                                             # fin du tour visiteur
    {"type": "reset"}                                              # nouvelle session
- serveur → client : événements de ``events.event_to_dict`` ; les chunks audio
  sont envoyés en {"type": "audio_chunk", "pcm16": "<base64 int16 24kHz>"}.

C'est le point de branchement de la Phase 4 : l'app Reachy Mini (fastrtc/WebRTC)
remplace ce transport, la boucle agent reste identique. Le VAD/turn-taking est
géré côté client pour l'instant (le client décide du ``commit``).

Lancement :
    python -m lfm2_audio.orchestrator.server --config configs/orchestrator.yaml
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build_agent_from_config(config: dict):
    """Assemble modèle + registre + agent depuis configs/orchestrator.yaml (GPU requis)."""
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

    from lfm2_audio.orchestrator.agent import AgentConfig, ReceptionAgent
    from lfm2_audio.orchestrator.fillers import FillerBank
    from lfm2_audio.tools.database import Database
    from lfm2_audio.tools.reception import (
        InMemoryReceptionBackend,
        PostgresReceptionBackend,
        build_reception_registry,
    )

    model_cfg = config.get("model", {})
    model_id = model_cfg.get("model_id", "LiquidAI/LFM2.5-Audio-1.5B")
    device = model_cfg.get("device", "cuda")

    proc = LFM2AudioProcessor.from_pretrained(model_id, device=device).eval()
    model = LFM2AudioModel.from_pretrained(model_id, device=device).eval()

    db = None
    dsn = config.get("postgres", {}).get("dsn")
    if dsn:
        db = Database(dsn, max_rows=config.get("postgres", {}).get("max_rows", 50))
        backend: PostgresReceptionBackend | InMemoryReceptionBackend = PostgresReceptionBackend(db)
    else:
        logger.warning("no postgres dsn configured, using in-memory demo backend")
        backend = InMemoryReceptionBackend()

    rag_search = None
    rag_cfg = config.get("rag", {})
    if rag_cfg.get("persist_dir"):
        from lfm2_audio.rag.retriever import KnowledgeBaseRetriever

        retriever = KnowledgeBaseRetriever(
            persist_dir=rag_cfg["persist_dir"],
            collection=rag_cfg.get("collection", "company_kb"),
            embedding_model=rag_cfg.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"),
            top_k=rag_cfg.get("top_k", 4),
        )
        rag_search = retriever.asearch

    registry = build_reception_registry(backend, db=db, rag_search=rag_search)

    agent_cfg = config.get("agent", {})
    fillers = FillerBank(filler_dir=Path(agent_cfg["filler_dir"]) if agent_cfg.get("filler_dir") else None)
    return ReceptionAgent(
        model,
        proc,
        registry,
        config=AgentConfig(
            max_new_tokens=agent_cfg.get("max_new_tokens", 2048),
            max_tool_rounds=agent_cfg.get("max_tool_rounds", 4),
            audio_temperature=agent_cfg.get("audio_temperature", 1.0),
            audio_top_k=agent_cfg.get("audio_top_k", 4),
            system_instructions=agent_cfg.get("system_instructions") or AgentConfig().system_instructions,
        ),
        fillers=fillers,
    )


def create_app(agent):
    import numpy as np
    import torch
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    from lfm2_audio.orchestrator.events import AudioChunk, event_to_dict

    app = FastAPI(title="s2s-toolcalling reception agent")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "tools": agent.registry.names}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        chat = agent.new_session()
        pcm_buffer: list[np.ndarray] = []
        input_rate = 16_000

        try:
            while True:
                msg = await ws.receive_json()
                mtype = msg.get("type")

                if mtype == "audio":
                    raw = base64.b64decode(msg["pcm16"])
                    pcm_buffer.append(np.frombuffer(raw, dtype=np.int16))
                    input_rate = int(msg.get("sample_rate", input_rate))

                elif mtype == "reset":
                    chat = agent.new_session()
                    pcm_buffer.clear()
                    await ws.send_json({"type": "session_reset"})

                elif mtype == "commit":
                    if not pcm_buffer:
                        await ws.send_json({"type": "error", "message": "no audio buffered"})
                        continue
                    pcm = np.concatenate(pcm_buffer)
                    pcm_buffer.clear()
                    wav = torch.tensor(pcm[None, :] / 32_768.0, dtype=torch.float32)

                    loop = asyncio.get_running_loop()
                    queue: asyncio.Queue = asyncio.Queue()

                    # Défauts d'arguments : fige les valeurs de CETTE itération.
                    # La closure est soumise à l'executor puis attendue avant le
                    # tour suivant, mais l'expliciter évite une capture tardive si
                    # la boucle venait à devenir concurrente.
                    def run_agent(chat=chat, wav=wav, input_rate=input_rate, loop=loop, queue=queue):
                        try:
                            for event in agent.respond(chat, wav, input_rate):
                                loop.call_soon_threadsafe(queue.put_nowait, event)
                        except Exception as e:  # remonté au client
                            logger.exception("agent turn failed")
                            loop.call_soon_threadsafe(queue.put_nowait, e)
                        finally:
                            loop.call_soon_threadsafe(queue.put_nowait, None)

                    task = loop.run_in_executor(None, run_agent)
                    while True:
                        event = await queue.get()
                        if event is None:
                            break
                        if isinstance(event, Exception):
                            await ws.send_json({"type": "error", "message": str(event), "recoverable": False})
                            break
                        payload = event_to_dict(event)
                        if isinstance(event, AudioChunk):
                            samples = (
                                event.samples.numpy() if hasattr(event.samples, "numpy") else np.asarray(event.samples)
                            )
                            pcm16 = (np.clip(samples, -1, 1) * 32_767).astype(np.int16)
                            payload["pcm16"] = base64.b64encode(pcm16.tobytes()).decode("ascii")
                        await ws.send_json(payload)
                    await task

                else:
                    await ws.send_json({"type": "error", "message": f"unknown message type: {mtype}"})

        except WebSocketDisconnect:
            logger.info("client disconnected")

    return app
