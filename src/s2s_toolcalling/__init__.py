"""Phases 2 & 3 — fine-tuning FR + tool calling de LFM2.5-Audio et orchestration d'outils.

Sous-paquets :
- ``data``        : format ChatML LFM2.5 (tool calls) et conversion vers liquid-audio (Phase 2).
- ``training``    : trainer SFT LoRA avec politiques de gel encodeur / têtes audio (Phase 2).
- ``tools``       : registre des outils métier de l'agent d'accueil (Phase 3).
- ``orchestrator``: parsing streaming des tool calls, boucle agent, serveur WebSocket (Phase 3).
- ``rag``         : ingestion / retrieval ChromaDB pour la base de connaissances (Phase 3).
- ``evaluation``  : scoring BFCL-style des tool calls sur set FR (Phases 2 & 5).
"""

__version__ = "0.1.0"
