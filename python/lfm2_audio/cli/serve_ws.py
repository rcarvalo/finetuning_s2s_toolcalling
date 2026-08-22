"""Serveur WebSocket de l'orchestrateur temps réel.

Point d'entrée : ``lfm2-serve-ws``.
La logique vit dans :mod:`lfm2_audio.orchestrator.server` — ce module ne porte que la CLI.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from lfm2_audio.orchestrator.server import (
    build_agent_from_config,
    create_app,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/orchestrator.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}

    agent = build_agent_from_config(config)
    app = create_app(agent)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
