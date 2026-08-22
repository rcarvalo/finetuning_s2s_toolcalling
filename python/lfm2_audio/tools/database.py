"""``Database`` — accès PostgreSQL en LECTURE SEULE pour l'outil ``query_database``.

``asyncpg`` est importé en tête : ce module n'est chargé que si on ouvre
réellement une base. Le garde syntaxique, lui, vit dans
:mod:`lfm2_audio.tools.sql_guard` et reste testable sans installation.

Trois verrous indépendants, aucun suffisant seul : le garde syntaxique, la
session ``default_transaction_read_only``, et un rôle SQL sans droit d'écriture.
"""

from __future__ import annotations

import logging
from typing import Any

import asyncpg

from lfm2_audio.tools.sql_guard import ensure_read_only

logger = logging.getLogger(__name__)


class Database:
    """Pool asyncpg en lecture seule (lazy)."""

    def __init__(self, dsn: str, *, statement_timeout_ms: int = 5_000, max_rows: int = 50) -> None:
        self.dsn = dsn
        self.statement_timeout_ms = statement_timeout_ms
        self.max_rows = max_rows
        self._pool = None

    async def _get_pool(self) -> Any:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.dsn,
                min_size=1,
                max_size=4,
                server_settings={
                    "default_transaction_read_only": "on",
                    "statement_timeout": str(self.statement_timeout_ms),
                },
            )
        return self._pool

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [dict(r) for r in rows]

    async def safe_query(self, sql: str) -> dict[str, Any]:
        """Point d'entrée de l'outil ``query_database``."""
        cleaned = ensure_read_only(sql)
        rows = await self.fetch(cleaned)
        truncated = len(rows) > self.max_rows
        return {
            "rows": rows[: self.max_rows],
            "row_count": min(len(rows), self.max_rows),
            "truncated": truncated,
        }

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
