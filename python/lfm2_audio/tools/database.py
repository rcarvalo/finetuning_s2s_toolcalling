"""Accès PostgreSQL en lecture seule pour l'outil ``query_database`` (Phase 3).

Double protection :
1. garde syntaxique (`ensure_read_only`) : une seule instruction, SELECT/WITH
   uniquement, mots-clés d'écriture interdits ;
2. session PostgreSQL ouverte avec ``default_transaction_read_only=on`` +
   ``statement_timeout`` court, donc même une requête qui passerait la garde
   serait refusée côté serveur.
"""

from __future__ import annotations

import re
from typing import Any

# Mots-clés interdits où qu'ils apparaissent (hors littéraux — la garde est
# volontairement conservatrice : un faux positif est préférable à une écriture).
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|vacuum|merge|call|do|execute|listen|notify|set|reset|lock)\b",
    re.IGNORECASE,
)
_COMMENT = re.compile(r"(--[^\n]*|/\*.*?\*/)", re.DOTALL)


class UnsafeQueryError(ValueError):
    pass


def ensure_read_only(sql: str) -> str:
    """Valide qu'une requête est une instruction SELECT unique. Retourne le SQL nettoyé."""
    cleaned = _COMMENT.sub(" ", sql).strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeQueryError("empty query")
    if ";" in cleaned:
        raise UnsafeQueryError("multiple statements are not allowed")
    first_word = cleaned.split(None, 1)[0].lower()
    if first_word not in ("select", "with"):
        raise UnsafeQueryError("only SELECT queries are allowed")
    if _FORBIDDEN.search(cleaned):
        raise UnsafeQueryError("query contains a forbidden keyword")
    return cleaned


class Database:
    """Pool asyncpg en lecture seule (lazy)."""

    def __init__(self, dsn: str, *, statement_timeout_ms: int = 5_000, max_rows: int = 50) -> None:
        self.dsn = dsn
        self.statement_timeout_ms = statement_timeout_ms
        self.max_rows = max_rows
        self._pool = None

    async def _get_pool(self) -> Any:
        if self._pool is None:
            import asyncpg

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
