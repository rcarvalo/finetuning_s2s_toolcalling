"""``PostgresReceptionBackend`` — accueil visiteurs adossé à PostgreSQL.

Séparé de :mod:`lfm2_audio.tools.reception`, qui reste en Python pur : le
registre d'outils, le backend en mémoire et l'orchestrateur se testent donc sans
asyncpg ni base à lancer.
"""

from __future__ import annotations

import logging
from typing import Any

from lfm2_audio.tools.database import Database

logger = logging.getLogger(__name__)


class PostgresReceptionBackend:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def find_appointments(self, visitor_name: str, host_name: str | None, time_hint: str | None) -> list[dict]:
        sql = """
            SELECT a.id, a.visitor_name, e.full_name AS host, a.scheduled_at::text,
                   a.location, a.status
            FROM appointments a
            JOIN employees e ON e.id = a.employee_id
            WHERE a.scheduled_at::date = CURRENT_DATE
              AND a.status != 'cancelled'
              AND immutable_unaccent(lower(a.visitor_name)) LIKE '%' || immutable_unaccent(lower($1)) || '%'
        """
        args: list[Any] = [visitor_name]
        if host_name:
            sql += " AND immutable_unaccent(lower(e.full_name)) LIKE '%' || immutable_unaccent(lower($2)) || '%'"
            args.append(host_name)
        sql += " ORDER BY a.scheduled_at"
        return await self.db.fetch(sql, *args)

    async def find_employee(self, name: str) -> dict | None:
        rows = await self.db.fetch(
            """
            SELECT id, full_name, team, office_location
            FROM employees
            WHERE immutable_unaccent(lower(full_name)) LIKE '%' || immutable_unaccent(lower($1)) || '%'
            LIMIT 1
            """,
            name,
        )
        return rows[0] if rows else None

    async def get_directions(self, destination: str) -> dict | None:
        rows = await self.db.fetch(
            """
            SELECT name, floor, directions
            FROM locations
            WHERE immutable_unaccent(lower(name)) LIKE '%' || immutable_unaccent(lower($1)) || '%'
            LIMIT 1
            """,
            destination,
        )
        return rows[0] if rows else None

    async def get_guest_wifi(self) -> dict:
        rows = await self.db.fetch(
            "SELECT ssid, password, valid_until::text FROM guest_wifi WHERE valid_until > now() ORDER BY "
            "valid_until DESC LIMIT 1"
        )
        return rows[0] if rows else {"error": "no active guest wifi"}

    async def log_notification(self, recipient_kind: str, recipient: str, message: str) -> None:
        # La table est alimentée par le notifier externe ; trace locale en repli.
        logger.info("notification (%s -> %s): %s", recipient_kind, recipient, message)
