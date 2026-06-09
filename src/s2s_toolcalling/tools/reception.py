"""Outils métier de l'agent d'accueil et leur câblage (Phase 3).

Deux backends :
- ``InMemoryReceptionBackend`` : données de démo, aucun service externe —
  utilisé pour le POC, les tests et la génération de données synthétiques ;
- ``PostgresReceptionBackend`` : branché sur le schéma ``sql/schema.sql``
  (rendez-vous, employés, lieux) ; les notifications restent des hooks à
  connecter (Slack/Teams/email) via ``notifier``.

``build_reception_registry`` assemble le ToolRegistry complet utilisé par
l'orchestrateur ET dont les ``definitions()`` doivent servir à générer les
données de la Phase 2 (contrat unique entraînement/inférence).
"""

from __future__ import annotations

import datetime as dt
import logging
import unicodedata
from typing import Any, Awaitable, Callable, Protocol

from s2s_toolcalling.tools import schemas
from s2s_toolcalling.tools.database import Database
from s2s_toolcalling.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

Notifier = Callable[[str, str, str], Awaitable[dict[str, Any]]]
"""(recipient_kind, recipient, message) -> status dict. recipient_kind: employee|receptionist"""


def _norm(s: str) -> str:
    """Normalisation accent/casse pour la recherche de noms."""
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn").strip()


class ReceptionBackend(Protocol):
    async def find_appointments(self, visitor_name: str, host_name: str | None, time_hint: str | None) -> list[dict]: ...
    async def find_employee(self, name: str) -> dict | None: ...
    async def get_directions(self, destination: str) -> dict | None: ...
    async def get_guest_wifi(self) -> dict: ...
    async def log_notification(self, recipient_kind: str, recipient: str, message: str) -> None: ...


# --------------------------------------------------------------------------- #
# Backend de démonstration (POC / tests / synthèse de données)
# --------------------------------------------------------------------------- #


class InMemoryReceptionBackend:
    def __init__(self, *, now: Callable[[], dt.datetime] | None = None) -> None:
        self._now = now or dt.datetime.now
        today = self._now().date().isoformat()
        self.employees = [
            {"id": 1, "full_name": "Claire Martin", "team": "Direction Produit", "office_location": "2e étage, aile B"},
            {"id": 2, "full_name": "Karim Benali", "team": "Ingénierie", "office_location": "3e étage, aile A"},
            {"id": 3, "full_name": "Sophie Nguyen", "team": "Ressources Humaines", "office_location": "1er étage, accueil RH"},
        ]
        self.appointments = [
            {"id": 1, "visitor_name": "Marie Dupont", "host": "Claire Martin", "scheduled_at": f"{today}T14:00:00",
             "location": "salle B2", "status": "confirmed"},
            {"id": 2, "visitor_name": "Jean Petit", "host": "Karim Benali", "scheduled_at": f"{today}T10:30:00",
             "location": "salle A1", "status": "confirmed"},
        ]
        self.locations = {
            "salle b2": {"name": "salle B2", "floor": "2e étage",
                         "directions": "Prenez l'ascenseur jusqu'au 2e étage, puis à droite : la salle B2 est la deuxième porte sur votre gauche."},
            "salle a1": {"name": "salle A1", "floor": "3e étage",
                         "directions": "Montez au 3e étage, la salle A1 est face aux ascenseurs."},
            "cafeteria": {"name": "cafétéria", "floor": "rez-de-chaussée",
                          "directions": "Traversez le hall : la cafétéria est au fond à gauche, après les portiques."},
            "toilettes": {"name": "toilettes", "floor": "rez-de-chaussée",
                          "directions": "Les toilettes sont dans le couloir à droite de l'accueil."},
        }
        self.guest_wifi = {"ssid": "Entreprise-Guest", "password": "Bienvenue2026!", "valid_until": f"{today}T23:59:59"}
        self.notifications: list[dict] = []

    async def find_appointments(self, visitor_name: str, host_name: str | None, time_hint: str | None) -> list[dict]:
        v = _norm(visitor_name)
        results = [a for a in self.appointments if _norm(a["visitor_name"]) == v or v in _norm(a["visitor_name"])]
        if host_name:
            h = _norm(host_name)
            results = [a for a in results if h in _norm(a["host"])]
        return results

    async def find_employee(self, name: str) -> dict | None:
        n = _norm(name)
        for e in self.employees:
            if n in _norm(e["full_name"]):
                return e
        return None

    async def get_directions(self, destination: str) -> dict | None:
        return self.locations.get(_norm(destination))

    async def get_guest_wifi(self) -> dict:
        return dict(self.guest_wifi)

    async def log_notification(self, recipient_kind: str, recipient: str, message: str) -> None:
        self.notifications.append(
            {"at": self._now().isoformat(), "kind": recipient_kind, "recipient": recipient, "message": message}
        )


# --------------------------------------------------------------------------- #
# Backend PostgreSQL (schéma sql/schema.sql)
# --------------------------------------------------------------------------- #


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
            "SELECT ssid, password, valid_until::text FROM guest_wifi WHERE valid_until > now() ORDER BY valid_until DESC LIMIT 1"
        )
        return rows[0] if rows else {"error": "no active guest wifi"}

    async def log_notification(self, recipient_kind: str, recipient: str, message: str) -> None:
        # La table est alimentée par le notifier externe ; trace locale en repli.
        logger.info("notification (%s -> %s): %s", recipient_kind, recipient, message)


# --------------------------------------------------------------------------- #
# Assemblage du registre
# --------------------------------------------------------------------------- #


async def _default_notifier(recipient_kind: str, recipient: str, message: str) -> dict[str, Any]:
    """Hook par défaut : log uniquement. À remplacer par Slack/Teams/email en prod."""
    logger.info("NOTIFY [%s] %s: %s", recipient_kind, recipient, message)
    return {"delivered": True, "channel": "log"}


def build_reception_registry(
    backend: ReceptionBackend,
    *,
    db: Database | None = None,
    rag_search: Callable[[str], Awaitable[list[dict]]] | None = None,
    notifier: Notifier | None = None,
    robot_gesture: Callable[[str], Awaitable[None]] | None = None,
    timeout_s: float = 10.0,
) -> ToolRegistry:
    """Construit le ToolRegistry complet de l'agent d'accueil.

    ``robot_gesture`` est le hook Phase 4 (geste d'orientation Reachy Mini) ;
    ``rag_search`` vient de ``s2s_toolcalling.rag.retriever``.
    """
    notify = notifier or _default_notifier
    registry = ToolRegistry(timeout_s=timeout_s)

    async def check_appointment(visitor_name: str, host_name: str | None = None, time_hint: str | None = None) -> dict:
        appointments = await backend.find_appointments(visitor_name, host_name, time_hint)
        return {"found": bool(appointments), "appointments": appointments}

    async def notify_employee(employee_name: str, message: str) -> dict:
        employee = await backend.find_employee(employee_name)
        if employee is None:
            return {"delivered": False, "error": f"employé introuvable : {employee_name}"}
        status = await notify("employee", employee["full_name"], message)
        await backend.log_notification("employee", employee["full_name"], message)
        return {"delivered": status.get("delivered", False), "employee": employee["full_name"]}

    async def guide_visitor(destination: str) -> dict:
        info = await backend.get_directions(destination)
        if info is None:
            return {"found": False, "error": f"lieu inconnu : {destination}"}
        if robot_gesture is not None:
            try:
                await robot_gesture(info["name"])
            except Exception:
                logger.exception("robot gesture failed (non bloquant)")
        return {"found": True, **info}

    async def get_guest_wifi() -> dict:
        return await backend.get_guest_wifi()

    async def notify_receptionist(reason: str, urgency: str = "normal") -> dict:
        status = await notify("receptionist", "accueil", f"[{urgency}] {reason}")
        await backend.log_notification("receptionist", "accueil", reason)
        return {"delivered": status.get("delivered", False), "urgency": urgency}

    registry.register(schemas.CHECK_APPOINTMENT, check_appointment)
    registry.register(schemas.NOTIFY_EMPLOYEE, notify_employee)
    registry.register(schemas.GUIDE_VISITOR, guide_visitor)
    registry.register(schemas.GET_GUEST_WIFI, get_guest_wifi)
    registry.register(schemas.NOTIFY_RECEPTIONIST, notify_receptionist)

    if db is not None:
        async def query_database(sql: str) -> dict:
            return await db.safe_query(sql)

        registry.register(schemas.QUERY_DATABASE, query_database)

    if rag_search is not None:
        async def search_knowledge_base(query: str) -> dict:
            passages = await rag_search(query)
            return {"passages": passages}

        registry.register(schemas.SEARCH_KNOWLEDGE_BASE, search_knowledge_base)

    return registry
