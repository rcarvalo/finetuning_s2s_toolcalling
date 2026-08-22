"""Définitions JSON des outils de l'agent d'accueil.

Ces schémas servent à la fois :
- en Phase 2, injectés entre ``<|tool_list_start|>/<|tool_list_end|>`` dans le
  system prompt des données d'entraînement ;
- en Phase 3, comme contrat de validation du registre d'exécution.

Ils doivent rester STRICTEMENT identiques entre entraînement et inférence.
"""

from __future__ import annotations

CHECK_APPOINTMENT = {
    "name": "check_appointment",
    "description": "Vérifie si un visiteur a un rendez-vous planifié dans les systèmes internes.",
    "parameters": {
        "type": "object",
        "properties": {
            "visitor_name": {"type": "string", "description": "Nom du visiteur tel qu'annoncé."},
            "host_name": {"type": "string", "description": "Nom du collaborateur visité, si connu."},
            "time_hint": {
                "type": "string",
                "description": "Indication horaire donnée par le visiteur (ex. '14h', 'cet après-midi').",
            },
        },
        "required": ["visitor_name"],
    },
}

NOTIFY_EMPLOYEE = {
    "name": "notify_employee",
    "description": "Prévient un collaborateur que son visiteur est arrivé (message instantané + email).",
    "parameters": {
        "type": "object",
        "properties": {
            "employee_name": {"type": "string", "description": "Nom du collaborateur à prévenir."},
            "message": {"type": "string", "description": "Message court à transmettre."},
        },
        "required": ["employee_name", "message"],
    },
}

GUIDE_VISITOR = {
    "name": "guide_visitor",
    "description": (
        "Donne l'itinéraire vers un lieu des locaux (salle, accueil, cafétéria...) et déclenche le geste "
        "d'orientation du robot."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "destination": {"type": "string", "description": "Lieu demandé (ex. 'salle B2', 'cafétéria')."},
        },
        "required": ["destination"],
    },
}

GET_GUEST_WIFI = {
    "name": "get_guest_wifi",
    "description": "Retourne le réseau wifi invité et son mot de passe en vigueur.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

NOTIFY_RECEPTIONIST = {
    "name": "notify_receptionist",
    "description": "Alerte le ou la réceptionniste quand l'agent ne peut pas répondre au besoin du visiteur.",
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "Résumé du besoin non couvert."},
            "urgency": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "description": "Niveau d'urgence.",
            },
        },
        "required": ["reason"],
    },
}

QUERY_DATABASE = {
    "name": "query_database",
    "description": (
        "Exécute une requête SQL en LECTURE SEULE (SELECT) sur la base PostgreSQL interne (employés, rendez-vous, "
        "lieux)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "Requête SELECT (une seule instruction, sans modification de données).",
            },
        },
        "required": ["sql"],
    },
}

SEARCH_KNOWLEDGE_BASE = {
    "name": "search_knowledge_base",
    "description": (
        "Recherche dans la base de connaissances de l'entreprise (projets, équipes, FAQ) et retourne les passages "
        "pertinents."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Question ou mots-clés à rechercher."},
        },
        "required": ["query"],
    },
}

RECEPTION_TOOL_DEFINITIONS: list[dict] = [
    CHECK_APPOINTMENT,
    NOTIFY_EMPLOYEE,
    GUIDE_VISITOR,
    GET_GUEST_WIFI,
    NOTIFY_RECEPTIONIST,
    QUERY_DATABASE,
    SEARCH_KNOWLEDGE_BASE,
]

RECEPTION_TOOL_NAMES = [t["name"] for t in RECEPTION_TOOL_DEFINITIONS]


# --------------------------------------------------------------------------- #
# Outils anglais — capacité tool calling vocal v1 (web_search + db_query).
# Descriptions EN, injectées telles quelles dans <|tool_list_start|>. db_query
# prend une QUESTION en langage naturel (NL→SQL fait côté backend, hors chemin
# du modèle assistant) — distinct de QUERY_DATABASE(sql) ci-dessus.
# --------------------------------------------------------------------------- #

WEB_SEARCH = {
    "name": "web_search",
    "description": (
        "Search the public web for current or general information and return the most "
        "relevant results. Use for news, prices, facts, definitions, or anything that is "
        "not stored in the internal company database."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query, phrased as concise keywords or a short question.",
            },
        },
        "required": ["query"],
    },
}

DB_QUERY = {
    "name": "db_query",
    "description": (
        "Answer a question from the internal company database (customers, orders, products, "
        "employees, meetings) by asking it in plain English. Use for internal or business "
        "data, never for public information."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to answer from the database, in natural language.",
            },
        },
        "required": ["question"],
    },
}

TOOLCALLING_EN_TOOL_DEFINITIONS: list[dict] = [WEB_SEARCH, DB_QUERY]

TOOLCALLING_EN_TOOL_NAMES = [t["name"] for t in TOOLCALLING_EN_TOOL_DEFINITIONS]
