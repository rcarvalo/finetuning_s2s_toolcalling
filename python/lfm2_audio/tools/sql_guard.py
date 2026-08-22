"""Garde syntaxique SELECT-only de l'outil ``query_database``.

Python pur, aucune dépendance base : c'est le verrou le plus facile à tester,
donc celui qu'on teste le plus. Une seule instruction, ``SELECT``/``WITH``
uniquement, mots-clés d'écriture interdits.

Ce n'est **pas** la seule protection : la session PostgreSQL est ouverte en
``default_transaction_read_only`` et le rôle SQL n'a aucun droit d'écriture
(cf. :mod:`lfm2_audio.tools.database`). Trois verrous indépendants, aucun
suffisant seul.
"""

from __future__ import annotations

import re

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
