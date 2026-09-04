"""Refuse a bad Gemini key before spending a Colab session on it.

The first operator run died on `401 UNAUTHENTICATED — ACCESS_TOKEN_TYPE_UNSUPPORTED`:
the Colab secret did not hold the API key the laptop's .env holds. The generator
only found out at its first real call, inside a retry loop, after the whole
bootstrap. One tiny request here says so in one line, and prints the key's
shape (never the key) so it can be compared with the one that works.
"""

from __future__ import annotations

import os
import sys


def preflight(model: str = "gemini-2.5-flash") -> None:
    key = os.environ.get("GEMINI_API_KEY", "")
    shape = f"préfixe {key[:3]!r}, {len(key)} caractères"
    if not key.strip():
        sys.exit("GEMINI_API_KEY manquant ou vide — Colab : icône clé, secret GEMINI_API_KEY, accès notebook activé")
    from google import genai

    try:
        genai.Client(api_key=key).models.generate_content(model=model, contents="Reply with: ok")
    except Exception as error:  # the SDK raises several classes; any of them means "not usable"
        sys.exit(
            f"GEMINI_API_KEY refusée par Gemini ({type(error).__name__}: {str(error)[:140]}). "
            f"Valeur reçue : {shape}. La clé qui marche est celle du .env local — "
            "recopiez-la telle quelle dans le secret Colab (pas de jeton OAuth, pas d'espace)."
        )
    print(f"Gemini OK ({shape})", flush=True)
