"""Découpage de documents en passages indexables.

Python pur : c'est la logique qui décide de ce que le modèle recevra comme
contexte, donc celle qui mérite le plus d'être testée — et elle doit l'être sans
installer ChromaDB.
"""

from __future__ import annotations

from collections.abc import Iterator


def chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 150) -> Iterator[str]:
    """Chunking par paragraphes, avec taille cible et recouvrement (caractères).

    Le recouvrement évite qu'une réponse tombe pile sur une frontière de chunk et
    se retrouve coupée en deux passages dont aucun ne suffit.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
            continue
        if current:
            yield current
            current = current[-overlap:] if overlap else ""

        # Paragraphe plus long que chunk_size : découpe dure.
        remaining = para
        while len(remaining) > chunk_size:
            head = remaining[:chunk_size]
            yield (current + "\n\n" + head).strip() if current else head
            remaining = remaining[chunk_size - overlap :]
            current = ""
        current = f"{current}\n\n{remaining}".strip() if current else remaining

    if current.strip():
        yield current.strip()
