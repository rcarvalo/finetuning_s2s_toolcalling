"""Démo sans GPU de la plomberie Phase 3 (parser → registre → réinjection).

Simule le flux texte qu'émettrait LFM2.5-Audio fine-tuné (token par token) et
déroule le round-trip complet : détection du tool call, filler, exécution sur
le backend de démo, rendu du tour ``tool`` réinjecté au modèle.

    lfm2-orchestrator-demo
"""

from __future__ import annotations

import asyncio

from lfm2_audio.core import chat_format
from lfm2_audio.orchestrator.fillers import FillerBank
from lfm2_audio.orchestrator.tool_parser import StreamingToolCallParser
from lfm2_audio.tools.reception import InMemoryReceptionBackend, build_reception_registry

# Flux texte simulé du modèle (ce que generate_interleaved produirait dans le
# canal texte après le SFT Phase 2b), découpé grossièrement comme des tokens.
SIMULATED_MODEL_TEXT = (
    '<|tool_call_start|>[check_appointment(visitor_name="Marie Dupont", host_name="Claire Martin")]<|tool_call_end|>'
)


def tokenize_roughly(text: str) -> list[str]:
    """Découpe en morceaux de 1 à 7 caractères pour simuler le décodage token par token
    (les marqueurs spéciaux arrivent entiers, comme de vrais tokens)."""
    pieces: list[str] = []
    i = 0
    for marker in (chat_format.TOOL_CALL_START, chat_format.TOOL_CALL_END):
        text = text.replace(marker, f"\x00{marker}\x00")
    for chunk in text.split("\x00"):
        if chunk in (chat_format.TOOL_CALL_START, chat_format.TOOL_CALL_END):
            pieces.append(chunk)
        else:
            while i < len(chunk):
                pieces.append(chunk[i : i + 5])
                i += 5
            i = 0
    return [p for p in pieces if p]


async def main() -> None:
    backend = InMemoryReceptionBackend()
    registry = build_reception_registry(backend)
    fillers = FillerBank()

    print("Outils enregistrés :", ", ".join(registry.names))
    print("\nSystem prompt (extrait tool list) :")
    print(" ", chat_format.render_tool_list(registry.definitions())[:120], "...\n")

    print(">>> Visiteur : « Bonjour, je suis Marie Dupont, j'ai rendez-vous avec Claire Martin. »\n")
    print(">>> Le modèle génère (flux texte interleaved) :")

    parser = StreamingToolCallParser()
    calls = []
    for piece in tokenize_roughly(SIMULATED_MODEL_TEXT):
        print(f"    token: {piece!r}")
        calls.extend(parser.feed(piece))
        if calls:
            break

    assert calls, "le tool call aurait dû être détecté"
    call = calls[0]
    print(f"\n>>> Tool call détecté : {call.name}({call.arguments})")

    filler = fillers.get(call.name)
    print(f">>> Filler vocal joué pendant l'exécution : « {filler.phrase} »")

    result = await registry.execute(call.name, call.arguments)
    print(f">>> Résultat ({result.elapsed_ms:.1f} ms, ok={result.ok}) : {result.payload()}")

    reinjection = chat_format.render_tool_response(result.payload())
    print("\n>>> Réinjecté au modèle (tour rôle `tool`) :")
    print(f"    <|im_start|>tool\\n{reinjection}<|im_end|>")
    print("\n>>> Le modèle reprendrait alors la génération interleaved pour la réponse audio :")
    print("    « Oui Madame Dupont, vous êtes attendue à 14 heures en salle B2. Je préviens Claire Martin ? »")


if __name__ == "__main__":
    asyncio.run(main())
