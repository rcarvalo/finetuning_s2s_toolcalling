"""LFM2.5-Audio — fine-tuning, serving vLLM et orchestration tool-calling.

Inférence en trois lignes :

>>> from lfm2_audio import LFM2Audio                                # doctest: +SKIP
>>> model = LFM2Audio.from_pretrained("Rcarvalo/lfm25-tc-en-s2s")   # doctest: +SKIP
>>> text, audio = model.reply(audio="question.wav")                 # doctest: +SKIP

Sous-paquets :

- ``core``        : abstractions transverses (erreurs, environnement, prompt ChatML) ;
- ``ds``          : structures de données (pydantic aux frontières, value objects) ;
- ``serving``     : chargement du modèle et backends d'inférence ;
- ``vllm_plugin`` : plugin out-of-tree vLLM-Omni (chargé dans les workers) ;
- ``training``    : SFT LoRA (gel encodeur / têtes audio) ;
- ``data_prep``   : préparation des datasets d'entraînement ;
- ``tools``       : outils métier appelables par le modèle ;
- ``orchestrator``: boucle agent tool-calling et transport temps réel ;
- ``rag``         : base de connaissances ;
- ``evaluation``  : scoring BFCL-style des tool calls ;
- ``cli``        : points d'entrée en ligne de commande.

``LFM2Audio`` est exposé paresseusement : importer ce paquet ne tire ni torch,
ni vLLM, ni liquid-audio — le plugin vLLM-Omni est chargé dans chaque process
worker et doit rester léger.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lfm2_audio.serving.model import LFM2Audio

__version__ = "0.1.0"

__all__ = ["LFM2Audio", "__version__"]


def __getattr__(name: str) -> Any:  # noqa: ANN401 — signature imposée par PEP 562
    """PEP 562 : ``from lfm2_audio import LFM2Audio`` sans import lourd au chargement."""
    if name == "LFM2Audio":
        from lfm2_audio.serving.model import LFM2Audio as _LFM2Audio

        return _LFM2Audio
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
