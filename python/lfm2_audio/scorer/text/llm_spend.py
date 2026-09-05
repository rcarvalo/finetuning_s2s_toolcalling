"""``SpendMeter`` — ce qu'un run a coûté, et le plafond qu'il ne franchit pas.

Le budget du projet est de 10 € tous services confondus. Un run de génération
non attendu qui dépasse ne se rattrape pas : les tokens sont facturés au moment
où ils sortent. Le compteur additionne l'usage rapporté par l'API à chaque
réponse, et refuse de lancer l'appel SUIVANT dès que le plafond est atteint —
le dépassement est donc borné par une seule réponse.

Les prix sont ceux de l'API Anthropic de première main (USD par million de
tokens) ; le mode Batch les divise par deux.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
"""(entrée, sortie) en USD par million de tokens."""

BATCH_DISCOUNT = 0.5


class SpendCapReachedError(RuntimeError):
    """Le plafond est atteint : aucun nouvel appel ne part."""


class UnknownModelPriceError(LookupError):
    """Un modèle sans tarif connu ne peut pas être plafonné — on refuse plutôt que de compter faux."""


def price_of(model_id: str) -> tuple[float, float]:
    try:
        return PRICES_USD_PER_MTOK[model_id]
    except KeyError as error:
        known = ", ".join(sorted(PRICES_USD_PER_MTOK))
        raise UnknownModelPriceError(f"tarif inconnu pour {model_id!r} (connus : {known})") from error


@dataclass
class SpendMeter:
    """Cumul des tokens d'un modèle, converti en dollars."""

    model_id: str
    max_usd: float | None = None
    discount: float = 1.0
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def __post_init__(self) -> None:
        self._price_in, self._price_out = price_of(self.model_id)

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1

    @property
    def usd(self) -> float:
        raw = (self.input_tokens * self._price_in + self.output_tokens * self._price_out) / 1_000_000
        return raw * self.discount

    def check(self) -> None:
        """À appeler AVANT chaque requête."""
        if self.max_usd is not None and self.usd >= self.max_usd:
            raise SpendCapReachedError(
                f"plafond {self.max_usd:.2f} $ atteint ({self.usd:.4f} $ après {self.calls} appels)"
            )

    def summary(self) -> str:
        """Une ligne à marqueur, lisible par un job qui additionne plusieurs runs."""
        return (
            f"===SPEND=== model={self.model_id} calls={self.calls} in={self.input_tokens} "
            f"out={self.output_tokens} usd={self.usd:.4f}"
        )
