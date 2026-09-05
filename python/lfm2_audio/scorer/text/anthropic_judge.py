"""``AnthropicJudge`` — juge/générateur LLM adossé à l'API Anthropic.

Même contrat que ``GeminiJudge`` (``Judge``), pour que le générateur de
dialogues change de fournisseur par un drapeau et rien d'autre.

Choix mesurés le 05/09 :
- réponse en streaming (``get_final_message``) : une sortie de 10 dialogues
  fait 5 000 tokens, hors de portée d'un timeout HTTP sans streaming ;
- effort ``low`` par défaut : écrire des dialogues n'est pas du raisonnement,
  et les tokens de réflexion sont facturés comme de la sortie ;
- ``max_tokens`` 16 000 : une réponse coupée est un JSON invalide, donc un lot
  payé pour rien.
"""

from __future__ import annotations

import logging
import os
from typing import Literal, get_args

import anthropic
from anthropic.types import Message
from anthropic.types.message_create_params import MessageCreateParamsBase

from lfm2_audio.scorer.text.llm_spend import SpendMeter

logger = logging.getLogger(__name__)

API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"  # pragma: allowlist secret
DEFAULT_MODEL_ID = "claude-opus-5"
DEFAULT_MAX_TOKENS = 16_000

Effort = Literal["low", "medium", "high", "xhigh", "max"]
EFFORTS: tuple[str, ...] = get_args(Effort)
DEFAULT_EFFORT: Effort = "low"


def parse_effort(value: str) -> Effort:
    """Validation à la frontière (CLI) ; le reste du code manipule le littéral."""
    if value not in EFFORTS:
        raise ValueError(f"effort inconnu : {value!r} (attendu : {', '.join(EFFORTS)})")
    return value  # type: ignore[return-value]


class AnthropicJudge:
    """Un prompt, une réponse texte, l'usage compté. Satisfait ``Judge``."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        api_key: str | None = None,
        effort: Effort = DEFAULT_EFFORT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        meter: SpendMeter | None = None,
    ) -> None:
        self._model_id = model_id
        self._api_key = api_key or os.environ.get(API_KEY_ENV_VAR, "")
        self._effort: Effort = effort
        self._max_tokens = max_tokens
        self.meter = meter or SpendMeter(model_id)
        self._client: anthropic.Anthropic | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def has_credentials(self) -> bool:
        return bool(self._api_key)

    def request_params(self, prompt: str) -> MessageCreateParamsBase:
        """Les paramètres d'UNE requête — partagés avec le mode Batch, qui les
        envoie tels quels : la seule différence entre les deux modes est le prix."""
        return MessageCreateParamsBase(
            model=self._model_id,
            max_tokens=self._max_tokens,
            output_config={"effort": self._effort},
            messages=[{"role": "user", "content": prompt}],
        )

    def judge(self, prompt: str) -> str:
        self.meter.check()
        with self.client.messages.stream(**self.request_params(prompt)) as stream:
            message = stream.get_final_message()
        self.meter.add(message.usage.input_tokens, message.usage.output_tokens)
        return text_of(message)

    @property
    def client(self) -> anthropic.Anthropic:
        """Client construit au premier usage puis conservé."""
        if self._client is None:
            logger.info("juge LLM : %s (effort %s)", self._model_id, self._effort)
            self._client = anthropic.Anthropic(api_key=self._api_key or None)
        return self._client


def text_of(message: Message) -> str:
    """Le texte d'une réponse ; vide si le modèle a refusé ou a été coupé.

    Une réponse coupée (``max_tokens``) est un JSON tronqué : autant la jeter
    ici, avec un avertissement qui dit pourquoi, qu'obtenir plus loin un
    « lot ignoré » muet.
    """
    if message.stop_reason == "refusal":
        logger.warning("réponse refusée par le modèle (%s)", getattr(message, "stop_details", None))
        return ""
    if message.stop_reason == "max_tokens":
        logger.warning("réponse coupée à max_tokens — lot perdu, augmenter --max-tokens")
        return ""
    return "".join(block.text for block in message.content if block.type == "text")
