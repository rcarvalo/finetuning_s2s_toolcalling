"""``AnthropicBatchJudge`` — les mêmes requêtes, par l'API Message Batches, à moitié prix.

Générer un corpus n'a aucune contrainte de latence : le lot part, le job
attend, les résultats restent disponibles 29 jours. Pour 10 € de budget, le
demi-tarif est la différence entre un modèle capable et un modèle au rabais.

Un lot par shard (30 requêtes) et non un lot par famille : le plafond de
dépense se vérifie entre deux lots, et un shard reste l'unité qui va au Hub.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import cast

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from lfm2_audio.scorer.text.anthropic_judge import AnthropicJudge, Effort, text_of
from lfm2_audio.scorer.text.llm_spend import BATCH_DISCOUNT, SpendMeter

logger = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 30.0


class AnthropicBatchJudge:
    """Satisfait ``Judge`` et ajoute ``judge_many`` : N prompts, un seul lot."""

    def __init__(
        self,
        model_id: str,
        *,
        api_key: str | None = None,
        effort: Effort = "low",
        max_tokens: int = 16_000,
        max_usd: float | None = None,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.meter = SpendMeter(model_id, max_usd=max_usd, discount=BATCH_DISCOUNT)
        self._single = AnthropicJudge(model_id, api_key=api_key, effort=effort, max_tokens=max_tokens, meter=self.meter)
        self._poll_seconds = poll_seconds
        self._sleep = sleep

    @property
    def model_id(self) -> str:
        return self._single.model_id

    @property
    def has_credentials(self) -> bool:
        return self._single.has_credentials

    def judge(self, prompt: str) -> str:
        return self.judge_many([prompt])[0]

    def judge_many(self, prompts: list[str]) -> list[str]:
        """Une réponse par prompt, dans l'ordre ; vide pour un échec individuel."""
        self.meter.check()
        if not prompts:
            return []
        client = self._single.client
        requests = [
            # The base params ARE non-streaming params (`stream` is optional and
            # absent); the cast only says so to the type checker.
            Request(custom_id=f"p{index}", params=cast(MessageCreateParamsNonStreaming, self._single.request_params(p)))
            for index, p in enumerate(prompts)
        ]
        batch = client.messages.batches.create(requests=requests)
        logger.info("lot %s : %d requêtes", batch.id, len(prompts))
        self._wait(client, batch.id)
        return self._collect(client, batch.id, len(prompts))

    def _wait(self, client: anthropic.Anthropic, batch_id: str) -> None:
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                return
            counts = batch.request_counts
            logger.info("lot %s : %s (%d en cours)", batch_id, batch.processing_status, counts.processing)
            self._sleep(self._poll_seconds)

    def _collect(self, client: anthropic.Anthropic, batch_id: str, count: int) -> list[str]:
        # Les résultats arrivent dans N'IMPORTE quel ordre : indexés par custom_id.
        texts = [""] * count
        for result in client.messages.batches.results(batch_id):
            index = int(result.custom_id[1:])
            if result.result.type == "succeeded":
                message = result.result.message
                self.meter.add(message.usage.input_tokens, message.usage.output_tokens)
                texts[index] = text_of(message)
            else:
                logger.warning("requête %s du lot %s : %s", result.custom_id, batch_id, result.result.type)
        return texts
