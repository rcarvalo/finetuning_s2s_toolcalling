"""``HfJudge`` — juge LLM adossé aux Inference Providers Hugging Face.

Existe parce que le projet a un ``HF_TOKEN`` mais pas de clé Gemini : le juge
est un ``Protocol`` précisément pour que la source du jugement soit
interchangeable sans toucher au scorer ni à la rubrique.

Le modèle par défaut est un modèle instruct de grande taille : juger la
pertinence d'une réponse par rapport à une question ET à un résultat d'outil
demande de la compréhension, pas de la fluidité — un petit modèle note au
hasard et fabrique une métrique qui semble marcher.
"""

from __future__ import annotations

import logging
import os

from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

API_KEY_ENV_VAR = "HF_TOKEN"  # pragma: allowlist secret
DEFAULT_MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"


class HfJudge:
    """Juge Hugging Face. Satisfait le protocole ``Judge``."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        api_key: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        self._model_id = model_id
        self._api_key = api_key or os.environ.get(API_KEY_ENV_VAR, "")
        self._max_tokens = max_tokens
        self._client: InferenceClient | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def has_credentials(self) -> bool:
        return bool(self._api_key)

    def judge(self, prompt: str) -> str:
        response = self._inference().chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self._model_id,
            max_tokens=self._max_tokens,
            temperature=0.0,
        )
        return str(response.choices[0].message.content or "")

    def _inference(self) -> InferenceClient:
        """Client d'inférence, construit au premier usage puis conservé."""
        if self._client is None:
            logger.info("juge LLM : %s (Hugging Face)", self._model_id)
            self._client = InferenceClient(api_key=self._api_key)
        return self._client
