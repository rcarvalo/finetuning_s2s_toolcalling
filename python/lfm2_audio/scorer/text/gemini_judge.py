"""``GeminiJudge`` — juge LLM adossé à l'API Gemini.

Choisi parce que la clé est déjà présente dans le projet (génération du dataset)
et que le coût d'une campagne de quelques centaines de cas y est négligeable.

``google-genai`` est importé en tête : ce module n'est chargé que par qui
construit réellement le juge.
"""

from __future__ import annotations

import logging
import os

from google.genai.client import Client

logger = logging.getLogger(__name__)

API_KEY_ENV_VAR = "GEMINI_API_KEY"
# gemini-2.0-flash a été retiré du service (404 « no longer available ») ; flash
# reste le palier adapté à une notation par rubrique — la tâche demande de la
# compréhension, pas du raisonnement long, et le coût par campagne reste marginal.
DEFAULT_MODEL_ID = "gemini-3.6-flash"


class GeminiJudge:
    """Juge Gemini. Satisfait le protocole ``Judge``."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, *, api_key: str | None = None) -> None:
        self._model_id = model_id
        self._api_key = api_key or os.environ.get(API_KEY_ENV_VAR, "")
        self._client: Client | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def has_credentials(self) -> bool:
        return bool(self._api_key)

    def judge(self, prompt: str) -> str:
        response = self._gemini().models.generate_content(model=self._model_id, contents=prompt)
        return str(response.text or "")

    def _gemini(self) -> Client:
        """Client Gemini, construit au premier usage puis conservé."""
        if self._client is None:
            logger.info("juge LLM : %s", self._model_id)
            self._client = Client(api_key=self._api_key)
        return self._client
