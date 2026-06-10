"""Config HF du checkpoint vLLM-Omni de LFM2.5-Audio.

vLLM core charge le checkpoint via ``transformers.AutoConfig`` : un
``model_type`` inconnu fait échouer ``ModelConfig`` avant même d'atteindre le
registre de modèles (vérifié sur runtime : ValidationError pydantic au
démarrage de l'orchestrateur). Cette classe est enregistrée par
``register()`` (entry point du plugin), même pattern que les configs in-tree
de vllm-omni (``transformers_utils/configs/glm_tts.py`` etc.).

La section ``lfm`` du config liquid-audio est un config Lfm2 HF complet :
elle est matérialisée en ``Lfm2Config`` et exposée via ``get_text_config()``
→ vLLM dimensionne le KV/conv cache du backbone sans rien savoir des modules
audio. Les autres sections (encoder, depthformer, preprocessor) restent des
dicts consommés par les modules liquid-audio du plugin.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, Lfm2Config, PretrainedConfig

from vllm_omni_lfm2_audio.constants import (
    AUDIO_EOA_PLACEHOLDER_ID,
    AUDIO_FRAME_PLACEHOLDER_ID,
    CODEBOOKS,
    DEFAULT_INTERLEAVED_N_AUDIO,
    DEFAULT_INTERLEAVED_N_TEXT,
)


class Lfm2AudioConfig(PretrainedConfig):
    """Config du modèle 2-stages (AR interleaved + code2wav)."""

    model_type = "lfm2_audio"

    def __init__(
        self,
        lfm: dict[str, Any] | None = None,
        encoder: dict[str, Any] | None = None,
        depthformer: dict[str, Any] | None = None,
        preprocessor: dict[str, Any] | None = None,
        codebooks: int = CODEBOOKS,
        interleaved_n_text: int = DEFAULT_INTERLEAVED_N_TEXT,
        interleaved_n_audio: int = DEFAULT_INTERLEAVED_N_AUDIO,
        audio_frame_token_id: int = AUDIO_FRAME_PLACEHOLDER_ID,
        audio_eoa_token_id: int = AUDIO_EOA_PLACEHOLDER_ID,
        audio_temperature: float = 1.0,
        audio_top_k: int = 4,
        **kwargs: Any,
    ) -> None:
        self.lfm = lfm if isinstance(lfm, Lfm2Config) else Lfm2Config(**(lfm or {}))
        self.encoder = dict(encoder or {})
        self.depthformer = dict(depthformer or {})
        self.preprocessor = dict(preprocessor or {})
        self.codebooks = codebooks
        self.interleaved_n_text = interleaved_n_text
        self.interleaved_n_audio = interleaved_n_audio
        self.audio_frame_token_id = audio_frame_token_id
        self.audio_eoa_token_id = audio_eoa_token_id
        self.audio_temperature = audio_temperature
        self.audio_top_k = audio_top_k
        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:  # noqa: ARG002
        """Backbone Lfm2 — utilisé par vLLM pour dimensionner KV/conv cache."""
        return self.lfm

    def to_dict(self) -> dict[str, Any]:
        # PretrainedConfig.to_dict ne sérialise pas les sous-configs non
        # déclarés dans sub_configs : on aplatit nous-mêmes pour le round-trip.
        output = super().to_dict()
        output["lfm"] = self.lfm.to_dict() if isinstance(self.lfm, PretrainedConfig) else self.lfm
        return output


def register_config() -> None:
    """Idempotent : le plugin est chargé dans chaque process (engine + workers)."""
    try:
        AutoConfig.register(Lfm2AudioConfig.model_type, Lfm2AudioConfig)
    except ValueError:
        pass  # déjà enregistré dans ce process
