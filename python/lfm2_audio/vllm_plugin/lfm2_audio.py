"""Classe d'architecture enregistrée : dispatch par ``model_stage``.

Pattern identique à ``MiMoAudioForConditionalGeneration`` : une seule
architecture HF (``Lfm2AudioOmniModel``) ; chaque stage du pipeline instancie
cette classe avec son ``model_stage`` et ne construit que son sous-modèle.
"""

from __future__ import annotations

import logging
import os

from torch import nn
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY

from lfm2_audio.vllm_plugin.multimodal import (
    Lfm2AudioDummyInputsBuilder,
    Lfm2AudioMultiModalProcessor,
    Lfm2AudioProcessingInfo,
)

logger = logging.getLogger(__name__)

AR_STAGE = "ar_interleaved"
CODE2WAV_STAGE = "code2wav"


@MULTIMODAL_REGISTRY.register_processor(
    Lfm2AudioMultiModalProcessor,
    info=Lfm2AudioProcessingInfo,
    dummy_inputs=Lfm2AudioDummyInputsBuilder,
)
class Lfm2AudioOmniForConditionalGeneration(nn.Module, SupportsMultiModal):
    # Protocoles IsHybrid / HasInnerState de vLLM (attributs de CLASSE, lus par
    # le registre sans instanciation) : le backbone Lfm2 a des couches ShortConv
    # (état type mamba) → sans ces marqueurs, mamba_block_size n'est jamais
    # dérivé et get_kv_cache_spec assert (vu sur Colab T4).
    is_hybrid = True
    has_inner_state = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """Placeholder textuel par item (1 token audio_in par audio dans le
        prompt — remplacé par ceil(T_mel/8) positions par le processor)."""
        if modality.startswith("audio"):
            return "<|audio_in|>"
        raise ValueError("Only audio modality is supported")

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        from vllm.model_executor.models.lfm2 import Lfm2ForCausalLM

        return Lfm2ForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        # Réimplémenté (et non délégué) : la version Lfm2 lit conv_dim /
        # conv_L_cache sur hf_config DIRECTEMENT ; chez nous ils vivent dans la
        # section lfm (exposée par get_text_config()).
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

        lfm = vllm_config.model_config.hf_config.get_text_config()
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
            intermediate_size=lfm.conv_dim,
            conv_kernel=lfm.conv_L_cache,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.model_executor.models.lfm2 import Lfm2ForCausalLM

        return Lfm2ForCausalLM.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        from vllm.model_executor.models import SupportsPP  # noqa: F401  (interface vérifiée au runtime)

        self.vllm_config = vllm_config
        self.have_multimodal_outputs = True
        self.requires_raw_input_tokens = True

        # Nom de variable imposé par vLLM-Omni (minuscules) — non modifiable.
        self.model_stage = os.environ.get(
            "model_stage",  # noqa: SIM112 — nom de variable imposé par vLLM-Omni
            getattr(vllm_config.model_config, "model_stage", AR_STAGE),
        )

        if self.model_stage == AR_STAGE:
            from lfm2_audio.vllm_plugin.lfm2_audio_ar import Lfm2AudioARForConditionalGeneration

            self.model = Lfm2AudioARForConditionalGeneration(vllm_config=vllm_config, prefix=prefix)
            # hooks du runner (lus sur CETTE instance par gpu_model_runner) :
            # preprocess par requête + sampler custom (machine à états interleaved)
            self.has_preprocess = True
            self.prefer_model_sampler = True
        elif self.model_stage == CODE2WAV_STAGE:
            from lfm2_audio.vllm_plugin.lfm2_audio_code2wav import Lfm2AudioCode2Wav

            self.model = Lfm2AudioCode2Wav(vllm_config=vllm_config, prefix=prefix)
            self.model.set_model_path(vllm_config.model_config.model)
        else:
            raise ValueError(f"invalid model_stage: {self.model_stage!r} (expected {AR_STAGE} or {CODE2WAV_STAGE})")

        self.make_empty_intermediate_tensors = getattr(self.model, "make_empty_intermediate_tensors", lambda: None)

    # Délégations (mêmes hooks que MiMoAudio) ------------------------------ #

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, is_multimodal: bool = False, **kwargs):
        if self.model_stage == CODE2WAV_STAGE:
            import torch

            hidden = self.vllm_config.model_config.get_hidden_size()
            return torch.zeros_like(input_ids).reshape(-1, 1).repeat(1, hidden)
        return self.model.embed_input_ids(input_ids, multimodal_embeddings, is_multimodal=is_multimodal, **kwargs)

    def get_language_model(self):
        """Requis par SupportsMultiModal (probe MoE du loader vLLM)."""
        if self.model_stage == AR_STAGE:
            return self.model.language_model
        return self.model

    def embed_multimodal(self, **kwargs):
        if self.model_stage == CODE2WAV_STAGE:
            return ()  # l'audio-in ne concerne que le stage AR
        return self.model.embed_multimodal(**kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_logits(self, hidden_states, sampling_metadata=None):
        from vllm_omni.model_executor.models.output_templates import OmniOutput

        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if self.model_stage == CODE2WAV_STAGE:
            return None
        return self.model.compute_logits(hidden_states, sampling_metadata)

    def preprocess_batch(self, **kwargs):
        if self.model_stage == AR_STAGE:
            self.model.preprocess_batch(**kwargs)

    def preprocess(self, **kwargs):
        return self.model.preprocess(**kwargs)

    def sample(self, logits, sampling_metadata):
        if self.model_stage == AR_STAGE:
            return self.model.sample(logits, sampling_metadata)
        return None  # code2wav : sampler standard

    def load_weights(self, weights, **kwargs):
        loaded = self.model.load_weights(weights, **kwargs) or set()
        # le sous-modèle est monté sous `model.` : préfixe pour le tracking vLLM
        return {f"model.{name}" for name in loaded}

    def free_request(self, request_id: str) -> None:
        if hasattr(self.model, "free_request"):
            self.model.free_request(request_id)
