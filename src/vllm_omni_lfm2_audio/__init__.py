"""Plugin vLLM-Omni pour LFM2.5-Audio (LiquidAI) — S2S interleaved FR + tool calling.

Enregistre, via le groupe d'entry points ``vllm_omni.general_plugins`` :
- l'architecture ``Lfm2AudioOmniModel`` dans le registre de modèles ;
- la topologie 2-stages ``lfm2_audio`` (AR interleaved → détokeniseur) via
  ``register_pipeline`` (mécanisme officiel out-of-tree, pas de fork requis).

Design : voir ``docs/vllm_omni_integration.md`` du repo. Le pattern suivi est
celui de MiMo-Audio (seul modèle S2S interleaved in-tree) :
- les steps audio apparaissent dans le flux d'ids comme un token *placeholder* ;
- les 8 codes Mimi de la frame sont produits en interne par le depthformer et
  exportés step par step via ``OmniOutput.multimodal_outputs`` vers le stage 1 ;
- l'embedding d'entrée du step suivant est la somme des 8 embeddings de
  codebooks, servie depuis un cache par requête (ou depuis les
  ``multi_modal_data`` au prefill).

Le plugin dépend de ``liquid-audio`` pour les modules audio (ConformerEncoder,
depthformer, détokeniseur) : mêmes classes, mêmes poids → parité numérique
avec l'implémentation de référence.
"""

__version__ = "0.1.0"


def register() -> None:
    """Entry point ``vllm_omni.general_plugins`` (chargé dans tous les process)."""
    from vllm.model_executor.models import ModelRegistry
    from vllm_omni.config.stage_config import register_pipeline
    from vllm_omni.model_executor.models.registry import OmniModelRegistry

    from vllm_omni_lfm2_audio.configuration import register_config
    from vllm_omni_lfm2_audio.pipeline import LFM2_AUDIO_PIPELINE

    arch = "Lfm2AudioOmniModel"
    target = "vllm_omni_lfm2_audio.lfm2_audio:Lfm2AudioOmniForConditionalGeneration"

    # AutoConfig d'abord : vLLM core charge le config HF avant de résoudre
    # l'architecture (un model_type inconnu tue l'orchestrateur au démarrage).
    register_config()
    OmniModelRegistry.register_model(arch, target)
    # register_omni_models_to_vllm() (runtime) ne pousse vers le ModelRegistry
    # de vLLM core QUE le dict statique _OMNI_MODELS — pas les enregistrements
    # dynamiques : la validation de ModelConfig passe par le registre core,
    # donc on s'y enregistre aussi (mécanisme out-of-tree officiel de vLLM).
    if arch not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(arch, target)
    register_pipeline(LFM2_AUDIO_PIPELINE)
