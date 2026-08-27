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

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """Entry point ``vllm_omni.general_plugins`` (chargé dans tous les process).

    **Décline si l'hôte n'expose pas l'API 0.22.** Ce plugin est chargé par
    vLLM-Omni dans *tous* ses process, y compris ceux qui servent un tout autre
    modèle. Sur vllm-omni 0.26, ``register_pipeline`` n'existe plus à cette
    adresse, et l'ImportError remontait en « Orchestrator initialization
    failed » : installer ce repo suffisait à casser Voxtral sur la même machine,
    six lancements de suite, pour un plugin dont ce job n'avait aucun besoin.
    Un plugin ne doit jamais tuer l'hôte qu'il ne sait pas étendre.
    """
    from vllm.model_executor.models import ModelRegistry

    try:
        from vllm_omni.config.stage_config import register_pipeline
    except ImportError as error:
        logger.warning(
            "plugin lfm2_audio désactivé : vLLM-Omni de cet environnement n'expose pas "
            "register_pipeline (%s). Servir LFM2.5-Audio échouera sur une architecture "
            "inconnue — les autres modèles ne sont pas affectés.",
            error,
        )
        return

    from vllm_omni.model_executor.models.registry import OmniModelRegistry

    # IMPORT EAGER de la classe d'archi → exécute son décorateur
    # @MULTIMODAL_REGISTRY.register_processor DÈS le chargement du plugin (dans
    # tous les process). Sinon l'enregistrement (paresseux, par chaîne) arrive
    # potentiellement APRÈS la construction du ModelConfig qui décide
    # is_multimodal_model → supports_mm_inputs=False → l'encodeur audio (mel/
    # conformer) n'est JAMAIS appelé → l'audio d'entrée est ignoré (réponses
    # génériques). Les imports lourds (liquid_audio) restent paresseux (méthodes).
    import lfm2_audio.vllm_plugin.omni_model  # noqa: F401
    from lfm2_audio.vllm_plugin.configuration import register_config
    from lfm2_audio.vllm_plugin.pipeline import LFM2_AUDIO_PIPELINE

    arch = "Lfm2AudioOmniModel"
    target = "lfm2_audio.vllm_plugin.omni_model:Lfm2AudioOmniForConditionalGeneration"

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
