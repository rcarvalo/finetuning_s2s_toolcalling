"""Topologie du pipeline LFM2.5-Audio (enregistrée via ``register_pipeline``).

Stage 0 : AR interleaved — compréhension multimodale + texte (tool calls) +
          codes Mimi exportés step par step.
Stage 1 : code2wav — détokeniseur LFM2 alimenté en chunks (async_chunk).

Modelé sur ``mimo_audio/pipeline.py`` (seul S2S interleaved in-tree).
"""

from __future__ import annotations

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

from vllm_omni_lfm2_audio.constants import IM_END_TOKEN_ID

_PROC = "vllm_omni_lfm2_audio.stage_input_processors"

LFM2_AUDIO_PIPELINE = PipelineConfig(
    model_type="lfm2_audio",
    model_arch="Lfm2AudioOmniModel",
    hf_architectures=("Lfm2AudioOmniModel",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="ar_interleaved",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=f"{_PROC}.ar2code2wav_async_chunk",
            custom_process_next_stage_input_func=f"{_PROC}.ar2code2wav_full_payload",
            sampling_constraints={
                "detokenize": True,
                # fin de tour : les tours de tool call (texte seul) s'arrêtent ici
                "stop_token_ids": [IM_END_TOKEN_ID],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            custom_process_input_func=f"{_PROC}.ar2code2wav",
            sync_process_input_func=f"{_PROC}.ar2code2wav_token_only",
            sampling_constraints={"detokenize": False},
        ),
    ),
)
