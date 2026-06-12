"""Audio-in natif : processor multimodal vLLM pour LFM2.5-Audio.

Pattern in-tree ``mimo_audio_llm.py`` (MiMo réutilise le processor HF de
Qwen2-Audio ; LFM2.5 n'en a pas → on calcule le mel nous-mêmes avec le
préprocesseur de référence ``liquid_audio`` — déjà dépendance runtime du
plugin via le ConformerEncoder — config lue dans ``config.json["preprocessor"]``
du checkpoint, parité garantie).

Contrat (vérifié dans liquid-audio 1.3.0, cf. docs/audio_in_spec.md) :
- wave 16 kHz → mel ``(128, T_mel)`` ;
- l'audio occupe ``mel2emb_len(T_mel) = ceil(T_mel / 8)`` positions
  d'embedding (downsampling ×8 du conformer) ;
- le prompt porte 1 token ``audio_in_token_id`` par audio, remplacé par
  ``ceil(T/8)`` placeholders ; les embeddings conformer+adapter sont mergés à
  ces positions par le chemin standard ``SupportsMultiModal``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers import BatchFeature
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)

from vllm_omni_lfm2_audio.constants import AUDIO_FRAME_PLACEHOLDER_ID

AUDIO_IN_SAMPLE_RATE = 16_000
# Plus petite longueur mel acceptée par l'encodeur (source liquid utils).
MIN_MEL_LEN = 9


def mel2emb_len(mel_len: int) -> int:
    """Longueur mel → nombre de positions d'embedding LFM (ceil-div par 8)."""
    return -(mel_len // -8)


def audio_in_token_id(hf_config: Any) -> int:
    """Token placeholder audio-in du prompt (1 par audio, côté utilisateur).

    Réutilise le placeholder de frame (token spécial, jamais produit par
    l'encodage de texte) sauf si le checkpoint en dédie un autre."""
    return int(getattr(hf_config, "audio_in_token_id", AUDIO_FRAME_PLACEHOLDER_ID))


_MEL_CACHE: dict[int, Any] = {}


def _mel_preprocessor(hf_config: Any):
    """AudioToMelSpectrogramPreprocessor de référence (CPU, fp32, caché)."""
    key = id(hf_config)
    proc = _MEL_CACHE.get(key)
    if proc is None:
        from liquid_audio.model.conformer.processor import AudioToMelSpectrogramPreprocessor

        cfg = getattr(hf_config, "preprocessor", None) or {}
        if not cfg:
            raise ValueError(
                "checkpoint sans section `preprocessor` dans config.json — "
                "re-exporter avec convert_checkpoint (elle est copiée du modèle HF)"
            )
        proc = AudioToMelSpectrogramPreprocessor(**dict(cfg)).eval()
        _MEL_CACHE[key] = proc
    return proc


def extract_mel(hf_config: Any, audio) -> torch.Tensor:
    """Wave 16 kHz (np/torch 1D) → mel ``(128, T)`` fp32 CPU.

    Même chemin que ``ChatState.add_audio`` (le resampling vers 16 kHz est
    fait en amont par le data parser vLLM)."""
    wave = torch.as_tensor(audio, dtype=torch.float32).reshape(1, -1)
    length = torch.tensor([wave.shape[1]], dtype=torch.long)
    with torch.no_grad():
        mel, mel_len = _mel_preprocessor(hf_config)(wave, length)
    t = int(mel_len[0].item())
    if t < MIN_MEL_LEN:
        raise ValueError(f"audio trop court pour l'encodeur ({t} frames mel < {MIN_MEL_LEN})")
    return mel[0, :, :t].contiguous()


class Lfm2AudioProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_data_parser(self) -> MultiModalDataParser:
        # Hook vLLM 0.22.1 : BaseProcessingInfo.get_data_parser, consommé par
        # le renderer (info.parse_mm_data) ET le processor (cf. Qwen2Audio).
        # resample systématique vers 16 kHz (contrat ChatState.add_audio).
        return MultiModalDataParser(target_sr=AUDIO_IN_SAMPLE_RATE)

    def get_hf_processor(self, **kwargs):  # pragma: no cover - garde-fou
        raise RuntimeError(
            "LFM2.5-Audio n'a pas de processor HF — le mel est calculé par "
            "Lfm2AudioMultiModalProcessor._call_hf_processor (liquid_audio)."
        )


class Lfm2AudioDummyInputsBuilder(BaseDummyInputsBuilder[Lfm2AudioProcessingInfo]):
    DUMMY_AUDIO_SECONDS = 5

    # vLLM 0.22 passe ``mm_options`` en positionnel (profiling du budget mm).
    def get_dummy_text(self, mm_counts: Mapping[str, int], mm_options: Any = None, **kwargs: Any) -> str:
        tok = self.info.get_tokenizer()
        token = tok.decode([audio_in_token_id(self.info.get_hf_config())])
        return token * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Any = None, **kwargs: Any
    ):
        num_audios = mm_counts.get("audio", 0)
        return {
            "audio": self._get_dummy_audios(
                length=AUDIO_IN_SAMPLE_RATE * self.DUMMY_AUDIO_SECONDS, num_audios=num_audios
            )
        }


class Lfm2AudioMultiModalProcessor(BaseMultiModalProcessor[Lfm2AudioProcessingInfo]):
    # data parser : hérité de info.get_data_parser() (vLLM 0.22.1,
    # BaseMultiModalProcessor.__init__ → self.data_parser = info.get_data_parser())

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        audios = mm_data.get("audios") or mm_data.get("audio") or []
        if not isinstance(audios, (list, tuple)):
            audios = [audios]
        if not audios:
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        hf_config = self.info.get_hf_config()
        feats = [extract_mel(hf_config, a) for a in audios]
        lens = torch.tensor([f.shape[1] for f in feats], dtype=torch.long)
        return BatchFeature(
            {
                "input_ids": torch.tensor([prompt_ids], dtype=torch.long),
                # listes (formes variables) — champ batched par item audio
                "input_audio_features": feats,
                "input_audio_lens": lens,
            }
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "input_audio_features": MultiModalFieldConfig.batched("audio"),
            "input_audio_lens": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        token_id = audio_in_token_id(self.info.get_hf_config())
        lens = out_mm_kwargs.get_data().get("input_audio_lens")

        def replacement(item_idx: int) -> list[int]:
            mel_len = int(lens[item_idx])
            return [token_id] * mel2emb_len(mel_len)

        return [PromptReplacement(modality="audio", target=[token_id], replacement=replacement)]
