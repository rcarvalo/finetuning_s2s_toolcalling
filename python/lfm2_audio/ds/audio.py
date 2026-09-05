"""``Waveform`` now lives in the evaluation toolkit; this module keeps the historical import path.

The value object (signal + its sample rate, immutable) is shared with the
scorers, so it belongs to the package both sides depend on.
"""

from __future__ import annotations

from avet.audio.waveform import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE, Waveform

__all__ = ["INPUT_SAMPLE_RATE", "OUTPUT_SAMPLE_RATE", "Waveform"]
