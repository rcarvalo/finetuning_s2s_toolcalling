"""Tests du value object ``Waveform``."""

from __future__ import annotations

import numpy as np
import pytest

from lfm2_audio.ds.audio import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE, Waveform


def test_should_keep_mono_signal_unchanged():
    samples = np.array([0.1, -0.2, 0.3], dtype=np.float32)

    waveform = Waveform.of(samples, 16_000)

    assert waveform.samples.shape == (3,)
    assert waveform.sample_rate == 16_000


@pytest.mark.parametrize("shape", [(100, 2), (2, 100)])
def test_should_downmix_stereo_whatever_the_channel_axis(shape):
    stereo = np.ones(shape, dtype=np.float32)

    waveform = Waveform.of(stereo, 48_000)

    assert waveform.samples.shape == (100,)


def test_should_reject_non_positive_sample_rate():
    with pytest.raises(ValueError, match="sample_rate"):
        Waveform.of(np.zeros(4, dtype=np.float32), 0)


def test_should_scale_pcm16_into_unit_range():
    pcm = np.array([0, 32_767, -32_768], dtype=np.int16)

    waveform = Waveform.from_pcm16(pcm, 16_000)

    assert waveform.samples.max() <= 1.0
    assert waveform.samples.min() >= -1.0


def test_should_report_duration_and_rms():
    waveform = Waveform.of(np.ones(24_000, dtype=np.float32), OUTPUT_SAMPLE_RATE)

    assert waveform.duration_s == pytest.approx(1.0)
    assert waveform.rms == pytest.approx(1.0)


def test_should_report_empty_signal():
    assert Waveform.of(np.array([], dtype=np.float32), 16_000).is_empty
    assert Waveform.of(np.zeros(1, dtype=np.float32), 16_000).rms == 0.0


def test_resample_should_be_a_noop_at_target_rate():
    # Chemin sans torch : `for_encoder` sur du 16 kHz ne doit rien importer.
    waveform = Waveform.of(np.ones(160, dtype=np.float32), INPUT_SAMPLE_RATE)

    assert waveform.for_encoder() is waveform


def test_concat_should_join_chunks_and_keep_rate():
    chunks = [Waveform.of(np.ones(n, dtype=np.float32), OUTPUT_SAMPLE_RATE) for n in (10, 5)]

    joined = Waveform.concat(chunks)

    assert joined is not None
    assert joined.samples.shape == (15,)
    assert joined.sample_rate == OUTPUT_SAMPLE_RATE


def test_concat_should_return_none_when_nothing_usable():
    empty = Waveform.of(np.array([], dtype=np.float32), OUTPUT_SAMPLE_RATE)

    assert Waveform.concat([]) is None
    assert Waveform.concat([empty]) is None


def test_concat_should_reject_mixed_sample_rates():
    # Concaténer du 16 et du 24 kHz produirait un son accéléré, sans erreur visible.
    chunks = [
        Waveform.of(np.ones(4, dtype=np.float32), OUTPUT_SAMPLE_RATE),
        Waveform.of(np.ones(4, dtype=np.float32), INPUT_SAMPLE_RATE),
    ]

    with pytest.raises(ValueError, match="hétérogènes"):
        Waveform.concat(chunks)


def test_should_roundtrip_through_a_wav_file(tmp_path):
    original = Waveform.of(np.linspace(-0.5, 0.5, 480, dtype=np.float32), OUTPUT_SAMPLE_RATE)

    path = original.save(tmp_path / "nested" / "out.wav")
    reloaded = Waveform.from_file(path)

    assert path.exists()
    assert reloaded.sample_rate == OUTPUT_SAMPLE_RATE
    assert reloaded.samples.shape == original.samples.shape
    np.testing.assert_allclose(reloaded.samples, original.samples, atol=1e-4)


def test_as_model_input_should_expose_the_raw_tuple():
    waveform = Waveform.of(np.ones(3, dtype=np.float32), 16_000)

    samples, rate = waveform.as_model_input()

    assert rate == 16_000
    assert samples.shape == (3,)
