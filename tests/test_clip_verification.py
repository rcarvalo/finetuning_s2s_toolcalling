"""La règle d'acceptation d'un clip : forme parlée, puis WER ou CER."""

from __future__ import annotations

import pytest

from lfm2_audio.data_prep.clip_verification import accepted, verification_rates


def test_should_agree_when_only_the_number_spelling_differs() -> None:
    assert verification_rates("L'accueil ferme à dix-neuf heures pile.", "L'accueil ferme à 19h pile.", "fr") == (
        0.0,
        0.0,
    )


def test_should_still_measure_a_real_miss() -> None:
    wer, cer = verification_rates("Bon passage parmi nous !", "Bon passage.", "fr")

    assert wer == pytest.approx(0.5)
    assert cer > 0.10


def test_should_keep_on_either_rate_and_refuse_on_both() -> None:
    assert accepted(0.15, 0.5, max_wer=0.15, max_cer=0.10)
    assert accepted(0.22, 0.03, max_wer=0.15, max_cer=0.10)
    assert not accepted(0.4, 0.3, max_wer=0.15, max_cer=0.10)
