"""Tests du registre et de la fabrique de scorers."""

from __future__ import annotations

import pytest

from lfm2_audio.core.errors import Lfm2AudioError
from lfm2_audio.ds.scoring_config import ScorerConfig, ScoringConfig
from lfm2_audio.scorer.factory import ScorerFactory
from lfm2_audio.scorer.missing import MissingScorer
from lfm2_audio.scorer.registry import SCORERS, ScorerRegistry, UnknownScorerError
from lfm2_audio.scorer.sample import EvalSample
from lfm2_audio.scorer.spec import ScorerSpec
from lfm2_audio.scorer.status import ScoreStatus

PRESENT = ScorerSpec(
    name="present",
    module="lfm2_audio.scorer.text.tool_call",
    class_name="ToolCallScorer",
    requires=("json",),
)
ABSENT = ScorerSpec(
    name="absent",
    module="lfm2_audio.scorer.text.tool_call",
    class_name="ToolCallScorer",
    requires=("definitely_not_a_module",),
)


# --------------------------------------------------------------------------- #
# Registre
# --------------------------------------------------------------------------- #


def test_availability_should_not_import_the_module():
    """Lister les métriques doit marcher sur une machine sans torch."""
    assert ABSENT.unavailable_reason() is not None
    assert "definitely_not_a_module" in ABSENT.unavailable_reason()
    assert PRESENT.unavailable_reason() is None


def test_registry_should_separate_known_from_available():
    registry = ScorerRegistry((PRESENT, ABSENT))

    assert registry.names == ("present", "absent")
    assert registry.available() == ("present",)


def test_registry_should_reject_an_unknown_name():
    with pytest.raises(UnknownScorerError, match="scorer inconnu"):
        ScorerRegistry((PRESENT,)).describe("nope")


def test_describe_should_work_for_an_uninstallable_scorer():
    assert ScorerRegistry((ABSENT,)).describe("absent").name == "absent"


def test_default_registry_should_expose_every_known_metric():
    assert set(SCORERS.names) == {"wer", "asr_wer", "dnsmos", "utmos", "nisqa", "tool_call", "reasoning", "lang_match"}


# --------------------------------------------------------------------------- #
# Fabrique
# --------------------------------------------------------------------------- #


def test_should_build_an_available_scorer():
    registry = ScorerRegistry((PRESENT,))
    config = ScoringConfig(scorers=(ScorerConfig(name="present"),))

    scorers = ScorerFactory(config, registry=registry).build_all()

    assert len(scorers) == 1
    assert not isinstance(scorers[0], MissingScorer)


def test_should_substitute_a_missing_scorer_rather_than_crash():
    registry = ScorerRegistry((ABSENT,))
    config = ScoringConfig(scorers=(ScorerConfig(name="absent"),))

    scorers = ScorerFactory(config, registry=registry).build_all()

    assert isinstance(scorers[0], MissingScorer)
    assert scorers[0].name == "absent"  # la clé du rapport reste celle attendue


def test_missing_scorer_should_report_unavailable_with_its_reason():
    registry = ScorerRegistry((ABSENT,))
    config = ScoringConfig(scorers=(ScorerConfig(name="absent"),))

    result = (
        ScorerFactory(config, registry=registry).build_all()[0].score(EvalSample(sample_id="s1", predicted_text="hi"))
    )

    assert result.status is ScoreStatus.UNAVAILABLE
    assert "definitely_not_a_module" in result.reason


def test_fail_on_unavailable_should_raise():
    registry = ScorerRegistry((ABSENT,))
    config = ScoringConfig(scorers=(ScorerConfig(name="absent"),), fail_on_unavailable=True)

    with pytest.raises(Lfm2AudioError, match="indisponible"):
        ScorerFactory(config, registry=registry).build_all()


def test_disabled_scorers_should_not_be_built():
    registry = ScorerRegistry((PRESENT,))
    config = ScoringConfig(scorers=(ScorerConfig(name="present", enabled=False),))

    assert ScorerFactory(config, registry=registry).build_all() == []


def test_options_should_reach_the_constructor():
    registry = ScorerRegistry((PRESENT,))
    config = ScoringConfig(scorers=(ScorerConfig(name="present", options={"arg_match": "exact"}),))

    scorer = ScorerFactory(config, registry=registry).build_all()[0]

    assert scorer._arg_match == "exact"


def test_default_config_should_enable_every_runnable_metric():
    """``nisqa`` reste enregistré mais hors du jeu par défaut : son architecture
    n'est pas distribuée, donc il ne peut produire aucun score (cf.
    ``NisqaScorer.unavailable_reason``). ``utmos`` couvre le même besoin.
    ``asr_wer`` reste hors défauts aussi : comparer une réponse libre à une
    transcription de référence n'a de sens que sur un benchmark ASR."""
    assert set(ScoringConfig.with_defaults().enabled_names) == set(SCORERS.names) - {"nisqa", "asr_wer", "lang_match"}
