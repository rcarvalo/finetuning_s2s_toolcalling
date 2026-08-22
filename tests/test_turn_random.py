"""Per-turn deterministic draws (`lfm2_audio.data_prep.turn_sampling`).

A resumed synthesis run skips turns whose WAV already exists. With a shared
RNG that shift would re-voice every later turn, so the corpus would depend on
where the previous run happened to die.
"""

from __future__ import annotations

from lfm2_audio.data_prep.turn_sampling import turn_random


def test_should_be_stable_for_the_same_turn() -> None:
    first = turn_random(0, "dlg_1", 0).random()
    second = turn_random(0, "dlg_1", 0).random()

    assert first == second


def test_should_differ_across_turns_of_one_dialogue() -> None:
    assert turn_random(0, "dlg_1", 0).random() != turn_random(0, "dlg_1", 1).random()


def test_should_differ_across_dialogues() -> None:
    assert turn_random(0, "dlg_1", 0).random() != turn_random(0, "dlg_2", 0).random()


def test_should_depend_on_the_seed() -> None:
    assert turn_random(0, "dlg_1", 0).random() != turn_random(1, "dlg_1", 0).random()


def test_should_not_depend_on_how_many_turns_were_drawn_before() -> None:
    """The point of the change: an interrupted run must not re-voice the rest."""
    full = [turn_random(0, f"dlg_{i}", 0).randrange(3) for i in range(50)]
    resumed = [turn_random(0, f"dlg_{i}", 0).randrange(3) for i in range(25, 50)]

    assert full[25:] == resumed
