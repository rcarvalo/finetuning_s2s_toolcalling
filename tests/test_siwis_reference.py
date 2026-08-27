"""Tests of the SIWIS reference resolution."""

from __future__ import annotations

from lfm2_audio.data_prep.siwis_reference import pair_stems

FILES = [
    "wavs/part1/neut_parl_s01_0106.wav",
    "wavs/part4/emph_book_s01_0343.wav",
    "wavs/part4/chap_full.wav",
    "labs/part1/neut_parl_s01_0149.lab",
    "labs/part1/neut_parl_s01_0106.lab",
    "text/part4/emph_book_s01_0343.txt",
    "text/part4/chap_full.txt",
    "README.txt",
    "lists/all_wavs.list",
]


def test_should_pair_on_transcripts_not_phonetic_labels() -> None:
    """The .lab for neut_parl_s01_0106 exists, but labs are phonetic labels for
    another set of utterances; pairing on them returns an empty intersection
    that reads as 'nothing available' and cost two runs their main candidate."""
    pairs = pair_stems(FILES)

    stems = [stem for stem, _, _ in pairs]
    assert stems == ["chap_full", "emph_book_s01_0343"]
    assert "neut_parl_s01_0106" not in stems


def test_should_return_the_matching_paths() -> None:
    pairs = dict((stem, (wav, text)) for stem, wav, text in pair_stems(FILES))

    assert pairs["emph_book_s01_0343"] == (
        "wavs/part4/emph_book_s01_0343.wav",
        "text/part4/emph_book_s01_0343.txt",
    )


def test_should_ignore_text_files_outside_the_transcript_directory() -> None:
    """README.txt is a text file and not a transcript."""
    pairs = pair_stems([*FILES, "wavs/part1/README.wav"])

    assert "README" not in [stem for stem, _, _ in pairs]


def test_should_return_nothing_when_no_clip_has_a_transcript() -> None:
    assert pair_stems(["wavs/a.wav", "labs/a.lab"]) == []


def test_neutral_register_should_come_first() -> None:
    """Cloning carries the speaking style over with the timbre: the first
    reference used here was an emphatic book reading and the clone was judged
    'very robotic' by ear. Naturalness metrics cannot see a register mismatch."""
    from lfm2_audio.data_prep.siwis_reference import candidates

    files = [
        "wavs/part1/emph_book_s01_0343.wav",
        "text/part1/emph_book_s01_0343.txt",
        "wavs/part1/neut_parl_s01_0106.wav",
        "wavs/part1/expr_book_s01_0020.wav",
    ]

    order = [stem for stem, _, _ in candidates(files)]

    assert order[0] == "neut_parl_s01_0106"
    assert order.index("expr_book_s01_0020") < order.index("emph_book_s01_0343")


def test_a_clip_without_transcript_must_stay_a_candidate() -> None:
    """No neutral clip in this corpus ships a transcript, so dropping
    transcript-less clips silently forces the theatrical registers."""
    from lfm2_audio.data_prep.siwis_reference import candidates

    listed = candidates(["wavs/part1/neut_parl_s01_0106.wav"])

    assert listed == [("neut_parl_s01_0106", "wavs/part1/neut_parl_s01_0106.wav", None)]
