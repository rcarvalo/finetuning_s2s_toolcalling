import lfm2_audio.cli.analyze_dataset as ad


def _pos(tool, q, **meta):
    key = "query" if tool == "web_search" else "question"
    return {
        "id": "x",
        "meta": {"target": tool, **meta},
        "turns": [
            {"role": "user", "text": q, "voice": meta.get("voice")},
            {"role": "assistant", "tool_calls": [{"name": tool, "arguments": {key: q}}]},
        ],
    }


def _neg(utt, ans, **meta):
    return {
        "id": "n",
        "meta": {"target": "none", **meta},
        "turns": [{"role": "user", "text": utt}, {"role": "assistant", "text": ans}],
    }


def test_distribution_counts_and_neg_ratio():
    rows = [_pos("web_search", "a"), _pos("db_query", "b"), _neg("hi", "hello")]
    d = ad.distribution(rows)
    assert d["n"] == 3
    assert d["targets"] == {"web_search": 1, "db_query": 1, "none": 1}
    assert round(d["neg_ratio"], 2) == 0.33


def test_text_quality_dups_and_empty_and_placeholder():
    rows = [
        _pos("web_search", "search cats"),
        _pos("web_search", "Search Cats"),  # doublon (normalisé)
        _pos("db_query", ""),  # arg vide
        _neg("what time", "It's [current time]."),  # négatif à trou
    ]
    t = ad.text_quality(rows)
    assert t["duplicate_utterances"] == 1
    assert t["placeholder_negatives"] == 1
    assert t["args"]["db_query"]["empty"] == 1


def test_voices():
    rows = [_pos("web_search", "a", voice="casual_male"), _pos("db_query", "b", voice="neutral_female")]
    assert ad.voices(rows) == {"casual_male": 1, "neutral_female": 1}


def test_flags_clean_dataset_has_none():
    rows = (
        [_pos("web_search", f"query number {i}") for i in range(7)]
        + [_pos("db_query", f"question number {i}") for i in range(7)]
        + [_neg(f"chit chat {i}", f"reply {i}") for i in range(6)]
    )
    assert ad.flags(ad.analyze(rows)) == []


def test_flags_detects_problems():
    rows = [
        _pos("web_search", "same"),
        _pos("web_search", "same"),  # dup + low diversity
        _neg("t", "It's [x]."),
    ]  # placeholder + neg ratio off
    issues = ad.flags(ad.analyze(rows))
    assert any("dupliqu" in f for f in issues)
    assert any("placeholder" in f for f in issues)


def test_target_inferred_without_meta():
    row = {
        "id": "x",
        "turns": [
            {"role": "user", "text": "q"},
            {"role": "assistant", "tool_calls": [{"name": "web_search", "arguments": {"query": "q"}}]},
        ],
    }
    assert ad._target(row) == "web_search"
