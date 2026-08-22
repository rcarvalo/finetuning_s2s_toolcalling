import json

import lfm2_audio.data_prep.hf_dataset as bhd


def test_dialogue_to_row_positive():
    dlg = {
        "id": "tc_1",
        "meta": {"target": "web_search", "style": "direct command", "depth": "explicit arguments"},
        "turns": [
            {"role": "user", "text": "search dogs", "audio": "tc_1_u0.wav", "voice": "casual_male"},
            {"role": "assistant", "tool_calls": [{"name": "web_search", "arguments": {"query": "dogs"}}]},
        ],
    }
    row = bhd.dialogue_to_row(dlg, "/audio")
    assert row["audio"] == "/audio/tc_1_u0.wav"
    assert row["has_tool_call"] is True
    assert row["tool_name"] == "web_search"
    assert json.loads(row["arguments"]) == {"query": "dogs"}
    assert json.loads(row["expected_calls"]) == [{"name": "web_search", "arguments": {"query": "dogs"}}]
    assert row["assistant_text"] is None
    assert row["voice"] == "casual_male" and row["target"] == "web_search"


def test_dialogue_to_row_negative():
    dlg = {
        "id": "tc_2",
        "meta": {"target": "none"},
        "turns": [
            {"role": "user", "text": "hello", "audio": "tc_2_u0.wav"},
            {"role": "assistant", "text": "Hi there!"},
        ],
    }
    row = bhd.dialogue_to_row(dlg, "/audio")
    assert row["has_tool_call"] is False
    assert row["tool_name"] is None and row["arguments"] is None
    assert row["assistant_text"] == "Hi there!"
    assert json.loads(row["expected_calls"]) == []


def test_dialogue_to_row_missing_audio_raises():
    dlg = {"id": "x", "turns": [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "yo"}]}
    try:
        bhd.dialogue_to_row(dlg, "/audio")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "audio" in str(e)


def test_load_rows(tmp_path):
    dlg = {
        "id": "a",
        "turns": [
            {"role": "user", "text": "search cats", "audio": "a_u0.wav"},
            {"role": "assistant", "tool_calls": [{"name": "web_search", "arguments": {"query": "cats"}}]},
        ],
    }
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps(dlg) + "\n", encoding="utf-8")
    rows = bhd.load_rows(p, "/audio")
    assert len(rows) == 1 and rows[0]["id"] == "a"


def test_dataset_card_is_neutral_about_engine():
    card = bhd.dataset_card("Rcarvalo/tc-en-audio", "cc-by-nc-4.0", 100, 20)
    assert "cc-by-nc-4.0" in card
    assert "voxtral" not in card.lower()  # carte neutre : pas de mention du moteur TTS
