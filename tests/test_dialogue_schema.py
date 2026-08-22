from pathlib import Path

import pytest

from lfm2_audio.ds.dialogue import DialogueValidationError, load_dialogues, parse_dialogue

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "examples" / "dialogues.sample.jsonl"


def test_sample_file_parses():
    dialogues = list(load_dialogues(SAMPLE))
    assert len(dialogues) == 4
    d = dialogues[0]
    assert d.id == "dlg_0001"
    assert d.turns[1].tool_calls[0].name == "check_appointment"
    assert d.turns[2].role == "tool"
    assert d.turns[3].audio == "audio/dlg_0001_a1.wav"


def test_tool_call_turn_with_audio_rejected():
    obj = {
        "id": "bad",
        "turns": [
            {"role": "user", "text": "hi"},
            {"role": "assistant", "audio": "a.wav", "tool_calls": [{"name": "f", "arguments": {}}]},
        ],
    }
    with pytest.raises(DialogueValidationError, match="texte seul"):
        parse_dialogue(obj)


def test_tool_turn_requires_content():
    obj = {"id": "bad", "turns": [{"role": "tool"}]}
    with pytest.raises(DialogueValidationError, match="content"):
        parse_dialogue(obj)


def test_invalid_role_rejected():
    obj = {"id": "bad", "turns": [{"role": "wizard", "text": "x"}]}
    with pytest.raises(DialogueValidationError, match="role"):
        parse_dialogue(obj)


def test_empty_user_turn_rejected():
    obj = {"id": "bad", "turns": [{"role": "user"}]}
    with pytest.raises(DialogueValidationError):
        parse_dialogue(obj)


def test_invalid_json_line_reports_position(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "ok", "turns": [{"role": "user", "text": "x"}]}\n{broken\n', encoding="utf-8")
    with pytest.raises(DialogueValidationError, match=":2:"):
        list(load_dialogues(p))


def test_negative_dialogue_has_no_tool_calls():
    dialogues = {d.id: d for d in load_dialogues(SAMPLE)}
    negative = dialogues["dlg_0003_negatif"]
    assert all(not t.tool_calls for t in negative.turns)
