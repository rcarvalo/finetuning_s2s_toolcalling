import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_hf_dataset as bhd  # noqa: E402
import hf_to_dialogues as h2d  # noqa: E402


def _roundtrip(dlg):
    row = bhd.dialogue_to_row(dlg, "/audio")
    audio_rel = Path(row["audio"]).name
    return h2d.row_to_dialogue(row, audio_rel)


def test_roundtrip_positive():
    dlg = {"id": "tc_1", "meta": {"target": "web_search", "style": "direct command", "depth": "explicit arguments"},
           "turns": [{"role": "user", "text": "search dogs", "audio": "tc_1_u0.wav", "voice": "casual_male"},
                     {"role": "assistant", "tool_calls": [{"name": "web_search", "arguments": {"query": "dogs"}}]}]}
    back = _roundtrip(dlg)
    assert back["id"] == "tc_1"
    assert back["turns"][0]["text"] == "search dogs" and back["turns"][0]["audio"] == "tc_1_u0.wav"
    assert back["turns"][0]["voice"] == "casual_male"
    assert back["turns"][1]["tool_calls"] == [{"name": "web_search", "arguments": {"query": "dogs"}}]
    assert back["meta"]["target"] == "web_search"


def test_roundtrip_negative():
    dlg = {"id": "n1", "meta": {"target": "none"},
           "turns": [{"role": "user", "text": "thanks", "audio": "n1_u0.wav"},
                     {"role": "assistant", "text": "You're welcome!"}]}
    back = _roundtrip(dlg)
    assert back["turns"][1] == {"role": "assistant", "text": "You're welcome!"}
    assert "tool_calls" not in back["turns"][1]


def test_roundtrip_parses_as_valid_dialogue():
    from s2s_toolcalling.data.dialogue_schema import parse_dialogue

    dlg = {"id": "x", "meta": {"target": "db_query"},
           "turns": [{"role": "user", "text": "how many orders", "audio": "x_u0.wav"},
                     {"role": "assistant", "tool_calls": [{"name": "db_query", "arguments": {"question": "how many orders"}}]}]}
    parsed = parse_dialogue(_roundtrip(dlg))  # ne doit pas lever
    assert parsed.turns[1].tool_calls[0].name == "db_query"
