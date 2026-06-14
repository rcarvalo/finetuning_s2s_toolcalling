import json
import sys
from pathlib import Path

from s2s_toolcalling.data.chat_format import render_tool_calls

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import eval_audio_toolcalling as eah  # noqa: E402


def _cases_and_predictions():
    """3 cas : positif correct, négatif correct, positif avec MAUVAIS outil."""
    cases = [
        {"id": "c1", "audio": "c1.wav",
         "expected_calls": [{"name": "web_search", "arguments": {"query": "news about mars"}}]},
        {"id": "c2", "audio": "c2.wav", "expected_calls": []},
        {"id": "c3", "audio": "c3.wav",
         "expected_calls": [{"name": "db_query", "arguments": {"question": "how many open orders"}}]},
    ]
    predicted = {
        "c1": render_tool_calls([("web_search", {"query": "news about mars"})]),
        "c2": "Sure, I can help with that.",  # pas d'appel → bon pour un négatif
        "c3": render_tool_calls([("web_search", {"query": "open orders"})]),  # mauvais outil
    }
    return cases, predicted


def test_run_audio_eval_with_mock_model():
    cases, predicted = _cases_and_predictions()
    predictions, report = eah.run_audio_eval(cases, lambda case: predicted[case["id"]])

    assert [p["id"] for p in predictions] == ["c1", "c2", "c3"]
    summary = report.summary()
    assert summary["cases"] == 3
    assert summary["positives"] == 2 and summary["negatives"] == 1
    assert summary["parse_rate"] == 1.0
    assert summary["relevance_accuracy"] == 1.0  # appel/abstention corrects partout
    assert summary["name_accuracy"] == 0.5       # c3 appelle le mauvais outil
    assert summary["call_accuracy"] == 0.5


def test_load_cases(tmp_path):
    p = tmp_path / "cases.jsonl"
    p.write_text('{"id": "a", "expected_calls": []}\n\n{"id": "b", "expected_calls": []}\n', encoding="utf-8")
    cases = eah.load_cases(p)
    assert [c["id"] for c in cases] == ["a", "b"]


def test_load_cases_accepts_dialogue_schema(tmp_path):
    # un dialogue single-turn (TTS-able) doit aussi être éval-able directement
    dlg = {
        "id": "d1",
        "turns": [
            {"role": "user", "text": "search the news", "audio": "d1_u0.wav"},
            {"role": "assistant", "tool_calls": [{"name": "web_search", "arguments": {"query": "the news"}}]},
        ],
    }
    neg = {"id": "d2", "turns": [
        {"role": "user", "text": "hello", "audio": "d2_u0.wav"},
        {"role": "assistant", "text": "Hi there!"},
    ]}
    p = tmp_path / "bench.jsonl"
    p.write_text(json.dumps(dlg) + "\n" + json.dumps(neg) + "\n", encoding="utf-8")
    cases = eah.load_cases(p)
    assert cases[0] == {"id": "d1", "audio": "d1_u0.wav",
                        "expected_calls": [{"name": "web_search", "arguments": {"query": "the news"}}]}
    assert cases[1] == {"id": "d2", "audio": "d2_u0.wav", "expected_calls": []}
