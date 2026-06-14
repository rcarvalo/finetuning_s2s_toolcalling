from s2s_toolcalling.evaluation.eval_toolcalling import calls_match, score_case, token_f1


def test_token_f1_bounds():
    assert token_f1("open orders count", "open orders count") == 1.0
    assert token_f1("apples", "oranges") == 0.0


def test_token_f1_partial_overlap():
    # 2 tokens communs ({open, orders}), 2 ensembles de 3 → P=R=2/3 → F1 = 0.667
    assert round(token_f1("open orders count", "open orders today"), 2) == 0.67


def test_exact_match_rejects_paraphrase():
    pred = {"name": "db_query", "arguments": {"question": "how many open orders are there"}}
    exp = {"name": "db_query", "arguments": {"question": "how many open orders"}}
    assert not calls_match(pred, exp, arg_match="exact")


def test_token_f1_accepts_paraphrase():
    pred = {"name": "db_query", "arguments": {"question": "how many open orders are there"}}
    exp = {"name": "db_query", "arguments": {"question": "how many open orders"}}
    assert calls_match(pred, exp, arg_match="token_f1", threshold=0.6)


def test_token_f1_rejects_different_meaning():
    pred = {"name": "web_search", "arguments": {"query": "weather in Tokyo"}}
    exp = {"name": "web_search", "arguments": {"query": "stock price of Apple"}}
    assert not calls_match(pred, exp, arg_match="token_f1", threshold=0.6)


def test_name_mismatch_never_matches():
    pred = {"name": "web_search", "arguments": {"query": "x"}}
    exp = {"name": "db_query", "arguments": {"question": "x"}}
    assert not calls_match(pred, exp, arg_match="token_f1", threshold=0.0)


def test_score_case_uses_tolerant_args():
    from s2s_toolcalling.data.chat_format import render_tool_calls

    predicted_text = render_tool_calls([("web_search", {"query": "latest news about mars rover"})])
    expected = [{"name": "web_search", "arguments": {"query": "news about mars rover"}}]
    exact = score_case("x", predicted_text, expected, arg_match="exact")
    tol = score_case("x", predicted_text, expected, arg_match="token_f1", threshold=0.6)
    assert exact.name_correct and not exact.call_correct
    assert tol.call_correct
