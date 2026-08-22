from lfm2_audio.evaluation.toolcalling import Report, score_case

CALL = '<|tool_call_start|>[check_appointment(visitor_name="Marie Dupont")]<|tool_call_end|>'


def test_exact_match():
    r = score_case("c1", CALL, [{"name": "check_appointment", "arguments": {"visitor_name": "Marie Dupont"}}])
    assert r.call_correct and r.name_correct and r.predicted_call


def test_accent_and_case_tolerant():
    r = score_case("c2", CALL, [{"name": "check_appointment", "arguments": {"visitor_name": "marie dupont"}}])
    assert r.call_correct


def test_wrong_arguments():
    r = score_case("c3", CALL, [{"name": "check_appointment", "arguments": {"visitor_name": "Jean Petit"}}])
    assert r.name_correct and not r.call_correct


def test_wrong_function():
    r = score_case("c4", CALL, [{"name": "get_guest_wifi", "arguments": {}}])
    assert not r.name_correct and not r.call_correct
    assert r.expected_call == r.predicted_call  # pertinence OK (un appel était attendu)


def test_true_negative():
    r = score_case("c5", "Bonjour, bienvenue !", [])
    assert not r.predicted_call and not r.expected_call


def test_false_positive_on_negative_case():
    r = score_case("c6", CALL, [])
    assert r.predicted_call and not r.expected_call


def test_malformed_call_counts_as_attempt():
    r = score_case("c7", "<|tool_call_start|>[broken(]<|tool_call_end|>", [])
    assert not r.parsed
    assert r.predicted_call  # tentative malformée = tentative


def test_report_summary():
    report = Report()
    report.add(score_case("a", CALL, [{"name": "check_appointment", "arguments": {"visitor_name": "Marie Dupont"}}]))
    report.add(score_case("b", "Bonjour !", []))
    s = report.summary()
    assert s["cases"] == 2
    assert s["relevance_accuracy"] == 1.0
    assert s["call_accuracy"] == 1.0
    assert s["parse_rate"] == 1.0
