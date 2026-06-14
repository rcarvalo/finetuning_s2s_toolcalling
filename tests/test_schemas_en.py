from s2s_toolcalling.tools.schemas import (
    DB_QUERY,
    TOOLCALLING_EN_TOOL_DEFINITIONS,
    TOOLCALLING_EN_TOOL_NAMES,
    WEB_SEARCH,
)


def test_en_tool_set_has_two_tools():
    assert TOOLCALLING_EN_TOOL_NAMES == ["web_search", "db_query"]
    assert TOOLCALLING_EN_TOOL_DEFINITIONS == [WEB_SEARCH, DB_QUERY]


def test_web_search_schema_shape():
    assert WEB_SEARCH["name"] == "web_search"
    assert WEB_SEARCH["parameters"]["required"] == ["query"]
    assert "query" in WEB_SEARCH["parameters"]["properties"]


def test_db_query_takes_natural_language_question():
    assert DB_QUERY["name"] == "db_query"
    assert DB_QUERY["parameters"]["required"] == ["question"]
    # db_query est NL (question), pas SQL : pas de paramètre sql.
    assert "sql" not in DB_QUERY["parameters"]["properties"]


def test_every_definition_has_required_fields():
    for d in TOOLCALLING_EN_TOOL_DEFINITIONS:
        assert {"name", "description", "parameters"} <= set(d)
        assert d["parameters"]["type"] == "object"
