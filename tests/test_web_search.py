import pytest

from s2s_toolcalling.tools.toolcalling_en import StubDbQueryBackend, build_toolcalling_en_registry
from s2s_toolcalling.tools.web_search import StubWebSearchBackend, web_search_handler


async def test_stub_backend_returns_k_results():
    backend = StubWebSearchBackend(max_results=3)
    results = await backend.search("weather in Paris")
    assert len(results) == 3
    assert all({"title", "url", "snippet"} <= set(r) for r in results)
    assert "weather" in results[0]["snippet"]


async def test_web_search_handler_shape():
    out = await web_search_handler(StubWebSearchBackend(), "latest news")
    assert out["query"] == "latest news"
    assert isinstance(out["results"], list) and out["results"]


async def test_stub_db_backend_answers_question():
    out = await StubDbQueryBackend().answer("how many open orders")
    assert out["question"] == "how many open orders"
    assert "answer" in out


def test_registry_registers_both_tools():
    registry = build_toolcalling_en_registry()
    assert registry.names == ["db_query", "web_search"]
    assert {d["name"] for d in registry.definitions()} == {"web_search", "db_query"}


async def test_registry_executes_web_search():
    registry = build_toolcalling_en_registry()
    result = await registry.execute("web_search", {"query": "BFCL leaderboard"})
    assert result.ok
    assert result.content["query"] == "BFCL leaderboard"


async def test_registry_executes_db_query():
    registry = build_toolcalling_en_registry()
    result = await registry.execute("db_query", {"question": "top 3 customers by revenue"})
    assert result.ok and "answer" in result.content


async def test_missing_required_argument_rejected():
    registry = build_toolcalling_en_registry()
    result = await registry.execute("web_search", {})
    assert not result.ok and "missing required" in result.error


async def test_unknown_argument_rejected():
    registry = build_toolcalling_en_registry()
    result = await registry.execute("db_query", {"question": "x", "sql": "DROP TABLE"})
    assert not result.ok and "unknown argument" in result.error


@pytest.mark.parametrize("name", ["web_search", "db_query"])
def test_definitions_match_schemas(name):
    registry = build_toolcalling_en_registry()
    assert name in registry
