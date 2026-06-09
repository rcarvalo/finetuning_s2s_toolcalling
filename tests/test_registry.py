import asyncio

import pytest

from s2s_toolcalling.tools.reception import InMemoryReceptionBackend, build_reception_registry
from s2s_toolcalling.tools.registry import ToolRegistry
from s2s_toolcalling.tools.schemas import RECEPTION_TOOL_DEFINITIONS


@pytest.fixture
def registry():
    return build_reception_registry(InMemoryReceptionBackend())


async def test_check_appointment_found(registry):
    result = await registry.execute("check_appointment", {"visitor_name": "Marie Dupont"})
    assert result.ok
    assert result.content["found"] is True
    assert result.content["appointments"][0]["host"] == "Claire Martin"


async def test_check_appointment_accent_insensitive(registry):
    result = await registry.execute("check_appointment", {"visitor_name": "marie dupont", "host_name": "claire"})
    assert result.ok and result.content["found"]


async def test_check_appointment_not_found(registry):
    result = await registry.execute("check_appointment", {"visitor_name": "Personne Inconnue"})
    assert result.ok
    assert result.content["found"] is False


async def test_notify_employee(registry):
    result = await registry.execute("notify_employee", {"employee_name": "Karim", "message": "Visiteur arrivé"})
    assert result.ok and result.content["delivered"]
    assert result.content["employee"] == "Karim Benali"


async def test_notify_unknown_employee(registry):
    result = await registry.execute("notify_employee", {"employee_name": "Inconnu", "message": "x"})
    assert result.ok
    assert result.content["delivered"] is False


async def test_guide_visitor(registry):
    result = await registry.execute("guide_visitor", {"destination": "cafétéria"})
    assert result.ok and result.content["found"]
    assert "cafétéria" in result.content["name"]


async def test_get_guest_wifi(registry):
    result = await registry.execute("get_guest_wifi", {})
    assert result.ok and result.content["ssid"]


async def test_notify_receptionist(registry):
    result = await registry.execute("notify_receptionist", {"reason": "demande complexe", "urgency": "high"})
    assert result.ok and result.content["delivered"]


async def test_missing_required_argument(registry):
    result = await registry.execute("check_appointment", {})
    assert not result.ok
    assert "missing required" in result.error


async def test_unknown_argument_rejected(registry):
    result = await registry.execute("get_guest_wifi", {"hack": 1})
    assert not result.ok and "unknown argument" in result.error


async def test_unknown_tool(registry):
    result = await registry.execute("rm_rf", {})
    assert not result.ok and "unknown tool" in result.error


async def test_handler_exception_is_contained():
    reg = ToolRegistry()

    async def boom():
        raise RuntimeError("kaboom")

    reg.register({"name": "boom", "description": "", "parameters": {"type": "object", "properties": {}}}, boom)
    result = await reg.execute("boom", {})
    assert not result.ok and "kaboom" in result.error
    assert result.payload() == {"error": "RuntimeError: kaboom"}


async def test_timeout():
    reg = ToolRegistry(timeout_s=0.05)

    async def slow():
        await asyncio.sleep(1)

    reg.register({"name": "slow", "description": "", "parameters": {"type": "object", "properties": {}}}, slow)
    result = await reg.execute("slow", {})
    assert not result.ok and "timed out" in result.error


def test_definitions_match_schemas(registry):
    base_names = {t["name"] for t in RECEPTION_TOOL_DEFINITIONS} - {"query_database", "search_knowledge_base"}
    assert {d["name"] for d in registry.definitions()} == base_names


def test_execute_sync(registry):
    result = registry.execute_sync("get_guest_wifi", {})
    assert result.ok
