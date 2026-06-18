import pytest
from s2s_toolcalling.tools.fake_db import FakeDbBackend


@pytest.fixture
def db():
    return FakeDbBackend()


async def test_routes_orders(db):
    out = await db.answer("how many open orders do we have")
    assert out["table"] == "orders"
    assert out["row_count"] == len(out["rows"]) > 0
    assert {"customer", "product", "status"} <= set(out["rows"][0])  # jointures lisibles


async def test_routes_products(db):
    out = await db.answer("which products are out of stock")
    assert out["table"] == "products"
    assert any(r["in_stock"] == 0 for r in out["rows"])


async def test_routes_employees(db):
    out = await db.answer("who on the sales team was hired most recently")
    assert out["table"] == "employees"


async def test_routes_customers(db):
    out = await db.answer("list our enterprise clients")
    assert out["table"] == "customers"


async def test_routes_meetings(db):
    out = await db.answer("what meetings are scheduled this week")
    assert out["table"] == "meetings"


async def test_unmatched_returns_snapshot(db):
    out = await db.answer("tell me something")
    assert out["table"] is None
    assert set(out["rows"]) == {"customers", "products", "employees", "orders", "meetings"}


async def test_orders_have_resolved_names(db):
    out = await db.answer("show recent orders")
    customers = {r["customer"] for r in out["rows"]}
    assert "Acme Corp" in customers  # ids résolus en noms
