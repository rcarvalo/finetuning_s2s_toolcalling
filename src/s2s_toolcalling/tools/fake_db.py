"""Fake DB en mémoire pour la démo ``db_query`` (données de ``sql/schema_en.sql``).

NL→SQL est dur ; pour une démo on route la question par mots-clés vers la bonne
table et on renvoie les lignes pertinentes (jointures lisibles). C'est l'ASSISTANT
qui formule la réponse parlée à partir de ces données réinjectées — le backend
fournit la DONNÉE, pas la phrase. Déterministe, sans réseau, testable.
"""

from __future__ import annotations

from typing import Any

CUSTOMERS = [
    {"id": 1, "name": "Acme Corp", "country": "US", "segment": "enterprise"},
    {"id": 2, "name": "Globex", "country": "UK", "segment": "enterprise"},
    {"id": 3, "name": "Initech", "country": "US", "segment": "smb"},
    {"id": 4, "name": "Hooli", "country": "US", "segment": "startup"},
    {"id": 5, "name": "Umbrella Ltd", "country": "DE", "segment": "smb"},
]
PRODUCTS = [
    {"id": 1, "name": "Widget Pro", "category": "hardware", "unit_price": 49.99, "in_stock": 1200},
    {"id": 2, "name": "Widget Lite", "category": "hardware", "unit_price": 19.99, "in_stock": 0},
    {"id": 3, "name": "Cloud Plan", "category": "service", "unit_price": 99.00, "in_stock": 9999},
    {"id": 4, "name": "Support Gold", "category": "service", "unit_price": 499.00, "in_stock": 9999},
    {"id": 5, "name": "Gadget X", "category": "hardware", "unit_price": 149.00, "in_stock": 30},
]
EMPLOYEES = [
    {"id": 1, "full_name": "Alice Johnson", "team": "Sales", "title": "Account Executive", "hired_at": "2022-03-01"},
    {"id": 2, "full_name": "Bob Smith", "team": "Engineering", "title": "Backend Engineer", "hired_at": "2021-09-15"},
    {"id": 3, "full_name": "Carol Lee", "team": "Support", "title": "Support Lead", "hired_at": "2023-01-10"},
    {"id": 4, "full_name": "David Kim", "team": "Sales", "title": "Sales Manager", "hired_at": "2020-06-20"},
]
_ORDERS = [
    {"id": 1, "customer_id": 1, "product_id": 1, "quantity": 10, "total_amount": 499.90, "status": "shipped"},
    {"id": 2, "customer_id": 1, "product_id": 3, "quantity": 1, "total_amount": 99.00, "status": "open"},
    {"id": 3, "customer_id": 2, "product_id": 5, "quantity": 2, "total_amount": 298.00, "status": "delivered"},
    {"id": 4, "customer_id": 3, "product_id": 1, "quantity": 5, "total_amount": 249.95, "status": "open"},
    {"id": 5, "customer_id": 4, "product_id": 4, "quantity": 1, "total_amount": 499.00, "status": "cancelled"},
]
MEETINGS = [
    {"id": 1, "employee": "Alice Johnson", "customer": "Acme Corp", "subject": "Quarterly review", "in_days": 2},
    {"id": 2, "employee": "David Kim", "customer": "Globex", "subject": "Renewal discussion", "in_days": 5},
]

_CUST = {c["id"]: c["name"] for c in CUSTOMERS}
_PROD = {p["id"]: p["name"] for p in PRODUCTS}
# orders avec noms (lisible par le modèle)
ORDERS = [
    {"id": o["id"], "customer": _CUST[o["customer_id"]], "product": _PROD[o["product_id"]],
     "quantity": o["quantity"], "total_amount": o["total_amount"], "status": o["status"]}
    for o in _ORDERS
]

# (table, mots-clés) — ordre = priorité (orders avant customers/products car « order » domine)
_ROUTES: list[tuple[str, list[str], list[dict]]] = [
    ("orders", ["order", "orders", "invoice", "shipped", "delivered", "cancelled", "open order", "purchase"], ORDERS),
    ("meetings", ["meeting", "schedule", "appointment", "calendar"], MEETINGS),
    ("employees", ["employee", "employees", "staff", "team", "hire", "hired", "sales team", "engineer"], EMPLOYEES),
    ("products", ["product", "products", "stock", "inventory", "price", "catalog"], PRODUCTS),
    ("customers", ["customer", "customers", "client", "clients", "account"], CUSTOMERS),
]


class FakeDbBackend:
    """Route la question NL vers une table et renvoie ses lignes (+ count)."""

    async def answer(self, question: str) -> dict[str, Any]:
        q = question.lower()
        for table, keywords, rows in _ROUTES:
            if any(k in q for k in keywords):
                return {"question": question, "table": table, "rows": rows, "row_count": len(rows)}
        # repli : renvoie un aperçu de chaque table
        return {"question": question, "table": None,
                "rows": {"customers": CUSTOMERS, "products": PRODUCTS, "employees": EMPLOYEES,
                         "orders": ORDERS, "meetings": MEETINGS},
                "note": "no table matched the question; full snapshot returned"}
