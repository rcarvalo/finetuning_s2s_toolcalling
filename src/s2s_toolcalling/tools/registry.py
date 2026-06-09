"""Registre et dispatch des outils (Phase 3).

Le registre associe chaque schéma JSON à un handler async, valide les
arguments requis et normalise le résultat en ``ToolResult`` — c'est ce
payload qui est réinjecté dans le contexte du modèle via
``chat_format.render_tool_response``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

ToolHandler = Callable[..., Awaitable[Any]]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    @property
    def required(self) -> list[str]:
        return list(self.parameters.get("required", []))

    @property
    def known_params(self) -> set[str]:
        return set(self.parameters.get("properties", {}).keys())


@dataclass(slots=True)
class ToolResult:
    name: str
    ok: bool
    content: Any = None
    error: str | None = None
    elapsed_ms: float = 0.0

    def payload(self) -> Any:
        """Payload réinjecté au modèle (rôle ``tool``)."""
        if self.ok:
            return self.content
        return {"error": self.error or "tool execution failed"}


@dataclass
class ToolRegistry:
    timeout_s: float = 10.0
    _tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, definition: dict[str, Any], handler: ToolHandler) -> None:
        name = definition["name"]
        if name in self._tools:
            raise ValueError(f"tool {name!r} already registered")
        if not inspect.iscoroutinefunction(handler):
            raise TypeError(f"handler for {name!r} must be async")
        self._tools[name] = ToolSpec(
            name=name,
            description=definition.get("description", ""),
            parameters=definition.get("parameters", {"type": "object", "properties": {}}),
            handler=handler,
        )

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def definitions(self) -> list[dict[str, Any]]:
        """Schémas JSON (pour ``<|tool_list_start|>``), dans l'ordre d'enregistrement."""
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters} for t in self._tools.values()
        ]

    def validate(self, name: str, arguments: dict[str, Any]) -> str | None:
        """Retourne un message d'erreur, ou None si l'appel est valide."""
        spec = self._tools.get(name)
        if spec is None:
            return f"unknown tool: {name}"
        missing = [p for p in spec.required if p not in arguments]
        if missing:
            return f"missing required argument(s) for {name}: {', '.join(missing)}"
        unknown = set(arguments) - spec.known_params
        if unknown:
            return f"unknown argument(s) for {name}: {', '.join(sorted(unknown))}"
        return None

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        start = time.monotonic()
        error = self.validate(name, arguments)
        if error:
            logger.warning("tool call rejected: %s", error)
            return ToolResult(name=name, ok=False, error=error)

        spec = self._tools[name]
        try:
            content = await asyncio.wait_for(spec.handler(**arguments), timeout=self.timeout_s)
            return ToolResult(name=name, ok=True, content=content, elapsed_ms=(time.monotonic() - start) * 1000)
        except asyncio.TimeoutError:
            logger.error("tool %s timed out after %.1fs", name, self.timeout_s)
            return ToolResult(name=name, ok=False, error=f"{name} timed out", elapsed_ms=(time.monotonic() - start) * 1000)
        except Exception as e:  # un outil ne doit jamais faire tomber l'agent
            logger.exception("tool %s failed", name)
            return ToolResult(name=name, ok=False, error=f"{type(e).__name__}: {e}", elapsed_ms=(time.monotonic() - start) * 1000)

    def execute_sync(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Exécution depuis un thread sans event loop (boucle de génération GPU)."""
        return asyncio.run(self.execute(name, arguments))
