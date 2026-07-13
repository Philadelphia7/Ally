from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        self._tools[definition.name] = definition

    def list(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.openai_schema() for tool in self.list()]

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].handler(arguments)


def build_default_registry(database_url: str | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_build_medication_adherence_tool(database_url))
    return registry


def _build_medication_adherence_tool(database_url: str | None) -> ToolDefinition:
    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        date = arguments.get("date", "the requested date")
        if not database_url:
            return {
                "status": "unavailable",
                "date": date,
                "message": (
                    "Medication adherence database is not configured yet. "
                    "Connect MEDICATION_DATABASE_URL to answer this from device data."
                ),
            }
        return {
            "status": "unimplemented",
            "date": date,
            "message": "Database URL is configured, but the medication adapter is not implemented yet.",
        }

    return ToolDefinition(
        name="get_medication_adherence",
        description="Check whether the patient used their medication on a requested date.",
        parameters={
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Natural language or ISO date, for example 'yesterday' or '2026-07-12'.",
                }
            },
            "required": ["date"],
            "additionalProperties": False,
        },
        handler=handler,
    )
