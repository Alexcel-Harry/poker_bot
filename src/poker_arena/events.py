from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poker_arena.cards import Card


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def _json_safe(value: Any) -> JsonValue:
    if isinstance(value, Card):
        return value.to_str()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value"):
        raw = value.value
        if isinstance(raw, (str, int, float, bool)):
            return raw
    raise TypeError(f"Value is not JSON-safe: {value!r}")


@dataclass(frozen=True)
class PokerEvent:
    event_type: str
    data: dict[str, JsonValue]

    def to_dict(self) -> dict[str, JsonValue]:
        return {"event_type": self.event_type, "data": _json_safe(self.data)}


def event_from_dict(data: dict[str, Any]) -> PokerEvent:
    return PokerEvent(event_type=str(data["event_type"]), data=_json_safe(data.get("data", {})))  # type: ignore[arg-type]
