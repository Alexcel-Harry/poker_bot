from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE_TO = "raise_to"


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    total: int | None = None

    @classmethod
    def fold(cls) -> Action:
        return cls(ActionType.FOLD)

    @classmethod
    def check(cls) -> Action:
        return cls(ActionType.CHECK)

    @classmethod
    def call(cls) -> Action:
        return cls(ActionType.CALL)

    @classmethod
    def raise_to(cls, total: int) -> Action:
        if not isinstance(total, int):
            raise TypeError("raise_to total must be an integer")
        if total <= 0:
            raise ValueError("raise_to total must be positive")
        return cls(ActionType.RAISE_TO, total=total)

    def to_dict(self) -> dict[str, int | str | None]:
        return {"type": self.action_type.value, "total": self.total}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Action:
        action_type = ActionType(str(data["type"]))
        total = data.get("total")
        return cls(action_type, int(total) if total is not None else None)


@dataclass(frozen=True)
class LegalActions:
    can_fold: bool
    can_check: bool
    can_call: bool
    call_amount: int
    min_raise_to: int | None
    max_raise_to: int | None
    current_bet: int
    actor_commitment: int

    @property
    def can_raise(self) -> bool:
        return self.min_raise_to is not None and self.max_raise_to is not None

    def to_dict(self) -> dict[str, int | bool | None]:
        return {
            "can_fold": self.can_fold,
            "can_check": self.can_check,
            "can_call": self.can_call,
            "can_raise": self.can_raise,
            "call_amount": self.call_amount,
            "min_raise_to": self.min_raise_to,
            "max_raise_to": self.max_raise_to,
            "current_bet": self.current_bet,
            "actor_commitment": self.actor_commitment,
        }
