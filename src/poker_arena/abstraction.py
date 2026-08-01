from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from poker_arena.actions import Action
from poker_arena.actions import LegalActions
from poker_arena.state import HandState


class AbstractActionKind(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    MIN_RAISE = "min_raise"
    THIRD_POT = "third_pot"
    HALF_POT = "half_pot"
    THREE_QUARTER_POT = "three_quarter_pot"
    POT = "pot"
    OVERBET = "overbet"
    ALL_IN = "all_in"
    CUSTOM_RAISE = "custom_raise"


@dataclass(frozen=True)
class AbstractAction:
    kind: AbstractActionKind
    total: int | None = None
    pot_ratio: float = 0.0
    stack_ratio: float = 0.0

    @property
    def label(self) -> str:
        return self.kind.value

    def to_action(self) -> Action:
        if self.kind == AbstractActionKind.FOLD:
            return Action.fold()
        if self.kind == AbstractActionKind.CHECK:
            return Action.check()
        if self.kind == AbstractActionKind.CALL:
            return Action.call()
        if self.total is None:
            raise ValueError(f"{self.kind.value} requires a concrete raise_to total")
        return Action.raise_to(self.total)


class ActionAbstraction:
    """Maps no-limit integer actions into a finite CFR action set."""

    def __init__(
        self,
        pot_fractions: tuple[tuple[AbstractActionKind, float], ...] | None = None,
    ) -> None:
        self.pot_fractions = pot_fractions or (
            (AbstractActionKind.HALF_POT, 0.5),
            (AbstractActionKind.POT, 1.0),
        )
        if any(fraction <= 0 for _kind, fraction in self.pot_fractions):
            raise ValueError("Pot fractions must be positive")
        labels = [kind for kind, _fraction in self.pot_fractions]
        if len(labels) != len(set(labels)):
            raise ValueError("Pot-fraction action kinds must be unique")

    @classmethod
    def compact(cls) -> ActionAbstraction:
        """Action set for recursive solving: three sizes plus min-raise/all-in."""

        return cls(
            (
                (AbstractActionKind.THIRD_POT, 1.0 / 3.0),
                (AbstractActionKind.THREE_QUARTER_POT, 0.75),
                (AbstractActionKind.OVERBET, 1.5),
            )
        )

    def actions_for(self, state: HandState, seat_id: int) -> list[AbstractAction]:
        legal = state.legal_actions(seat_id)
        player = state.player_by_seat(seat_id)
        return self.actions_from_legal(
            legal,
            pot=state.total_pot,
            stack=player.stack + legal.actor_commitment,
        )

    def actions_from_legal(self, legal: LegalActions, pot: int, stack: int) -> list[AbstractAction]:
        if pot < 0 or stack <= 0:
            raise ValueError("pot must be non-negative and stack must be positive")
        actions: list[AbstractAction] = []
        if legal.can_fold:
            actions.append(AbstractAction(AbstractActionKind.FOLD))
        if legal.can_check:
            actions.append(AbstractAction(AbstractActionKind.CHECK))
        if legal.can_call:
            actions.append(AbstractAction(AbstractActionKind.CALL))
        if legal.can_raise and legal.min_raise_to is not None and legal.max_raise_to is not None:
            pot_after_call = pot + legal.call_amount
            candidates = [(AbstractActionKind.MIN_RAISE, legal.min_raise_to)]
            candidates.extend(
                (kind, legal.current_bet + max(1, round(pot_after_call * fraction)))
                for kind, fraction in self.pot_fractions
            )
            candidates.append((AbstractActionKind.ALL_IN, legal.max_raise_to))
            seen_totals: set[int] = set()
            for kind, raw_total in candidates:
                total = max(legal.min_raise_to, min(legal.max_raise_to, raw_total))
                if total in seen_totals:
                    continue
                seen_totals.add(total)
                actions.append(self.describe_concrete_raise(total, pot, stack, legal.current_bet, kind))
        return actions

    def describe_concrete_raise(
        self,
        total: int,
        pot: int,
        stack: int,
        current_bet: int,
        kind: AbstractActionKind = AbstractActionKind.CUSTOM_RAISE,
    ) -> AbstractAction:
        added = max(0, total - current_bet)
        pot_ratio = added / max(1, pot)
        stack_ratio = total / max(1, stack)
        return AbstractAction(kind=kind, total=total, pot_ratio=pot_ratio, stack_ratio=stack_ratio)
