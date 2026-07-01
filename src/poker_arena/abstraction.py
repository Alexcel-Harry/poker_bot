from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from poker_arena.actions import Action
from poker_arena.state import HandState


class AbstractActionKind(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    MIN_RAISE = "min_raise"
    HALF_POT = "half_pot"
    POT = "pot"
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

    def actions_for(self, state: HandState, seat_id: int) -> list[AbstractAction]:
        legal = state.legal_actions(seat_id)
        player = state.player_by_seat(seat_id)
        actions: list[AbstractAction] = []
        if legal.can_fold:
            actions.append(AbstractAction(AbstractActionKind.FOLD))
        if legal.can_check:
            actions.append(AbstractAction(AbstractActionKind.CHECK))
        if legal.can_call:
            actions.append(AbstractAction(AbstractActionKind.CALL))
        if legal.can_raise and legal.min_raise_to is not None and legal.max_raise_to is not None:
            candidates = [
                (AbstractActionKind.MIN_RAISE, legal.min_raise_to),
                (AbstractActionKind.HALF_POT, legal.current_bet + max(1, state.total_pot // 2)),
                (AbstractActionKind.POT, legal.current_bet + max(1, state.total_pot)),
                (AbstractActionKind.ALL_IN, legal.max_raise_to),
            ]
            seen_totals: set[int] = set()
            for kind, raw_total in candidates:
                total = max(legal.min_raise_to, min(legal.max_raise_to, raw_total))
                if total in seen_totals and kind != AbstractActionKind.ALL_IN:
                    continue
                seen_totals.add(total)
                actions.append(self.describe_concrete_raise(total, state.total_pot, player.stack + legal.actor_commitment, legal.current_bet, kind))
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
