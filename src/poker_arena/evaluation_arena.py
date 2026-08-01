from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import fmean, stdev
from typing import Callable, Mapping, Sequence

from poker_arena.actions import Action, LegalActions
from poker_arena.bots import BotPolicy
from poker_arena.cards import Deck
from poker_arena.state import HandState
from poker_arena.table import Table, TableConfig


@dataclass
class ActionTelemetry:
    decisions: int = 0
    by_street: dict[str, int] = field(default_factory=dict)
    by_action: dict[str, int] = field(default_factory=dict)
    by_street_action: dict[str, int] = field(default_factory=dict)
    facing_bet: dict[str, int] = field(default_factory=dict)
    facing_bet_by_street: dict[str, int] = field(default_factory=dict)
    raise_to_pot_ratios: list[float] = field(default_factory=list)
    raise_to_pot_ratios_by_street: dict[str, list[float]] = field(default_factory=dict)

    def record(self, state: HandState, action: Action, legal: LegalActions) -> None:
        self.decisions += 1
        street = state.street.value
        action_name = action.action_type.value
        self.by_street[street] = self.by_street.get(street, 0) + 1
        self.by_action[action_name] = self.by_action.get(action_name, 0) + 1
        street_action = f"{street}:{action_name}"
        self.by_street_action[street_action] = self.by_street_action.get(street_action, 0) + 1
        facing_key = "facing_bet" if legal.call_amount > 0 else "not_facing_bet"
        combined = f"{facing_key}:{action_name}"
        self.facing_bet[combined] = self.facing_bet.get(combined, 0) + 1
        street_combined = f"{street}:{combined}"
        self.facing_bet_by_street[street_combined] = self.facing_bet_by_street.get(street_combined, 0) + 1
        if action.total is not None:
            added_beyond_call = max(0, action.total - legal.current_bet)
            pot_after_call = state.total_pot + legal.call_amount
            ratio = added_beyond_call / max(1.0, float(pot_after_call))
            self.raise_to_pot_ratios.append(ratio)
            self.raise_to_pot_ratios_by_street.setdefault(street, []).append(ratio)

    def to_dict(self) -> dict[str, object]:
        ratios = self.raise_to_pot_ratios
        return {
            "decisions": self.decisions,
            "by_street": dict(sorted(self.by_street.items())),
            "by_action": dict(sorted(self.by_action.items())),
            "by_street_action": dict(sorted(self.by_street_action.items())),
            "facing_bet": dict(sorted(self.facing_bet.items())),
            "facing_bet_by_street": dict(sorted(self.facing_bet_by_street.items())),
            "mean_raise_to_pot_ratio": fmean(ratios) if ratios else None,
            "min_raise_to_pot_ratio": min(ratios) if ratios else None,
            "max_raise_to_pot_ratio": max(ratios) if ratios else None,
            "mean_raise_to_pot_ratio_by_street": {
                street: fmean(values)
                for street, values in sorted(self.raise_to_pot_ratios_by_street.items())
            },
        }


@dataclass(frozen=True)
class DuplicateMatchResult:
    deals: int
    hands: int
    mean_chips_per_hand: float
    standard_error: float
    confidence_95: tuple[float, float]
    big_blinds_per_100: float
    big_blinds_per_100_confidence_95: tuple[float, float]
    paired_scores: tuple[float, ...]
    telemetry: Mapping[str, Mapping[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "deals": self.deals,
            "hands": self.hands,
            "mean_chips_per_hand": self.mean_chips_per_hand,
            "standard_error": self.standard_error,
            "confidence_95": list(self.confidence_95),
            "big_blinds_per_100": self.big_blinds_per_100,
            "big_blinds_per_100_confidence_95": list(self.big_blinds_per_100_confidence_95),
            "paired_scores": list(self.paired_scores),
            "telemetry": {name: dict(values) for name, values in self.telemetry.items()},
        }


def play_duplicate_match(
    first_factory: Callable[[], BotPolicy],
    second_factory: Callable[[], BotPolicy],
    deals: int,
    small_blind: int = 5,
    big_blind: int = 10,
    starting_stack: int = 200,
    random_seed: int = 17,
    max_actions_per_hand: int = 256,
) -> DuplicateMatchResult:
    """Play every deck twice with seats swapped and return paired uncertainty."""

    if deals <= 0:
        raise ValueError("deals must be positive")
    if max_actions_per_hand <= 0:
        raise ValueError("max_actions_per_hand must be positive")
    first = first_factory()
    second = second_factory()
    telemetry = {"first": ActionTelemetry(), "second": ActionTelemetry()}
    paired_scores: list[float] = []

    for deal_index in range(deals):
        deck = Deck(seed=random_seed + deal_index)
        order = [card for card in deck.draw(deck.remaining())]
        first_score = _play_fixed_hand(
            order,
            bots=(first, second),
            tracked_player=0,
            names=("first", "second"),
            telemetry=telemetry,
            small_blind=small_blind,
            big_blind=big_blind,
            starting_stack=starting_stack,
            max_actions=max_actions_per_hand,
        )
        second_score = _play_fixed_hand(
            order,
            bots=(second, first),
            tracked_player=1,
            names=("second", "first"),
            telemetry=telemetry,
            small_blind=small_blind,
            big_blind=big_blind,
            starting_stack=starting_stack,
            max_actions=max_actions_per_hand,
        )
        paired_scores.append(first_score + second_score)

    return _summarize_rotations(paired_scores, 2, big_blind, telemetry)


def play_rotating_match(
    tracked_factory: Callable[[], BotPolicy],
    opponent_factory: Callable[[], BotPolicy],
    deals: int,
    seats: int = 3,
    small_blind: int = 5,
    big_blind: int = 10,
    starting_stack: int = 200,
    random_seed: int = 17,
    max_actions_per_hand: int = 256,
) -> DuplicateMatchResult:
    """Replay each deck with the tracked policy rotated through every seat."""

    if deals <= 0:
        raise ValueError("deals must be positive")
    if not 3 <= seats <= 9:
        raise ValueError("rotating multi-player evaluation requires 3-9 seats")
    if max_actions_per_hand <= 0:
        raise ValueError("max_actions_per_hand must be positive")
    tracked = tracked_factory()
    opponents = [opponent_factory() for _ in range(seats - 1)]
    telemetry = {"tracked": ActionTelemetry(), "opponent": ActionTelemetry()}
    grouped_scores: list[float] = []

    for deal_index in range(deals):
        deck = Deck(seed=random_seed + deal_index)
        order = [card for card in deck.draw(deck.remaining())]
        deal_score = 0.0
        for tracked_seat in range(seats):
            bots: list[BotPolicy] = []
            names: list[str] = []
            opponent_index = 0
            for seat in range(seats):
                if seat == tracked_seat:
                    bots.append(tracked)
                    names.append("tracked")
                else:
                    bots.append(opponents[opponent_index])
                    names.append("opponent")
                    opponent_index += 1
            deal_score += _play_fixed_hand(
                order,
                bots=bots,
                tracked_player=tracked_seat,
                names=names,
                telemetry=telemetry,
                small_blind=small_blind,
                big_blind=big_blind,
                starting_stack=starting_stack,
                max_actions=max_actions_per_hand,
            )
        grouped_scores.append(deal_score)

    return _summarize_rotations(grouped_scores, seats, big_blind, telemetry)


def _summarize_rotations(
    grouped_scores: Sequence[float],
    hands_per_deal: int,
    big_blind: int,
    telemetry: Mapping[str, ActionTelemetry],
) -> DuplicateMatchResult:
    deals = len(grouped_scores)
    mean_group = fmean(grouped_scores)
    standard_error_group = stdev(grouped_scores) / math.sqrt(deals) if deals > 1 else 0.0
    mean_hand = mean_group / hands_per_deal
    standard_error_hand = standard_error_group / hands_per_deal
    margin = 1.96 * standard_error_hand
    bb100_scale = 100.0 / big_blind
    return DuplicateMatchResult(
        deals=deals,
        hands=deals * hands_per_deal,
        mean_chips_per_hand=mean_hand,
        standard_error=standard_error_hand,
        confidence_95=(mean_hand - margin, mean_hand + margin),
        big_blinds_per_100=mean_hand * bb100_scale,
        big_blinds_per_100_confidence_95=(
            (mean_hand - margin) * bb100_scale,
            (mean_hand + margin) * bb100_scale,
        ),
        paired_scores=tuple(grouped_scores),
        telemetry={name: collector.to_dict() for name, collector in telemetry.items()},
    )


def _play_fixed_hand(
    deck_order: list[object],
    bots: Sequence[BotPolicy],
    tracked_player: int,
    names: Sequence[str],
    telemetry: Mapping[str, ActionTelemetry],
    small_blind: int,
    big_blind: int,
    starting_stack: int,
    max_actions: int,
) -> float:
    config = TableConfig(
        seats=len(bots),
        small_blind=small_blind,
        big_blind=big_blind,
        starting_stacks=[starting_stack] * len(bots),
        deck_order=deck_order,  # type: ignore[arg-type]
    )
    table = Table(config)
    state = table.start_hand()
    for _ in range(max_actions):
        if state.is_terminal or state.current_actor is None:
            break
        actor = state.current_actor
        legal = state.legal_actions(actor)
        action = bots[actor].choose_action(state.player_view(actor), legal)
        telemetry[names[actor]].record(state, action, legal)
        state = table.apply(action)
    if not state.is_terminal:
        raise RuntimeError(f"Duplicate evaluation exceeded {max_actions} actions")
    return float(state.player_by_seat(tracked_player).stack - starting_stack)
