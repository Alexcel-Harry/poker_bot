from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from poker_arena.actions import LegalActions
from poker_arena.cards import Card
from poker_arena.events import JsonValue, PokerEvent

if TYPE_CHECKING:
    from poker_arena.table import TableConfig


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


@dataclass
class Player:
    seat_id: int
    stack: int
    hole_cards: list[Card] = field(default_factory=list)
    folded: bool = False
    all_in: bool = False

    def to_dict(self, include_hole_cards: bool = True) -> dict[str, JsonValue]:
        return {
            "seat_id": self.seat_id,
            "stack": self.stack,
            "hole_cards": [card.to_str() for card in self.hole_cards] if include_hole_cards else None,
            "folded": self.folded,
            "all_in": self.all_in,
        }


@dataclass(frozen=True)
class Pot:
    amount: int
    eligible_seats: tuple[int, ...]

    def to_dict(self) -> dict[str, JsonValue]:
        return {"amount": self.amount, "eligible_seats": list(self.eligible_seats)}


@dataclass(frozen=True)
class PublicHandView:
    button: int
    street: Street
    board: tuple[Card, ...]
    current_actor: int | None
    pot: int
    stacks: dict[int, int]
    folded: dict[int, bool]
    all_in: dict[int, bool]
    hole_cards: dict[int, tuple[Card, ...]]
    events: tuple[PokerEvent, ...]

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "button": self.button,
            "street": self.street.value,
            "board": [card.to_str() for card in self.board],
            "current_actor": self.current_actor,
            "pot": self.pot,
            "stacks": self.stacks,
            "folded": self.folded,
            "all_in": self.all_in,
            "hole_cards": {seat: [card.to_str() for card in cards] for seat, cards in self.hole_cards.items()},
            "events": [event.to_dict() for event in self.events],
        }


@dataclass(frozen=True)
class PlayerHandView(PublicHandView):
    hole_cards: dict[int, tuple[Card, ...] | None]

    def to_dict(self) -> dict[str, JsonValue]:
        data = super().to_dict()
        data["hole_cards"] = {
            seat: [card.to_str() for card in cards] if cards is not None else None
            for seat, cards in self.hole_cards.items()
        }
        return data


@dataclass
class HandState:
    config: TableConfig
    players: list[Player]
    button: int
    small_blind_seat: int
    big_blind_seat: int
    street: Street
    board: list[Card]
    deck_cards: list[str]
    current_actor: int | None
    events: list[PokerEvent]
    committed_this_street: dict[int, int]
    total_committed: dict[int, int]
    current_bet: int
    last_full_raise: int
    acted_this_round: set[int]
    carryover_in_pot: int = 0
    is_terminal: bool = False
    pots: list[Pot] = field(default_factory=list)

    @property
    def total_pot(self) -> int:
        if self.is_terminal:
            return 0
        return self.carryover_in_pot + sum(self.total_committed.values())

    @property
    def active_seats(self) -> list[int]:
        return [player.seat_id for player in self.players if not player.folded]

    def player_by_seat(self, seat_id: int) -> Player:
        for player in self.players:
            if player.seat_id == seat_id:
                return player
        raise ValueError(f"Unknown seat {seat_id}")

    def legal_actions(self, seat_id: int) -> LegalActions:
        from poker_arena.rules import legal_actions_for

        return legal_actions_for(self, seat_id)

    def public_view(self) -> PublicHandView:
        return PublicHandView(
            button=self.button,
            street=self.street,
            board=tuple(self.board),
            current_actor=self.current_actor,
            pot=self.total_pot,
            stacks={player.seat_id: player.stack for player in self.players},
            folded={player.seat_id: player.folded for player in self.players},
            all_in={player.seat_id: player.all_in for player in self.players},
            hole_cards={},
            events=tuple(event for event in self.events if event.event_type != "snapshot"),
        )

    def player_view(self, seat_id: int) -> PlayerHandView:
        hole_cards: dict[int, tuple[Card, ...] | None] = {}
        for player in self.players:
            hole_cards[player.seat_id] = tuple(player.hole_cards) if player.seat_id == seat_id else None
        return PlayerHandView(
            button=self.button,
            street=self.street,
            board=tuple(self.board),
            current_actor=self.current_actor,
            pot=self.total_pot,
            stacks={player.seat_id: player.stack for player in self.players},
            folded={player.seat_id: player.folded for player in self.players},
            all_in={player.seat_id: player.all_in for player in self.players},
            hole_cards=hole_cards,
            events=tuple(event for event in self.events if event.event_type != "snapshot"),
        )

    def to_snapshot(self) -> dict[str, JsonValue]:
        return {
            "players": [player.to_dict(include_hole_cards=True) for player in self.players],
            "button": self.button,
            "small_blind_seat": self.small_blind_seat,
            "big_blind_seat": self.big_blind_seat,
            "street": self.street.value,
            "board": [card.to_str() for card in self.board],
            "deck_cards": list(self.deck_cards),
            "current_actor": self.current_actor,
            "committed_this_street": dict(self.committed_this_street),
            "total_committed": dict(self.total_committed),
            "current_bet": self.current_bet,
            "last_full_raise": self.last_full_raise,
            "acted_this_round": sorted(self.acted_this_round),
            "carryover_in_pot": self.carryover_in_pot,
            "is_terminal": self.is_terminal,
            "pots": [pot.to_dict() for pot in self.pots],
        }

    @classmethod
    def from_snapshot(cls, config: TableConfig, snapshot: dict[str, JsonValue], events: list[PokerEvent]) -> HandState:
        players = []
        for raw_player in snapshot["players"]:  # type: ignore[index]
            player_data = raw_player  # type: ignore[assignment]
            players.append(
                Player(
                    seat_id=int(player_data["seat_id"]),  # type: ignore[index]
                    stack=int(player_data["stack"]),  # type: ignore[index]
                    hole_cards=[Card.from_str(str(card)) for card in player_data["hole_cards"]],  # type: ignore[index]
                    folded=bool(player_data["folded"]),  # type: ignore[index]
                    all_in=bool(player_data["all_in"]),  # type: ignore[index]
                )
            )
        pots = [
            Pot(int(raw_pot["amount"]), tuple(int(seat) for seat in raw_pot["eligible_seats"]))  # type: ignore[index]
            for raw_pot in snapshot["pots"]  # type: ignore[index]
        ]
        return cls(
            config=config,
            players=players,
            button=int(snapshot["button"]),
            small_blind_seat=int(snapshot["small_blind_seat"]),
            big_blind_seat=int(snapshot["big_blind_seat"]),
            street=Street(str(snapshot["street"])),
            board=[Card.from_str(str(card)) for card in snapshot["board"]],  # type: ignore[index]
            deck_cards=[str(card) for card in snapshot["deck_cards"]],  # type: ignore[index]
            current_actor=int(snapshot["current_actor"]) if snapshot["current_actor"] is not None else None,
            events=events,
            committed_this_street={int(key): int(value) for key, value in snapshot["committed_this_street"].items()},  # type: ignore[union-attr]
            total_committed={int(key): int(value) for key, value in snapshot["total_committed"].items()},  # type: ignore[union-attr]
            current_bet=int(snapshot["current_bet"]),
            last_full_raise=int(snapshot["last_full_raise"]),
            acted_this_round={int(seat) for seat in snapshot["acted_this_round"]},  # type: ignore[arg-type]
            carryover_in_pot=int(snapshot["carryover_in_pot"]),
            is_terminal=bool(snapshot["is_terminal"]),
            pots=pots,
        )
