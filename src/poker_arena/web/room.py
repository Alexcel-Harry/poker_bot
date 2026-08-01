from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import secrets
import string
from typing import Any

from poker_arena.actions import Action
from poker_arena.bots import BotPolicy
from poker_arena.cards import Card
from poker_arena.events import PokerEvent
from poker_arena.table import Table, TableConfig


STARTING_STACK = 2000
SMALL_BLIND = 10
BIG_BLIND = 20
MAX_SEATS = 9


def _room_code(length: int = 6) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


@dataclass
class RoomSeat:
    seat_id: int
    kind: str = "empty"
    nickname: str | None = None
    token: str | None = None
    stack: int = STARTING_STACK
    bot_policy: BotPolicy | None = None

    @property
    def occupied(self) -> bool:
        return self.kind in {"human", "bot"}

    def public_dict(self) -> dict[str, Any]:
        return {
            "seat_id": self.seat_id,
            "kind": self.kind,
            "nickname": self.nickname,
            "stack": self.stack,
        }


class PokerRoom:
    def __init__(
        self,
        seed: int | None = None,
        bot_policy_factory: Callable[[int], BotPolicy | None] | None = None,
        reveal_all_hole_cards: bool = False,
    ) -> None:
        self.room_code = _room_code()
        self.host_token = secrets.token_urlsafe(24)
        self.seed = seed
        self.bot_policy_factory = bot_policy_factory
        self.reveal_all_hole_cards = reveal_all_hole_cards
        self.game_started = False
        self.seats = [RoomSeat(seat_id=seat_id) for seat_id in range(MAX_SEATS)]
        self.table: Table | None = None
        self.display_to_engine: dict[int, int] = {}
        self.engine_to_display: dict[int, int] = {}
        self.completed_hands: list[dict[str, Any]] = []
        self._hand_seed_counter = 0
        self._last_button_display: int | None = None
        self._terminal_hand_recorded = False
        self._hand_number = 0

    def join(self, room_code: str, seat_id: int, nickname: str) -> dict[str, Any]:
        if room_code != self.room_code:
            raise PermissionError("Invalid room code")
        self._validate_seat_id(seat_id)
        cleaned = nickname.strip()
        if not cleaned or len(cleaned) > 24:
            raise ValueError("Nickname must be 1-24 characters")
        seat = self.seats[seat_id]
        if seat.occupied:
            raise ValueError("Seat is already occupied")
        token = secrets.token_urlsafe(24)
        seat.kind = "human"
        seat.nickname = cleaned
        seat.token = token
        seat.bot_policy = None
        if self.game_started:
            self.advance_bots()
        return {"seat_id": seat_id, "nickname": cleaned, "seat_token": token}

    def reserve_bot(self, host_token: str, seat_id: int) -> dict[str, Any]:
        self._require_host(host_token)
        self._validate_seat_id(seat_id)
        seat = self.seats[seat_id]
        if seat.occupied:
            raise ValueError("Seat is already occupied")
        policy = self.bot_policy_factory(seat_id) if self.bot_policy_factory is not None else None
        seat.kind = "bot"
        seat.nickname = f"Bot {seat_id + 1}" if policy is not None else f"Bot {seat_id + 1} (.pt pending)"
        seat.token = None
        seat.bot_policy = policy
        if self.game_started and any(existing.kind == "human" for existing in self.seats):
            self.advance_bots()
        return seat.public_dict()

    def start_game(self, host_token: str) -> dict[str, Any]:
        self._require_host(host_token)
        if self.game_started:
            raise ValueError("The game has already started")
        active_seats = [seat for seat in self.seats if seat.occupied and seat.stack > 0]
        if len(active_seats) < 2:
            raise ValueError("At least two occupied seats are required to start the game")

        self.game_started = True
        self._ensure_hand_started()
        self.advance_bots()
        return self.snapshot_for()

    def submit_action(self, seat_token: str, action: Action) -> dict[str, Any]:
        display_seat = self._seat_for_token(seat_token)
        if self.table is None or self.table.current_hand is None:
            raise ValueError("No hand is active")
        state = self.table.current_hand
        if state.current_actor is None:
            raise ValueError("No player is currently acting")
        current_display = self.engine_to_display[state.current_actor]
        if display_seat != current_display:
            raise PermissionError("It is not this seat's turn")
        if self.seats[display_seat].kind != "human":
            raise PermissionError("Only human seats can submit actions")

        state = self.table.apply(action)
        self._sync_display_stacks()
        self._complete_terminal_hand(state)
        self.advance_bots()
        return self.snapshot_for(seat_token=seat_token)

    def start_next_hand(self, seat_token: str) -> dict[str, Any]:
        self._seat_for_token(seat_token)
        if self.table is None or self.table.current_hand is None or not self.table.current_hand.is_terminal:
            raise ValueError("The current hand is not finished")

        self.table = None
        self.display_to_engine = {}
        self.engine_to_display = {}
        self._terminal_hand_recorded = False
        self._ensure_hand_started()
        self.advance_bots()
        return self.snapshot_for(seat_token=seat_token)

    def advance_bots(self, max_actions: int = 100) -> None:
        actions_taken = 0
        while True:
            if self.table is None:
                self._ensure_hand_started()
            if self.table is None or self.table.current_hand is None:
                return
            state = self.table.current_hand
            if state.is_terminal:
                self._complete_terminal_hand(state)
                return
            if state.current_actor is None:
                return
            display_seat = self.engine_to_display[state.current_actor]
            seat = self.seats[display_seat]
            if seat.kind != "bot" or seat.bot_policy is None:
                return
            if actions_taken >= max_actions:
                raise RuntimeError("Bot advancement exceeded max_actions")
            engine_seat = state.current_actor
            legal_actions = state.legal_actions(engine_seat)
            action = seat.bot_policy.choose_action(state.player_view(engine_seat), legal_actions)
            state = self.table.apply(action)
            actions_taken += 1
            self._sync_display_stacks()
            self._complete_terminal_hand(state)

    def snapshot_for(self, seat_token: str | None = None) -> dict[str, Any]:
        viewer_display = self._seat_for_token(seat_token) if seat_token else None
        current_actor = self._current_actor_display()
        status = self._status(current_actor)
        board: list[str] = []
        street = "waiting"
        pot = 0
        current_bet = 0
        legal_actions: dict[str, Any] | None = None
        private_cards: list[str] | None = None
        revealed_hole_cards: dict[int, list[str]] = {}
        log_events: list[dict[str, Any]] = []

        if self.table is not None and self.table.current_hand is not None:
            hand = self.table.current_hand
            public_view = hand.public_view()
            board = [card.to_str() for card in public_view.board]
            street = public_view.street.value
            pot = public_view.pot
            current_bet = hand.current_bet
            log_events = [self._event_for_display(event) for event in public_view.events][-40:]
            if self.reveal_all_hole_cards:
                for engine_seat, display_seat in self.engine_to_display.items():
                    revealed_hole_cards[display_seat] = [
                        card.to_str() for card in hand.player_by_seat(engine_seat).hole_cards
                    ]
            if viewer_display in self.display_to_engine:
                engine_seat = self.display_to_engine[viewer_display]
                cards = hand.player_view(engine_seat).hole_cards[engine_seat]
                private_cards = [card.to_str() for card in cards] if cards else []
                if current_actor == viewer_display and self.seats[viewer_display].kind == "human":
                    legal_actions = hand.legal_actions(engine_seat).to_dict()

        return {
            "room_code": self.room_code,
            "settings": {"starting_stack": STARTING_STACK, "small_blind": SMALL_BLIND, "big_blind": BIG_BLIND},
            "game_started": self.game_started,
            "can_start_game": not self.game_started
            and sum(seat.occupied and seat.stack > 0 for seat in self.seats) >= 2,
            "status": status["status"],
            "paused_reason": status["paused_reason"],
            "seats": self._seat_payloads(),
            "button": self._button_display(),
            "small_blind": self._blind_display("small"),
            "big_blind": self._blind_display("big"),
            "current_actor": current_actor,
            "street": street,
            "board": board,
            "pot": pot,
            "current_bet": current_bet,
            "legal_actions": legal_actions,
            "private_hole_cards": private_cards,
            "revealed_hole_cards": revealed_hole_cards,
            "debug_reveal": self.reveal_all_hole_cards,
            "log": log_events,
        }

    def session_log(self, host_token: str) -> dict[str, Any]:
        self._require_host(host_token)
        hands = list(self.completed_hands)
        if (
            self.table is not None
            and self.table.current_hand is not None
            and not (self.table.current_hand.is_terminal and self._terminal_hand_recorded)
        ):
            hands.append(self._hand_record(self.table.current_hand))
        return {
            "room_code": self.room_code,
            "seat_id_space": "display",
            "game_started": self.game_started,
            "settings": {"starting_stack": STARTING_STACK, "small_blind": SMALL_BLIND, "big_blind": BIG_BLIND},
            "seats": [seat.public_dict() for seat in self.seats],
            "hands": hands,
        }

    def _ensure_hand_started(self) -> None:
        if not self.game_started or self.table is not None:
            return
        active_display_seats = [seat.seat_id for seat in self.seats if seat.occupied and seat.stack > 0]
        if len(active_display_seats) < 2:
            return

        self.display_to_engine = {display: engine for engine, display in enumerate(active_display_seats)}
        self.engine_to_display = {engine: display for display, engine in self.display_to_engine.items()}
        seed = None if self.seed is None else self.seed + self._hand_seed_counter
        self._hand_seed_counter += 1
        config = TableConfig(
            seats=len(active_display_seats),
            small_blind=SMALL_BLIND,
            big_blind=BIG_BLIND,
            starting_stacks=[self.seats[display].stack for display in active_display_seats],
            seed=seed,
        )
        table = Table(config)
        table.hand_number = self._hand_number
        if self._last_button_display in self.display_to_engine:
            table.button = self.display_to_engine[self._last_button_display]
        table.start_hand()
        self._hand_number = table.hand_number
        self.table = table
        self._terminal_hand_recorded = False
        if table.current_hand is not None:
            self._last_button_display = self.engine_to_display[table.current_hand.button]
            self._sync_display_stacks()

    def _complete_terminal_hand(self, state: Any) -> bool:
        if not state.is_terminal:
            return False
        if not self._terminal_hand_recorded:
            self.completed_hands.append(self._hand_record(state))
            self._terminal_hand_recorded = True
        return True

    def _sync_display_stacks(self) -> None:
        if self.table is None or self.table.current_hand is None:
            return
        for engine_seat, display_seat in self.engine_to_display.items():
            self.seats[display_seat].stack = self.table.current_hand.player_by_seat(engine_seat).stack

    def _hand_record(self, state: Any) -> dict[str, Any]:
        hole_cards = []
        for player in state.players:
            display_seat = self.engine_to_display.get(player.seat_id, player.seat_id)
            seat = self.seats[display_seat]
            hole_cards.append(
                {
                    "seat_id": display_seat,
                    "nickname": seat.nickname,
                    "kind": seat.kind,
                    "cards": [card.to_str() for card in player.hole_cards],
                }
            )
        return {
            "button": self.engine_to_display.get(state.button),
            "board": [card.to_str() for card in state.board],
            "hole_cards": hole_cards,
            "events": [
                self._event_for_display(event)
                for event in state.events
                if event.event_type != "snapshot"
            ],
        }

    def _event_for_display(self, event: PokerEvent) -> dict[str, Any]:
        data = dict(event.data)
        seat_fields = ("button", "small_blind_seat", "big_blind_seat") if event.event_type == "hand_started" else ("seat_id",)
        for field in seat_fields:
            value = data.get(field)
            if isinstance(value, int):
                data[field] = self.engine_to_display.get(value, value)

        eligible = data.get("eligible_seats")
        if isinstance(eligible, list):
            data["eligible_seats"] = [
                self.engine_to_display.get(seat_id, seat_id)
                for seat_id in eligible
                if isinstance(seat_id, int)
            ]

        stacks = data.get("stacks")
        if event.event_type == "hand_finished" and isinstance(stacks, list):
            data["stacks_by_seat"] = [
                {"seat_id": self.engine_to_display.get(engine_seat, engine_seat), "stack": stack}
                for engine_seat, stack in enumerate(stacks)
            ]

        return {"event_type": event.event_type, "data": data}

    def _seat_payloads(self) -> list[dict[str, Any]]:
        payloads = []
        folded: dict[int, bool] = {}
        all_in: dict[int, bool] = {}
        if self.table is not None and self.table.current_hand is not None:
            for engine_seat, display_seat in self.engine_to_display.items():
                player = self.table.current_hand.player_by_seat(engine_seat)
                folded[display_seat] = player.folded
                all_in[display_seat] = player.all_in
        for seat in self.seats:
            data = seat.public_dict()
            data["folded"] = folded.get(seat.seat_id, False)
            data["all_in"] = all_in.get(seat.seat_id, False)
            data["in_hand"] = seat.seat_id in self.display_to_engine
            payloads.append(data)
        return payloads

    def _current_actor_display(self) -> int | None:
        if self.table is None or self.table.current_hand is None or self.table.current_hand.current_actor is None:
            return None
        return self.engine_to_display[self.table.current_hand.current_actor]

    def _button_display(self) -> int | None:
        if self.table is None or self.table.current_hand is None:
            return None
        return self.engine_to_display[self.table.current_hand.button]

    def _blind_display(self, blind: str) -> int | None:
        if self.table is None or self.table.current_hand is None:
            return None
        engine_seat = self.table.current_hand.small_blind_seat if blind == "small" else self.table.current_hand.big_blind_seat
        return self.engine_to_display[engine_seat]

    def _status(self, current_actor: int | None) -> dict[str, str | None]:
        if not self.game_started:
            occupied = [seat for seat in self.seats if seat.occupied and seat.stack > 0]
            reason = "Waiting for host to start game" if len(occupied) >= 2 else "Waiting for players"
            return {"status": "waiting", "paused_reason": reason}
        if self.table is not None and self.table.current_hand is not None and self.table.current_hand.is_terminal:
            return {"status": "finished", "paused_reason": "Hand complete - review the river and result"}
        occupied = [seat for seat in self.seats if seat.occupied and seat.stack > 0]
        if len(occupied) < 2:
            return {"status": "waiting", "paused_reason": None}
        if (
            current_actor is not None
            and self.seats[current_actor].kind == "bot"
            and self.seats[current_actor].bot_policy is None
        ):
            return {"status": "paused", "paused_reason": "waiting for unavailable bot"}
        return {"status": "playing", "paused_reason": None}

    def _seat_for_token(self, token: str | None) -> int:
        if not token:
            raise PermissionError("Missing seat token")
        for seat in self.seats:
            if seat.kind == "human" and seat.token == token:
                return seat.seat_id
        raise PermissionError("Invalid seat token")

    def _require_host(self, host_token: str) -> None:
        if host_token != self.host_token:
            raise PermissionError("Invalid host token")

    @staticmethod
    def _validate_seat_id(seat_id: int) -> None:
        if not 0 <= seat_id < MAX_SEATS:
            raise ValueError("seat_id must be between 0 and 8")
