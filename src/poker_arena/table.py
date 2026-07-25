from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from poker_arena.actions import Action, ActionType
from poker_arena.cards import Card, Deck
from poker_arena.evaluator import evaluate_best
from poker_arena.events import PokerEvent, event_from_dict
from poker_arena.rules import active_not_all_in_seats, betting_round_complete, live_seats
from poker_arena.state import HandState, Player, Pot, Street


@dataclass(frozen=True)
class TableConfig:
    seats: int
    small_blind: int
    big_blind: int
    starting_stacks: list[int]
    seed: int | None = None
    deck_order: list[Card] | None = None

    def __post_init__(self) -> None:
        if not 2 <= self.seats <= 9:
            raise ValueError("No Limit Hold'em supports 2-9 seats in this engine")
        if self.small_blind <= 0 or self.big_blind <= 0:
            raise ValueError("Blinds must be positive")
        if self.small_blind > self.big_blind:
            raise ValueError("Small blind cannot exceed big blind")
        if len(self.starting_stacks) != self.seats:
            raise ValueError("starting_stacks length must match seats")
        if any(stack <= 0 for stack in self.starting_stacks):
            raise ValueError("All starting stacks must be positive")


class Table:
    def __init__(self, config: TableConfig) -> None:
        self.config = config
        self.stacks = list(config.starting_stacks)
        self.button = -1
        self.hand_number = 0
        self.carryover_chips = 0
        self.current_hand: HandState | None = None
        self._seed_counter = 0

    def start_hand(self) -> HandState:
        self.hand_number += 1
        self.button = self._next_seat(self.button)
        small_blind_seat, big_blind_seat = self._blind_seats(self.button)

        if self.config.deck_order is not None:
            deck = Deck(cards=self.config.deck_order)
        else:
            seed = None if self.config.seed is None else self.config.seed + self._seed_counter
            self._seed_counter += 1
            deck = Deck(seed=seed)

        players = [Player(seat_id=seat, stack=self.stacks[seat]) for seat in range(self.config.seats)]
        for _ in range(2):
            for player in players:
                player.hole_cards.append(deck.draw_one())

        committed = {seat: 0 for seat in range(self.config.seats)}
        total_committed = {seat: 0 for seat in range(self.config.seats)}
        events: list[PokerEvent] = [
            PokerEvent(
                "hand_started",
                {
                    "hand_number": self.hand_number,
                    "button": self.button,
                    "small_blind_seat": small_blind_seat,
                    "big_blind_seat": big_blind_seat,
                    "carryover_in_pot": self.carryover_chips,
                },
            )
        ]
        state = HandState(
            config=self.config,
            players=players,
            button=self.button,
            small_blind_seat=small_blind_seat,
            big_blind_seat=big_blind_seat,
            street=Street.PREFLOP,
            board=[],
            deck_cards=deck.to_list(),
            current_actor=None,
            events=events,
            committed_this_street=committed,
            total_committed=total_committed,
            current_bet=0,
            last_full_raise=self.config.big_blind,
            acted_this_round=set(),
            carryover_in_pot=self.carryover_chips,
        )
        self.carryover_chips = 0
        self._post_blind(state, small_blind_seat, self.config.small_blind, "small_blind")
        self._post_blind(state, big_blind_seat, self.config.big_blind, "big_blind")
        state.current_bet = max(state.committed_this_street.values())
        state.current_actor = self._first_preflop_actor(small_blind_seat, big_blind_seat)
        self._append_snapshot(state)
        self.current_hand = state
        return state

    def apply(self, action: Action) -> HandState:
        if self.current_hand is None:
            raise ValueError("No hand is in progress")
        state = self.current_hand
        if state.is_terminal or state.current_actor is None:
            raise ValueError("Hand is already terminal")

        actor = state.current_actor
        legal = state.legal_actions(actor)
        player = state.player_by_seat(actor)

        if action.action_type == ActionType.FOLD:
            player.folded = True
            state.acted_this_round.add(actor)
        elif action.action_type == ActionType.CHECK:
            if not legal.can_check:
                raise ValueError("Cannot check while facing a bet")
            state.acted_this_round.add(actor)
        elif action.action_type == ActionType.CALL:
            if legal.call_amount <= 0:
                raise ValueError("Cannot call when there is no bet")
            self._commit_chips(state, actor, legal.call_amount)
            state.acted_this_round.add(actor)
        elif action.action_type == ActionType.RAISE_TO:
            self._apply_raise_to(state, actor, action.total, legal.max_raise_to)
        else:
            raise ValueError(f"Unsupported action {action.action_type}")

        state.events.append(PokerEvent("action", {"seat_id": actor, "action": action.to_dict()}))
        self._after_action(state, actor)
        self._append_snapshot(state)
        self.current_hand = state
        return state

    @classmethod
    def replay(cls, config: TableConfig, events: Iterable[PokerEvent | dict[str, object]]) -> HandState:
        normalized = [event if isinstance(event, PokerEvent) else event_from_dict(event) for event in events]
        for event in reversed(normalized):
            if event.event_type == "snapshot":
                return HandState.from_snapshot(config, event.data["state"], normalized)  # type: ignore[arg-type]
        raise ValueError("Cannot replay events without a snapshot event")

    def _post_blind(self, state: HandState, seat_id: int, amount: int, event_type: str) -> None:
        paid = self._commit_chips(state, seat_id, amount)
        state.events.append(PokerEvent(event_type, {"seat_id": seat_id, "amount": paid}))

    def _commit_chips(self, state: HandState, seat_id: int, amount: int) -> int:
        player = state.player_by_seat(seat_id)
        paid = min(amount, player.stack)
        if paid < 0:
            raise ValueError("Cannot commit negative chips")
        player.stack -= paid
        state.committed_this_street[seat_id] += paid
        state.total_committed[seat_id] += paid
        if player.stack == 0:
            player.all_in = True
        return paid

    def _apply_raise_to(self, state: HandState, actor: int, total: int | None, max_raise_to: int | None) -> None:
        if total is None:
            raise ValueError("raise_to requires a total")
        if max_raise_to is None or total > max_raise_to:
            raise ValueError("raise_to exceeds available stack")
        if total <= state.current_bet:
            raise ValueError("raise_to must exceed the current bet")

        previous_bet = state.current_bet
        full_minimum = state.config.big_blind if previous_bet == 0 else previous_bet + state.last_full_raise
        if total < full_minimum and total != max_raise_to:
            raise ValueError(f"raise_to must be at least {full_minimum}, unless it is all-in")

        actor_commitment = state.committed_this_street[actor]
        self._commit_chips(state, actor, total - actor_commitment)
        state.current_bet = total
        raise_size = total - previous_bet
        if raise_size >= state.last_full_raise:
            state.last_full_raise = raise_size
            state.acted_this_round = {actor}
        else:
            state.acted_this_round.add(actor)

    def _after_action(self, state: HandState, actor: int) -> None:
        if len(live_seats(state)) == 1:
            self._award_without_showdown(state)
            return

        active_actors = active_not_all_in_seats(state)
        if len(active_actors) == 0:
            self._deal_remaining_board(state)
            self._showdown(state)
            return

        if betting_round_complete(state):
            # Once every opponent is all-in, there is nobody left who can
            # respond to another wager. Run out the board instead of asking
            # the sole player with chips to check through empty betting rounds.
            if len(active_actors) < 2:
                self._deal_remaining_board(state)
                self._showdown(state)
                return
            self._advance_street_or_showdown(state)
            return

        state.current_actor = self._next_actor_after(state, actor)

    def _advance_street_or_showdown(self, state: HandState) -> None:
        if state.street == Street.PREFLOP:
            self._advance_to_street(state, Street.FLOP, 3)
        elif state.street == Street.FLOP:
            self._advance_to_street(state, Street.TURN, 1)
        elif state.street == Street.TURN:
            self._advance_to_street(state, Street.RIVER, 1)
        else:
            self._showdown(state)
            return

        if len(active_not_all_in_seats(state)) < 2:
            self._deal_remaining_board(state)
            self._showdown(state)
        else:
            state.current_actor = self._first_postflop_actor(state)

    def _advance_to_street(self, state: HandState, street: Street, draw_count: int) -> None:
        deck = Deck(cards=[Card.from_str(text) for text in state.deck_cards])
        state.board.extend(deck.draw(draw_count))
        state.deck_cards = deck.to_list()
        state.street = street
        state.current_bet = 0
        state.last_full_raise = state.config.big_blind
        state.committed_this_street = {seat: 0 for seat in range(state.config.seats)}
        state.acted_this_round = set()
        state.events.append(PokerEvent("street_dealt", {"street": street.value, "board": [card.to_str() for card in state.board]}))

    def _deal_remaining_board(self, state: HandState) -> None:
        while len(state.board) < 5:
            street = Street.FLOP if len(state.board) == 0 else Street.TURN if len(state.board) == 3 else Street.RIVER
            draw_count = 3 if street == Street.FLOP else 1
            self._advance_to_street(state, street, draw_count)

    def _award_without_showdown(self, state: HandState) -> None:
        winner = live_seats(state)[0]
        amount = state.total_pot
        state.player_by_seat(winner).stack += amount
        state.pots = [Pot(amount, (winner,))]
        state.events.append(PokerEvent("pot_awarded", {"seat_id": winner, "amount": amount, "showdown": False}))
        self._finish_hand(state)

    def _showdown(self, state: HandState) -> None:
        state.street = Street.SHOWDOWN
        state.pots = self._build_pots(state)
        values = {
            seat_id: evaluate_best(state.player_by_seat(seat_id).hole_cards + state.board)
            for seat_id in live_seats(state)
        }
        for pot in state.pots:
            eligible = [seat for seat in pot.eligible_seats if seat in values]
            best = max(values[seat] for seat in eligible)
            winners = [seat for seat in eligible if values[seat] == best]
            share = pot.amount // len(winners)
            remainder = pot.amount % len(winners)
            for winner in winners:
                state.player_by_seat(winner).stack += share
                state.events.append(PokerEvent("pot_awarded", {"seat_id": winner, "amount": share, "showdown": True}))
            if remainder:
                self.carryover_chips += remainder
                state.events.append(PokerEvent("odd_chip_carryover", {"amount": remainder}))
        self._finish_hand(state)

    def _finish_hand(self, state: HandState) -> None:
        state.is_terminal = True
        state.current_actor = None
        for player in state.players:
            self.stacks[player.seat_id] = player.stack
        state.events.append(PokerEvent("hand_finished", {"stacks": list(self.stacks), "carryover_chips": self.carryover_chips}))

    def _build_pots(self, state: HandState) -> list[Pot]:
        contributions = {seat: amount for seat, amount in state.total_committed.items() if amount > 0}
        if not contributions and state.carryover_in_pot:
            return [Pot(state.carryover_in_pot, tuple(live_seats(state)))]
        pots: list[Pot] = []
        previous = 0
        for level in sorted(set(contributions.values())):
            contributors = [seat for seat, amount in contributions.items() if amount >= level]
            amount = (level - previous) * len(contributors)
            eligible = tuple(seat for seat in contributors if not state.player_by_seat(seat).folded)
            if amount and eligible:
                pots.append(Pot(amount, eligible))
            previous = level
        if state.carryover_in_pot:
            if pots:
                first = pots[0]
                pots[0] = Pot(first.amount + state.carryover_in_pot, first.eligible_seats)
            else:
                pots.append(Pot(state.carryover_in_pot, tuple(live_seats(state))))
        return pots

    def _append_snapshot(self, state: HandState) -> None:
        state.events.append(PokerEvent("snapshot", {"state": state.to_snapshot()}))

    def _blind_seats(self, button: int) -> tuple[int, int]:
        if self.config.seats == 2:
            return button, self._next_seat(button)
        small_blind = self._next_seat(button)
        return small_blind, self._next_seat(small_blind)

    def _first_preflop_actor(self, small_blind_seat: int, big_blind_seat: int) -> int:
        if self.config.seats == 2:
            return small_blind_seat
        return self._next_seat(big_blind_seat)

    def _first_postflop_actor(self, state: HandState) -> int:
        return self._next_actor_from(state, self._next_seat(state.button))

    def _next_actor_after(self, state: HandState, seat_id: int) -> int:
        return self._next_actor_from(state, self._next_seat(seat_id))

    def _next_actor_from(self, state: HandState, start: int) -> int:
        seat = start
        for _ in range(state.config.seats):
            player = state.player_by_seat(seat)
            needs_action = (
                not player.folded
                and not player.all_in
                and (state.committed_this_street[seat] != state.current_bet or seat not in state.acted_this_round)
            )
            if needs_action:
                return seat
            seat = self._next_seat(seat)
        raise RuntimeError("No next actor found despite incomplete betting round")

    def _next_seat(self, seat_id: int) -> int:
        return (seat_id + 1) % self.config.seats
