from __future__ import annotations

from poker_arena.actions import LegalActions
from poker_arena.state import HandState


def legal_actions_for(state: HandState, seat_id: int) -> LegalActions:
    if state.is_terminal or state.current_actor is None:
        raise ValueError("Hand is terminal")
    if seat_id != state.current_actor:
        raise ValueError(f"Seat {seat_id} is not the current actor")

    player = state.player_by_seat(seat_id)
    if player.folded or player.all_in:
        raise ValueError(f"Seat {seat_id} cannot act")

    actor_commitment = state.committed_this_street[seat_id]
    call_amount = max(0, state.current_bet - actor_commitment)
    max_raise_to = actor_commitment + player.stack
    can_raise = max_raise_to > state.current_bet

    min_full_raise_to: int | None
    if not can_raise:
        min_full_raise_to = None
        max_total = None
    elif state.current_bet == 0:
        min_full_raise_to = min(state.config.big_blind, max_raise_to)
        max_total = max_raise_to
    else:
        full_minimum = state.current_bet + state.last_full_raise
        min_full_raise_to = min(full_minimum, max_raise_to) if max_raise_to < full_minimum else full_minimum
        max_total = max_raise_to

    return LegalActions(
        can_fold=True,
        can_check=call_amount == 0,
        can_call=call_amount > 0 and player.stack > 0,
        call_amount=min(call_amount, player.stack),
        min_raise_to=min_full_raise_to,
        max_raise_to=max_total,
        current_bet=state.current_bet,
        actor_commitment=actor_commitment,
    )


def active_not_all_in_seats(state: HandState) -> list[int]:
    return [player.seat_id for player in state.players if not player.folded and not player.all_in]


def live_seats(state: HandState) -> list[int]:
    return [player.seat_id for player in state.players if not player.folded]


def betting_round_complete(state: HandState) -> bool:
    actors = active_not_all_in_seats(state)
    if not actors:
        return True
    for seat_id in actors:
        if state.committed_this_street[seat_id] != state.current_bet:
            return False
        if seat_id not in state.acted_this_round:
            return False
    return True
