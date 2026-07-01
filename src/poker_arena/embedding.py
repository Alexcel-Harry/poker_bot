from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
import random
from typing import Iterable, Mapping, Sequence

from poker_arena.events import PokerEvent


ACTION_INDEX = {
    "fold": 0,
    "check": 1,
    "call": 2,
    "raise_to": 3,
}
STREET_INDEX = {
    "preflop": 0,
    "flop": 1,
    "turn": 2,
    "river": 3,
    "showdown": 4,
}


class TrajectoryEncoder:
    """Deterministic numeric trajectory encoder for hand histories."""

    dimension = 14

    def event_vector(
        self,
        event: PokerEvent | Mapping[str, object] | EventContext,
        pot_before: int = 0,
        stack_before: int = 1,
        street: str = "preflop",
    ) -> list[float]:
        if isinstance(event, EventContext):
            pot_before = event.pot_before
            stack_before = event.actor_stack_before
            street = event.street_before
            raw = event.event.to_dict() if isinstance(event.event, PokerEvent) else event.event
        else:
            raw = event.to_dict() if isinstance(event, PokerEvent) else event
        event_type = str(raw.get("event_type"))
        data = raw.get("data", {})
        if not isinstance(data, Mapping):
            data = {}
        vector = [0.0] * self.dimension

        if event_type == "action":
            action = data.get("action", {})
            if isinstance(action, Mapping):
                action_type = str(action.get("type"))
                if action_type in ACTION_INDEX:
                    vector[ACTION_INDEX[action_type]] = 1.0
                total = action.get("total")
                if isinstance(total, int):
                    vector[4] = total / max(1.0, float(pot_before + stack_before))
                    vector[5] = total / max(1.0, float(stack_before))
                    vector[6] = total / float(pot_before) if pot_before > 0 else 0.0
            seat_id = data.get("seat_id")
            if isinstance(seat_id, int):
                vector[7] = seat_id / 8.0
        elif event_type in {"small_blind", "big_blind", "pot_awarded"}:
            amount = data.get("amount")
            if isinstance(amount, int):
                vector[8] = amount / max(1.0, float(pot_before + stack_before))
        elif event_type == "street_dealt":
            street = str(data.get("street", street))

        street_pos = STREET_INDEX.get(street, 0)
        vector[9 + street_pos] = 1.0
        return vector

    def encode_events(self, events: Iterable[PokerEvent | Mapping[str, object]]) -> list[float]:
        total = [0.0] * self.dimension
        count = 0
        for context in EventContextBuilder().build(events):
            vector = self.event_vector(context)
            total = [left + right for left, right in zip(total, vector)]
            count += 1
        if count == 0:
            return total
        return [value / count for value in total]

    @staticmethod
    def distance(first: Sequence[float], second: Sequence[float]) -> float:
        if len(first) != len(second):
            raise ValueError("Vectors must have the same dimension")
        return sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))


@dataclass(frozen=True)
class EventContext:
    event: PokerEvent | Mapping[str, object]
    event_type: str
    street_before: str
    pot_before: int
    current_bet_before: int
    stacks_before: tuple[int, ...]
    actor_stack_before: int


class EventContextBuilder:
    """Reconstructs pot/stack context from event snapshots."""

    def __init__(self, default_stack: int = 2000) -> None:
        self.default_stack = default_stack

    def build(self, events: Iterable[PokerEvent | Mapping[str, object]]) -> list[EventContext]:
        contexts: list[EventContext] = []
        street = "preflop"
        pot = 0
        current_bet = 0
        stacks: tuple[int, ...] = ()
        current_actor: int | None = None
        street_committed: dict[int, int] = {}
        for event in events:
            raw = event.to_dict() if isinstance(event, PokerEvent) else event
            event_type = str(raw.get("event_type"))
            data = raw.get("data", {})
            if not isinstance(data, Mapping):
                data = {}

            if event_type == "snapshot":
                state = data.get("state")
                if isinstance(state, Mapping):
                    street = str(state.get("street", street))
                    current_bet = int(state.get("current_bet", current_bet) or 0)
                    current_actor = int(state["current_actor"]) if state.get("current_actor") is not None else None
                    players = state.get("players", [])
                    if isinstance(players, list):
                        stacks = tuple(int(player.get("stack", self.default_stack)) for player in players if isinstance(player, Mapping))
                    total_committed = state.get("total_committed", {})
                    carryover = int(state.get("carryover_in_pot", 0) or 0)
                    if isinstance(total_committed, Mapping):
                        pot = carryover + sum(int(value) for value in total_committed.values())
                    committed_this_street = state.get("committed_this_street", {})
                    if isinstance(committed_this_street, Mapping):
                        street_committed = {int(key): int(value) for key, value in committed_this_street.items()}
                continue

            seat_id = data.get("seat_id")
            actor = int(seat_id) if isinstance(seat_id, int) else current_actor
            if actor is not None and 0 <= actor < len(stacks):
                actor_stack = stacks[actor]
            else:
                actor_stack = self.default_stack
            contexts.append(
                EventContext(
                    event=event,
                    event_type=event_type,
                    street_before=street,
                    pot_before=pot,
                    current_bet_before=current_bet,
                    stacks_before=stacks,
                    actor_stack_before=actor_stack,
                )
            )

            if event_type == "hand_started":
                pot = int(data.get("carryover_in_pot", pot) or 0)
                street = "preflop"
                current_bet = 0
            elif event_type in {"small_blind", "big_blind"} and isinstance(data.get("amount"), int):
                amount = int(data["amount"])
                pot += amount
                if actor is not None:
                    stacks = self._adjust_stack(stacks, actor, -amount)
                    street_committed[actor] = street_committed.get(actor, 0) + amount
                current_bet = max(current_bet, amount)
            elif event_type == "action":
                action = data.get("action", {})
                if actor is not None and isinstance(action, Mapping):
                    committed = street_committed.get(actor, 0)
                    action_type = str(action.get("type"))
                    paid = 0
                    if action_type == "call":
                        paid = max(0, current_bet - committed)
                    elif action_type == "raise_to" and isinstance(action.get("total"), int):
                        total = int(action["total"])
                        paid = max(0, total - committed)
                        current_bet = max(current_bet, total)
                    if paid:
                        pot += paid
                        stacks = self._adjust_stack(stacks, actor, -paid)
                        street_committed[actor] = committed + paid
            elif event_type == "street_dealt":
                street = str(data.get("street", street))
                current_bet = 0
                street_committed = {}
            elif event_type == "pot_awarded" and actor is not None and isinstance(data.get("amount"), int):
                amount = int(data["amount"])
                pot = max(0, pot - amount)
                stacks = self._adjust_stack(stacks, actor, amount)
        return contexts

    def _adjust_stack(self, stacks: tuple[int, ...], seat_id: int, delta: int) -> tuple[int, ...]:
        values = list(stacks)
        if seat_id >= len(values):
            values.extend([self.default_stack] * (seat_id + 1 - len(values)))
        values[seat_id] = max(0, values[seat_id] + delta)
        return tuple(values)


class TrainableTrajectoryEncoder:
    """Small pure-Python linear autoencoder over trajectory vectors."""

    def __init__(self, embedding_dim: int = 8, random_seed: int | None = None) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.base_encoder = TrajectoryEncoder()
        self.embedding_dim = embedding_dim
        self.input_dim = self.base_encoder.dimension
        rng = random.Random(random_seed)
        self.encoder_weights = [
            [rng.uniform(-0.08, 0.08) for _ in range(embedding_dim)]
            for _ in range(self.input_dim)
        ]
        self.decoder_weights = [
            [rng.uniform(-0.08, 0.08) for _ in range(self.input_dim)]
            for _ in range(embedding_dim)
        ]
        self.decoder_bias = [0.0 for _ in range(self.input_dim)]

    def transform(self, events: Iterable[PokerEvent | Mapping[str, object]]) -> list[float]:
        vector = self.base_encoder.encode_events(events)
        return self._encode_vector(vector)

    def fit(
        self,
        trajectories: Iterable[Iterable[PokerEvent | Mapping[str, object]]],
        epochs: int = 20,
        learning_rate: float = 0.03,
    ) -> list[float]:
        vectors = [self.base_encoder.encode_events(events) for events in trajectories]
        history: list[float] = []
        if not vectors:
            return history
        for _ in range(epochs):
            for vector in vectors:
                embedding = self._encode_vector(vector)
                reconstruction = self._decode_vector(embedding)
                errors = [reconstruction[index] - vector[index] for index in range(self.input_dim)]

                decoder_before = [row[:] for row in self.decoder_weights]
                for hidden in range(self.embedding_dim):
                    for index in range(self.input_dim):
                        gradient = 2.0 * errors[index] * embedding[hidden] / self.input_dim
                        self.decoder_weights[hidden][index] -= learning_rate * gradient
                for index in range(self.input_dim):
                    self.decoder_bias[index] -= learning_rate * 2.0 * errors[index] / self.input_dim

                hidden_errors = []
                for hidden in range(self.embedding_dim):
                    hidden_errors.append(
                        sum(2.0 * errors[index] * decoder_before[hidden][index] / self.input_dim for index in range(self.input_dim))
                    )
                for index in range(self.input_dim):
                    for hidden in range(self.embedding_dim):
                        self.encoder_weights[index][hidden] -= learning_rate * hidden_errors[hidden] * vector[index]
            history.append(self.reconstruction_loss_from_vectors(vectors))
        return history

    def reconstruction_loss(self, trajectories: Iterable[Iterable[PokerEvent | Mapping[str, object]]]) -> float:
        vectors = [self.base_encoder.encode_events(events) for events in trajectories]
        return self.reconstruction_loss_from_vectors(vectors)

    def reconstruction_loss_from_vectors(self, vectors: Sequence[Sequence[float]]) -> float:
        if not vectors:
            return 0.0
        total = 0.0
        for vector in vectors:
            embedding = self._encode_vector(vector)
            reconstruction = self._decode_vector(embedding)
            total += sum((reconstruction[index] - vector[index]) ** 2 for index in range(self.input_dim)) / self.input_dim
        return total / len(vectors)

    def _encode_vector(self, vector: Sequence[float]) -> list[float]:
        return [
            sum(vector[index] * self.encoder_weights[index][hidden] for index in range(self.input_dim))
            for hidden in range(self.embedding_dim)
        ]

    def _decode_vector(self, embedding: Sequence[float]) -> list[float]:
        return [
            self.decoder_bias[index] + sum(embedding[hidden] * self.decoder_weights[hidden][index] for hidden in range(self.embedding_dim))
            for index in range(self.input_dim)
        ]

    @staticmethod
    def distance(first: Sequence[float], second: Sequence[float]) -> float:
        return TrajectoryEncoder.distance(first, second)
